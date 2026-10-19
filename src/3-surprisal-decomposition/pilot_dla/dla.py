"""Pilot: direct logit attribution (DLA) of W&I high-surprisal tokens.

Stratified sample of the `wi` bench (level x {ERR, OK}, up to 1,000 per cell,
POST excluded), scored in fp32 with Study 3's window. For each token, the
logit gap logit(obs) - E_p[logit] is split exactly over 273 residual-stream
writes at the prediction position: embedding, 16 x 16 attention heads, 16 MLPs
(see NOTES.md section 1). Final and post-attention RMSNorm scales are frozen at
their actual values, which makes the split exact.

Outputs (data/study3/pilot_dla/):
    sample.parquet    one row per token: bench row, gold, level, essay, S, H, checks
    contrib.npy       (n, 273) float32 contributions in nats; columns in COMPONENTS
    contrib_obs.npy   same, post hoc: splits logit(obs) - mean_v logit_v instead
    components.json   component names

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/3-surprisal-decomposition/pilot_dla/dla.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.bench import CACHE, S3  # noqa: E402
from lib.lm import HF_ID, LOG2E, REVISION, WINDOW  # noqa: E402

OUT = S3 / "pilot_dla"
SEED = 20261019
PER_CELL = 1000
BATCH = 128
N_LAYERS, N_HEADS, HEAD_DIM = 16, 16, 128
COMPONENTS = (["emb"] + [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
              + [f"L{l}MLP" for l in range(N_LAYERS)])


def sample() -> pd.DataFrame:
    bench = pd.read_parquet(CACHE / "bench_wi.parquet")
    meta = pd.read_parquet(CACHE / "base_meta_wi.parquet")
    base = np.load(CACHE / "base_wi.npz")
    gold = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet",
                           columns=["row_id", "level", "gold", "surprisal_bits", "entropy_bits"])
    att = pd.read_parquet(CACHE / "attestation_wi.parquet", columns=["a_log_uni"])
    h = bench.merge(gold, on="row_id", how="left")
    h["k"] = meta.k.to_numpy()
    h["ok_scored"] = meta.has_target.to_numpy()
    h["stored_S_bench"] = base["t_surprisal"]
    h["stored_H_bench"] = base["t_entropy"]
    h["a_log_uni"] = att.a_log_uni.to_numpy()
    h = h[h.ok_scored & (h.gold != "POST")].copy()
    h["cls"] = np.where(h.gold == "OK", "OK", "ERR")
    rng = np.random.default_rng(SEED)
    parts = []
    for (lvl, cls), d in h.groupby(["level", "cls"]):
        n = min(PER_CELL, len(d))
        parts.append(d.iloc[np.sort(rng.choice(len(d), n, replace=False))])
    return pd.concat(parts).reset_index(drop=True)


class DLA:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_ID, revision=REVISION, dtype=torch.float32, attn_implementation="eager").cuda().eval()
        m = self.model.model
        self.W_U = self.model.lm_head.weight.data          # (V, D)
        assert self.model.lm_head.bias is None
        self.w_f = m.norm.weight.data
        self.eps = m.norm.variance_epsilon
        self.cap = {}
        hooks = []
        hooks.append(m.embed_tokens.register_forward_hook(lambda mod, i, o: self.cap.__setitem__("emb", o[:, -2])))
        hooks.append(m.norm.register_forward_pre_hook(lambda mod, i: self.cap.__setitem__("hL", i[0][:, -2])))
        for l, layer in enumerate(m.layers):
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda mod, i, l=l: self.cap.__setitem__(f"z{l}", i[0][:, -2])))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(
                lambda mod, i, o, l=l: self.cap.update({f"ain{l}": i[0][:, -2], f"aout{l}": o[:, -2]})))
            hooks.append(layer.post_feedforward_layernorm.register_forward_hook(
                lambda mod, i, o, l=l: self.cap.__setitem__(f"m{l}", o[:, -2])))
        self.hooks = hooks
        self.ablate = None  # (layer, head, vector) for patch.py

    def locate(self, text: str, char: int):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        for k, (s, e) in enumerate(enc.offset_mapping):
            piece = text[s:e]
            if s + len(piece) - len(piece.lstrip()) == char:
                return np.array(enc.input_ids), k
        raise ValueError(char)

    def forward(self, windows):
        L = max(len(w) for w in windows)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(windows), L), pad, dtype=torch.long)
        att = torch.zeros((len(windows), L), dtype=torch.long)
        pos = torch.zeros((len(windows), L), dtype=torch.long)
        for i, w in enumerate(windows):
            ids[i, L - len(w):] = torch.as_tensor(w)
            att[i, L - len(w):] = 1
            pos[i, L - len(w):] = torch.arange(len(w))
        out = self.model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda())
        return ids[:, -1].cuda(), out.logits[:, -2].float()

    @torch.no_grad()
    def run(self, windows):
        obs, logits = self.forward(windows)
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        S = -logp.gather(1, obs[:, None])[:, 0]
        H = -(p * logp).sum(-1)
        gap = logits.gather(1, obs[:, None])[:, 0] - (p * logits).sum(-1)
        out = {}
        hL = self.cap["hL"]
        rL = torch.sqrt(hL.pow(2).mean(-1, keepdim=True) + self.eps)
        # main: d = W_U[obs] - E_p W_U (sums to the logit gap);
        # post hoc: d = W_U[obs] - mean_v W_U (observed word's logit vs the vocabulary mean)
        for key, d in [("contrib", self.W_U[obs] - p @ self.W_U),
                       ("contrib_obs", self.W_U[obs] - self.W_U.mean(0))]:
            C, recon = self.attribute(d * self.w_f / rL)
            out[key] = C
        C = out["contrib"]
        resid_err = (recon - hL).abs().max(-1).values / hL.abs().max(-1).values
        top1 = p.max(-1).values
        obs_c = logits.gather(1, obs[:, None])[:, 0] - logits.mean(-1)
        return {"contrib": C.cpu().numpy().astype(np.float32),
                "contrib_obs": out["contrib_obs"].cpu().numpy().astype(np.float32),
                "obs_centered_logit": obs_c.cpu().numpy(), "obs_sum_err": (out["contrib_obs"].sum(1) - obs_c).cpu().numpy(),
                "S_nats": S.cpu().numpy(), "H_nats": H.cpu().numpy(), "gap": gap.cpu().numpy(),
                "sum_err": (C.sum(1) - gap).cpu().numpy(), "resid_rel_err": resid_err.cpu().numpy(),
                "log_top1": torch.log(top1).cpu().numpy(), "obs_id": obs.cpu().numpy(),
                "top1_id": p.argmax(-1).cpu().numpy()}

    def attribute(self, g):
        """Contributions of the 273 writes along g = d * w_f / r_L (B, D)."""
        cols = [(g * self.cap["emb"]).sum(-1, keepdim=True)]
        recon = self.cap["emb"].clone()
        mlps = []
        layers = self.model.model.layers
        for l, layer in enumerate(layers):
            ain = self.cap[f"ain{l}"]
            ra = torch.sqrt(ain.pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.variance_epsilon)
            u = g * layer.post_attention_layernorm.weight / ra  # (B, D)
            W_O = layer.self_attn.o_proj.weight                # (D, H*hd)
            zc = (u @ W_O) * self.cap[f"z{l}"]                 # (B, H*hd)
            cols.append(zc.view(-1, N_HEADS, HEAD_DIM).sum(-1))
            mlps.append((g * self.cap[f"m{l}"]).sum(-1, keepdim=True))
            recon += self.cap[f"aout{l}"] + self.cap[f"m{l}"]
        return torch.cat(cols + mlps, 1), recon


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = sample()
    print(s.groupby(["level", "cls"]).size().unstack(), flush=True)
    dla = DLA()
    wins = []
    for r in s.itertuples():
        ids, k = dla.locate(r.text, r.target_char)
        assert k == r.k
        wins.append(ids[max(0, k - WINDOW + 1):k + 1])
    t0 = time.time()
    res = [dla.run(wins[a:a + BATCH]) for a in range(0, len(wins), BATCH)]
    print(f"forward+DLA {time.time() - t0:.1f}s for {len(wins)} tokens", flush=True)
    out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    np.save(OUT / "contrib.npy", out.pop("contrib"))
    np.save(OUT / "contrib_obs.npy", out.pop("contrib_obs"))
    for k, v in out.items():
        s[k] = v
    s["S_bits"] = s.S_nats * LOG2E
    s["H_bits"] = s.H_nats * LOG2E
    s["identity_err"] = s.gap - (s.H_nats - s.S_nats)
    s.drop(columns=["text"]).to_parquet(OUT / "sample.parquet", index=False)
    (OUT / "components.json").write_text(json.dumps(COMPONENTS))
    print("max |sum - gap| nats:", np.abs(s.sum_err).max(),
          "| max rel residual recon err:", s.resid_rel_err.max(),
          "| max |gap - (H - S)|:", np.abs(s.identity_err).max())
    print("max |sum contrib_obs - centered obs logit|:", np.abs(s.obs_sum_err).max())
    print("fp32 S vs stored bench S (bits): mean abs diff", np.abs(s.S_bits - s.stored_S_bench).mean(),
          "max", np.abs(s.S_bits - s.stored_S_bench).max(),
          "r", np.corrcoef(s.S_bits, s.stored_S_bench)[0, 1])
    print("stored bench S vs stage-1 gold S: max abs diff", np.abs(s.stored_S_bench - s.surprisal_bits).max())
    print("obs id matches top1 share:", (s.obs_id == s.top1_id).mean())


if __name__ == "__main__":
    main()
