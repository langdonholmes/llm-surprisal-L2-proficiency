"""Context & capacity lab helper — rescore the bench target word under other views.

For every bench instance, the observed word's FIRST subword is scored under:

    olmo1b_w{W}   OLMo-2 1B (Study 1 revision), causal, left window W subwords
    olmo7b_w{W}   OLMo-2 7B (allenai/OLMo-2-1124-7B), causal, left window W
    mlm           ModernBERT-large, all subwords of the word masked, CTX subwords
                  of real text on each side: p(first subword | left + right)

Outputs per view (row-aligned with the bench; NaN where the word does not start
on a subword boundary):
    data/study3/lab/ctx/{view}{SUFFIX}.npz   surprisal (bits), entropy (bits)

Only windows ending at (or centred on) the bench positions are scored; essays
are tokenized once. No gold labels are read.

    CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/lab/ctx_score.py olmo1b_w64 mlm
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB, SUFFIX  # noqa: E402

CTX = LAB / "ctx"
LOG2E = 1.0 / np.log(2.0)
VIEWS = {
    "olmo1b_w8": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", 8),
    "olmo1b_w64": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", 64),
    "olmo7b_w8": ("allenai/OLMo-2-1124-7B", None, 8),
    "olmo7b_w64": ("allenai/OLMo-2-1124-7B", None, 64),
}
MLM_ID = "answerdotai/ModernBERT-large"
MLM_CTX = 32


def locate_all(tok, bench):
    """(ids per essay, target subword index, word end subword index) per instance."""
    enc = {}
    out_k, out_e = np.full(len(bench), -1), np.full(len(bench), -1)
    for tid, g in bench.groupby("text_id"):
        text = g.text.iat[0]
        e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.array(e.input_ids)
        starts = np.array([s + len(text[s:t]) - len(text[s:t].lstrip()) for s, t in e.offset_mapping])
        enc[tid] = ids
        pos = {int(c): k for k, c in enumerate(starts)}
        for r in g.itertuples():
            k = pos.get(int(r.target_char), -1)
            if k <= 0:
                continue
            end_char = r.target_char + len(r.observed)
            j = k + 1
            while j < len(ids) and starts[j] < end_char:
                j += 1
            out_k[r.inst], out_e[r.inst] = k, j
    return enc, out_k, out_e


@torch.no_grad()
def score_causal(view, bench, batch=None):
    hf, rev, W = VIEWS[view]
    batch = batch or max(16, 2048 // W)  # keeps full-sequence logits under ~0.5 GB
    tok = AutoTokenizer.from_pretrained(hf, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(hf, revision=rev, dtype=torch.bfloat16).cuda().eval()
    enc, K, _ = locate_all(tok, bench)
    surp = np.full(len(bench), np.nan, dtype=np.float32)
    ent = np.full(len(bench), np.nan, dtype=np.float32)
    todo = [i for i in range(len(bench)) if K[i] > 0]
    # sort by window length to limit padding
    lens = {i: min(W, K[i] + 1) for i in todo}
    todo.sort(key=lambda i: lens[i])
    pad = tok.pad_token_id or 0
    for a in range(0, len(todo), batch):
        idx = todo[a:a + batch]
        wins = [enc[bench.text_id.iat[i]][max(0, K[i] - W + 1):K[i] + 1] for i in idx]
        L = max(len(w) for w in wins)
        ids = torch.full((len(wins), L), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        pos = torch.zeros_like(ids)
        for j, w in enumerate(wins):
            ids[j, L - len(w):] = torch.as_tensor(w)
            att[j, L - len(w):] = 1
            pos[j, L - len(w):] = torch.arange(len(w))
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda()).logits[:, -2].float()
        lp = torch.log_softmax(logits, -1)
        tgt = ids[:, -1].cuda()
        surp[idx] = (-lp.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy()
        ent[idx] = (-(lp.exp() * lp).sum(-1) * LOG2E).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return surp, ent


@torch.no_grad()
def score_mlm(bench, batch=64):
    tok = AutoTokenizer.from_pretrained(MLM_ID)
    model = AutoModelForMaskedLM.from_pretrained(MLM_ID, dtype=torch.bfloat16).cuda().eval()
    enc, K, E = locate_all(tok, bench)
    surp = np.full(len(bench), np.nan, dtype=np.float32)
    ent = np.full(len(bench), np.nan, dtype=np.float32)
    todo = [i for i in range(len(bench)) if K[i] > 0 and E[i] > K[i]]
    for a in range(0, len(todo), batch):
        idx = todo[a:a + batch]
        seqs, mpos, tgts = [], [], []
        for i in idx:
            ids = enc[bench.text_id.iat[i]]
            k, e = K[i], E[i]
            left = ids[max(0, k - MLM_CTX):k]
            right = ids[e:e + MLM_CTX]
            word = [tok.mask_token_id] * (e - k)
            s = [tok.cls_token_id] + list(left) + word + list(right) + [tok.sep_token_id]
            seqs.append(s)
            mpos.append(1 + len(left))
            tgts.append(int(ids[k]))
        L = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids_t)
        for j, s in enumerate(seqs):
            ids_t[j, :len(s)] = torch.as_tensor(s)
            att[j, :len(s)] = 1
        logits = model(input_ids=ids_t.cuda(), attention_mask=att.cuda()).logits
        lg = logits[torch.arange(len(seqs)), torch.as_tensor(mpos)].float()
        lp = torch.log_softmax(lg, -1)
        t = torch.as_tensor(tgts).cuda()
        surp[idx] = (-lp.gather(1, t[:, None])[:, 0] * LOG2E).cpu().numpy()
        ent[idx] = (-(lp.exp() * lp).sum(-1) * LOG2E).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return surp, ent


def main():
    CTX.mkdir(parents=True, exist_ok=True)
    bench = pd.read_parquet(LAB / f"bench{SUFFIX}.parquet")
    for view in sys.argv[1:]:
        path = CTX / f"{view}{SUFFIX}.npz"
        if path.exists():
            print(view, "cached")
            continue
        t0 = time.time()
        s, e = score_mlm(bench) if view == "mlm" else score_causal(view, bench)
        np.savez(path, surprisal=s, entropy=e)
        dt = time.time() - t0
        print(f"{view}: {np.isfinite(s).sum()} scored, {dt:.0f}s = {1000 * dt / len(bench):.2f} ms/token", flush=True)


if __name__ == "__main__":
    main()
