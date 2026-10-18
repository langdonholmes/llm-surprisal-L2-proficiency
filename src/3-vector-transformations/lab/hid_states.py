"""Hidden-state lab helper: all-layer states for the observed and counterfactual target.

For each bench instance (target subword index k, observed word spanning
subwords k..kn-1 where kn is the next word's first subword):

    obs window   ids[k-7 : k+1]                      -> h_obs[L]   at last position
    cf window    same with ids[k] := top-1 prediction -> h_cf[L]   at last position
    next (obs)   ids[kn-7 : kn+1]                    -> next-word logprob, distribution
    next (cf)    ids[..k-1] + [top1] + ids[kn : kn+1], last 8 -> same, after the
                 model's own expected token instead of the writer's word

Stored per instance (layers 0..16, fp16), row-aligned with the bench:
    data/study3/lab/hidden/states{SUFFIX}.npz
        h_obs [n, 17, 2048], h_cf [n, 17, 2048],
        n_lp_obs, n_lp_cf   log2 p(actual next subword) after obs / after cf
        n_kl                KL(p_next_obs || p_next_cf) in bits
        n_js                Jensen-Shannon divergence in bits

No benchmark labels are read. Cost: four 8-token forward passes per token.

    CUDA_VISIBLE_DEVICES=1 python src/3-vector-transformations/lab/hid_states.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB, SUFFIX  # noqa: E402
from lab.lm import LM, LOG2E  # noqa: E402

HID = LAB / "hidden"
BATCH = 256


@torch.no_grad()
def forward(lm: LM, windows: list[np.ndarray]):
    """Left-padded forward; returns all-layer hidden at the last position and
    log-probs (fp32) predicting the position after the last."""
    L = max(len(w) for w in windows)
    ids = torch.zeros((len(windows), L), dtype=torch.long)
    att = torch.zeros_like(ids)
    pos = torch.zeros_like(ids)
    for i, w in enumerate(windows):
        ids[i, L - len(w):] = torch.as_tensor(w)
        att[i, L - len(w):] = 1
        pos[i, L - len(w):] = torch.arange(len(w))
    out = lm.model(input_ids=ids.to(lm.device), attention_mask=att.to(lm.device),
                   position_ids=pos.to(lm.device), output_hidden_states=True)
    hs = torch.stack([h[:, -1] for h in out.hidden_states], 1).half().cpu().numpy()
    return hs, out.logits


def main():
    HID.mkdir(parents=True, exist_ok=True)
    bench = pd.read_parquet(LAB / f"bench{SUFFIX}.parquet")
    base = np.load(LAB / f"base{SUFFIX}.npz")
    top1 = base["t_topk_ids"][:, 0]
    lm = LM()
    n = len(bench)
    h_obs = np.zeros((n, 17, 2048), np.float16)
    h_cf = np.zeros((n, 17, 2048), np.float16)
    lp_obs = np.full(n, np.nan, np.float32)
    lp_cf = np.full(n, np.nan, np.float32)
    kl = np.full(n, np.nan, np.float32)
    js = np.full(n, np.nan, np.float32)

    jobs = []  # (i, obs_win, cf_win, next_obs_win | None, next_cf_win | None)
    for i, r in enumerate(bench.itertuples()):
        try:
            ids, k = lm.locate(r.text, r.target_char)
        except ValueError:
            continue
        if k == 0:
            continue
        w = lm.window(ids, k)
        cf = w.copy()
        cf[-1] = top1[i]
        no = nc = None
        if r.next_char is not None and not pd.isna(r.next_char):
            try:
                _, kn = lm.locate(r.text, int(r.next_char))
                no = ids[max(0, kn - 7):kn + 1]
                nc = np.concatenate([ids[:k], [top1[i]], ids[kn:kn + 1]])[-8:]
            except ValueError:
                pass
        jobs.append((i, w, cf, no, nc))

    for a in range(0, len(jobs), BATCH):
        J = jobs[a:a + BATCH]
        idx = np.array([j[0] for j in J])
        h_obs[idx] = forward(lm, [j[1] for j in J])[0]
        h_cf[idx] = forward(lm, [j[2] for j in J])[0]
        Jn = [j for j in J if j[3] is not None]
        if Jn:
            ni = np.array([j[0] for j in Jn])
            _, lo = forward(lm, [j[3][:-1] for j in Jn])  # predicts the next word
            _, lc = forward(lm, [j[4][:-1] for j in Jn])
            po = torch.log_softmax(lo[:, -1].float(), -1)
            pc = torch.log_softmax(lc[:, -1].float(), -1)
            tgt = torch.as_tensor([int(j[3][-1]) for j in Jn], device=lm.device)
            lp_obs[ni] = (po.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy()
            lp_cf[ni] = (pc.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy()
            kl[ni] = ((po.exp() * (po - pc)).sum(-1) * LOG2E).cpu().numpy()
            m = torch.log((po.exp() + pc.exp()) / 2)
            js[ni] = (0.5 * ((po.exp() * (po - m)).sum(-1) + (pc.exp() * (pc - m)).sum(-1))
                      * LOG2E).cpu().numpy()
        if a // BATCH % 10 == 0:
            print(f"{a}/{len(jobs)}", flush=True)
    np.savez(HID / f"states{SUFFIX}.npz", h_obs=h_obs, h_cf=h_cf, n_lp_obs=lp_obs,
             n_lp_cf=lp_cf, n_kl=kl, n_js=js)


if __name__ == "__main__":
    main()
