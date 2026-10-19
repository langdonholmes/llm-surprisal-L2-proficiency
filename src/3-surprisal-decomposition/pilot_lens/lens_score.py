"""Layer-wise surprisal of each sampled W&I high token, logit lens and tuned lens.

Same window as the stored scores (<= 8 subwords ending at the target, the
second-to-last position predicts it). Per token and layer 0..16: surprisal and
entropy (bits) under both lenses.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/3-surprisal-decomposition/pilot_lens/lens_score.py
Writes data/study3/pilot_lens/lens_scores.npz (row-aligned with sample.parquet)
"""

import time

import numpy as np
import pandas as pd
import torch

from lens_common import (LOG2E, N_LAYERS, OUT, WINDOW, TunedLens, hidden_states, lens_logits,
                         load_model, pad_windows)

BATCH = 256


def locate_all(tok, s):
    """Subword windows ending at each sampled target (None if not locatable)."""
    wins = [None] * len(s)
    for tid, g in s.groupby("text_id"):
        text = g.text.iat[0]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.array(enc.input_ids)
        starts = {}
        for k, (a, b) in enumerate(enc.offset_mapping):
            piece = text[a:b]
            starts.setdefault(a + len(piece) - len(piece.lstrip()), k)
        for i, c in zip(g.index, g.target_char):
            k = starts.get(int(c), -1)
            if k >= 1:
                wins[i] = ids[max(0, k - WINDOW + 1):k + 1]
    return wins


@torch.no_grad()
def main():
    s = pd.read_parquet(OUT / "sample.parquet")
    tok, model = load_model()
    lens = TunedLens(model.config.hidden_size).cuda()
    lens.load_state_dict(torch.load(OUT / "tuned_lens.pt"))
    wins = locate_all(tok, s)
    ok = np.array([w is not None for w in wins])
    print(f"located {ok.mean():.4f}", flush=True)
    n, L = len(s), N_LAYERS + 1
    res = {k: np.full((n, L), np.nan, np.float32) for k in ["s_ll", "h_ll", "s_tl", "h_tl"]}
    res["final_surprisal"] = np.full(n, np.nan, np.float32)
    todo = np.flatnonzero(ok)
    t0 = time.time()
    pad = tok.pad_token_id or 0
    for a in range(0, len(todo), BATCH):
        idx = todo[a:a + BATCH]
        ids, att, pos = pad_windows([wins[i] for i in idx], pad)
        ids, att, pos = ids.cuda(), att.cuda(), pos.cuda()
        hs, logits = hidden_states(model, ids, att, pos)
        tgt = ids[:, -1]
        lp = torch.log_softmax(logits[:, -2].float(), -1)
        res["final_surprisal"][idx] = (-lp.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy()
        for layer in range(L):
            h = hs[layer][:, -2]
            for name, ln in [("ll", None), ("tl", lens)]:
                lp = torch.log_softmax(lens_logits(model, h, layer, ln), -1)
                res[f"s_{name}"][idx, layer] = (-lp.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy()
                res[f"h_{name}"][idx, layer] = (-(lp.exp() * lp).sum(-1) * LOG2E).cpu().numpy()
    np.savez(OUT / "lens_scores.npz", **res)
    dt = time.time() - t0
    d = res["s_tl"][:, -1] - s.surprisal_bits.to_numpy()
    print(f"{len(todo)} tokens, {dt:.0f}s; |s_16 - stored| median {np.nanmedian(np.abs(d)):.4f}, "
          f"p99 {np.nanpercentile(np.abs(d), 99):.4f}, max {np.nanmax(np.abs(d)):.4f}")


if __name__ == "__main__":
    main()
