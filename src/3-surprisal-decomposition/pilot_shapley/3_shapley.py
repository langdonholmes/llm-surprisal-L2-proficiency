"""Pilot step 3 (GPU) — exact Shapley values and leave-one-out over context players.

For each target, all 2^p subsets of its p context players are scored
(engine.py), in fp32 with TF32 off: bf16 values depend on batch composition
(median .02, max .5 bits between batch 32 and batch 4000), which is the size of
small attributions. Shapley values are W @ v (exact; no sampling), LOO_i =
v(full) - v(full minus i).

Runs:
    main          sample.parquet, scheme delete, word players (sharded)
    sens_mask     sensitivity subsample (1,500 cand + 1,500 ctrl), scheme mask
    sens_replace  same subsample, scheme replace
    sens_delete   same subsample, scheme delete (bit-identical rerun of main rows)
    sens_subword  cand+ctrl with fewer than 7 players (up to 2,000), one player
                  per subword, scheme delete
    check         equivalence checks (4D attention mask vs gapped positions,
                  bf16 batch-32 reproduction of stored surprisal)

Outputs: data/study3/pilot_shapley/shapley_<run>[_<shard>].parquet with
    text_id, spacy_idx, set, p, v (2^p list), phi, loo, v_empty, v_full

    CUDA_VISIBLE_DEVICES=0 python .../3_shapley.py main --shard 0 --of 2
    CUDA_VISIBLE_DEVICES=1 python .../3_shapley.py main --shard 1 --of 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, SEED  # noqa: E402
from engine import LOG2E, build_sequences, load, score_all, shapley_matrix, subset_masks, surprisal_batch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sens_subsample(s):
    rng = np.random.default_rng(SEED + 1)
    parts = [s[s.set == k].iloc[rng.choice((s.set == k).sum(), 1500, replace=False)] for k in ["cand", "ctrl"]]
    return pd.concat(parts)


def run(model, t, scheme, subword=False):
    seqs, spans = [], []
    for r in t.itertuples():
        pos = list(range(7)) if subword else r.player_of_sub
        p = 7 if subword else r.n_players
        q = build_sequences(r.window, pos, p, scheme)
        spans.append((len(seqs), len(q), p))
        seqs += q
    log(f"{len(t)} targets, {len(seqs)} sequences")
    v = score_all(model, seqs, batch=2048, log=log)
    rows = []
    for r, (a, n, p) in zip(t.itertuples(), spans):
        vv = v[a:a + n]
        full = 2 ** p - 1
        loo = np.array([vv[full] - vv[full ^ (1 << i)] for i in range(p)])
        rows.append({"text_id": r.text_id, "spacy_idx": r.spacy_idx, "set": r.set, "p": p,
                     "v": vv, "phi": shapley_matrix(p) @ vv, "loo": loo,
                     "v_empty": vv[0], "v_full": vv[full]})
    return pd.DataFrame(rows)


def checks(s):
    """Equivalence checks, written to checks_engine.json."""
    out = {}
    W = np.stack(s.window.to_numpy()[:4000])
    P = np.tile(np.arange(8), (len(W), 1))
    stored = s.surprisal_bits.to_numpy()[:4000]
    mb = load(dtype=torch.bfloat16)
    v = np.concatenate([surprisal_batch(mb, W[i:i + 32], P[i:i + 32]) for i in range(0, len(W), 32)])
    d = np.abs(v - stored)
    out["bf16_batch32_vs_stored"] = {"n": len(d), "median": float(np.median(d)), "share_lt_1e-3": float((d < 1e-3).mean()),
                                     "share_lt_.01": float((d < .01).mean()), "max": float(d.max())}
    v2 = np.concatenate([surprisal_batch(mb, W[i:i + 2048], P[i:i + 2048]) for i in range(0, len(W), 2048)])
    d2 = np.abs(v2 - v)
    out["bf16_batch2048_vs_batch32"] = {"median": float(np.median(d2)), "max": float(d2.max())}
    del mb
    m = load(dtype=torch.float32)
    v3 = np.concatenate([surprisal_batch(m, W[i:i + 2048], P[i:i + 2048]) for i in range(0, len(W), 2048)])
    d3 = np.abs(v3 - stored)
    out["fp32_vs_stored_bf16"] = {"median": float(np.median(d3)), "p99": float(np.quantile(d3, .99)),
                                  "max": float(d3.max()), "share_lt_.05": float((d3 < .05).mean())}
    v4 = surprisal_batch(m, W[:1000], P[:1000] + 5)
    out["fp32_position_shift_5_max"] = float(np.abs(v4 - v3[:1000]).max())
    rng = np.random.default_rng(0)
    eq, gap = [], []
    for j in range(300):
        w = W[j]
        keep = rng.random(7) < .6
        keep[rng.integers(7)] = True
        kp = np.flatnonzero(keep)
        q = kp[-1]
        ids = np.append(w[:q + 1], w[7])
        L = len(ids)
        allow = np.tril(np.ones((L, L), bool))
        rm = np.flatnonzero(~keep[:q + 1])
        allow[:, rm] = False
        allow[rm, rm] = True          # removed rows attend to themselves only (no NaN); nobody reads them
        pos = np.arange(L)
        pos[-1] = q + 1
        mask = torch.zeros(1, 1, L, L)
        mask[0, 0][~torch.as_tensor(allow)] = torch.finfo(torch.float32).min
        with torch.no_grad():
            lg = m(input_ids=torch.as_tensor(ids)[None].cuda(), attention_mask=mask.cuda(),
                   position_ids=torch.as_tensor(pos)[None].cuda()).logits[0, -2]
        a = -torch.log_softmax(lg.float(), -1)[ids[-1]].item() * LOG2E
        ids2 = np.append(w[:7][keep], w[7])
        b = surprisal_batch(m, ids2[None], np.append(kp, q + 1)[None])[0]
        c = surprisal_batch(m, ids2[None], np.arange(len(ids2))[None])[0]
        eq.append(abs(a - b))
        gap.append(abs(b - c))
    out["mask4d_vs_gapped_positions_fp32"] = {"n": 300, "max": float(max(eq))}
    out["gapped_vs_contiguous_positions_fp32"] = {"median": float(np.median(gap)), "max": float(max(gap))}
    (OUT / "checks_engine.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    s = pd.read_parquet(OUT / "sample.parquet")
    t0 = time.time()
    if a.run == "check":
        checks(s)
        return
    model = load(dtype=torch.float32)
    if a.run == "main":
        t = s.iloc[a.shard::a.of]
        df = run(model, t, "delete")
        name = f"shapley_main_{a.shard}"
    elif a.run.startswith("sens_"):
        sub = sens_subsample(s)
        if a.run == "sens_subword":
            c = s[s.set.isin(["cand", "ctrl"]) & (s.n_players < 7)]
            sub = c.iloc[np.random.default_rng(SEED + 2).choice(len(c), min(2000, len(c)), replace=False)]
            df = run(model, sub, "delete", subword=True)
        else:
            df = run(model, sub, a.run[5:])
        name = f"shapley_{a.run}"
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    log(f"{name}: {len(df)} targets in {time.time() - t0:.0f} s")
    (OUT / f"{name}.time.json").write_text(json.dumps({"seconds": time.time() - t0, "n": len(df)}))


if __name__ == "__main__":
    main()
