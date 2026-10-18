"""Lab, step 2 — base GPU pass over the benchmark.

For every benchmark instance, scores two positions with Study 1's windowing:
the target (where the edit is observable) and the next word. Stores everything
the CPU-only representation ideas need, row-aligned with bench.parquet:

    data/study3/lab/base.npz
        {pos}_target_id, {pos}_surprisal, {pos}_entropy, {pos}_topk_ids,
        {pos}_topk_probs, {pos}_h_pred_{8,16}, {pos}_h_obs_{8,16}
        for pos in (t, n)   [n rows are zero where there is no next word]
    data/study3/lab/base_meta.parquet   inst, has_target, has_next, k (target subword index)

    python src/3-vector-transformations/lab/extract.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import SUFFIX  # noqa: E402
from lab.lm import LM  # noqa: E402

from util.paths import DATA_DIR  # noqa: E402

LAB = DATA_DIR / "study3" / "lab"
BATCH = 256


def main():
    b = pd.read_parquet(LAB / f"bench{SUFFIX}.parquet")
    # LAB_LAYERS="" skips hidden states (real-corpus runs); default layers 8 and 16
    layers = os.environ.get("LAB_LAYERS", "8,16")
    lm = LM(layers=tuple(int(x) for x in layers.split(",") if x))
    wins = {"t": [], "n": []}
    meta = []
    # Words that do not start on a subword boundary are unscored, as in Study 1.
    for r in b.itertuples():
        try:
            ids, k = lm.locate(r.text, r.target_char)
        except ValueError:
            wins["t"].append(None)
            wins["n"].append(None)
            meta.append({"inst": r.inst, "has_target": False, "has_next": False, "k": -1})
            continue
        wins["t"].append(lm.window(ids, k) if k > 0 else None)
        has_next = r.next_char is not None and not pd.isna(r.next_char)
        if has_next:
            try:
                _, kn = lm.locate(r.text, int(r.next_char))
                wins["n"].append(lm.window(ids, kn))
            except ValueError:
                has_next = False
                wins["n"].append(None)
        else:
            wins["n"].append(None)
        meta.append({"inst": r.inst, "has_target": k > 0, "has_next": has_next, "k": k})
    out = {}
    for pos, ws in wins.items():
        idx = [i for i, w in enumerate(ws) if w is not None]
        chunks = []
        for a in range(0, len(idx), BATCH):
            chunks.append(lm.score([ws[i] for i in idx[a:a + BATCH]]))
        for key in chunks[0]:
            full = np.concatenate([c[key] for c in chunks])
            arr = np.zeros((len(ws),) + full.shape[1:], dtype=full.dtype)
            arr[idx] = full
            out[f"{pos}_{key}"] = arr.astype(np.float16) if key.startswith("h_") else arr
        print(pos, "done", flush=True)
    np.savez(LAB / f"base{SUFFIX}.npz", **out)
    pd.DataFrame(meta).to_parquet(LAB / f"base_meta{SUFFIX}.parquet", index=False)


if __name__ == "__main__":
    main()
