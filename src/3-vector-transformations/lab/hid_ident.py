"""Hidden-state lab helper: the observed-word identity component of h_obs, from real text.

The question behind it: how does *this* occurrence of a word differ from the
model's typical state after reading that word? Estimated on ordinary dev
ELLIPSE tokens, never on the benchmark:

  * Sample tokens from dev essays that host no benchmark instance (Overall < 4.0,
    so no essay overlaps the bench's host set), scored with Study 1 windowing.
  * For each layer in LAYERS store per-token-type means of h_obs (keyed by the
    observed first-subword id, with counts) and a ridge map from the input
    embedding W_E[obs] to h_obs, for types too rare to have a stable mean.

    data/study3/lab/hidden/ident.npz
        ids, counts, mean_L{l} [n_types, 2048], ridge_L{l} [2048 + 1, 2048]

Identity removal for a token: h_obs - (shrunk type mean), where the type mean
is shrunk toward the ridge prediction with weight n / (n + SHRINK).

    CUDA_VISIBLE_DEVICES=1 python src/3-vector-transformations/lab/hid_ident.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB  # noqa: E402
from lab.hid_states import forward  # noqa: E402
from lab.lm import LM  # noqa: E402

from util.paths import DATA_DIR  # noqa: E402

HID = LAB / "hidden"
LAYERS = [4, 6, 8, 12, 16]
N_ESSAYS = 1500
SHRINK = 20.0
BATCH = 512
SEED = 7


def main():
    split = pd.read_csv(Path(__file__).resolve().parents[1] / "split.csv")
    es = pd.read_parquet(DATA_DIR / "study3" / "ellipse" / "essays.parquet").merge(split)
    pool = es[(es.split == "dev") & (es.overall < 4.0)].sample(N_ESSAYS, random_state=SEED)
    lm = LM()
    wins = []
    for text in pool.text:
        ids = lm.tok(text, add_special_tokens=False).input_ids
        for k in range(1, len(ids)):
            wins.append(np.asarray(ids[max(0, k - 7):k + 1]))
    print(len(wins), "tokens", flush=True)
    obs = np.array([w[-1] for w in wins])
    sums = {l: {} for l in LAYERS}
    # streaming sufficient statistics: per-type sums + ridge normal equations
    types, inv = np.unique(obs, return_inverse=True)
    S = {l: np.zeros((len(types), 2048)) for l in LAYERS}
    WE = lm.W_E.cpu().numpy().astype(np.float64)
    XtX = np.zeros((2049, 2049))
    XtY = {l: np.zeros((2049, 2048)) for l in LAYERS}
    for a in range(0, len(wins), BATCH):
        hs, _ = forward(lm, wins[a:a + BATCH])
        o = obs[a:a + BATCH]
        X = np.hstack([WE[o], np.ones((len(o), 1))])
        XtX += X.T @ X
        for l in LAYERS:
            H = hs[:, l].astype(np.float64)
            np.add.at(S[l], inv[a:a + BATCH], H)
            XtY[l] += X.T @ H
        if a // BATCH % 50 == 0:
            print(a, flush=True)
    counts = np.bincount(inv, minlength=len(types))
    out = {"ids": types, "counts": counts}
    lam = 1e-2 * np.trace(XtX) / XtX.shape[0]
    for l in LAYERS:
        out[f"mean_L{l}"] = (S[l] / counts[:, None]).astype(np.float32)
        out[f"ridge_L{l}"] = np.linalg.solve(XtX + lam * np.eye(2049), XtY[l]).astype(np.float32)
    del sums
    np.savez(HID / "ident.npz", **out)


if __name__ == "__main__":
    with torch.no_grad():
        main()
