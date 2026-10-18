"""Study 3 appendix, stage A3 — NMF on the delta vectors: k sweep, final fit, comparisons.

Input is the dev ELLIPSE rows of ``data/study3/delta.npy``, split into positive
and negative parts, X = [max(delta, 0) | max(-delta, 0)] (2 x 2048 columns,
unscaled). Test and TOEFL 11 rows are only ever projected onto the dev-fitted
H (``transform``) and never refit.

    sweep   for each k in K_GRID, fit on two essay-level random halves of dev
            (N_SEEDS seeds), match components across halves by Hungarian
            assignment on the cosine of H rows, and record the median matched
            cosine and the relative reconstruction error.
            -> results/study3/nmf_ksweep_<k>.csv
    fit     final fit at --k on all dev rows, transform every row.
            -> data/study3/nmf/{H,W}_<rep>.npy
    The --rep option selects the representation: ``delta`` (primary), or the
    two halves of delta on their own, ``obs`` (-e_obs, i.e. the observed word
    alone) and ``pred`` (e_pred, the prediction alone). The last two are
    premise checks: if the observed-word profile does as well as the delta
    profile, the components describe which words were surprising, not how the
    prediction failed.

    python src/3-surprisal-decomposition/appendix/A3_nmf.py sweep
    python src/3-surprisal-decomposition/appendix/A3_nmf.py fit --k 30 --rep delta
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from util.paths import DATA_DIR, RESULTS_DIR

S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
NMF_DIR = S3 / "nmf"
SEED = 42
K_GRID = [10, 20, 30, 50, 80]
N_SEEDS = 2
NMF_KW = dict(init="nndsvda", solver="cd", beta_loss="frobenius", tol=1e-4, max_iter=1000)


def posneg(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.maximum(x, 0), np.maximum(-x, 0)]).astype(np.float32)


def load_rep(rep: str, rows: np.ndarray | None = None) -> np.ndarray:
    """Return the (rows x 2048) real-valued matrix for one representation."""
    delta = np.load(S3 / "delta.npy", mmap_mode="r")
    d = np.asarray(delta if rows is None else delta[rows])
    if rep == "delta":
        return d
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["obs_id"])
    obs_ids = sel.obs_id.to_numpy() if rows is None else sel.obs_id.to_numpy()[rows]
    W_U = np.load(S3 / "W_U.npy", mmap_mode="r")
    e_obs = np.asarray(W_U[obs_ids])
    return -e_obs if rep == "obs" else d + e_obs  # pred: e_pred = delta + e_obs


def fit_nmf(X: np.ndarray, k: int, seed: int = SEED) -> NMF:
    m = NMF(n_components=k, random_state=seed, **NMF_KW)
    m.fit(X)
    if m.n_iter_ >= NMF_KW["max_iter"]:
        raise RuntimeError(f"NMF k={k} did not converge")
    return m


def matched_cosine(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    A = H1 / np.linalg.norm(H1, axis=1, keepdims=True)
    B = H2 / np.linalg.norm(H2, axis=1, keepdims=True)
    C = A @ B.T
    r, c = linear_sum_assignment(-C)
    return C[r, c]


def rel_error(m: NMF, X: np.ndarray) -> float:
    return float(m.reconstruction_err_ / np.linalg.norm(X))


def dev_rows() -> tuple[pd.DataFrame, np.ndarray]:
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "text_id", "split"])
    rows = sel.row_id[sel.split == "dev"].to_numpy()
    return sel, rows


def cmd_sweep(args):
    sel, rows = dev_rows()
    X = posneg(load_rep("delta", rows))
    tids = sel.text_id.to_numpy()[rows]
    essays = np.unique(tids)
    out = []
    for k in args.k_grid:
        for s in range(N_SEEDS):
            rng = np.random.default_rng([SEED, s])
            half = set(rng.permutation(essays)[: len(essays) // 2])
            a = np.isin(tids, list(half))
            t0 = time.time()
            ma, mb = fit_nmf(X[a], k), fit_nmf(X[~a], k)
            cos = matched_cosine(ma.components_, mb.components_)
            out.append({"k": k, "seed": s, "median_matched_cos": float(np.median(cos)),
                        "min_matched_cos": float(cos.min()),
                        "share_cos_ge_80": float((cos >= .80).mean()),
                        "rel_error": (rel_error(ma, X[a]) + rel_error(mb, X[~a])) / 2,
                        "n_iter": max(ma.n_iter_, mb.n_iter_),
                        "seconds": time.time() - t0})
            print(out[-1], flush=True)
            pd.DataFrame(out).to_csv(OUT / f"nmf_ksweep_{'_'.join(map(str, args.k_grid))}.csv", index=False)


def cmd_fit(args):
    sel, rows = dev_rows()
    NMF_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = fit_nmf(posneg(load_rep(args.rep, rows)), args.k)
    print(f"fit {args.rep} k={args.k}: {m.n_iter_} iters, {time.time() - t0:.0f}s", flush=True)
    np.save(NMF_DIR / f"H_{args.rep}.npy", m.components_.astype(np.float32))
    W = np.zeros((len(sel), args.k), dtype=np.float32)
    # squared residual and squared norm per row, for per-split relative error
    sq_res, sq_norm = np.zeros(len(sel)), np.zeros(len(sel))
    for a in range(0, len(sel), 50_000):
        b = min(a + 50_000, len(sel))
        X = posneg(load_rep(args.rep, np.arange(a, b)))
        W[a:b] = m.transform(X)
        sq_res[a:b] = ((X - W[a:b] @ m.components_) ** 2).sum(1)
        sq_norm[a:b] = (X ** 2).sum(1)
    np.save(NMF_DIR / f"W_{args.rep}.npy", W)
    split = sel.split.to_numpy()
    rel_err = {s: float(np.sqrt(sq_res[split == s].sum() / sq_norm[split == s].sum()))
               for s in np.unique(split)}
    info = {"rep": args.rep, "k": args.k, "n_iter": int(m.n_iter_),
            "rel_error_dev_fit": rel_error(m, posneg(load_rep(args.rep, rows))),
            "rel_error_by_split_transform": rel_err,
            "n_dev_rows": int(len(rows)), "seconds": time.time() - t0,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"), **NMF_KW}
    (NMF_DIR / f"fit_{args.rep}.json").write_text(json.dumps(info, indent=2))
    print(info)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--k-grid", type=int, nargs="+", default=K_GRID)
    f = sub.add_parser("fit")
    f.add_argument("--k", type=int, required=True)
    f.add_argument("--rep", default="delta", choices=["delta", "obs", "pred"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"sweep": cmd_sweep, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
