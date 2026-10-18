"""Geometry lab — per-word statistics from REAL ELLIPSE dev tokens (all tokens,
including those below tau), from the stage-1 shards. Never touches the benchmark.

For every observed first-subword id w seen in dev:
    n[w]            occurrences
    pbar[w]         mean renormalized top-50 prediction (sparse over vocab), so that
                    mean_e_pred[w] = pbar[w] @ W_U  ("what is usually predicted
                    where this word occurs")
    s_mean[w], s_sd[w]  mean / sd of surprisal (bits) at this word
Also:
    vocab_freq      dev counts of observed ids (for frequency-weighted whitening)
    freq_ids        the 300 most frequent observed ids (a fixed, interpretable basis)

    -> data/study3/lab/geo/stats.npz

    python src/3-vector-transformations/lab/geo_stats.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from util.paths import DATA_DIR

S3 = DATA_DIR / "study3"
GEO = S3 / "lab" / "geo"
HERE_SPLIT = DATA_DIR.parent / "src/3-vector-transformations/split.csv"
V = 100352


def main():
    GEO.mkdir(parents=True, exist_ok=True)
    dev = set(pd.read_csv(HERE_SPLIT).query("split == 'dev'").text_id)
    rows, cols, vals = [], [], []
    obs_all, s_all = [], []
    for pq in sorted((S3 / "ellipse" / "shards").glob("shard_*.parquet")):
        t = pd.read_parquet(pq, columns=["text_id", "obs_id", "surprisal_bits", "is_space"])
        z = np.load(pq.with_suffix(".npz"))
        m = (t.text_id.isin(dev) & ~t.is_space).to_numpy()
        ids, p = z["topk_ids"][m], z["topk_probs"][m].astype(np.float64)
        p /= p.sum(1, keepdims=True)
        obs = t.obs_id.to_numpy()[m]
        rows.append(np.repeat(obs, ids.shape[1]))
        cols.append(ids.ravel())
        vals.append(p.ravel().astype(np.float32))
        obs_all.append(obs)
        s_all.append(t.surprisal_bits.to_numpy()[m])
        print(pq.name, m.sum(), flush=True)
    obs = np.concatenate(obs_all)
    s = np.concatenate(s_all)
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(V, V)).tocsr()
    n = np.bincount(obs, minlength=V)
    s_sum = np.bincount(obs, weights=s, minlength=V)
    s_sq = np.bincount(obs, weights=s * s, minlength=V)
    seen = np.flatnonzero(n)
    W_U = np.load(S3 / "W_U.npy")
    mean_pred = (A[seen] @ W_U) / n[seen, None]
    glob_pred = (np.asarray(A.sum(0)).ravel() @ W_U) / n.sum()
    s_mean = s_sum[seen] / n[seen]
    s_sd = np.sqrt(np.maximum(s_sq[seen] / n[seen] - s_mean ** 2, 0))
    np.savez(GEO / "stats.npz", seen=seen, n=n[seen], mean_pred=mean_pred.astype(np.float32),
             glob_pred=glob_pred.astype(np.float32), s_mean=s_mean, s_sd=s_sd,
             s_glob_mean=s.mean(), s_glob_sd=s.std(), vocab_freq=n,
             freq_ids=np.argsort(-n)[:300])
    print("tokens", len(obs), "distinct obs ids", len(seen))


if __name__ == "__main__":
    main()
