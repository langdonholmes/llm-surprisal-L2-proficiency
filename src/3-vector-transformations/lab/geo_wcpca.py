"""Geometry lab — PCA basis for word-centered e_pred, fitted on REAL dev selected rows.

wc = e_pred - word_mean_pred(obs), e_pred = delta + W_U[obs]. 100k random dev
rows of data/study3/delta.npy.  -> data/study3/lab/geo/wc_pca.npz
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lab.geo_common import GEO, word_mean_pred
from util.paths import DATA_DIR

S3 = DATA_DIR / "study3"


def main():
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "split", "obs_id"])
    dev = sel[sel.split == "dev"]
    rows = np.sort(np.random.default_rng(42).choice(dev.row_id.to_numpy(), 100_000, replace=False))
    D = np.asarray(np.load(S3 / "delta.npy", mmap_mode="r")[rows])
    obs = sel.obs_id.to_numpy()[rows]
    W = np.load(S3 / "W_U.npy", mmap_mode="r")
    wc = D + np.asarray(W[obs]) - word_mean_pred(obs)
    p = PCA(64, random_state=42).fit(wc)
    np.savez(GEO / "wc_pca.npz", components=p.components_, mean=p.mean_, evr=p.explained_variance_ratio_)
    print(p.explained_variance_ratio_[:10].round(3), p.explained_variance_ratio_.sum())


if __name__ == "__main__":
    main()
