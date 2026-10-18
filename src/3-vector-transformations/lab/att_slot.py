"""Attestation lab — slot attestation: is the word an attested filler of its frame?

From the Dolma 3-gram tables (corpus a = edited prose, b = unedited):

    slot  (p1, _, n1):  lp_slot   = log2 (c(p1,w,n1) + .5) / (c(p1,*,n1) + 1)
                        slot_logn = log1p c(p1,*,n1)      (how common the frame is)
                        slot_types= log1p #distinct fillers
    fwd   (p2, p1, _):  lp_fwd    = log2 (c(p2,p1,w) + .5) / (c(p2,p1,*) + 1)
    bwd   (_, n1, n2):  lp_bwd    = log2 (c(w,n1,n2) + .5) / (c(*,n1,n2) + 1)
                        (backward: does what follows license the word?)
    bwd bigram          lp_bwd_bi = log2 (c(w,n1) + .5) / (c(*,n1) + 1)

Row-aligned cache data/study3/lab/att/slot{SUFFIX}.parquet. No labels read.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.att_build import ATT, FREQ, PAD, token_context  # noqa: E402
from lab.harness import LAB, SUFFIX  # noqa: E402


def context(bench):
    p = ATT / f"context{SUFFIX}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = token_context(bench)[["w", "p1", "p2", "n1", "n2"]].fillna(PAD)
    c.to_parquet(p, index=False)
    return c


def slot_feats(c: pd.DataFrame, corpus: str) -> pd.DataFrame:
    lf = pl.scan_parquet(str(FREQ / f"corpus_{corpus}" / "3grams.parquet"))
    cc = pl.from_pandas(c)
    frames = cc.select(pl.col("p1").alias("a"), pl.col("n1").alias("b")).unique()
    fr = (lf.join(frames.lazy(), left_on=["token_0", "token_2"], right_on=["a", "b"], how="semi")
          .group_by(["token_0", "token_2"]).agg(pl.col("count").sum().alias("n"), pl.len().alias("types"))
          .collect(engine="streaming"))
    lefts = cc.select(pl.col("p2").alias("a"), pl.col("p1").alias("b")).unique()
    lt = (lf.join(lefts.lazy(), left_on=["token_0", "token_1"], right_on=["a", "b"], how="semi")
          .group_by(["token_0", "token_1"]).agg(pl.col("count").sum().alias("n")).collect(engine="streaming"))
    rights = cc.select(pl.col("n1").alias("a"), pl.col("n2").alias("b")).unique()
    rt = (lf.join(rights.lazy(), left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
          .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("n")).collect(engine="streaming"))
    tri_keys = pl.concat([cc.select(pl.col("p2").alias("x"), pl.col("p1").alias("y"), pl.col("w").alias("z")),
                          cc.select(pl.col("p1").alias("x"), pl.col("w").alias("y"), pl.col("n1").alias("z")),
                          cc.select(pl.col("w").alias("x"), pl.col("n1").alias("y"), pl.col("n2").alias("z"))]).unique()
    tri = (lf.join(tri_keys.lazy(), left_on=["token_0", "token_1", "token_2"], right_on=["x", "y", "z"], how="semi")
           .collect(engine="streaming"))
    # bigram (w, n1) and unigram-as-second (*, n1) for the backward bigram
    bi_keys = cc.select(pl.col("w").alias("a"), pl.col("n1").alias("b")).unique()
    bi = (lf.join(bi_keys.lazy(), left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
          .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("n")).collect(engine="streaming"))
    second = cc.select(pl.col("n1").alias("a")).unique()
    sec = (lf.join(second.lazy(), left_on="token_2", right_on="a", how="semi")
           .group_by("token_2").agg(pl.col("count").sum().alias("n")).collect(engine="streaming"))

    def d2(df, k):
        return dict(zip(zip(*(df[x].to_list() for x in k)), df["n"].to_list()))

    F, FT = d2(fr, ["token_0", "token_2"]), dict(zip(zip(fr["token_0"].to_list(), fr["token_2"].to_list()),
                                                     fr["types"].to_list()))
    L, R = d2(lt, ["token_0", "token_1"]), d2(rt, ["token_1", "token_2"])
    T = dict(zip(zip(tri["token_0"].to_list(), tri["token_1"].to_list(), tri["token_2"].to_list()),
                 tri["count"].to_list()))
    BI, SEC = d2(bi, ["token_1", "token_2"]), dict(zip(sec["token_2"].to_list(), sec["n"].to_list()))
    w, p1, p2, n1, n2 = (c[k].tolist() for k in ["w", "p1", "p2", "n1", "n2"])
    o = pd.DataFrame(index=c.index)
    fn = np.array([F.get(k, 0) for k in zip(p1, n1)], float)
    o[f"{corpus}_lp_slot"] = np.log2((np.array([T.get(k, 0) for k in zip(p1, w, n1)]) + .5) / (fn + 1))
    o[f"{corpus}_slot_logn"] = np.log1p(fn)
    o[f"{corpus}_slot_types"] = np.log1p([FT.get(k, 0) for k in zip(p1, n1)])
    ln = np.array([L.get(k, 0) for k in zip(p2, p1)], float)
    o[f"{corpus}_lp_fwd"] = np.log2((np.array([T.get(k, 0) for k in zip(p2, p1, w)]) + .5) / (ln + 1))
    o[f"{corpus}_fwd_logn"] = np.log1p(ln)
    rn = np.array([R.get(k, 0) for k in zip(n1, n2)], float)
    o[f"{corpus}_lp_bwd"] = np.log2((np.array([T.get(k, 0) for k in zip(w, n1, n2)]) + .5) / (rn + 1))
    o[f"{corpus}_bwd_logn"] = np.log1p(rn)
    o[f"{corpus}_lp_bwd_bi"] = np.log2((np.array([BI.get(k, 0) for k in zip(w, n1)]) + .5)
                                       / (np.array([SEC.get(x, 0) for x in n1]) + 1))
    return o


def build(bench):
    p = ATT / f"slot{SUFFIX}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t0 = time.time()
    c = context(bench)
    f = pd.concat([slot_feats(c, "a"), slot_feats(c, "b")], axis=1)
    f["seconds_total"] = time.time() - t0
    f.to_parquet(p, index=False)
    return f


if __name__ == "__main__":
    f = build(pd.read_parquet(LAB / f"bench{SUFFIX}.parquet"))
    print(f.describe().T.round(2).to_string())
