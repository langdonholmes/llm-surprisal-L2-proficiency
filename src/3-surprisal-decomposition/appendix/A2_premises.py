"""Study 3 appendix, stage A2 — check the premises the delta decomposition rests on.

Run before interpreting any component. Everything uses ELLIPSE dev rows only,
except the token-composition table, which describes all ELLIPSE essays.
Writes CSVs to results/study3/ for premises.qmd.

1. What gets selected. Composition of the high-surprisal tokens by POS and by
   coarse form class (punctuation, non-alphabetic, proper noun, other), within
   each Overall tertile, against the base rate of all scored tokens. Also
   whether the observed token is in the model's top-50 at all.
   -> premise_composition.csv, premise_topk.csv
2. Where the delta's variance comes from. delta = e_pred - e_obs, so
   tr Cov(delta) = tr Cov(e_pred) + tr Cov(e_obs) - 2 tr Cov(e_pred, e_obs).
   A top-50 mixture is pulled toward the centroid of the vocabulary, so
   e_pred should vary much less than a single row e_obs does.
   -> premise_variance.csv
3. How much of delta is explained by the identity of the observed word (and,
   for contrast, of the top-1 predicted word): the between-group share of
   total sum of squares, R^2 = 1 - SS_within / SS_total, over all 2048 dims.
   -> premise_identity_r2.csv
4. Essay level. How the count and rate of high-surprisal tokens relate to
   Overall and to length.
   -> premise_essay.csv

    python src/3-surprisal-decomposition/appendix/A2_premises.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from util.paths import DATA_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"


def ss_total(X: np.ndarray) -> float:
    return float(((X - X.mean(0)) ** 2).sum())


def between_r2(X: np.ndarray, groups: np.ndarray) -> float:
    """Share of total SS between groups (one-way ANOVA R^2, summed over columns)."""
    codes, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(codes), X.shape[1]))
    np.add.at(sums, inv, X)
    n = np.bincount(inv)[:, None]
    grand = X.mean(0)
    ss_between = float((n * (sums / n - grand) ** 2).sum())
    return ss_between / ss_total(X)


def tertile(overall: pd.Series, cuts) -> pd.Series:
    return pd.cut(overall, [-np.inf, cuts[0], cuts[1], np.inf], right=False,
                  labels=["low", "mid", "high"])


def main():
    sel = pd.read_parquet(S3 / "tokens_selected.parquet")
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet").drop(columns="text")
    split = pd.read_csv(HERE.parent / "split.csv")
    cuts = essays.merge(split).query("split == 'dev'").overall.quantile([1 / 3, 2 / 3]).to_list()
    essays["band"] = tertile(essays.overall, cuts)

    # --- 1. composition -------------------------------------------------------
    def form_class(df):
        return np.select(
            [df.is_punct, ~df.is_alpha, df.pos == "PROPN"],
            ["punctuation", "other non-alphabetic", "proper noun"], "word")

    rows = []
    for s in sorted((S3 / "ellipse" / "shards").glob("shard_*.parquet")):
        t = pd.read_parquet(s, columns=["text_id", "pos", "is_punct", "is_alpha", "is_space"])
        t = t[~t.is_space].merge(essays[["text_id", "band"]])
        t["form"] = form_class(t)
        rows.append(t.groupby(["band", "pos"], observed=True).size().rename("n").reset_index()
                    .assign(level="pos", set="all"))
        rows.append(t.groupby(["band", "form"], observed=True).size().rename("n").reset_index()
                    .rename(columns={"form": "pos"}).assign(level="form", set="all"))
    el = sel[sel.corpus == "ellipse"].merge(essays[["text_id", "band"]])
    el["form"] = form_class(el)
    rows.append(el.groupby(["band", "pos"], observed=True).size().rename("n").reset_index()
                .assign(level="pos", set="high"))
    rows.append(el.groupby(["band", "form"], observed=True).size().rename("n").reset_index()
                .rename(columns={"form": "pos"}).assign(level="form", set="high"))
    comp = pd.concat(rows).groupby(["set", "level", "band", "pos"], observed=True).n.sum().reset_index()
    comp["share"] = comp.n / comp.groupby(["set", "level", "band"], observed=True).n.transform("sum")
    comp.to_csv(OUT / "premise_composition.csv", index=False)

    topk = el.groupby("band", observed=True).agg(
        n=("row_id", "size"),
        share_obs_in_top50=("obs_rank_in_topk", lambda r: (r >= 0).mean()),
        mean_top50_mass=("topk_mass", "mean"),
        median_surprisal=("surprisal_bits", "median"),
        mean_pred1_p=("pred1_p", "mean"),
    ).reset_index()
    topk.to_csv(OUT / "premise_topk.csv", index=False)

    # --- 2-3. delta geometry (dev rows) ---------------------------------------
    dev = sel[sel.split == "dev"]
    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(dev.row_id.to_numpy(), size=min(100_000, len(dev)), replace=False))
    delta = np.asarray(np.load(S3 / "delta.npy", mmap_mode="r")[sub], dtype=np.float64)
    W_U = np.load(S3 / "W_U.npy", mmap_mode="r")
    obs_ids = sel.obs_id.to_numpy()[sub]
    e_obs = np.asarray(W_U[obs_ids], dtype=np.float64)
    e_pred = delta + e_obs

    def tr_cov(A, B=None):
        A = A - A.mean(0)
        B = A if B is None else B - B.mean(0)
        return float((A * B).sum() / (len(A) - 1))

    var = pd.DataFrame([
        {"term": "tr Cov(delta)", "value": tr_cov(delta)},
        {"term": "tr Cov(e_obs)", "value": tr_cov(e_obs)},
        {"term": "tr Cov(e_pred)", "value": tr_cov(e_pred)},
        {"term": "-2 tr Cov(e_pred, e_obs)", "value": -2 * tr_cov(e_pred, e_obs)},
        {"term": "tr Cov(W_U rows, whole vocabulary)", "value": tr_cov(np.asarray(W_U, dtype=np.float64))},
        {"term": "mean ||delta||", "value": float(np.linalg.norm(delta, axis=1).mean())},
        {"term": "mean ||e_obs||", "value": float(np.linalg.norm(e_obs, axis=1).mean())},
        {"term": "mean ||e_pred||", "value": float(np.linalg.norm(e_pred, axis=1).mean())},
        {"term": "mean cos(delta, -e_obs)", "value": float(np.mean(
            (delta * -e_obs).sum(1) / np.linalg.norm(delta, axis=1) / np.linalg.norm(e_obs, axis=1)))},
    ])
    var["n_rows"] = len(sub)
    var.to_csv(OUT / "premise_variance.csv", index=False)
    print(var)

    s = sel.iloc[sub]
    groupings = {"observed first subword": s.obs_id.to_numpy(),
                 "observed lemma (lowercased)": s.lemma.str.lower().to_numpy(),
                 "observed POS": s.pos.to_numpy(),
                 "top-1 predicted subword": s.pred1_id.to_numpy()}
    r2 = []
    for target, X in [("delta", delta), ("e_pred", e_pred), ("e_obs", e_obs)]:
        for gname, g in groupings.items():
            r2.append({"target": target, "grouping": gname, "n_groups": len(np.unique(g)),
                       "r2": between_r2(X, g)})
    # chance level for many small groups: same grouping with labels permuted
    perm = rng.permutation(groupings["observed first subword"])
    r2.append({"target": "delta", "grouping": "observed first subword, permuted",
               "n_groups": len(np.unique(perm)), "r2": between_r2(delta, perm)})
    r2 = pd.DataFrame(r2)
    r2.to_csv(OUT / "premise_identity_r2.csv", index=False)
    print(r2)

    # --- 4. essay level --------------------------------------------------------
    es = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == 'ellipse'")
    es = es.merge(essays).merge(split)
    es["rate_high"] = es.n_high / es.n_scored
    es.to_csv(OUT / "premise_essay.csv", index=False)
    for v in ["rate_high", "n_high", "n_scored", "mean_surprisal"]:
        print(v, "rho with overall =", round(spearmanr(es[v], es.overall)[0], 3))


if __name__ == "__main__":
    main()
