"""FCE step 4 (CPU) — the D24 exact split, against the FCE exam score.

``split`` is ``pilot_corrected/5_decompose.split`` unchanged; its "level"
column carries the score here. Primary unit: the script (both answers'
tokens pooled) against ``script-s``. Secondary: the answer against the
``answer-s`` ordinal, bootstrapping scripts. See NOTES.md for the rule.

Writes, under data/study3/pilot_fce/tables/:
    decompose_by_band.csv     part means by script-score quartile band (plus all)
    decompose_tests.json      shares of cov(total, score), Spearman, CIs; level-free summaries
    wi_vs_fce.csv             part means: W&I A/B/C (D24 table) next to FCE bands

    python src/3-surprisal-decomposition/pilot_fce/4_decompose.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fce_common import N_BOOT, SEED, TAB, OUT, pc  # noqa: E402

D = pc("5_decompose")
PARTS = D.PARTS
COLS = ["total"] + PARTS
BANDS = [(1, 23), (24, 27), (28, 31), (32, 40)]
MEASURES = [("mean_surprisal_bits", "surprisal_bits", "s_corr", 1.0),
            ("high_per_1000", "high_o_f", "high_c_f", 1000.0)]


def answer_ordinal(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.extract(r"^([1-5])\.([1-3])T?$").astype(float)
    return (m[0] - 1) * 3 + m[1]


def spearman_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman of each column of X with y."""
    rx = np.apply_along_axis(rankdata, 0, X)
    ry = rankdata(y)
    rx = rx - rx.mean(0)
    ry = ry - ry.mean()
    return (rx * ry[:, None]).sum(0) / np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())


def shares_and_rho(e: pd.DataFrame, score: str, cluster: str) -> dict:
    """Share of cov(total, score) per part and Spearman with score; cluster bootstrap."""
    X = e[COLS].to_numpy(float)
    y = e[score].to_numpy(float)
    cl = e[cluster].to_numpy()
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(SEED)

    def stat(i):
        xc = X[i] - X[i].mean(0)
        c = (xc * (y[i] - y[i].mean())[:, None]).mean(0)
        return np.concatenate([c[1:] / c[0], spearman_rows(X[i], y[i])])

    pt = stat(np.arange(len(e)))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        bs.append(stat(np.concatenate([members[k] for k in pick])))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    k = len(PARTS)
    corr_share = bs[:, 1:k].sum(1)
    return {
        "n": int(len(e)), "n_clusters": int(len(uniq)),
        "share_of_cov_with_score": {p: {"share": float(pt[j]), "ci": [float(lo[j]), float(hi[j])]}
                                    for j, p in enumerate(PARTS)},
        "corrections_share": {"share": float(pt[1:k].sum()),
                              "ci": [float(x) for x in np.percentile(corr_share, [2.5, 97.5])]},
        "spearman_with_score": {c: {"rho": float(pt[k + j]), "ci": [float(lo[k + j]), float(hi[k + j])]}
                                for j, c in enumerate(COLS)},
    }


def verdict(r: dict) -> dict:
    s = r["share_of_cov_with_score"]["clean"]
    a = (s["ci"][0] <= 0 <= s["ci"][1]) or abs(s["share"]) < 0.15
    b = r["corrections_share"]["share"] >= 0.85
    return {"a_clean_ci_has_0_or_small": bool(a), "b_corrections_ge_85": bool(b), "replicates": bool(a and b)}


def main():
    a = pd.read_parquet(OUT / "aligned.parquet")
    a = a[~a.is_space]
    t = D.tau()
    a["high_o_f"] = (a.surprisal_bits > t).astype(float)
    a["high_c_f"] = (a.s_corr > t).astype(float)
    a["script_s"] = pd.to_numeric(a.script_s, errors="coerce")
    a["answer_o"] = answer_ordinal(a.answer_s)

    # primary: script units (split groups by text_id, so key the tokens by script)
    sa = a[a.script_s > 0].assign(text_id=lambda x: x.script_id, level=lambda x: x.script_s)
    # secondary: answer units
    aa = a[a.answer_o.notna()].assign(level=lambda x: x.answer_o)
    res, rows = {"tau": t}, []
    for name, so, sc, k in MEASURES:
        es = D.split(sa, so, sc)
        es[COLS] *= k
        es = es.rename(columns={"level": "score"})
        es["script_id"] = es.index
        ea = D.split(aa, so, sc)
        ea[COLS] *= k
        ea = ea.rename(columns={"level": "score"})
        ea["script_id"] = aa.groupby("text_id").script_id.first().reindex(ea.index)
        es["band"] = pd.cut(es.score, [b[0] - .5 for b in BANDS] + [40.5],
                            labels=[f"{lo}-{hi}" for lo, hi in BANDS])
        tb = es.groupby("band", observed=True)[COLS].mean()
        tb.loc["all"] = es[COLS].mean()
        tb["n_scripts"] = es.groupby("band", observed=True).size().tolist() + [len(es)]
        rows.append(tb.assign(measure=name))
        res[name] = {
            "script": {**shares_and_rho(es, "score", "script_id")},
            "answer": {**shares_and_rho(ea, "score", "script_id")},
            "level_free": {
                "mean": es[COLS].mean().to_dict(), "sd": es[COLS].std().to_dict(),
                "share_of_total_mean": (es[PARTS].mean() / es.total.mean()).to_dict(),
                "clean_total_pearson": float(es.clean.corr(es.total)),
                "clean_q": es.clean.quantile([.1, .5, .9]).to_dict(),
                "total_q": es.total.quantile([.1, .5, .9]).to_dict(),
                "var_share_of_total": {p: float(np.cov(es[p], es.total)[0, 1] / es.total.var())
                                       for p in PARTS}},
        }
        res[name]["script"]["verdict"] = verdict(res[name]["script"])
        res[name]["answer"]["verdict"] = verdict(res[name]["answer"])
        if name == "mean_surprisal_bits":
            l1 = sa.groupby("script_id").l1.first()
            es["l1"] = l1.reindex(es.index)
            top = es.l1.value_counts().head(6).index
            res["l1_clean_mean_by_top6"] = es[es.l1.isin(top)].groupby("l1").agg(
                n=("clean", "size"), clean=("clean", "mean"), total=("total", "mean"),
                score=("score", "mean")).round(3).to_dict(orient="index")
            es.to_parquet(OUT / "script_parts.parquet")
    tab = pd.concat(rows).reset_index()
    tab.to_csv(TAB / "decompose_by_band.csv", index=False)
    (TAB / "decompose_tests.json").write_text(json.dumps(res, indent=2, default=float))

    wi = pd.read_csv(OUT.parent / "pilot_corrected" / "tables" / "decompose_by_level.csv")
    wi = wi[wi.level != "N"].rename(columns={"level": "group"}).assign(corpus="W&I")
    fc = tab.rename(columns={"band": "group"}).assign(corpus="FCE")[["group"] + COLS + ["measure", "corpus"]]
    pd.concat([wi, fc]).to_csv(TAB / "wi_vs_fce.csv", index=False)

    print(tab.round(3).to_string())
    for name, *_ in MEASURES:
        for u in ("script", "answer"):
            r = res[name][u]
            print(name, u, r["n"], {p: (round(v["share"], 3), [round(x, 3) for x in v["ci"]])
                                    for p, v in r["share_of_cov_with_score"].items()},
                  "corr", round(r["corrections_share"]["share"], 3),
                  {c: round(v["rho"], 3) for c, v in r["spearman_with_score"].items()}, r["verdict"])
    print(json.dumps({k: res[k]["level_free"] for k in ("mean_surprisal_bits", "high_per_1000")}, indent=1, default=float))
    print(res["l1_clean_mean_by_top6"])


if __name__ == "__main__":
    main()
