"""Study 3, stage 7 (CPU) — the ELLIPSE analytic scores against the parts of surprisal (D42).

Per essay (units.parquet from stage 5 with --corpus ellipse): the clean part and
each ERRANT group's and bucket's part (at site + spill), which add up to the
essay's mean surprisal. Against each of the seven rated scores (overall and the
six analytic: cohesion, syntax, vocabulary, phraseology, grammar, conventions):

shares    each part's share of cov(mean surprisal, score), as in stage 5, and its
          difference from the same part's share for overall
partial   Spearman of each part with each analytic score, controlling overall
          (the score's own signal beyond the overall rating, as in D35's A3)

2,000 essay bootstraps, the same resamples for every score, so the differences
from overall are paired.

Writes results/study3/analytic_shares_ellipse.csv and analytic_partial_ellipse.csv.

    python src/3-surprisal-decomposition/7_analytic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import RES, out  # noqa: E402
from lib.errant import BUCKETS, GROUPS  # noqa: E402
from lib.stats import boot, partial_rank  # noqa: E402
from util.paths import ELLIPSE_DIR  # noqa: E402

SCORES = ["overall", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
CUTS = {"bucket": BUCKETS, "group": GROUPS}


def load():
    U = pd.read_parquet(out("ellipse") / "units.parquet")
    m = pd.read_csv(ELLIPSE_DIR / "ELLIPSE_Final_github.csv", usecols=["text_id_kaggle"] + [s.title() for s in SCORES])
    m = m.rename(columns={"text_id_kaggle": "unit", **{s.title(): s for s in SCORES}}).set_index("unit")
    U = U.join(m, how="inner")
    assert len(U) == 6482 and np.allclose(U.y, U.overall)
    return U


def shares(U):
    rows = []
    Y = U[SCORES].to_numpy(float)
    for cut, cats in CUTS.items():
        cols = ["clean"] + [f"{c}|total" for c in cats]
        X = U[["total"] + cols].to_numpy(float)

        def st(i):
            x, y = X[i] - X[i].mean(0), Y[i] - Y[i].mean(0)
            c = x.T @ y  # parts x scores
            sh = c[1:] / c[0]
            return np.concatenate([sh.ravel(), (sh - sh[:, :1]).ravel()])
        pt, lo, hi = boot(st, len(U))
        k = len(cols) * len(SCORES)
        for j, (col, s) in enumerate((col, s) for col in cols for s in SCORES):
            rows.append({"cut": cut, "col": col.removesuffix("|total"), "score": s,
                         "share": pt[j], "lo": lo[j], "hi": hi[j],
                         "diff_vs_overall": pt[k + j], "diff_lo": lo[k + j], "diff_hi": hi[k + j]})
    return pd.DataFrame(rows)


def partials(U):
    rows = []
    z = U.overall.to_numpy(float)
    for cut, cats in CUTS.items():
        cols = ["clean"] + [f"{c}|total" for c in cats]
        X = U[cols].to_numpy(float)
        for s in SCORES[1:]:
            y = U[s].to_numpy(float)
            pt, lo, hi = boot(lambda i: [partial_rank(X[i, j], y[i], z[i]) for j in range(len(cols))], len(U))
            rows += [{"cut": cut, "col": c.removesuffix("|total"), "score": s, "partial": pt[j], "lo": lo[j],
                      "hi": hi[j]} for j, c in enumerate(cols)]
    return pd.DataFrame(rows)


def main():
    U = load()
    shares(U).to_csv(RES / "analytic_shares_ellipse.csv", index=False)
    partials(U).to_csv(RES / "analytic_partial_ellipse.csv", index=False)


if __name__ == "__main__":
    main()
