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
from lib.corpora import RESULTS, data_dir  # noqa: E402
from lib.errant import BUCKETS, GROUPS  # noqa: E402
from lib.stats import boot, partial_rank  # noqa: E402
from util.paths import ELLIPSE_DIR  # noqa: E402

SCORES = ["overall", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
CUTS = {"bucket": BUCKETS, "group": GROUPS}


def load():
    units = pd.read_parquet(data_dir("ellipse") / "units.parquet")
    scores = pd.read_csv(ELLIPSE_DIR / "ELLIPSE_Final_github.csv",
                         usecols=["text_id_kaggle"] + [s.title() for s in SCORES])
    scores = scores.rename(columns={"text_id_kaggle": "unit", **{s.title(): s for s in SCORES}}).set_index("unit")
    units = units.join(scores, how="inner")
    assert len(units) == 6482 and np.allclose(units.y, units.overall)
    return units


def shares(units):
    rows = []
    Y = units[SCORES].to_numpy(float)
    for cut, cats in CUTS.items():
        cols = ["clean"] + [f"{c}|total" for c in cats]
        X = units[["total"] + cols].to_numpy(float)

        def stat(idx):
            x, y = X[idx] - X[idx].mean(0), Y[idx] - Y[idx].mean(0)
            cov = x.T @ y  # (total + parts) x scores
            share = cov[1:] / cov[0]
            # every share, then every share minus overall's (column 0)
            return np.concatenate([share.ravel(), (share - share[:, :1]).ravel()])
        point, lo, hi = boot(stat, len(units))
        k = len(cols) * len(SCORES)  # where the differences start
        for j, (col, score) in enumerate((col, score) for col in cols for score in SCORES):
            rows.append({"cut": cut, "col": col.removesuffix("|total"), "score": score,
                         "share": point[j], "lo": lo[j], "hi": hi[j],
                         "diff_vs_overall": point[k + j], "diff_lo": lo[k + j], "diff_hi": hi[k + j]})
    return pd.DataFrame(rows)


def partials(units):
    rows = []
    z = units.overall.to_numpy(float)
    for cut, cats in CUTS.items():
        cols = ["clean"] + [f"{c}|total" for c in cats]
        X = units[cols].to_numpy(float)
        for score in SCORES[1:]:
            y = units[score].to_numpy(float)
            point, lo, hi = boot(lambda idx: [partial_rank(X[idx, j], y[idx], z[idx]) for j in range(len(cols))],
                                 len(units))
            rows += [{"cut": cut, "col": c.removesuffix("|total"), "score": score, "partial": point[j], "lo": lo[j],
                      "hi": hi[j]} for j, c in enumerate(cols)]
    return pd.DataFrame(rows)


def main():
    units = load()
    shares(units).to_csv(RESULTS / "analytic_shares_ellipse.csv", index=False)
    partials(units).to_csv(RESULTS / "analytic_partial_ellipse.csv", index=False)


if __name__ == "__main__":
    main()
