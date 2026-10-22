"""Study 3, stage 6 (CPU) — RQ3: does surprisal predict proficiency beyond the error counts?

Per unit (learners; units.parquet from stage 5):
    B            the human annotation only: edits per token of each of the nine
                 ERRANT groups, flagged spans per token, and log n
    S compact    clean (mean surprisal of the writer's own words in the corrected
                 essay), mean at-site bits per error, mean spill bits per error
    mean surprisal  the aggregate measure of Studies 1 and 2

Ridge (imputation, scaling and alpha fitted in-fold), 10 x 10-fold CV
stratified by level (W&I) or score band (FCE), the same folds for every model.
ΔR² = R²(B + S) − R²(B) with a CI from 2,000 unit bootstraps of the stored
out-of-fold predictions (D31). Standardized coefficients refit on all units.

Writes results/study3/increment.csv and coefs.csv.

    python src/3-surprisal-decomposition/6_increment.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import CORPORA, ERROR_COUNTS, RESULTS, S_COMPACT, SEED, data_dir  # noqa: E402
from lib.stats import delta_r2, oof, r2, ridge  # noqa: E402

MODELS = {"B": ERROR_COUNTS, "B + S compact": ERROR_COUNTS + S_COMPACT, "B + mean surprisal": ERROR_COUNTS + ["total"],
          "mean surprisal": ["total"], "S compact": S_COMPACT}


def strata(units, corpus):
    """CV strata: CEFR level (W&I) or score band (FCE)."""
    y = units.y.to_numpy()
    return y.astype(int) if corpus == "wi" else np.digitize(y, [0, 23.5, 27.5, 31.5, 40.5])


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    rows, coefs = [], []
    for corpus in CORPORA:
        units = pd.read_parquet(data_dir(corpus) / "units.parquet")
        units = units[units.y.notna()]
        y = units.y.to_numpy(float)
        folds = strata(units, corpus)
        preds = {model: oof(units[features], y, folds) for model, features in MODELS.items()}
        for model in MODELS:
            row = {"corpus": corpus, "model": model, "n": len(y), "r2": float(r2(preds[model], y).mean())}
            if model.startswith("B +"):
                row.update(delta_r2(preds[model], preds["B"], y))
            rows.append(row)
        for model in ["B", "B + S compact"]:
            fit = ridge(SEED).fit(units[MODELS[model]].to_numpy(float), y)
            coefs += [{"corpus": corpus, "model": model, "feature": f, "coef_sd": c}
                      for f, c in zip(MODELS[model], fit[-1].coef_)]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "increment.csv", index=False)
    pd.DataFrame(coefs).to_csv(RESULTS / "coefs.csv", index=False)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
