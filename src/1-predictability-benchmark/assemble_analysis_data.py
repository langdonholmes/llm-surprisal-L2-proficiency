"""Export tidy long tables for the Study 1 fairness-validity Pareto analysis.

Reads the per-model surprisal parquets under data/predictability/ and writes two
gzipped long CSVs to results/predictability/ that the R/Quarto analysis consumes
(so the R side needs only tidyverse -- no arrow/parquet dependency).

    ellipse_long.csv.gz  text_id, window, mean_surprisal_bits, overall, truncated, model
    toefl_long.csv.gz    text_id, window, mean_surprisal_bits, l1, level, truncated, model

Run from the repo root:  python src/1-predictability-benchmark/assemble_analysis_data.py
"""

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent))
from models import MODEL_REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "predictability"
OUT = ROOT / "results" / "predictability"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ellipse, toefl = [], []
    for key in MODEL_REGISTRY:
        e = pd.read_parquet(PRED / "ellipse" / key / "surprisal.parquet")
        e = e[["text_id", "window", "mean_surprisal_bits", "overall", "truncated"]].copy()
        e["model"] = key
        ellipse.append(e)

        t = pd.read_parquet(PRED / "toefl11" / key / "surprisal.parquet")
        t = t[["text_id", "window", "mean_surprisal_bits", "l1", "level", "truncated"]].copy()
        t["model"] = key
        toefl.append(t)

    ell = pd.concat(ellipse, ignore_index=True)
    toe = pd.concat(toefl, ignore_index=True)
    ell.to_csv(OUT / "ellipse_long.csv.gz", index=False, compression="gzip")
    toe.to_csv(OUT / "toefl_long.csv.gz", index=False, compression="gzip")
    print(f"wrote {OUT/'ellipse_long.csv.gz'}  {ell.shape}")
    print(f"wrote {OUT/'toefl_long.csv.gz'}  {toe.shape}")


if __name__ == "__main__":
    main()
