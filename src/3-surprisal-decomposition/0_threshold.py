"""Study 3, stage 0 (CPU) — the high-surprisal threshold τ.

τ is the 90th percentile of token surprisal (bits, whitespace tokens excluded)
among ELLIPSE dev essays (``split.csv``) whose Overall score is at least the
dev 2/3 quantile, as in ``appendix/A0_select.py``. Token surprisal is the Study 1
configuration (OLMo-2 1B, 8-subword window) as stored at token level by
``appendix/A0_extract.py --corpus ellipse`` (data/study3/appendix/ellipse/shards). τ is used here
for the high-token rate only.

Writes results/study3/threshold.json.

    python src/3-surprisal-decomposition/0_threshold.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.corpora import ELLIPSE_APPENDIX, RESULTS  # noqa: E402

TAU_PCT = 90


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(HERE / "appendix" / "split.csv", dtype={"text_id": str})
    essays = pd.read_parquet(ELLIPSE_APPENDIX / "essays.parquet", columns=["text_id", "overall"])
    dev = essays.merge(split[split.split == "dev"], on="text_id")
    cut = float(dev.overall.quantile(2 / 3))
    ref = set(dev.text_id[dev.overall >= cut])
    tokens = pd.concat([pd.read_parquet(p, columns=["text_id", "is_space", "surprisal_bits"])
                        for p in sorted((ELLIPSE_APPENDIX / "shards").glob("shard_*.parquet"))])
    ref_bits = tokens.surprisal_bits[tokens.text_id.isin(ref) & ~tokens.is_space]
    tau = float(np.percentile(ref_bits, TAU_PCT))
    res = {"tau_bits": tau, "tau_percentile": TAU_PCT, "dev_top_tertile_overall_cut": cut,
           "n_reference_essays": len(ref), "n_reference_tokens": int(len(ref_bits))}
    (RESULTS / "threshold.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
