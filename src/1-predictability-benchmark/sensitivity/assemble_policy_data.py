"""Export the subword-policy parquets as one tidy long CSV for the R report.

Mirrors `assemble_analysis_data.py`: the R side reads only gzipped CSVs, so the
per-model parquets written by `subword_policy.py` are collapsed here into one
table with a `policy` column.

Policies are given their reporting names, which differ by architecture:

    first        the current convention (both architectures)
    joint        -log P(word | context): the chain rule for causal models,
                 Kauf & Ivanova within-word autoregressive scoring for masked
    density      per-subword mean, which is what the masked path does today
    joint_indep  masked only: sum of the all-masked marginals, i.e. the joint
                 you get if you ignore within-word dependence

Run from the repo root:
    python src/1-predictability-benchmark/sensitivity/assemble_policy_data.py
"""

import pandas as pd

from util.paths import PREDICTABILITY_DIR, RESULTS_DIR

IN_DIR = PREDICTABILITY_DIR / "sensitivity" / "subword_policy"
OUT = RESULTS_DIR / "predictability" / "subword_policy_long.csv.gz"

COLUMN_POLICY = {
    "first_bits": "first",
    "sum_bits": "joint",          # causal: exact chain rule
    "sum_chain_bits": "joint",    # masked: Kauf & Ivanova
    "mean_bits": "density",
    "sum_indep_bits": "joint_indep",
}
ID_COLS = ["corpus", "text_id", "model", "arch", "overall", "l1", "level",
           "n_scored", "n_split", "n_noncontig", "n_subwords"]


def main() -> None:
    files = sorted(IN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets in {IN_DIR} — run subword_policy.py first")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        for col in ID_COLS:
            if col not in df:
                df[col] = pd.NA
        for col, policy in COLUMN_POLICY.items():
            if col not in df:
                continue
            split_col = col.replace("_bits", "_bits_split")
            frames.append(df.assign(
                policy=policy,
                bits=df[col],
                bits_split=df[split_col] if split_col in df else pd.NA,
            )[ID_COLS + ["policy", "bits", "bits_split"]])

    long = pd.concat(frames, ignore_index=True).dropna(subset=["bits"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT}  {long.shape}")
    print(long.groupby(["arch", "policy"]).size().to_string())


if __name__ == "__main__":
    main()
