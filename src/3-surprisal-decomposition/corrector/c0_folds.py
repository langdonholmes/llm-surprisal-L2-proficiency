"""Corrector, stage c0 (CPU) — the training pairs and the five cross-fitting folds.

One pair per text: the original and the first annotator's correction, both
taken from stage 1 (data/study3/<corpus>/essays.parquet), so the corrector
learns exactly the corrections the decomposition uses. W&I+LOCNESS texts are
essays and FCE texts are answers (two per script).

Folds are assigned by unit (the W&I essay, the FCE script), so both answers of
a script fall in the same fold, and stratified by corpus x group (W&I CEFR
level or LOCNESS; FCE score band). Each fold's model is trained on the other
four and corrects its own fold, so every W&I/FCE text gets a correction from a
model that never saw it. Fold 0 also carries the tuning split: 10% of the units
of folds 1-4, stratified the same way, held out from fold 0's training data to
choose the hyperparameters (README).

Writes data/study3/corrector/pairs.parquet:
    text_id, corpus, unit, group, fold (0-4), tune (bool; inner validation for
    fold 0), text, corrected

    python src/3-surprisal-decomposition/corrector/c0_folds.py
"""

import numpy as np
import pandas as pd

from util.paths import DATA_DIR

S3 = DATA_DIR / "study3"
OUT = S3 / "corrector"
SEED = 42
K = 5
TUNE_FRAC = 0.10


def stratified_units(units: pd.DataFrame, k: int, rng) -> pd.Series:
    """Round-robin over shuffled units within each stratum -> fold per unit."""
    fold = pd.Series(-1, index=units.index)
    for _, g in units.groupby("stratum", sort=True):
        idx = rng.permutation(g.index.to_numpy())
        start = rng.integers(k)  # so small strata do not all start in fold 0
        fold[idx] = (start + np.arange(len(idx))) % k
    return fold


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frames = []
    for corpus in ["wi", "fce"]:
        e = pd.read_parquet(S3 / corpus / "essays.parquet", columns=["text_id", "unit", "group", "text", "corrected"])
        frames.append(e.assign(corpus=corpus))
    pairs = pd.concat(frames, ignore_index=True)

    units = pairs.groupby(["corpus", "unit"], sort=True, dropna=False).group.first().reset_index()
    units["stratum"] = units.corpus + "|" + units.group.fillna("none").astype(str)
    units["fold"] = stratified_units(units, K, rng)

    rest = units[units.fold != 0]
    tune = []
    for _, g in rest.groupby("stratum", sort=True):
        n = int(round(TUNE_FRAC * len(g)))
        tune.extend(rng.choice(g.index.to_numpy(), n, replace=False))
    units["tune"] = units.index.isin(tune)

    pairs = pairs.merge(units[["corpus", "unit", "fold", "tune"]], on=["corpus", "unit"])
    pairs = pairs[["text_id", "corpus", "unit", "group", "fold", "tune", "text", "corrected"]]
    pairs.to_parquet(OUT / "pairs.parquet", index=False)

    print(pairs.groupby(["corpus", "fold"]).size().unstack())
    print(pd.crosstab(pairs.group, pairs.fold))
    print("tune texts:", int(pairs.tune.sum()), " unchanged texts:", int((pairs.text == pairs.corrected).sum()))


if __name__ == "__main__":
    main()
