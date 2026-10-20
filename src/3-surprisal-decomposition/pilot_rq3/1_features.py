"""RQ3 pilot, step 1 (CPU) — essay / script features from the D28 split.

Per unit (W&I essay, learners only; FCE script, both answers pooled, score > 0),
n = scored tokens:

  B (human annotation only)
      rate_<g>      first-annotator edits of ERRANT group g per token (count / n)
      rate_flagged  detection-only spans per token
      log_n         log of n
  S (surprisal components, D28)
      clean         mean surprisal of the writer's own words in the corrected essay
      cost_<g>      at-site bits per error: at_site_g * n / count_g
      spill_<g>     spill bits per error:   spill_g * n / count_g
      cost_flagged  excess bits per flagged span
      cost_mean     pooled at-site bits per typed error: sum_g at_site_g * n / sum_g count_g
      spill_mean    pooled spill bits per typed error
  total             mean surprisal (S_orig), for the aggregate baseline
  level             W&I A = 1, B = 2, C = 3; FCE script score (1-40)

Per-error terms are NaN when the unit has no error of that kind; step 2
imputes them with the training-fold mean.

Writes data/study3/pilot_rq3/features_{wi,fce}.parquet.

    python src/3-surprisal-decomposition/pilot_rq3/1_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_common import CORPORA, GROUPS, OUT, TYPES_OUT, slug  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402


def counts(corpus: str, units: pd.Index) -> pd.DataFrame:
    et = pd.read_parquet(CORPORA[corpus]["edit_types"])
    if corpus == "fce":
        a = pd.read_parquet(DATA_DIR / "study3" / "pilot_fce" / "aligned.parquet",
                            columns=["text_id", "script_id"]).drop_duplicates()
        et["unit"] = et.text_id.map(a.set_index("text_id").script_id)
    else:
        et["unit"] = et.text_id
    et["g"] = np.where(et.kind == "flagged", "flagged", et.group.replace({"unmapped": "other"}))
    c = et.groupby(["unit", "g"]).size().unstack(fill_value=0)
    return c.reindex(units).fillna(0).reindex(columns=GROUPS + ["flagged"], fill_value=0)


def features(corpus: str) -> pd.DataFrame:
    U = pd.read_parquet(TYPES_OUT / f"units_{corpus}.parquet")
    if corpus == "wi":
        U = U[U.level != "N"].copy()
        U["level"] = U.level.map({"A": 1, "B": 2, "C": 3}).astype(float)
    C = counts(corpus, U.index)
    n = U.n.astype(float)
    F = pd.DataFrame(index=U.index)
    F["level"] = U.level.astype(float)
    F["n"] = n
    F["log_n"] = np.log(n)
    F["total"] = U.total
    F["clean"] = U.clean
    checks = {}
    for g in GROUPS + ["flagged"]:
        k = C[g].to_numpy(float)
        F[f"rate_{slug(g)}"] = k / n
        at = U[f"{g}|at_site"] * n
        sp = U[f"{g}|spill"] * n
        # a part with no edit behind it would be a bookkeeping error
        checks[g] = float(max(at[k == 0].abs().max() if (k == 0).any() else 0,
                              sp[k == 0].abs().max() if (k == 0).any() else 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            F[f"cost_{slug(g)}"] = np.where(k > 0, at / k, np.nan)
            if g != "flagged":
                F[f"spill_{slug(g)}"] = np.where(k > 0, sp / k, np.nan)
    k_all = C[GROUPS].sum(axis=1).to_numpy(float)
    at_all = sum(U[f"{g}|at_site"] for g in GROUPS) * n
    sp_all = sum(U[f"{g}|spill"] for g in GROUPS) * n
    with np.errstate(divide="ignore", invalid="ignore"):
        F["cost_mean"] = np.where(k_all > 0, at_all / k_all, np.nan)
        F["spill_mean"] = np.where(k_all > 0, sp_all / k_all, np.nan)
    F["n_typed_edits"] = k_all
    F["n_flagged"] = C["flagged"].to_numpy(float)
    print(corpus, "units", len(F), "| max |part| where count = 0:", {g: round(v, 6) for g, v in checks.items()})
    print("  units with no typed edit:", int((k_all == 0).sum()),
          "| share missing per group:", (C[GROUPS] == 0).mean().round(3).to_dict(),
          "| flagged:", round(float((C.flagged == 0).mean()), 3))
    assert max(checks.values()) < 1e-6, checks
    return F


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for corpus in ["wi", "fce"]:
        F = features(corpus)
        F.to_parquet(OUT / f"features_{corpus}.parquet")
        print(F.level.value_counts().sort_index().to_dict() if corpus == "wi" else F.level.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
