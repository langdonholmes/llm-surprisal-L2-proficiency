"""Bucket pilot, step 1 (CPU) — the D28 surprisal split regrouped by correction bucket.

Rebuilds the D28 credit table (``pilot_types/3_decompose.credits``, primary rule:
an error token's whole excess is its own edit's at-site part; spill = edit-level
Shapley within the 8-subword window, correction as baseline) and sums the
credits by bucket instead of ERRANT group:

    S_orig = clean + sum_b (at_site_b + spill_b) + flagged + residual

exactly per unit. Shares of cov(total, level) with 2,000 unit-bootstrap CIs.
Also the reference rows: each bucket's share of all edits, and of cov(edit
rate, level) where edit rate = edits per scored token (the bucket rates add up).

Writes data/study3/pilot_buckets/:
    surprisal_units_<corpus>.parquet     per unit: level, n, total, clean, <bucket>|<comp>, edits_<bucket>
    tables/surprisal_<corpus>.json       shares, means, checks

    python src/3-surprisal-decomposition/pilot_buckets/1_surprisal.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bucket_common import OUT, PLAYERS, S3_SRC, TAB, bucket_of, cov_shares, load  # noqa: E402

dec = load("types_decompose", S3_SRC / "pilot_types" / "3_decompose.py")
COMP = ["at_site", "spill"]
CATS = PLAYERS + ["flagged", "residual"]


def run(corpus):
    a, cr, et, checks = dec.credits(corpus)
    cr = cr[cr.comp.isin(COMP)].copy()
    cr["bucket"] = cr.errant.map(bucket_of)
    cr.loc[cr.k == -1, "bucket"] = "residual"
    cr.loc[cr.k == -2, "bucket"] = "flagged"
    U, parts = dec.unit_table(a, cr, "bucket", "primary")
    for cat in CATS:
        for comp in COMP:
            if f"{cat}|{comp}" not in U:
                U[f"{cat}|{comp}"] = 0.0
        U[f"{cat}|total"] = U[f"{cat}|at_site"] + U[f"{cat}|spill"]
    cols3 = [f"{c}|{p}" for c in CATS for p in COMP]
    gap = float((U.clean + U[cols3].sum(1) - U.total).abs().max())
    assert gap < 1e-9

    # edit counts per unit and bucket
    et = et.assign(bucket=et.errant.map(bucket_of))
    t2u = a.groupby("text_id").unit.first()
    et["unit"] = et.text_id.map(t2u)
    cnt = et.dropna(subset=["unit"]).groupby(["unit", "bucket"]).size().unstack().reindex(columns=PLAYERS)
    cnt = cnt.reindex(U.index).fillna(0)
    for b in PLAYERS:
        U[f"edits_{b}"] = cnt[b]
    U.to_parquet(OUT / f"surprisal_units_{corpus}.parquet")

    L, lv = dec.learners(U, corpus)
    cols = ["total", "clean"] + [f"{c}|{p}" for c in CATS for p in COMP + ["total"]]
    sh = cov_shares(L[cols].to_numpy(float), lv, cols)
    # reference rows: edit counts
    rate = L[[f"edits_{b}" for b in PLAYERS]].div(L.n, axis=0)
    rate.insert(0, "all", rate.sum(1))
    sh_rate = cov_shares(rate.to_numpy(float), lv, ["all"] + PLAYERS)
    tot = L[[f"edits_{b}" for b in PLAYERS]].sum()
    res = {
        "n_units": int(len(L)),
        "exact_sum_max_abs": gap,
        "shares": sh,
        "edit_share_of_all_edits": {b: float(tot[f"edits_{b}"] / tot.sum()) for b in PLAYERS},
        "edit_rate_cov_shares": sh_rate,
        "edits_total": int(tot.sum()),
        "means_learners": L[["total", "clean"] + [f"{c}|total" for c in CATS]].mean().to_dict(),
        "d28_checks": checks,
    }
    if corpus == "wi":
        res["means_by_level"] = L.groupby("level")[["total", "clean"] + [f"{c}|total" for c in CATS]].mean().to_dict()
    (TAB / f"surprisal_{corpus}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    for corpus in ["wi", "fce"]:
        r = run(corpus)
        print(corpus, r["n_units"], "exact", r["exact_sum_max_abs"])
        for k, v in r["shares"].items():
            print(f"  {k:32s} {100 * v[0]:6.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]")
        print("  edit share", {k: round(v, 3) for k, v in r["edit_share_of_all_edits"].items()})
        print("  edit-rate cov", {k: round(v[0], 3) for k, v in r["edit_rate_cov_shares"].items()})


if __name__ == "__main__":
    main()
