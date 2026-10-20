"""RQ3 pilot, step 3 (CPU) — the clean part against proficiency, raw and partial.

For mean surprisal and for the high-token rate (the D24 split, s replaced by
the indicator s > tau), the Spearman rho of the clean part with the criterion
and its partial rho controlling

    err_rate    5_gold.py's err_rate: tokens above tau and corrected (ERR_TOK or ERR_INS) / n
    corr_tok    sensitivity: corrected tokens (ERR_TOK or ERR_INS), any surprisal / n
    correction  the correction part of the same measure, total - clean
                (omission + context + error tokens)

Partial = Pearson of the residuals of rank(clean) and rank(criterion) after a
linear regression on rank(control). Unit bootstrap, 2,000 resamples, seed 42.
W&I (learners, A = 1 ... C = 3) is run first to reproduce the coordinator's
numbers; FCE uses scripts against script-s, as in D25.

Writes data/study3/pilot_rq3/tables/reconcile.csv and reconcile.json.

    python src/3-surprisal-decomposition/pilot_rq3/3_reconcile.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pilot_fce"))
from fce_common import pc  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402

# not imported from rq3_common: it loads pilot_shapley's `common`, which would
# shadow the pilot_corrected `common` that 5_decompose imports
D = pc("5_decompose")
SEED, N_BOOT = 42, 2000
TAB = DATA_DIR / "study3" / "pilot_rq3" / "tables"

MEASURES = [("mean_surprisal", "surprisal_bits", "s_corr"), ("high_rate", "high_o_f", "high_c_f")]


def tokens(corpus):
    if corpus == "wi":
        a = pd.read_parquet(DATA_DIR / "study3" / "pilot_corrected" / "aligned.parquet")
        a = a[~a.is_space & (a.level != "N")].copy()
        a["level"] = a.level.map({"A": 1, "B": 2, "C": 3}).astype(float)
    else:
        a = pd.read_parquet(DATA_DIR / "study3" / "pilot_fce" / "aligned.parquet")
        a = a[~a.is_space].copy()
        a["level"] = pd.to_numeric(a.script_s, errors="coerce")
        a = a[a.level > 0].assign(text_id=lambda x: x.script_id)
    t = D.tau()
    a["high_o_f"] = (a.surprisal_bits > t).astype(float)
    a["high_c_f"] = (a.s_corr > t).astype(float)
    a["err"] = a.gold.isin(["ERR_TOK", "ERR_INS"])
    a["err_high"] = a.err & (a.high_o_f > 0)
    return a


def partial(rx, ry, rz):
    def res(u):
        zc = rz - rz.mean()
        return u - u.mean() - zc * (zc @ (u - u.mean())) / (zc @ zc)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def stats(x, y, ctrls, i):
    rx, ry = rankdata(x[i]), rankdata(y[i])
    out = [float(np.corrcoef(rx, ry)[0, 1])]
    out += [partial(rx, ry, rankdata(z[i])) for z in ctrls]
    return out


def run(corpus):
    a = tokens(corpus)
    g = a.groupby("text_id")
    rate = pd.DataFrame({"err_rate": g.err_high.mean(), "corr_tok": g.err.mean()})
    rows = []
    for name, so, sc in MEASURES:
        e = D.split(a, so, sc)
        y = e.level.to_numpy(float)
        x = e.clean.to_numpy(float)
        ctrls = {"err_rate": rate.err_rate.reindex(e.index).to_numpy(float),
                 "corr_tok": rate.corr_tok.reindex(e.index).to_numpy(float),
                 "correction": (e.total - e.clean).to_numpy(float)}
        pt = stats(x, y, list(ctrls.values()), np.arange(len(y)))
        rng = np.random.default_rng(SEED)
        bs = np.array([stats(x, y, list(ctrls.values()), rng.integers(0, len(y), len(y))) for _ in range(N_BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for j, lab in enumerate(["rho", *[f"partial | {c}" for c in ctrls]]):
            rows.append({"corpus": corpus, "measure": name, "stat": lab, "value": pt[j],
                         "lo": lo[j], "hi": hi[j], "n": len(y)})
    return rows


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows = run("wi") + run("fce")
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "reconcile.csv", index=False)
    (TAB / "reconcile.json").write_text(json.dumps(rows, indent=1))
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
