"""Pilot, step 5 (CPU) — an exact split of each essay's mean surprisal into
what the gold corrections remove and what they leave.

Per essay, over its n scored original tokens. O is the writer's own tokens that
align to the corrected essay, with surprisal s_o in the original and s_c in the
corrected essay. E is everything else: corrected tokens (ERR_TOK) and the few
own tokens that do not align. Then

    S_orig = clean + omission + context + error_tokens

    clean        = mean_O(s_c)                       own words, corrected context
    omission     = sum_{O, ERR_INS}(s_o - s_c) / n   a word is missing just before
    context      = sum_{O, other}(s_o - s_c) / n     errors elsewhere in the window
    error_tokens = sum_E(s_o - clean) / n            corrected tokens' excess

which holds exactly. The same split is applied to the high-token rate (s replaced
by the indicator s > tau). Because the parts add up, so do their covariances with
level, so each part's share of cov(S_orig, level) among learners (A=1, B=2, C=3)
is an exact split of the proficiency signal. The N - C difference splits the
same way.

Writes tables/decompose_by_level.csv and tables/decompose_tests.json.

    python src/3-surprisal-decomposition/pilot_corrected/5_decompose.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import LEVELS, OUT, SEED, tau  # noqa: E402

TAB = OUT / "tables"
N_BOOT = 2000
PARTS = ["clean", "omission", "context", "error_tokens"]


def split(a: pd.DataFrame, so: str, sc: str) -> pd.DataFrame:
    """Per-essay parts of the mean of `so`, with `sc` the corrected-essay value."""
    a = a.assign(o=a.own & a.aligned)
    a["d"] = np.where(a.o, a[so] - a[sc], 0.0)
    g = a.groupby("text_id")
    e = pd.DataFrame({"level": g["level"].first(), "n": g.size(), "total": g[so].mean()})
    e["clean"] = a[a.o].groupby("text_id")[sc].mean()
    e["omission"] = a[a.o & (a.gold == "ERR_INS")].groupby("text_id").d.sum()
    e["context"] = a[a.o & (a.gold != "ERR_INS")].groupby("text_id").d.sum()
    a = a.join(e.clean, on="text_id")
    e["error_tokens"] = a[~a.o].assign(x=lambda x: x[so] - x.clean).groupby("text_id").x.sum()
    e = e.fillna({"omission": 0.0, "context": 0.0, "error_tokens": 0.0})
    for p in PARTS[1:]:
        e[p] /= e.n
    assert np.allclose(e[PARTS].sum(1), e.total, atol=1e-9)
    return e


def cov_shares(e: pd.DataFrame) -> dict:
    """Each part's share of cov(total, level) among learners, with essay-bootstrap CIs."""
    L = e[e.level != "N"]
    lv = L.level.map({"A": 1, "B": 2, "C": 3}).to_numpy(float)
    X = L[["total"] + PARTS].to_numpy()
    rng = np.random.default_rng(SEED)

    def shares(i):
        c = ((X[i] - X[i].mean(0)) * (lv[i] - lv[i].mean())[:, None]).mean(0)
        return c[1:] / c[0]

    pt = shares(np.arange(len(L)))
    bs = np.array([shares(rng.integers(0, len(L), len(L))) for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    rho = {c: float(pd.Series(L[c].to_numpy()).corr(pd.Series(lv), method="spearman"))
           for c in ["total"] + PARTS}
    return {"share_of_cov_with_level": {p: {"share": float(s), "ci": [float(a), float(b)]}
                                        for p, s, a, b in zip(PARTS, pt, lo, hi)},
            "spearman_with_level": rho}


def n_minus_c(e: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    N, C = e[e.level == "N"], e[e.level == "C"]
    iN = rng.integers(0, len(N), (N_BOOT, len(N)))
    iC = rng.integers(0, len(C), (N_BOOT, len(C)))
    out = {}
    for c in ["total"] + PARTS:
        n, k = N[c].to_numpy(), C[c].to_numpy()
        bs = n[iN].mean(1) - k[iC].mean(1)
        out[c] = {"diff": float(n.mean() - k.mean()), "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
    return out


def main():
    a = pd.read_parquet(OUT / "aligned.parquet")
    a = a[~a.is_space]
    t = tau()
    a["high_o_f"] = (a.surprisal_bits > t).astype(float)
    a["high_c_f"] = (a.s_corr > t).astype(float)

    rows, res = [], {}
    for name, so, sc, k in [("mean_surprisal_bits", "surprisal_bits", "s_corr", 1.0),
                            ("high_per_1000", "high_o_f", "high_c_f", 1000.0)]:
        e = split(a, so, sc)
        e[["total"] + PARTS] *= k
        rows.append(e.groupby("level")[["total"] + PARTS].mean().reindex(LEVELS).assign(measure=name))
        res[name] = {**cov_shares(e), "native_minus_C": n_minus_c(e)}
    pd.concat(rows).reset_index().to_csv(TAB / "decompose_by_level.csv", index=False)
    (TAB / "decompose_tests.json").write_text(json.dumps(res, indent=2))
    print(pd.concat(rows).round(3).to_string())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
