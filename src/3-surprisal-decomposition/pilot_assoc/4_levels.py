"""Association pilot, step 4 (CPU) — every measure against level / score, before and after correction.

Per measure and corpus (W&I learners vs level A=1..C=3; FCE scripts vs script-s):

    rho_orig, rho_corr       Spearman, 2,000 unit bootstraps (seed 42)
    removed                  token-mean measures: 1 - cov(clean, y) / cov(total, y) (exact D24 split);
                             other measures: 1 - rho_corr / rho_orig
    partial_orig/_corr       partial rho | plain corrected-token share (ERR_TOK or ERR_INS / non-space tokens)
    dR2_corr / dR2_orig      R2(B + measure) - R2(B), the pilot_rq3 framework unchanged (2_increment.py)

The reading rule is in NOTES.md (written before this step was run).

Writes data/study3/pilot_assoc/tables/levels_<corpus>.csv, levels.json, units_<corpus>.parquet

    python src/3-surprisal-decomposition/pilot_assoc/4_levels.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assoc_common import N_BOOT, OUT, S3_SRC, SEED, TAB, load  # noqa: E402

sys.path.insert(0, str(S3_SRC / "pilot_rq3"))
INC = load("rq3_2_increment", S3_SRC / "pilot_rq3" / "2_increment.py")
RQ3 = sys.modules["rq3_common"]

CLASSIC = ["amod_mi", "dobj_mi", "advmod_mi", "log_freq_s2", "content_freq", "bigram_mi", "bigram_dp", "bigram_t",
           "share_mi3", "share_t2", "share_unatt"]
TOKEN = ["surprisal", "surprisal_sw", "llm_pmi", "llm_dp", "llm_pmi_content", "llm_pmi_fadj", "llm_pmi_ge3",
         "high_s_content", "llm_pmi_att1", "llm_pmi_att2", "llm_dp_att1", "llm_dp_att2", "surprisal_att2",
         "llm_pmi_lowS", "surprisal_lowS", "llm_pmi_pbar", "llm_pmi_pbar_b", "llm_pmi_subfreq"]
PARTS = ["omission", "context", "error_tokens"]


def unit_table(corpus):
    """One row per unit: y, corrected-token share, each measure (orig, corr) and the split parts."""
    t = pd.read_parquet(OUT / f"tokens_{corpus}.parquet", columns=["unit", "y", "gold"])
    g = t.groupby("unit")
    U = pd.DataFrame({"y": g.y.first(), "corr_share": g.gold.agg(lambda s: s.isin(["ERR_TOK", "ERR_INS"]).mean())})
    s = pd.read_parquet(OUT / f"split_{corpus}.parquet")
    for m in TOKEN:
        x = s[s.measure == m].set_index("unit")
        U[f"{m}|orig"] = x.total
        U[f"{m}|corr"] = x.clean
        for p in PARTS:
            U[f"{m}|{p}"] = x[p]
    for v in ["orig", "corr"]:
        c = pd.read_parquet(OUT / f"classic_{corpus}_{v}.parquet")
        for m in CLASSIC:
            U[f"{m}|{v}"] = c[m].reindex(U.index)
    return U


def _rho(x, y):
    ok = ~np.isnan(x)
    return np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]


def _partial(x, y, z):
    ok = ~np.isnan(x)
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    zc = rz - rz.mean()

    def res(u):
        u = u - u.mean()
        return u - zc * (zc @ u) / (zc @ zc)
    return np.corrcoef(res(rx), res(ry))[0, 1]


def _cov(x, y):
    return ((x - x.mean()) * (y - y.mean())).mean()


def stats(U, m, i):
    y, z = U.y.to_numpy(float)[i], U.corr_share.to_numpy(float)[i]
    xo, xc = U[f"{m}|orig"].to_numpy(float)[i], U[f"{m}|corr"].to_numpy(float)[i]
    ro, rc = _rho(xo, y), _rho(xc, y)
    if m in TOKEN:
        removed = 1 - _cov(xc, y) / _cov(xo, y)
    else:
        removed = 1 - rc / ro
    return [ro, rc, removed, _partial(xo, y, z), _partial(xc, y, z)]


STATS = ["rho_orig", "rho_corr", "removed", "partial_orig", "partial_corr"]


def correlations(U):
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(U), len(U)) for _ in range(N_BOOT)]
    out = {}
    for m in TOKEN + CLASSIC:
        pt = stats(U, m, np.arange(len(U)))
        bs = np.array([stats(U, m, i) for i in idx])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        out[m] = {s: {"est": float(p), "ci": [float(a), float(b)]} for s, p, a, b in zip(STATS, pt, lo, hi)}
        out[m]["n_missing_orig"] = int(U[f"{m}|orig"].isna().sum())
        out[m]["n_missing_corr"] = int(U[f"{m}|corr"].isna().sum())
    return out


_CTX = None


def _one(col):
    """B + one column vs B (forked worker; reads the module-level _CTX)."""
    warnings.filterwarnings("ignore")
    F, y, st, PB = _CTX
    P, _ = INC.oof(F, y, st, RQ3.B + [col])
    d = INC.r2(P, y) - INC.r2(PB, y)
    bs = INC.boot_delta(P, PB, y)
    return col, {"dR2": float(d.mean()), "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                 "share_repeats_gt0": float((d > 0).mean())}


def increments(U, corpus):
    F = pd.read_parquet(RQ3.OUT / f"features_{corpus}.parquet")
    assert set(F.index) == set(U.index), (len(F), len(U), len(set(F.index) & set(U.index)))
    F = F.join(U.drop(columns=["y"]))
    assert np.allclose(F.level, U.y.reindex(F.index))
    y = F.level.to_numpy(float)
    st = INC.strata(F, corpus)
    PB, _ = INC.oof(F, y, st, RQ3.B)
    rB = INC.r2(PB, y).mean()
    out = {"B_r2": float(rB)}

    global _CTX
    _CTX = (F, y, st, PB)
    cols = [f"{m}|{v}" for m in TOKEN + CLASSIC for v in ["corr", "orig"]]
    with threadpool_limits(1), mp.get_context("fork").Pool(28) as pool:
        res = pool.map(_one, cols)
    out.update(dict(res))
    return out


def verdict(c, inc):
    a = c["rho_orig"]["ci"][0] > 0 or c["rho_orig"]["ci"][1] < 0
    if not a:
        return "no signal"
    ro, rc = c["rho_orig"]["est"], c["rho_corr"]["est"]
    b = np.sign(rc) == np.sign(ro) and abs(rc) >= 0.5 * abs(ro)
    cc = inc["ci"][0] > 0
    why = [w for w, ok in [("rho", b), ("dR2", cc)] if not ok]
    return "survives" if not why else f"fails ({'+'.join(why)})"


def main():
    warnings.filterwarnings("ignore")
    TAB.mkdir(parents=True, exist_ok=True)
    res = {}
    for corpus in ["wi", "fce"]:
        U = unit_table(corpus)
        U = U.copy()
        U.to_parquet(OUT / f"units_{corpus}.parquet")
        cor = correlations(U)
        inc = increments(U, corpus)
        rows = []
        for m in TOKEN + CLASSIC:
            c = cor[m]
            r = {"measure": m, "kind": "token" if m in TOKEN else "classic"}
            for s in STATS:
                r[s] = c[s]["est"]
                r[f"{s}_lo"], r[f"{s}_hi"] = c[s]["ci"]
            for v in ["corr", "orig"]:
                r[f"dR2_{v}"] = inc[f"{m}|{v}"]["dR2"]
                r[f"dR2_{v}_lo"], r[f"dR2_{v}_hi"] = inc[f"{m}|{v}"]["ci"]
            r["verdict"] = verdict(c, inc[f"{m}|corr"])
            r["n_missing_orig"], r["n_missing_corr"] = c["n_missing_orig"], c["n_missing_corr"]
            rows.append(r)
        T = pd.DataFrame(rows)
        T.to_csv(TAB / f"levels_{corpus}.csv", index=False)
        res[corpus] = {"n": len(U), "B_r2": inc["B_r2"], "rows": rows}
        print(T[["measure", "rho_orig", "rho_corr", "removed", "partial_orig", "partial_corr", "dR2_corr",
                 "dR2_corr_lo", "dR2_corr_hi", "verdict"]].round(3).to_string())
    both = {}
    for m in TOKEN + CLASSIC:
        v = [next(r for r in res[c]["rows"] if r["measure"] == m)["verdict"] for c in ["wi", "fce"]]
        both[m] = "survives" if all(x == "survives" for x in v) else " / ".join(v)
    res["overall"] = both
    (TAB / "levels.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(both, indent=1))


if __name__ == "__main__":
    main()
