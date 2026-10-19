"""Pilot analysis: sanity checks, Q1 (errors vs OK controlled on surprisal), Q2 (OK by level), KWIC.

    .venv/bin/python src/3-surprisal-decomposition/pilot_lens/analyze.py
Writes data/study3/pilot_lens/{features.parquet, results.json, kwic.md} and prints tables.
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from lens_common import OUT

SEED = 42
LEVELS = ["A", "B", "C", "N"]
FOCAL = ["TL_late", "TL_early", "LL_late", "LL_early", "C_gain", "pool_ctx", "MLM_gain"]
SECONDARY = ["C_sent_gain", "kl_w8_full", "MLM_ctx", "s_full", "s_mlm_full"]
BASE = ["s16", "H", "logfreq"]


def load():
    s = pd.read_parquet(OUT / "sample.parquet")
    z = dict(np.load(OUT / "lens_scores.npz"))
    c = pd.read_parquet(OUT / "ctx_causal.parquet")
    m = pd.read_parquet(OUT / "ctx_mlm.parquet")
    f = s[["row_id", "text_id", "level", "gold", "err", "observed", "pos", "target_char", "text",
           "a_unseen_uni", "is_punct"]].copy()
    f["s16"] = s.surprisal_bits
    f["H"] = s.entropy_bits
    f["logfreq"] = s.a_log_uni
    for k in range(z["s_tl"].shape[1]):
        f[f"tl{k}"] = z["s_tl"][:, k]
        f[f"ll{k}"] = z["s_ll"][:, k]
        f[f"htl{k}"] = z["h_tl"][:, k]
    f["s16_lens"] = z["s_tl"][:, 16]
    f["TL_late"] = z["s_tl"][:, 8] - z["s_tl"][:, 16]
    f["TL_early"] = z["s_tl"][:, 1] - z["s_tl"][:, 8]
    f["LL_late"] = z["s_ll"][:, 8] - z["s_ll"][:, 16]
    f["LL_early"] = z["s_ll"][:, 1] - z["s_ll"][:, 8]
    f = pd.concat([f, c, m], axis=1)
    f["C_gain"] = f.s_w8 - f.s_full
    f["C_sent_gain"] = f.s_sent - f.s_full
    f["MLM_gain"] = f.s_mlm_sent - f.s_mlm_full
    zeta, psi = 2.0 ** -f.s_mlm_full, 2.0 ** -f.s_mlm_sent
    f["MLM_ctx"] = zeta * np.log2(zeta / psi)
    return f


def sanity(f):
    out = {}
    d = (f.s16_lens - f.s16).abs()
    out["lens16_vs_stored"] = {"median": d.median(), "p99": d.quantile(.99), "max": d.max()}
    out["ll16_eq_tl16_max"] = float((f.ll16 - f.tl16).abs().max())
    d = (f.s_w8 - f.s16).abs()
    out["ctx_w8_vs_stored"] = {"median": d.median(), "p99": d.quantile(.99), "max": d.max()}
    rows = []
    for k in range(17):
        rows.append({"layer": k,
                     "rho_tl_negfreq": spearmanr(f[f"tl{k}"], -f.logfreq, nan_policy="omit")[0],
                     "rho_ll_negfreq": spearmanr(f[f"ll{k}"], -f.logfreq, nan_policy="omit")[0],
                     "rho_tl_s16": spearmanr(f[f"tl{k}"], f.s16, nan_policy="omit")[0],
                     "mean_tl": f[f"tl{k}"].mean(), "mean_ll": f[f"ll{k}"].mean()})
    out["by_layer"] = pd.DataFrame(rows)
    # how much of each focal measure do the baselines already explain?
    r2 = {}
    for m in FOCAL + SECONDARY:
        d = f[BASE + [m]].dropna()
        X = spline_base().fit_transform(d[BASE])
        r2[m] = LinearRegression().fit(X, d[m]).score(X, d[m])
    out["r2_from_baselines"] = r2
    return out


def spline_base():
    return ColumnTransformer([("sp", SplineTransformer(n_knots=5, degree=3), ["s16", "H"]),
                              ("lin", StandardScaler(), ["logfreq"])])


def design(d, extra):
    X = spline_base().fit_transform(d[BASE])
    if extra:
        E = d[extra].to_numpy(float)
        miss = np.isnan(E)
        E = np.where(miss, np.nanmedian(E, 0), E)
        E = (E - E.mean(0)) / (E.std(0) + 1e-9)
        X = np.hstack([X, E] + ([miss[:, miss.any(0)].astype(float)] if miss.any() else []))
    return X


def oof(d, extra):
    X, y, g = design(d, extra), d.err.to_numpy(), d.text_id.to_numpy()
    p = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
        mdl = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
        p[te] = mdl.predict_proba(X[te])[:, 1]
    return p


def boot_auc_diff(y, p0, p1, groups, n=1000):
    rng = np.random.default_rng(SEED)
    ess = np.unique(groups)
    idx_by = {e: np.flatnonzero(groups == e) for e in ess}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([idx_by[e] for e in rng.choice(ess, len(ess))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    return np.percentile(diffs, [2.5, 97.5])


def q1(f):
    d = f[f.gold != "POST"].reset_index(drop=True)
    y, g = d.err.to_numpy(), d.text_id.to_numpy()
    res = {"n": len(d), "n_err": int(y.sum())}
    p_s = oof(d.assign(H=0.0, logfreq=0.0), [])  # surprisal only (H, freq zeroed)
    p0 = oof(d, [])
    res["auc_surprisal_only"] = roc_auc_score(y, p_s)
    res["auc_B1"] = roc_auc_score(y, p0)
    rows = []
    sets = {m: [m] for m in FOCAL + SECONDARY}
    sets["TL_trajectory"] = [f"tl{k}" for k in range(16)]
    sets["LL_trajectory"] = [f"ll{k}" for k in range(16)]
    sets["TL_traj+entropies"] = [f"tl{k}" for k in range(16)] + [f"htl{k}" for k in range(16)]
    sets["all_ctx"] = ["s_full", "s_sent", "h_full", "pool_ctx", "kl_w8_full", "s_mlm_sent", "s_mlm_full",
                       "h_mlm_full"]
    sets["all"] = sets["TL_trajectory"] + sets["all_ctx"]
    q = pd.qcut(d.s16, 5, labels=False)
    for name, cols in sets.items():
        p1 = oof(d, cols)
        lo, hi = boot_auc_diff(y, p0, p1, g)
        row = {"measure": name, "auc": roc_auc_score(y, p1), "d_auc": roc_auc_score(y, p1) - res["auc_B1"],
               "lo": lo, "hi": hi}
        if len(cols) == 1:
            v = d[cols[0]]
            aucs = [roc_auc_score(y[(q == b) & v.notna()], v[(q == b) & v.notna()]) for b in range(5)]
            row["raw_auc_within_s16_quintiles"] = float(np.mean(aucs))
        rows.append(row)
    res["table"] = pd.DataFrame(rows)
    return res


def essay_boot_means(d, col, n=2000):
    """Mean of col (token-weighted) with CI from essay resampling."""
    e = d.groupby("text_id")[col].agg(["sum", "count"])
    s, c = e["sum"].to_numpy(), e["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(e), (n, len(e)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), b


def q2(f):
    d = f[f.gold == "OK"].copy()
    cols = BASE + FOCAL + SECONDARY
    for m in FOCAL + SECONDARY:
        ok = d[m].notna()
        X = spline_base().fit_transform(d.loc[ok, BASE])
        d.loc[ok, f"{m}_res"] = d.loc[ok, m] - LinearRegression().fit(X, d.loc[ok, m]).predict(X)
    rows = []
    for m in cols + [f"{x}_res" for x in FOCAL + SECONDARY]:
        sd = d[m].std()
        boots = {}
        row = {"measure": m, "sd_ok": sd}
        for lv in LEVELS:
            dd = d[(d.level == lv) & d[m].notna()]
            mu, b = essay_boot_means(dd, m)
            boots[lv] = b
            row[lv] = mu
            row[f"{lv}_ci"] = f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
        for a, bb in [("C", "A"), ("N", "C")]:
            diff = boots[a] - boots[bb]
            row[f"{a}-{bb}"] = row[a] - row[bb]
            row[f"{a}-{bb}_ci"] = f"[{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]"
            row[f"{a}-{bb}_sd"] = (row[a] - row[bb]) / sd
            row[f"{a}-{bb}_excl0"] = bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0)
        rows.append(row)
    return pd.DataFrame(rows), d


def kwic(r, width=60):
    t, c = r.text, int(r.target_char)
    e = c + len(r.observed)
    return (t[max(0, c - width):c] + "[[" + t[c:e] + "]]" + t[e:e + 25]).replace("\n", " / ")


def qualitative(d):
    lines = ["# KWIC (local only; W&I licence: excerpts < 100 words, not for publication)\n"]
    rng = np.random.default_rng(SEED)
    ok = d[d.gold == "OK"]
    groups = {
        "TL: rescued late (TL_late top decile)": ok[ok.TL_late >= ok.TL_late.quantile(.9)],
        "TL: pushed down late (TL_late bottom decile)": ok[ok.TL_late <= ok.TL_late.quantile(.1)],
        "TL: never supported (min over layers 4-16 of s_TL >= 14 bits)":
            ok[ok[[f"tl{k}" for k in range(4, 17)]].min(1) >= 14],
        "C_gain high (top decile: essay context helps)": ok[ok.C_gain >= ok.C_gain.quantile(.9)],
        "C_gain low (bottom decile: essay context hurts)": ok[ok.C_gain <= ok.C_gain.quantile(.1)],
        "MLM_gain high (top decile)": ok[ok.MLM_gain >= ok.MLM_gain.quantile(.9)],
        "MLM_gain low (bottom decile)": ok[ok.MLM_gain <= ok.MLM_gain.quantile(.1)],
        "pool_ctx high (top decile)": ok[ok.pool_ctx >= ok.pool_ctx.quantile(.9)],
    }
    comp = {}
    for name, g in groups.items():
        comp[name] = {"n": len(g), **g.level.value_counts(normalize=True).reindex(LEVELS).round(3).to_dict(),
                      "top_pos": g.pos.value_counts(normalize=True).head(4).round(2).to_dict(),
                      "mean_s16": round(g.s16.mean(), 2), "mean_logfreq": round(g.logfreq.mean(), 2)}
        lines.append(f"\n## {name} (n = {len(g)})\n")
        for r in g.iloc[rng.permutation(len(g))[:25]].itertuples():
            lines.append(f"- [{r.level}] s16={r.s16:.1f} TL_late={r.TL_late:.1f} C_gain={r.C_gain:.1f} "
                         f"MLM_gain={r.MLM_gain:.1f} :: {kwic(r)}")
    (OUT / "kwic.md").write_text("\n".join(lines))
    return comp


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = load()
    f.drop(columns=["text"]).to_parquet(OUT / "features.parquet", index=False)
    sc = sanity(f)
    print("SANITY", json.dumps({k: v for k, v in sc.items() if k != "by_layer"}, indent=1, default=float))
    print(sc["by_layer"].round(3).to_string(index=False))
    traj = f.groupby(np.where(f.gold == "OK", "OK_" + f.level, f.gold.where(f.gold == "POST", "ERR")))[
        [f"tl{k}" for k in range(17)]].mean().round(2)
    print("TL trajectory means\n", traj.to_string())
    if "sanity" in sys.argv:
        return
    r1 = q1(f)
    print("Q1", r1["auc_surprisal_only"], r1["auc_B1"])
    print(r1["table"].round(3).to_string(index=False))
    t2, d = q2(f)
    print(t2[["measure", "sd_ok"] + LEVELS + ["C-A", "C-A_ci", "C-A_sd", "N-C", "N-C_ci", "N-C_sd"]]
          .round(3).to_string(index=False))
    comp = qualitative(f.assign(**{c: d[c] for c in []}))
    print(json.dumps(comp, indent=1))
    json.dump({"sanity": {k: v for k, v in sc.items() if k != "by_layer"},
               "by_layer": sc["by_layer"].to_dict("records"), "trajectory": traj.to_dict(),
               "q1": {k: v for k, v in r1.items() if k != "table"}, "q1_table": r1["table"].to_dict("records"),
               "q2_table": t2.to_dict("records"), "qual": comp},
              open(OUT / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
