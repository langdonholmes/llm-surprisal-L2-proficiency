"""Pilot DLA: pre-declared statistics, the gold contrast and the by-level profile
of acceptable tokens (NOTES.md sections 2 and 4).

Outputs (src/3-surprisal-decomposition/pilot_dla/out/):
    auc_bins.csv        within-S-quintile AUC (ERR vs OK) for every stat and baseline
    cv_auc.csv          essay-grouped 5-fold logistic AUCs (baseline vs + DLA)
    ok_by_level.csv     OK tokens: by-level essay-bootstrap means, raw and S/H-residualised
    ok_tests.csv        rho with level, A - C, N - C (in token SD units)
    top_components.csv  components carrying most negative mass, by class

    .venv/bin/python src/3-surprisal-decomposition/pilot_dla/analyze.py
    .venv/bin/python src/3-surprisal-decomposition/pilot_dla/analyze.py obs   # post hoc variant

The `obs` variant (post hoc, after the main result) attributes the observed
word's logit relative to the vocabulary mean, logit(obs) - mean_v logit_v, so a
negative contribution means a write pushes obs itself down, not that it
promotes the predicted word. Outputs get an `obs_` prefix; the centred logit
itself is added to the DLA stats as `obs_logit`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.bench import S3  # noqa: E402

IN = S3 / "pilot_dla"
OUT = HERE / "out"
SEED = 20261019
LEVELS = ["A", "B", "C", "N"]
DLA_STATS = ["N", "P", "supp", "max_neg", "top5_neg", "negH", "neg_layer", "neg_mlp"]
PRIMARY = ["supp", "max_neg"]
BASE = ["S_bits", "H_bits", "log_top1", "a_log_uni"]


def stats(C: np.ndarray, comps: list[str]) -> pd.DataFrame:
    layer = np.array([0 if c == "emb" else int(c[1:].split("H")[0].split("M")[0]) + 1 for c in comps])
    is_mlp = np.array([c.endswith("MLP") for c in comps])
    neg = np.clip(-C, 0, None)
    pos = np.clip(C, 0, None)
    Nm, Pm = neg.sum(1), pos.sum(1)
    q = neg / np.maximum(Nm[:, None], 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(q > 0, q * np.log(q), 0), 1) / np.log(C.shape[1])
    return pd.DataFrame({
        "N": -Nm, "P": Pm, "supp": Nm / (Nm + Pm), "max_neg": neg.max(1),
        "top5_neg": np.sort(neg, 1)[:, -5:].sum(1) / Nm, "negH": ent,
        "neg_layer": (neg * layer).sum(1) / Nm, "neg_mlp": neg[:, is_mlp].sum(1) / Nm})


def auc_bins(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(bin=pd.qcut(d.S_bits, 5, labels=False))
    rows = []
    for v in DLA_STATS + BASE:
        r = {"stat": v}
        for b, g in d.groupby("bin"):
            r[f"bin{b}"] = roc_auc_score(g.err, g[v])
        r["mean_within_bin"] = np.mean([r[f"bin{b}"] for b in range(5)])
        r["pooled"] = roc_auc_score(d.err, d[v])
        rows.append(r)
    return pd.DataFrame(rows)


def cv_auc(d: pd.DataFrame, feats: list[str]) -> float:
    X, y, g = d[feats].to_numpy(), d.err.to_numpy(), d.text_id.to_numpy()
    pred = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred)


def residualise(ok: pd.DataFrame, v: str) -> np.ndarray:
    S, H = ok.S_bits.to_numpy(), ok.H_bits.to_numpy()
    X = np.column_stack([np.ones(len(ok))] + [S ** p for p in (1, 2, 3)] + [H ** p for p in (1, 2, 3)])
    X = (X - np.r_[0, X[:, 1:].mean(0)]) / np.r_[1, X[:, 1:].std(0)]
    beta, *_ = np.linalg.lstsq(X, ok[v].to_numpy(), rcond=None)
    return ok[v].to_numpy() - X @ beta


def boot_level(e: pd.DataFrame, col: str, rng) -> dict:
    out = {}
    for lv in LEVELS:
        x = e[e.level == lv][col].to_numpy()
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        out[lv] = (x.mean(), *np.percentile(bs, [2.5, 97.5]))
    return out


def ok_level(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = d[~d.err].copy()
    rng = np.random.default_rng(SEED)
    rows, tests = [], []
    for v in DLA_STATS + BASE:
        sd = ok[v].std()
        ok[f"{v}_res"] = residualise(ok, v) if v not in ("S_bits", "H_bits") else ok[v] - ok[v].mean()
        for kind, col in [("raw", v), ("resid", f"{v}_res")]:
            e = ok.groupby("text_id").agg(level=("level", "first"), x=(col, "mean")).reset_index()
            bl = boot_level(e, "x", rng)
            for lv, (m, lo, hi) in bl.items():
                rows.append({"stat": v, "kind": kind, "level": lv, "mean": m, "lo": lo, "hi": hi,
                             "n_essays": (e.level == lv).sum()})
            le = e[e.level != "N"]
            lvn = le.level.map({"A": 1, "B": 2, "C": 3}).to_numpy()
            x = le.x.to_numpy()
            idx = rng.integers(0, len(x), (2000, len(x)))
            rb = [spearmanr(x[i], lvn[i])[0] for i in idx]
            ea = {lv: e[e.level == lv].x.to_numpy() for lv in LEVELS}
            ac = [rng.choice(ea["A"], len(ea["A"])).mean() - rng.choice(ea["C"], len(ea["C"])).mean() for _ in range(2000)]
            nc = [rng.choice(ea["N"], len(ea["N"])).mean() - rng.choice(ea["C"], len(ea["C"])).mean() for _ in range(2000)]
            tests.append({"stat": v, "kind": kind, "rho": spearmanr(x, lvn)[0],
                          "rho_lo": np.percentile(rb, 2.5), "rho_hi": np.percentile(rb, 97.5),
                          "A_minus_C_sd": (ea["A"].mean() - ea["C"].mean()) / sd,
                          "AC_lo_sd": np.percentile(ac, 2.5) / sd, "AC_hi_sd": np.percentile(ac, 97.5) / sd,
                          "N_minus_C_sd": (ea["N"].mean() - ea["C"].mean()) / sd,
                          "NC_lo_sd": np.percentile(nc, 2.5) / sd, "NC_hi_sd": np.percentile(nc, 97.5) / sd})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def top_components(C, comps, d) -> pd.DataFrame:
    neg = np.clip(-C, 0, None)
    share = neg / neg.sum(1, keepdims=True)
    rows = []
    for name, m in [("ERR", d.err.to_numpy()), ("OK", ~d.err.to_numpy())]:
        s = share[m].mean(0)
        top = np.argsort(-neg[m], 1)[:, 0]
        argmax_share = np.bincount(top, minlength=len(comps)) / m.sum()
        for i in np.argsort(-s)[:15]:
            rows.append({"cls": name, "component": comps[i], "mean_neg_share": s[i],
                         "share_tokens_top_suppressor": argmax_share[i], "mean_contrib": C[m, i].mean()})
    return pd.DataFrame(rows)


def main():
    global DLA_STATS
    variant = sys.argv[1] if len(sys.argv) > 1 else "gap"
    pre = "" if variant == "gap" else "obs_"
    OUT.mkdir(exist_ok=True)
    d = pd.read_parquet(IN / "sample.parquet")
    C = np.load(IN / ("contrib.npy" if variant == "gap" else "contrib_obs.npy"))
    comps = json.loads((IN / "components.json").read_text())
    d = pd.concat([d, stats(C, comps)], axis=1)
    d["err"] = d.cls == "ERR"
    if variant == "obs":
        d["obs_logit"] = d.obs_centered_logit
        DLA_STATS = DLA_STATS + ["obs_logit"]
    d.to_parquet(IN / f"{pre}sample_stats.parquet", index=False)

    a = auc_bins(d)
    a.to_csv(OUT / f"{pre}auc_bins.csv", index=False)
    print(a.round(3).to_string(index=False))

    cv = []
    for name, feats in [("S", ["S_bits"]), ("S+H", ["S_bits", "H_bits"]), ("baseline", BASE),
                        ("baseline+supp+max_neg", BASE + PRIMARY), ("baseline+DLA", BASE + DLA_STATS),
                        ("S+H+DLA", ["S_bits", "H_bits"] + DLA_STATS), ("DLA only", DLA_STATS)]:
        cv.append({"model": name, "auc": cv_auc(d, feats)})
        # same, within level (level differences in base rate removed)
        cv[-1]["auc_mean_within_level"] = np.mean([cv_auc(d[d.level == lv], feats) for lv in LEVELS])
    cv = pd.DataFrame(cv)
    cv.to_csv(OUT / f"{pre}cv_auc.csv", index=False)
    print(cv.round(3).to_string(index=False))

    lev, tests = ok_level(d)
    lev.to_csv(OUT / f"{pre}ok_by_level.csv", index=False)
    tests.to_csv(OUT / f"{pre}ok_tests.csv", index=False)
    print(lev.pivot_table(index=["stat", "kind"], columns="level", values="mean").round(3).to_string())
    print(tests.round(3).to_string(index=False))

    tc = top_components(C, comps, d)
    tc.to_csv(OUT / f"{pre}top_components.csv", index=False)
    print(tc.round(3).to_string(index=False))
    print(d.groupby("cls")[DLA_STATS + BASE].mean().round(3).T)


if __name__ == "__main__":
    main()
