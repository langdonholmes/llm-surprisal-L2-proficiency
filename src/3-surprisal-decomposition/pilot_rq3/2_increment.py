"""RQ3 pilot, step 2 (CPU) — do surprisal components predict proficiency beyond error counts?

Ridge (mean imputation and standardisation in-fold, alpha by inner 5-fold CV),
10 x 10-fold CV stratified by level (W&I) or score band (FCE), the same folds
for every model. Per repeat: out-of-fold R2 and Spearman rho. Increment
Delta = R2(B + S) - R2(B), CI from a unit bootstrap of the stored out-of-fold
predictions (Delta averaged over repeats inside each resample). Drop-one-term
increments and standardized coefficients refit on all units. Design and rule:
NOTES.md.

Writes data/study3/pilot_rq3/tables/:
    models_<corpus>.csv     R2 and rho per model (mean, min, max over repeats)
    increment_<corpus>.csv  Delta R2 / Delta rho vs B, bootstrap CI, share of repeats > 0
    dropone_<corpus>.csv    Delta R2 when one S term (block) is dropped from B + S
    coefs_<corpus>.csv      standardized ridge coefficients, all units
    results.json            all of the above

    python src/3-surprisal-decomposition/pilot_rq3/2_increment.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_common import (ALPHAS, COST, MODELS, N_BOOT, N_FOLDS, N_REPEATS, OUT, S_COMPACT,  # noqa: E402
                        S_FULL, SEED, SPILL, TAB, B)

FCE_BANDS = [0, 23.5, 27.5, 31.5, 40.5]
DROP = {  # name -> (base model features, features dropped)
    "compact - clean": (B + S_COMPACT, ["clean"]),
    "compact - cost_mean": (B + S_COMPACT, ["cost_mean"]),
    "compact - spill_mean": (B + S_COMPACT, ["spill_mean"]),
    "full - clean": (B + S_FULL, ["clean"]),
    "full - cost block": (B + S_FULL, COST + ["cost_flagged"]),
    "full - spill block": (B + S_FULL, SPILL),
    "full - cost_flagged": (B + S_FULL, ["cost_flagged"]),
}


def pipe(seed):
    return make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=seed)))


def strata(F, corpus):
    if corpus == "wi":
        return F.level.to_numpy().astype(int)
    return np.digitize(F.level.to_numpy(), FCE_BANDS)


def oof(F, y, strat, feats):
    """Out-of-fold predictions, shape (repeats, units), and chosen alphas."""
    X = F[feats].to_numpy(float)
    P = np.empty((N_REPEATS, len(y)))
    alphas = []
    for r in range(N_REPEATS):
        for j, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(X, strat)):
            m = pipe(SEED + 1000 * r + j).fit(X[tr], y[tr])
            P[r, te] = m.predict(X[te])
            alphas.append(m[-1].alpha_)
    return P, alphas


def r2(P, y):
    return 1 - ((P - y) ** 2).sum(-1) / ((y - y.mean()) ** 2).sum()


def rho(P, y):
    ry = rankdata(y)
    return np.array([np.corrcoef(rankdata(p), ry)[0, 1] for p in P])


def boot_delta(Pa, Pb, y, rng_seed=SEED):
    """Bootstrap of mean-over-repeats R2(a) - R2(b), resampling units."""
    rng = np.random.default_rng(rng_seed)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        sst = ((yi - yi.mean()) ** 2).sum()
        out[b] = (((Pb[:, i] - yi) ** 2).sum(-1) - ((Pa[:, i] - yi) ** 2).sum(-1)).mean() / sst
    return out


def boot_drho(Pa, Pb, y, rng_seed=SEED, n=500):
    rng = np.random.default_rng(rng_seed)
    pa, pb = Pa.mean(0), Pb.mean(0)  # repeat-averaged predictions keep this cheap
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ry = rankdata(y[i])
        out.append(np.corrcoef(rankdata(pa[i]), ry)[0, 1] - np.corrcoef(rankdata(pb[i]), ry)[0, 1])
    return np.array(out)


def run(corpus):
    F = pd.read_parquet(OUT / f"features_{corpus}.parquet")
    y = F.level.to_numpy(float)
    st = strata(F, corpus)
    preds, alph = {}, {}
    for name, feats in {**MODELS, **{k: [f for f in v[0] if f not in v[1]] for k, v in DROP.items()}}.items():
        preds[name], alph[name] = oof(F, y, st, feats)
    np.savez(OUT / f"oof_{corpus}.npz", y=y, units=F.index.to_numpy(str), **preds)
    res = {"n": int(len(y)), "models": {}, "increment": {}, "dropone": {}}
    rows = []
    for name in MODELS:
        a, p = r2(preds[name], y), rho(preds[name], y)
        res["models"][name] = {"r2": float(a.mean()), "r2_range": [float(a.min()), float(a.max())],
                               "rho": float(p.mean()), "rho_range": [float(p.min()), float(p.max())],
                               "alpha_median": float(np.median(alph[name])), "n_features": len(MODELS[name])}
        rows.append({"model": name, **res["models"][name]})
    pd.DataFrame(rows).to_csv(TAB / f"models_{corpus}.csv", index=False)

    rows = []
    for name in ["B+S_compact", "B+S_full", "B+mean_surprisal"]:
        d_rep = r2(preds[name], y) - r2(preds["B"], y)
        bs = boot_delta(preds[name], preds["B"], y)
        drho = rho(preds[name], y) - rho(preds["B"], y)
        bsr = boot_drho(preds[name], preds["B"], y)
        rec = {"vs": "B", "model": name, "delta_r2": float(d_rep.mean()),
               "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
               "repeat_range": [float(d_rep.min()), float(d_rep.max())],
               "share_repeats_gt0": float((d_rep > 0).mean()),
               "delta_rho": float(drho.mean()), "delta_rho_ci": [float(x) for x in np.percentile(bsr, [2.5, 97.5])]}
        res["increment"][name] = rec
        rows.append(rec)
    pd.DataFrame(rows).to_csv(TAB / f"increment_{corpus}.csv", index=False)

    rows = []
    for name, (base, _) in DROP.items():
        full = "B+S_compact" if base == B + S_COMPACT else "B+S_full"
        d_rep = r2(preds[full], y) - r2(preds[name], y)
        bs = boot_delta(preds[full], preds[name], y)
        rec = {"drop": name, "loss_r2": float(d_rep.mean()),
               "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
               "share_repeats_gt0": float((d_rep > 0).mean())}
        res["dropone"][name] = rec
        rows.append(rec)
    pd.DataFrame(rows).to_csv(TAB / f"dropone_{corpus}.csv", index=False)

    rows = []
    for name in ["B", "B+S_compact", "B+S_full"]:
        m = pipe(SEED).fit(F[MODELS[name]].to_numpy(float), y)
        for f, c in zip(MODELS[name], m[-1].coef_):
            rows.append({"model": name, "feature": f, "coef_sd": float(c), "alpha": float(m[-1].alpha_)})
    co = pd.DataFrame(rows)
    co.to_csv(TAB / f"coefs_{corpus}.csv", index=False)
    res["coefs"] = co.to_dict(orient="records")
    # how much the S terms overlap with B: R2 of each compact S term on B (in-sample, OLS)
    Xb = np.column_stack([np.ones(len(F)), StandardScaler().fit_transform(F[B])])
    ov = {}
    for f in S_COMPACT + ["total"]:
        v = F[f].fillna(F[f].mean()).to_numpy(float)
        beta, *_ = np.linalg.lstsq(Xb, v, rcond=None)
        ov[f] = float(1 - ((v - Xb @ beta) ** 2).sum() / ((v - v.mean()) ** 2).sum())
    res["r2_of_S_term_on_B"] = ov
    return res


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    TAB.mkdir(parents=True, exist_ok=True)
    res = {c: run(c) for c in ["wi", "fce"]}
    for s in ["B+S_compact", "B+S_full"]:
        w, f = res["wi"]["increment"][s], res["fce"]["increment"][s]
        # W&I CI above 0; FCE CI excludes 0 on the same side
        res[f"verdict_{s}"] = bool(w["ci"][0] > 0 and f["ci"][0] > 0)
    (TAB / "results.json").write_text(json.dumps(res, indent=1))
    for c in ["wi", "fce"]:
        print(c, "n", res[c]["n"])
        print(pd.read_csv(TAB / f"models_{c}.csv").round(4).to_string())
        print(pd.read_csv(TAB / f"increment_{c}.csv").round(4).to_string())
        print(pd.read_csv(TAB / f"dropone_{c}.csv").round(4).to_string())
        print("R2 of S term on B:", {k: round(v, 3) for k, v in res[c]["r2_of_S_term_on_B"].items()})
    print("verdict compact:", res["verdict_B+S_compact"], "| full:", res["verdict_B+S_full"])


if __name__ == "__main__":
    main()
