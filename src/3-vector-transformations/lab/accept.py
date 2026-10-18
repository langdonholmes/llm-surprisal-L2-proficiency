"""Lab — acceptable-surprise battery on gold W&I+LOCNESS tokens above tau.

The question: among tokens the measure finds surprising, which are errors and
which are acceptable-but-unexpected? Gold comes from expert edits (9_wi.py).
A representation is any featurize(ctx) module in lab/reps/ run against
LAB_BENCH=real_wi (the bench built by `9_wi.py select`).

    python src/3-vector-transformations/lab/accept.py <rep> [...]          # development (W&I train)
    python src/3-vector-transformations/lab/accept.py --confirm <rep> ...  # held out: W&I dev + LOCNESS

Battery (logistic regression on standardized features, <= 64 PCs):

    auc_err_ok      AUC, error (ERR_TOK or ERR_INS) vs OK. Development: 5-fold
                    CV grouped by essay within W&I train. Confirmation: fit on
                    all of train, score W&I dev.
    auc_ins_ok      the same, ERR_INS vs OK (omissions surface one word late)
    prec_ok_at_50   precision of the "acceptable" call among the 50% of
                    tokens judged most acceptable (OK base rate in brackets)
    native_ok_rate  confirmation only: share of LOCNESS tokens above tau that
                    the model, at the threshold that gives 50% acceptable calls
                    on W&I dev, calls acceptable
    by level        auc_err_ok within each CEFR level A/B/C

--gbm swaps the probe for gradient boosting on the raw features
(HistGradientBoostingClassifier, fixed hyperparameters, no tuning), which
can use interactions and many dimensions that a 64-PC linear probe cannot.

Scores go to results/study3/accept_scoreboard.csv (one row per run).
POST tokens are excluded from fitting and scoring.
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LAB_BENCH", "real_wi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lab.harness import LAB, OUT, Ctx  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402

SEED = 42
GBM = False


def _model(d):
    if GBM:
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=SEED)
    return make_pipeline(StandardScaler(), PCA(min(64, d), random_state=SEED),
                         LogisticRegression(max_iter=3000))


def gold_frame(ctx) -> pd.DataFrame:
    g = pd.read_parquet(DATA_DIR / "study3" / "wi" / "tokens_gold.parquet",
                        columns=["row_id", "gold", "level", "split", "surprisal_bits"])
    return ctx.bench[["inst", "row_id", "text_id"]].merge(g, on="row_id", how="left")


def p_acceptable(X, g, fit, score, groups=None):
    """P(OK) for rows `score`, fit on rows `fit` (error vs OK only)."""
    y = (g.gold == "OK").to_numpy()
    out = np.full(len(g), np.nan)
    if groups is None:
        m = _model(X.shape[1]).fit(X[fit], y[fit])
        out[score] = m.predict_proba(X[score])[:, 1]
        return out
    idx = np.flatnonzero(fit)
    for tr, te in GroupKFold(5).split(idx, groups=groups[idx]):
        m = _model(X.shape[1]).fit(X[idx[tr]], y[idx[tr]])
        out[idx[te]] = m.predict_proba(X[idx[te]])[:, 1]
    return out


def battery(p, g, rows):
    err = g.gold.isin(["ERR_TOK", "ERR_INS"]).to_numpy()
    ok = (g.gold == "OK").to_numpy()
    ins = (g.gold == "ERR_INS").to_numpy()
    r = {}
    m = rows & (err | ok)
    r["auc_err_ok"] = roc_auc_score(ok[m], p[m])
    m2 = rows & (ins | ok)
    r["auc_ins_ok"] = roc_auc_score(ok[m2], p[m2])
    s = np.flatnonzero(m)
    top = s[np.argsort(-p[s])][: len(s) // 2]
    r["prec_ok_at_50"] = ok[top].mean()
    r["ok_base_rate"] = ok[s].mean()
    for lvl in ["A", "B", "C"]:
        mm = m & (g.level == lvl).to_numpy()
        r[f"auc_{lvl}"] = roc_auc_score(ok[mm], p[mm]) if mm.sum() > 10 else np.nan
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--gbm", action="store_true")
    args = ap.parse_args()
    global GBM
    GBM = args.gbm
    ctx = Ctx()
    g = gold_frame(ctx)
    train = (g.split == "train").to_numpy() & ctx.ok
    dev = (g.split == "dev").to_numpy() & (g.level != "N").to_numpy() & ctx.ok
    native = (g.level == "N").to_numpy() & ctx.ok
    usable = g.gold.isin(["ERR_TOK", "ERR_INS", "OK"]).to_numpy()
    for name in args.names:
        mod = importlib.import_module(f"lab.reps.{name}")
        cache = LAB / "reps" / f"{name}_real_wi.npy"
        X = np.load(cache) if cache.exists() else np.asarray(mod.featurize(ctx), dtype=np.float32)
        np.save(cache, X)
        X = np.nan_to_num(X.astype(np.float64))
        if args.confirm:
            p = p_acceptable(X, g, train & usable, dev | native)
            res = battery(p, g, dev)
            cut = np.nanquantile(p[dev & usable], 0.5)
            res["native_ok_rate"] = float((p[native] >= cut).mean())
            res["native_gold_ok_rate"] = float((g.gold[native] == "OK").mean())
            mode = "confirm"
        else:
            p = p_acceptable(X, g, train & usable, None, groups=g.text_id.to_numpy())
            res = battery(p, g, train)
            mode = "dev"
        mode += "_gbm" if GBM else ""
        row = {"rep": name, "mode": mode, "when": datetime.now().isoformat(timespec="seconds"),
               "dim": X.shape[1], **{k: round(float(v), 4) for k, v in res.items()}}
        sb = OUT / "accept_scoreboard.csv"
        pd.concat([pd.read_csv(sb) if sb.exists() else pd.DataFrame(), pd.DataFrame([row])]) \
            .to_csv(sb, index=False)
        np.save(LAB / "reps" / f"{name}_real_wi_p_{mode}.npy", p)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
