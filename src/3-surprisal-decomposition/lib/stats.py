"""Small statistics shared by the stage scripts."""

import numpy as np
from scipy.stats import rankdata

SEED = 42


# --------------------------------------------------------------------------- #
# unit bootstrap of covariance shares and rank correlations
# --------------------------------------------------------------------------- #
def boot(stat, n: int, n_boot: int = 2000, seed: int = SEED):
    """Point estimate and 95% percentile CI of ``stat(index)`` over unit resamples."""
    rng = np.random.default_rng(seed)
    point = np.asarray(stat(np.arange(n)), float)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def cov_shares(X, y, **kw):
    """Each column's share of cov(X[:, 0], y). The columns that add up to X[:, 0]
    split the covariance exactly, so their shares add up to 1."""
    X, y = np.asarray(X, float), np.asarray(y, float)

    def shares(idx):
        cov = ((X[idx] - X[idx].mean(0)) * (y[idx] - y[idx].mean())[:, None]).mean(0)
        return cov / cov[0]
    return boot(shares, len(y), **kw)


def spearman_cols(X, y):
    """Spearman rho of every column of X with y (no missing values)."""
    rank_x = np.apply_along_axis(rankdata, 0, np.asarray(X, float))
    rank_y = rankdata(y)
    rank_x, rank_y = rank_x - rank_x.mean(0), rank_y - rank_y.mean()
    return (rank_x * rank_y[:, None]).sum(0) / np.sqrt((rank_x ** 2).sum(0) * (rank_y ** 2).sum())


def partial_rank(x, y, z):
    """Partial Spearman: Pearson of rank(x) and rank(y) after regressing out rank(z)."""
    rank_x, rank_y, rank_z = rankdata(x), rankdata(y), rankdata(z)
    z_centred = rank_z - rank_z.mean()

    def residual(u):
        u = u - u.mean()
        return u - z_centred * (z_centred @ u) / (z_centred @ z_centred)
    return float(np.corrcoef(residual(rank_x), residual(rank_y))[0, 1])


# --------------------------------------------------------------------------- #
# incremental validity: ridge, repeated stratified CV, unit bootstrap of ΔR²
# --------------------------------------------------------------------------- #
N_REPEATS, N_FOLDS = 10, 10
ALPHAS = [10.0 ** k for k in range(-3, 6)]


def ridge(seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=seed)))


def oof(X, y, strata):
    """Out-of-fold predictions, shape (repeats, units). Mean imputation, scaling
    and alpha (inner 5-fold CV) are fitted in-fold; the folds are the same for
    every feature set."""
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, float)
    preds = np.empty((N_REPEATS, len(y)))
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep).split(X, strata)
        for fold, (train, test) in enumerate(folds):
            preds[rep, test] = ridge(SEED + 1000 * rep + fold).fit(X[train], y[train]).predict(X[test])
    return preds


def r2(preds, y):
    return 1 - ((preds - y) ** 2).sum(-1) / ((y - y.mean()) ** 2).sum()


def delta_r2(preds_a, preds_b, y, n_boot: int = 2000, seed: int = SEED) -> dict:
    """R²(a) − R²(b), averaged over repeats, with a unit-bootstrap CI."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        y_boot = y[idx]
        # (SSE_b - SSE_a) / SST, averaged over the CV repeats
        boots[b] = (((preds_b[:, idx] - y_boot) ** 2).sum(-1) - ((preds_a[:, idx] - y_boot) ** 2).sum(-1)).mean() \
            / ((y_boot - y_boot.mean()) ** 2).sum()
    diffs = r2(preds_a, y) - r2(preds_b, y)
    return {"delta_r2": float(diffs.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "share_repeats_gt0": float((diffs > 0).mean())}
