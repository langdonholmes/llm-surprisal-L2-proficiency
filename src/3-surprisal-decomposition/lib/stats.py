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
    pt = np.asarray(stat(np.arange(n)), float)
    bs = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi


def cov_shares(X, y, **kw):
    """Each column's share of cov(X[:, 0], y). The columns that add up to X[:, 0]
    split the covariance exactly, so their shares add up to 1."""
    X, y = np.asarray(X, float), np.asarray(y, float)

    def sh(i):
        c = ((X[i] - X[i].mean(0)) * (y[i] - y[i].mean())[:, None]).mean(0)
        return c / c[0]
    return boot(sh, len(y), **kw)


def spearman_cols(X, y):
    """Spearman rho of every column of X with y (no missing values)."""
    rx = np.apply_along_axis(rankdata, 0, np.asarray(X, float))
    ry = rankdata(y)
    rx, ry = rx - rx.mean(0), ry - ry.mean()
    return (rx * ry[:, None]).sum(0) / np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())


def partial_rank(x, y, z):
    """Partial Spearman: Pearson of rank(x) and rank(y) after regressing out rank(z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    zc = rz - rz.mean()

    def res(u):
        u = u - u.mean()
        return u - zc * (zc @ u) / (zc @ zc)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


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
    P = np.empty((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(X, strata)
        for j, (tr, te) in enumerate(folds):
            P[r, te] = ridge(SEED + 1000 * r + j).fit(X[tr], y[tr]).predict(X[te])
    return P


def r2(P, y):
    return 1 - ((P - y) ** 2).sum(-1) / ((y - y.mean()) ** 2).sum()


def delta_r2(Pa, Pb, y, n_boot: int = 2000, seed: int = SEED) -> dict:
    """R²(a) − R²(b), averaged over repeats, with a unit-bootstrap CI."""
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        bs[b] = (((Pb[:, i] - yi) ** 2).sum(-1) - ((Pa[:, i] - yi) ** 2).sum(-1)).mean() / ((yi - yi.mean()) ** 2).sum()
    d = r2(Pa, y) - r2(Pb, y)
    return {"delta_r2": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "share_repeats_gt0": float((d > 0).mean())}
