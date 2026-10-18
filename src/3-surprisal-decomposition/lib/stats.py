"""Small statistics shared by the stage scripts."""

import numpy as np
from scipy.stats import spearmanr

SEED = 42


def boot_spearman(x, y, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Spearman rho with a 95% percentile bootstrap CI over cases (essays)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    boots = [spearmanr(x[i], y[i])[0] for i in idx]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(spearmanr(x, y)[0]), float(lo), float(hi)


def weighted_d(x: np.ndarray, group: np.ndarray, level: np.ndarray) -> dict:
    """Study 1's L1 effect size: for each group, Cohen's d against all other
    essays within each level, averaged over levels weighted by the group's n."""
    out = {}
    for g in np.unique(group):
        num = wsum = 0.0
        for s in np.unique(level):
            foc = (level == s) & (group == g)
            rst = (level == s) & (group != g)
            n1, n2 = foc.sum(), rst.sum()
            if n1 < 2 or n2 < 2:
                continue
            sp = np.sqrt(((n1 - 1) * x[foc].var(ddof=1) + (n2 - 1) * x[rst].var(ddof=1))
                         / (n1 + n2 - 2))
            num += n1 * (x[foc].mean() - x[rst].mean()) / sp
            wsum += n1
        out[g] = num / wsum
    return out
