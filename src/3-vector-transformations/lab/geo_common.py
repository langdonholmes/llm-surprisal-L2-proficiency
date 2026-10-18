"""Geometry lab — shared helpers over data/study3/lab/geo/stats.npz (real dev stats)."""

from functools import lru_cache

import numpy as np

from util.paths import DATA_DIR

GEO = DATA_DIR / "study3" / "lab" / "geo"
SHRINK = 10.0  # pseudo-count pulling rare words toward the global mean


@lru_cache(maxsize=1)
def stats() -> dict:
    return dict(np.load(GEO / "stats.npz"))


def word_mean_pred(obs_ids: np.ndarray) -> np.ndarray:
    """Shrunken dev-mean of e_pred at occurrences of each observed id."""
    s = stats()
    pos = np.searchsorted(s["seen"], obs_ids)
    pos = np.clip(pos, 0, len(s["seen"]) - 1)
    hit = s["seen"][pos] == obs_ids
    n = np.where(hit, s["n"][pos], 0).astype(np.float64)[:, None]
    m = np.where(hit[:, None], s["mean_pred"][pos], 0)
    return ((n * m + SHRINK * s["glob_pred"]) / (n + SHRINK)).astype(np.float32)


def word_surprisal_z(obs_ids: np.ndarray, surprisal: np.ndarray) -> np.ndarray:
    """Surprisal relative to how surprising this word usually is (shrunken)."""
    s = stats()
    pos = np.clip(np.searchsorted(s["seen"], obs_ids), 0, len(s["seen"]) - 1)
    hit = s["seen"][pos] == obs_ids
    n = np.where(hit, s["n"][pos], 0)
    mu = (n * np.where(hit, s["s_mean"][pos], 0) + SHRINK * s["s_glob_mean"]) / (n + SHRINK)
    sd = (n * np.where(hit, s["s_sd"][pos], 0) + SHRINK * s["s_glob_sd"]) / (n + SHRINK)
    return (surprisal - mu) / np.maximum(sd, 0.5)


def scalars(ctx) -> np.ndarray:
    """Surprisal/entropy at target and next word plus word-relative surprisal z."""
    return np.column_stack([
        ctx.get("t_surprisal"), ctx.get("t_entropy"), ctx.get("n_surprisal"), ctx.get("n_entropy"),
        word_surprisal_z(ctx.get("t_target_id"), ctx.get("t_surprisal")),
        word_surprisal_z(ctx.get("n_target_id"), ctx.get("n_surprisal")),
    ])
