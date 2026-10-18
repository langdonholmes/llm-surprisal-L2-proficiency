"""Shared loader for the hidden-state lab cache (hid_states.py output)."""
from functools import lru_cache

import numpy as np

from lab.harness import LAB, SUFFIX


@lru_cache(1)
def states():
    return np.load(LAB / "hidden" / f"states{SUFFIX}.npz", mmap_mode="r")


def h_obs(layer):
    return np.asarray(states()["h_obs"][:, layer], dtype=np.float32)


def h_cf(layer):
    return np.asarray(states()["h_cf"][:, layer], dtype=np.float32)


def consequence(ctx):
    """Next-word consequence scalars: logprob of the actual next word after the
    writer's word and after the model's expected word, their difference, KL, JS."""
    s = states()
    lo, lc = np.nan_to_num(s["n_lp_obs"]), np.nan_to_num(s["n_lp_cf"])
    return np.column_stack([lo, lc, lo - lc, np.nan_to_num(s["n_kl"]), np.nan_to_num(s["n_js"])])


SHRINK = 20.0


@lru_cache(1)
def ident():
    return dict(np.load(LAB / "hidden" / "ident.npz"))


def identity_component(ctx, layer):
    """Expected h_obs for the observed token type, estimated on real dev text
    (hid_ident.py): the type mean shrunk toward a ridge prediction from W_E."""
    I = ident()
    obs = ctx.get("t_target_id")
    WE = ctx.W_E
    ridge = np.hstack([WE[obs], np.ones((len(obs), 1), np.float32)]) @ I[f"ridge_L{layer}"]
    pos = np.searchsorted(I["ids"], obs)
    pos = np.clip(pos, 0, len(I["ids"]) - 1)
    seen = I["ids"][pos] == obs
    n = np.where(seen, I["counts"][pos], 0).astype(np.float32)[:, None]
    mean = np.where(seen[:, None], I[f"mean_L{layer}"][pos], 0.0)
    w = n / (n + SHRINK)
    return w * mean + (1 - w) * ridge


def h_resid(ctx, layer):
    """h_obs minus its identity component: how this occurrence differs from a
    typical occurrence of the same token."""
    return h_obs(layer) - identity_component(ctx, layer)
