"""Surprisal scalars + next-word consequence (logp after obs vs after expected, KL, JS)."""
import numpy as np

from lab.hid_feats import consequence


def featurize(ctx):
    s = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([s, consequence(ctx)])
