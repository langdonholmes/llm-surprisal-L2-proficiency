"""Scalar baseline: surprisal and entropy at the target and at the next word."""
import numpy as np


def featurize(ctx):
    return np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
