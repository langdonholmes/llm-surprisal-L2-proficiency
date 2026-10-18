"""Repair profile (15) + best-repair properties (6) + surprisal/entropy at target and next (4)."""
import numpy as np

from lab.repair_feats import best_repair_props, gain_matrix, surprisal4


def featurize(ctx):
    return np.hstack([gain_matrix(ctx, ("g", "gx", "gr")), best_repair_props(ctx), surprisal4(ctx)])
