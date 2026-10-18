"""Repair subclass gains (27) + right-context gains (27) + best-repair props (6) + surprisal (4)."""
import numpy as np

from lab.repair_feats import best_repair_props, sub_matrix, surprisal4


def featurize(ctx):
    return np.hstack([sub_matrix(ctx, "g"), sub_matrix(ctx, "gr"), best_repair_props(ctx), surprisal4(ctx)])
