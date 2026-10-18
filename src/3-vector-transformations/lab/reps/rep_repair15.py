"""Repair profile, 15 dims: per edit class, best gain split into total / edited material / right context."""
from lab.repair_feats import gain_matrix


def featurize(ctx):
    return gain_matrix(ctx, ("g", "gx", "gr"))
