"""Discrete repair label: one-hot best edit subclass after channel cost, scaled by net gain (29 dims)."""
from lab.repair_feats import mdl_label


def featurize(ctx):
    return mdl_label(ctx)
