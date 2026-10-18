"""Repair subclass profile, positive part only: max(0, best gain) per nameable edit (27 dims, NMF-ready)."""
from lab.repair_feats import sub_matrix


def featurize(ctx):
    return sub_matrix(ctx, "g", positive=True)
