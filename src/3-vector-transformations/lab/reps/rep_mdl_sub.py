"""Channel-cost repair profile: best (LM gain - 1 bit/char edited) per nameable edit subclass (29 dims)."""
from lab.repair_feats import mdl_matrix


def featurize(ctx):
    return mdl_matrix(ctx, "gm")
