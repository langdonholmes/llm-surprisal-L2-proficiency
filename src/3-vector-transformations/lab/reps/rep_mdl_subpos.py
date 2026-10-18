"""Channel-cost repair profile, positive part: max(0, gain - 1 bit/char) per edit subclass (29 dims)."""
from lab.repair_feats import mdl_matrix


def featurize(ctx):
    return mdl_matrix(ctx, "gm", positive=True)
