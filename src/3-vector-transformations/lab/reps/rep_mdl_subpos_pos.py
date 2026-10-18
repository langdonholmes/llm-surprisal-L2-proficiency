"""Channel-cost repair profile, positive part, with -s inflection split by observed POS (noun/verb), 31 dims."""
from lab.repair_feats import mdl_matrix_pos


def featurize(ctx):
    return mdl_matrix_pos(ctx, "gm", positive=True)
