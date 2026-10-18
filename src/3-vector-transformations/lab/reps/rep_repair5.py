"""Repair profile, 5 dims: best LM gain (bits) per edit class MORPH/FORM/PRED/DEL/INS."""
from lab.repair_feats import gain_matrix


def featurize(ctx):
    return gain_matrix(ctx, ("g",))
