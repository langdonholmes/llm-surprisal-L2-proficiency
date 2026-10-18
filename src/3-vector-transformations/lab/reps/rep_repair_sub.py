"""Repair subclass profile, 27 dims: best LM gain per nameable edit (e.g. INS_DET, MORPH_S, FORM_NONWORD)."""
from lab.repair_feats import sub_matrix


def featurize(ctx):
    return sub_matrix(ctx, "g")
