"""Repair (rep_mdl_plus_pos, 99) + attestation and lexis (att_full_s, 92): the two evidence families together."""
import numpy as np

from lab.reps import att_full_s, rep_mdl_plus_pos


def featurize(ctx):
    return np.hstack([rep_mdl_plus_pos.featurize(ctx), att_full_s.featurize(ctx)])
