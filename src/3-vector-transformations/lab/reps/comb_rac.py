"""Repair + attestation + context/capacity (all views, with the base surprisal scalars)."""
import numpy as np

from lab.reps import att_full_s, ctx_excess_s4, rep_mdl_plus_pos


def featurize(ctx):
    return np.hstack([rep_mdl_plus_pos.featurize(ctx), att_full_s.featurize(ctx), ctx_excess_s4.featurize(ctx)])
