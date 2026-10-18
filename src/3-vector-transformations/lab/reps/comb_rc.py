"""Repair + context/capacity, no attestation (tests what Dolma lookups add)."""
import numpy as np

from lab.reps import ctx_excess_s4, rep_mdl_plus_pos


def featurize(ctx):
    return np.hstack([rep_mdl_plus_pos.featurize(ctx), ctx_excess_s4.featurize(ctx)])
