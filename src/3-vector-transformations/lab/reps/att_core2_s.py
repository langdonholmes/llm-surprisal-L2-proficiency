"""Compact named evidence: att_core + a-corpus slot/forward/backward attestation + surprisal scalars (32 dims)."""
import numpy as np

from lab.att_common import core, slot


def featurize(ctx):
    sl = slot(ctx)[["a_lp_slot", "a_slot_logn", "a_slot_types", "a_lp_fwd", "a_lp_bwd", "a_lp_bwd_bi"]]
    s = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([core(ctx).to_numpy("float32"), sl.to_numpy("float32"), s])
