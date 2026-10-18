"""att_full plus the four surprisal scalars (target and next word)."""
import numpy as np

from lab.att_common import block, slot


def featurize(ctx):
    s = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([block(ctx, prefixes=("",)), slot(ctx).to_numpy("float32"), s])
