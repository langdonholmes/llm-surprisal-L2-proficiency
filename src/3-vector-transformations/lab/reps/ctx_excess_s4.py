"""ctx_excess plus the four base surprisal/entropy scalars (target and next word)."""
import numpy as np

from lab.reps.ctx_excess import featurize as excess


def featurize(ctx):
    s4 = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([excess(ctx), s4])
