"""ctx_gain plus the four base surprisal/entropy scalars at the target and the next word."""
import numpy as np

from lab.reps.ctx_gain import featurize as gain


def featurize(ctx):
    s4 = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([gain(ctx), s4])
