"""h_obs at layer 8 plus the surprisal and next-word consequence scalars."""
import numpy as np

from lab.hid_feats import consequence, h_obs


def featurize(ctx):
    s = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    return np.hstack([h_obs(8), s, consequence(ctx)])
