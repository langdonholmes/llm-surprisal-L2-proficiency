"""att_core plus the four surprisal scalars and surprisal x attestation interactions."""
import numpy as np

from lab.att_common import core


def featurize(ctx):
    c = core(ctx)
    s = np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])
    inter = np.column_stack([s[:, 0] * c.attested_both_sides_a, s[:, 0] * c.rarity, s[:, 2] * c.unseen_bigram_any_a])
    return np.hstack([c.to_numpy("float32"), s, inter])
