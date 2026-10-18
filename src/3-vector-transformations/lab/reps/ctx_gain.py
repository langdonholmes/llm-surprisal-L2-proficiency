"""Base surprisal (1B, 8) plus how much it drops under more context, more capacity, both, and right context (MLM)."""
import numpy as np

from lab.ctx_feats import gains, load


def featurize(ctx):
    g = gains()
    return np.column_stack([load("olmo1b_w8")[0]] + list(g.values()))
