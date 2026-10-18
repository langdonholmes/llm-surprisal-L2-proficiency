"""Surprisal and entropy of the observed first subword under every available view (1B/7B x 8/64 window, bidirectional MLM)."""
import numpy as np

from lab.ctx_feats import available, load


def featurize(ctx):
    return np.column_stack([a for v in available() for a in load(v)])
