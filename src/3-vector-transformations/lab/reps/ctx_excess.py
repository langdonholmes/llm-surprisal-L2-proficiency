"""Surprisal in excess of the model's own entropy (s - H) under every available view, plus the bidirectional and context gains."""
import numpy as np

from lab.ctx_feats import available, gains, load


def featurize(ctx):
    cols = [np.subtract(*load(v)) for v in available()]
    return np.column_stack(cols + list(gains().values()))
