"""Relational features without the observed-word class one-hot (prediction side + relations + scalars only)."""
import numpy as np

from lab.geo_rel import features


def featurize(ctx):
    X, names = features(ctx)
    return X[:, [i for i, n in enumerate(names) if not n.startswith("obs_")]]
