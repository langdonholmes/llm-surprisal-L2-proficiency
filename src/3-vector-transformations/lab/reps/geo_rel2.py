"""Relational features + category mismatch (mass on observed word's class) + top-1 margin (42 dims)."""
import numpy as np

from lab.geo_rel import extra, features


def featurize(ctx):
    return np.hstack([features(ctx)[0], extra(ctx)[0]])
