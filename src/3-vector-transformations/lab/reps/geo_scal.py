"""Scalars only: surprisal/entropy at target and next word plus word-relative surprisal z (6 dims)."""
from lab.geo_common import scalars


def featurize(ctx):
    return scalars(ctx)
