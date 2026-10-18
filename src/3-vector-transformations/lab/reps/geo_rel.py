"""Relational features (39 named dims): predicted/observed word class, orthographic and embedding relation, surprisal scalars."""
from lab.geo_rel import features


def featurize(ctx):
    return features(ctx)[0]
