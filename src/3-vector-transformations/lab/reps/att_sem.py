"""Semantic fit: cosine of the word's static vector to the essay's content centroid and to its sentence."""
from lab.att_common import block


def featurize(ctx):
    return block(ctx, names=("sim_essay", "sim_sent", "has_vector"))
