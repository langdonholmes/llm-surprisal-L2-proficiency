"""Dolma n-gram attestation (corpus a edited, b unedited): log counts, PMI, conditional log-probs, unseen flags."""
from lab.att_common import block


def featurize(ctx):
    return block(ctx, prefixes=("a_log_", "a_pmi", "a_lp_", "a_unseen", "b_log_", "b_pmi", "b_lp_", "b_unseen"))
