"""Dolma dependency attestation: own arc MI/count and children's min MI / unseen share, corpus a and b."""
from lab.att_common import block


def featurize(ctx):
    return block(ctx, prefixes=("a_arc", "a_kid", "b_arc", "b_kid"), names=("n_children",))
