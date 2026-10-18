"""All attestation + lexis + semantic-fit features (no LM quantities)."""
from lab.att_common import block


def featurize(ctx):
    return block(ctx, prefixes=("",))
