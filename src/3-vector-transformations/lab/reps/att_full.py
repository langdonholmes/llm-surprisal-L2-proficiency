"""All attestation/lexis/semantic features + slot attestation (no LM quantities)."""
import numpy as np

from lab.att_common import block, slot


def featurize(ctx):
    return np.hstack([block(ctx, prefixes=("",)), slot(ctx).to_numpy("float32")])
