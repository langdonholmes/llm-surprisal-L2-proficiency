"""Mean of h_obs over layers 6, 8, 10, 12 (middle-layer state after reading the target)."""
import numpy as np

from lab.hid_feats import h_obs


def featurize(ctx):
    return np.mean([h_obs(l) for l in (6, 8, 10, 12)], axis=0)
