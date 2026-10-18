"""h_obs at layer 4 (state after reading the target)."""
from lab.hid_feats import h_obs


def featurize(ctx):
    return h_obs(4)
