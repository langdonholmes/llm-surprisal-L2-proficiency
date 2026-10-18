"""Identity-removed state at layer 4: h_obs minus the shrunk per-type mean from real dev text."""
from lab.hid_feats import h_resid


def featurize(ctx):
    return h_resid(ctx, 4)
