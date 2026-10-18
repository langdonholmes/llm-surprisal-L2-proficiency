"""Counterfactual contrast at layer 16: h_obs(writer word) - h_obs(model top-1 substituted)."""
from lab.hid_feats import h_cf, h_obs


def featurize(ctx):
    return h_obs(16) - h_cf(16)
