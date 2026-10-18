"""Counterfactual contrast at layer 8: h_obs(writer word) - h_obs(model top-1 substituted)."""
from lab.hid_feats import h_cf, h_obs


def featurize(ctx):
    return h_obs(8) - h_cf(8)
