"""geo_rel2 plus 16 PCs of word-centered e_pred (58 dims)."""
import numpy as np

from lab.geo_rel import extra, features, wc_pcs


def featurize(ctx):
    return np.hstack([features(ctx)[0], extra(ctx)[0], wc_pcs(ctx, 16)])
