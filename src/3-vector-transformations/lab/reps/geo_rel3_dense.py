"""geo_rel3 without one-hot blocks (no observed class, no UPOS): continuous masses, relations, morph, scalars, wc16."""
import numpy as np

from lab.geo_rel import extra, features, morph, wc_pcs


def featurize(ctx):
    X, names = features(ctx)
    keep = [i for i, n in enumerate(names) if not n.startswith("obs_")]
    return np.hstack([X[:, keep], extra(ctx)[0], morph(ctx)[0], wc_pcs(ctx, 16)])
