"""geo_rel2 + 16 word-centered PCs + inflectional-transition masses + spaCy UPOS of observed and next word."""
import numpy as np

from lab.geo_rel import UPOS, extra, features, morph, obs_pos, wc_pcs


def featurize(ctx):
    pos = obs_pos(ctx)
    return np.hstack([features(ctx)[0], extra(ctx)[0], wc_pcs(ctx, 16), morph(ctx)[0],
                      np.eye(len(UPOS))[pos[:, 0]], np.eye(len(UPOS))[pos[:, 1]]])
