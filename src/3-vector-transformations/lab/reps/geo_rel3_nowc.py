"""geo_rel3 without the word-centered PCs (all columns named)."""
import numpy as np

from lab.geo_rel import UPOS, extra, features, morph, obs_pos


def featurize(ctx):
    pos = obs_pos(ctx)
    return np.hstack([features(ctx)[0], extra(ctx)[0], morph(ctx)[0],
                      np.eye(len(UPOS))[pos[:, 0]], np.eye(len(UPOS))[pos[:, 1]]])
