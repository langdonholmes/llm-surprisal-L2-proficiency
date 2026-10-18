"""Word-centered delta plus surprisal scalars (target, next, word-relative z)."""
import numpy as np

from lab.geo_common import scalars, word_mean_pred


def featurize(ctx):
    wc = ctx.e_pred("t") - word_mean_pred(ctx.get("t_target_id"))
    sc = scalars(ctx)
    # scale scalars to the same total variance as the vector block
    sc = (sc - sc.mean(0)) / sc.std(0) * np.sqrt((wc.var(0).sum()) / sc.shape[1])
    return np.hstack([wc, sc])
