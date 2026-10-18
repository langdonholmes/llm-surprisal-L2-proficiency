"""Word-centered delta: e_pred minus the dev-mean e_pred where this word occurs (= delta minus its word mean)."""
from lab.geo_common import word_mean_pred


def featurize(ctx):
    return ctx.e_pred("t") - word_mean_pred(ctx.get("t_target_id"))
