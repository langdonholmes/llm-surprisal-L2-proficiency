"""Layer-8 state change on reading the target: h_obs_8 - h_pred_8 (same window)."""


def featurize(ctx):
    return ctx.get("t_h_obs_8") - ctx.get("t_h_pred_8")
