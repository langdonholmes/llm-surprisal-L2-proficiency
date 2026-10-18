"""layer-8 state after reading the next word."""


def featurize(ctx):
    return ctx.get("n_h_obs_8")
