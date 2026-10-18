"""Identity baseline: the observed word's output embedding, W_U[obs] (no prediction)."""


def featurize(ctx):
    return ctx.e_obs("t")
