"""Pipeline representation: delta = e_pred - e_obs in W_U, top-50 renormalized."""


def featurize(ctx):
    return ctx.e_pred("t") - ctx.e_obs("t")
