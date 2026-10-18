"""Input-space delta: top-50 renormalized e_pred minus e_obs in W_E (input embeddings)."""


def featurize(ctx):
    W = ctx.W_E
    return ctx.e_pred("t", W=W) - ctx.e_obs("t", W=W)
