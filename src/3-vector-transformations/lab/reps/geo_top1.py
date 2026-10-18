"""Top-1 contrast: W_U[top-1 prediction] - W_U[observed] (no mixture)."""


def featurize(ctx):
    return ctx.W_U[ctx.get("t_topk_ids")[:, 0]] - ctx.e_obs("t")
