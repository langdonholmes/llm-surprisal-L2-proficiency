"""Prediction only: top-50 renormalized expected output embedding, e_pred."""


def featurize(ctx):
    return ctx.e_pred("t")
