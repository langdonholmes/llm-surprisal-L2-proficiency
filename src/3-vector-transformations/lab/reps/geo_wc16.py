"""Word-centered e_pred projected on its top 16 PCs (basis fitted on real dev rows)."""
from lab.geo_rel import wc_pcs


def featurize(ctx):
    return wc_pcs(ctx, 16)
