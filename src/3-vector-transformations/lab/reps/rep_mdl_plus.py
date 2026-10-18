"""Channel-cost subclass gains (29) + raw subclass gains (29) + right-context gains (29) + props (6) + surprisal (4)."""
import numpy as np

from lab.repair_feats import best_repair_props, mdl_matrix, sub_matrix, surprisal4


def featurize(ctx):
    return np.hstack([mdl_matrix(ctx, "gm"), sub_matrix(ctx, "g"), sub_matrix(ctx, "gr"),
                      best_repair_props(ctx), surprisal4(ctx)])
