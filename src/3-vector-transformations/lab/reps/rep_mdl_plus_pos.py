"""Best supervised readout: POS-split channel-cost gains (31) + raw gains (29) + right-context gains (29) + props (6) + surprisal (4)."""
import numpy as np

from lab.repair_feats import best_repair_props, mdl_matrix_pos, sub_matrix, surprisal4


def featurize(ctx):
    return np.hstack([mdl_matrix_pos(ctx, "gm"), sub_matrix(ctx, "g"), sub_matrix(ctx, "gr"),
                      best_repair_props(ctx), surprisal4(ctx)])
