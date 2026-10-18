"""Analogy-style offset: e(best repair) - e(observed) in W_U, using the cost-adjusted best edit (zero vector if no edit pays; for an insertion, the inserted word's embedding)."""
import numpy as np
from transformers import AutoTokenizer

from lab.lm import HF_ID, REVISION
from lab.repair_feats import mdl_table


def featurize(ctx):
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    c = mdl_table(ctx)
    best = c.loc[c.groupby("inst").gm.idxmax()]
    W = ctx.W_U
    obs = ctx.get("t_target_id")
    X = np.zeros((ctx.n, W.shape[1]), dtype=np.float32)
    for r in best.itertuples():
        if r.gm <= 0:
            continue
        w = str(r.word).strip()
        if r.cls == "DEL" or not w:
            X[r.inst] = -W[obs[r.inst]]
            continue
        rid = tok(" " + w, add_special_tokens=False).input_ids[0]
        X[r.inst] = W[rid] if r.cls == "INS" else W[rid] - W[obs[r.inst]]
    return X
