"""Attestation lab — load the per-token attestation/lexis table and select blocks."""

import numpy as np
import pandas as pd

from lab.att_build import ATT, build
from lab.harness import SUFFIX


def feats(ctx) -> pd.DataFrame:
    p = ATT / f"feats{SUFFIX}.parquet"
    return pd.read_parquet(p) if p.exists() else build(ctx.bench)


def block(ctx, prefixes=(), names=(), exclude=("seconds_total", "found")) -> np.ndarray:
    """Columns whose name starts with any prefix or is listed; NaN -> column median + missing flag."""
    f = feats(ctx)
    cols = [c for c in f.columns if c not in exclude and (c in names or any(c.startswith(p) for p in prefixes))]
    X = f[cols].astype(float)
    miss = X.columns[X.isna().any()]
    flags = X[miss].isna().astype(float).add_suffix("_missing")
    X = X.fillna(X.median())
    return pd.concat([X, flags], axis=1).to_numpy(np.float32)


def core(ctx) -> pd.DataFrame:
    """Compact, named acceptability evidence (no LM quantities)."""
    f = feats(ctx)
    o = pd.DataFrame(index=f.index)
    o["attested_both_sides_a"] = np.minimum(f.a_log_bi_left, f.a_log_bi_right)
    o["attested_trigram_mid_a"] = f.a_log_tri_mid
    o["attested_any_trigram_a"] = f[["a_log_tri_left", "a_log_tri_mid", "a_log_tri_right"]].max(1)
    o["unseen_bigram_any_a"] = np.maximum(f.a_unseen_bi_left, f.a_unseen_bi_right)
    o["pmi_left_a"] = f.a_pmi_left
    o["pmi_right_a"] = f.a_pmi_right
    o["rarity"] = -f.a_log_uni
    o["nonword"] = np.maximum(f.a_unseen_uni, f.b_unseen_uni)
    o["edited_vs_unedited_uni"] = f.a_log_uni - f.b_log_uni        # formal register
    o["edited_vs_unedited_bi"] = (f.a_log_bi_left + f.a_log_bi_right) - (f.b_log_bi_left + f.b_log_bi_right)
    o["arc_mi_a"] = f.a_arc_mi.fillna(f.a_arc_mi.median())
    o["arc_unseen_a"] = (f.a_arc_logc.fillna(1) == 0).astype(float)
    o["kid_unseen_share_a"] = f.a_kid_unseen_share.fillna(0)
    o["length"] = f.length
    o["repeated_in_essay"] = np.log(f.word_count_in_essay)
    o["sim_sent"] = f.sim_sent.fillna(f.sim_sent.median())
    o["proper_or_num"] = np.maximum(f.pos_PROPN, f.like_num)
    o["capital_mid_not_propn"] = f.is_title_mid * (1 - f.pos_PROPN)
    o["function_word"] = f[["pos_DET", "pos_ADP", "pos_AUX", "pos_PRON", "pos_CCONJ", "pos_SCONJ", "pos_PART"]].sum(1)
    o["punct"] = f.is_punct
    o["content_word"] = f[["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV"]].sum(1)
    return o


def slot(ctx) -> pd.DataFrame:
    from lab.att_slot import ATT as _A, build as _b
    p = _A / f"slot{SUFFIX}.parquet"
    f = pd.read_parquet(p) if p.exists() else _b(ctx.bench)
    return f.drop(columns=["seconds_total"])
