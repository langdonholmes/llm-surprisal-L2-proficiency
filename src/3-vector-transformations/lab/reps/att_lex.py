"""Lexis: rarity (Dolma unigram), unseen form, length, POS, shape flags, repetition in essay."""
from lab.att_common import block


def featurize(ctx):
    return block(ctx, prefixes=("pos_",), names=("a_log_uni", "b_log_uni", "a_unseen_uni", "length",
                 "is_title_mid", "is_sent_start", "like_num", "is_alpha", "is_punct", "has_vector",
                 "word_count_in_essay"))
