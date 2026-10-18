"""Compact named acceptability evidence (21 features): attestation both sides, trigram, PMI, rarity, nonword, register, arcs, repetition, POS class."""
from lab.att_common import core


def featurize(ctx):
    return core(ctx).to_numpy("float32")
