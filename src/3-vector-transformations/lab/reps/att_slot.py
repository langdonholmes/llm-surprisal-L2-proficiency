"""Slot attestation only: P(w | p1 _ n1), forward trigram P(w | p2 p1), backward P(w | n1 n2) and bigram backward, frame frequencies (corpus a and b)."""
from lab.att_common import slot


def featurize(ctx):
    return slot(ctx).to_numpy("float32")
