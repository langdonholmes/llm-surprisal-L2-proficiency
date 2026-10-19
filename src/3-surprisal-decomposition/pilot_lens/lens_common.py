"""Shared pieces for the layer-wise pilot: model loading, hidden states per layer, the lens.

Layers are indexed 0 (embedding output) .. 16 (output of the last block, before
the final norm). The model's own logits are lm_head(norm(h_16)), so both lenses
are exact at layer 16.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from lib.bench import S3  # noqa: E402
from lib.lm import HF_ID, LOG2E, REVISION, WINDOW  # noqa: E402

OUT = S3 / "pilot_lens"
N_LAYERS = 16


def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(HF_ID, revision=REVISION, dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def hidden_states(model, ids, att, pos):
    """List of N_LAYERS + 1 tensors [B, T, d] (bf16) and the model's logits [B, T, V]."""
    hs = []
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: hs.append(o))]
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(
            lambda m, i, o: hs.append(o[0] if isinstance(o, tuple) else o)))
    try:
        out = model(input_ids=ids, attention_mask=att, position_ids=pos)
    finally:
        for h in hooks:
            h.remove()
    assert len(hs) == N_LAYERS + 1
    return hs, out.logits


def final_norm(model, h):
    """The model's final RMSNorm, in fp32."""
    n = model.model.norm
    h = h.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + n.variance_epsilon)
    return h * n.weight.float()


def unembed(model, h_normed):
    return torch.nn.functional.linear(h_normed.to(torch.bfloat16), model.lm_head.weight).float()


class TunedLens(nn.Module):
    """One affine translator per layer 0..15 (residual form, zero init)."""

    def __init__(self, d: int, n: int = N_LAYERS):
        super().__init__()
        self.A = nn.ParameterList([nn.Parameter(torch.zeros(d, d)) for _ in range(n)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n)])

    def forward(self, h, layer: int):
        if layer >= len(self.A):
            return h.float()
        h = h.float()
        return h + h @ self.A[layer].T + self.b[layer]


def lens_logits(model, h, layer, lens=None):
    """Logits of a lens at `layer`: logit lens if lens is None, else tuned lens."""
    z = h.float() if lens is None else lens(h, layer)
    return unembed(model, final_norm(model, z))


def pad_windows(windows, pad_id):
    L = max(len(w) for w in windows)
    ids = torch.full((len(windows), L), pad_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    pos = torch.zeros_like(ids)
    for i, w in enumerate(windows):
        ids[i, L - len(w):] = torch.as_tensor(np.asarray(w))
        att[i, L - len(w):] = 1
        pos[i, L - len(w):] = torch.arange(len(w))
    return ids, att, pos


__all__ = ["OUT", "N_LAYERS", "LOG2E", "WINDOW", "load_model", "hidden_states", "final_norm",
           "unembed", "TunedLens", "lens_logits", "pad_windows"]
