"""Lab — shared access to the Study 3 model, with Study 1's windowing.

A position is scored exactly as ``features.predictability.Predictor`` scores it
under the causal path: the window is the (up to) WINDOW subword ids ending at
the target's first subword, and the logits at the second-to-last position
predict it. Hidden states come from the same forward pass.

Usage (inside a representation module that needs extra forward passes):

    from lab.lm import LM
    lm = LM()                                   # loads OLMo-2 1B, bf16, cuda
    ids, k = lm.locate(text, char)              # subword ids + target index
    out = lm.score([ids[max(0, k - 7):k + 1]])  # dict of tensors, one row per window
"""

from functools import cached_property

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ID = "allenai/OLMo-2-0425-1B"
REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
WINDOW = 8
TOP_K = 50
LOG2E = 1.0 / np.log(2.0)


class LM:
    def __init__(self, device: str = "cuda", layers: tuple[int, ...] = (8, 16)):
        self.device = device
        self.layers = layers
        self.tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_ID, revision=REVISION, dtype=torch.bfloat16).to(device).eval()

    @cached_property
    def W_U(self) -> torch.Tensor:
        """Output embedding matrix, fp32 (vocab x 2048)."""
        return self.model.get_output_embeddings().weight.detach().float()

    @cached_property
    def W_E(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight.detach().float()

    def locate(self, text: str, char: int) -> tuple[np.ndarray, int]:
        """Token ids of ``text`` and the index of the subword starting at ``char``
        (leading whitespace stripped, as in Predictor.get_token_alignment)."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.array(enc.input_ids)
        for k, (s, e) in enumerate(enc.offset_mapping):
            piece = text[s:e]
            if s + len(piece) - len(piece.lstrip()) == char:
                return ids, k
        raise ValueError(f"no subword starts at char {char}")

    @staticmethod
    def window(ids: np.ndarray, k: int, width: int = WINDOW) -> np.ndarray:
        return ids[max(0, k - width + 1):k + 1]

    @torch.no_grad()
    def score(self, windows: list[np.ndarray]) -> dict:
        """Forward a batch of windows, each ending at its target id.

        Left-pads to a common length (positions are unaffected because the
        attention mask hides padding and OLMo-2 uses RoPE on real positions
        only when position_ids are given). Returns, per window:
          target_id, surprisal (bits), entropy (bits), topk_ids, topk_probs,
          h_pred[layer]  hidden state at the prediction position (predicts target)
          h_obs[layer]   hidden state at the target position (after reading it)
        """
        L = max(len(w) for w in windows)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(windows), L), pad, dtype=torch.long)
        att = torch.zeros((len(windows), L), dtype=torch.long)
        pos = torch.zeros((len(windows), L), dtype=torch.long)
        for i, w in enumerate(windows):
            ids[i, L - len(w):] = torch.as_tensor(w)
            att[i, L - len(w):] = 1
            pos[i, L - len(w):] = torch.arange(len(w))
        out = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                         position_ids=pos.to(self.device), output_hidden_states=True)
        logits = out.logits[:, -2].float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        tgt = ids[:, -1].to(self.device)
        top_p, top_i = torch.topk(p, TOP_K)
        res = {
            "target_id": tgt.cpu().numpy(),
            "surprisal": (-logp.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy(),
            "entropy": (-(p * logp).sum(-1) * LOG2E).cpu().numpy(),
            "topk_ids": top_i.int().cpu().numpy(),
            "topk_probs": top_p.cpu().numpy(),
        }
        for layer in self.layers:
            hs = out.hidden_states[layer]
            res[f"h_pred_{layer}"] = hs[:, -2].float().cpu().numpy()
            res[f"h_obs_{layer}"] = hs[:, -1].float().cpu().numpy()
        return res
