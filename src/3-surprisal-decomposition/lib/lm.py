"""Shared access to the Study 3 model, with Study 1's windowing.

``instrument()`` is the Study 1 scoring path itself (``Predictor``), used to
score whole essays. ``LM`` scores single windows.

A position is scored exactly as ``features.predictability.Predictor`` scores it
under the causal path: the window is the (up to) WINDOW subword ids ending at
the target's first subword, and the logits at the second-to-last position
predict it.

Usage:

    from lib.lm import LM
    lm = LM()                                   # loads OLMo-2 1B, bf16, cuda
    ids, k = lm.locate(text, char)              # subword ids + target index
    out = lm.score([ids[max(0, k - 7):k + 1]])  # dict of tensors, one row per window
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ID = "allenai/OLMo-2-0425-1B"
REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
WINDOW = 8
TOP_K = 50
LOG2E = 1.0 / np.log(2.0)


class LM:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_ID, revision=REVISION, dtype=torch.bfloat16).to(device).eval()

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
          target_id, surprisal (bits), entropy (bits), topk_ids, topk_probs
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
                         position_ids=pos.to(self.device))
        logits = out.logits[:, -2].float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        tgt = ids[:, -1].to(self.device)
        top_p, top_i = torch.topk(p, TOP_K)
        return {
            "target_id": tgt.cpu().numpy(),
            "surprisal": (-logp.gather(1, tgt[:, None])[:, 0] * LOG2E).cpu().numpy(),
            "entropy": (-(p * logp).sum(-1) * LOG2E).cpu().numpy(),
            "topk_ids": top_i.int().cpu().numpy(),
            "topk_probs": top_p.cpu().numpy(),
        }


def instrument():
    """The Study 1 instrument as a callable: ``features.predictability.Predictor``
    with OLMo-2 1B, an 8-subword window ending at each word's first subword,
    batch 32, bf16 forward and fp32 softmax. Returns score(docs, ids), which gives
    one row per scored token: text_id, spacy_idx, idx, text, is_space, surprisal_bits."""
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1-predictability-benchmark"))
    from models import MODEL_REGISTRY
    from run_surprisal import load_predictor, window_int

    spec = MODEL_REGISTRY["olmo2-1b"]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    predictor, commit = load_predictor(spec, torch.bfloat16, "cuda", 32)
    assert commit == REVISION, commit
    w = window_int(spec, str(WINDOW))
    rows = []

    def record(spacy_idx, target_id, pred_logits):
        logp = torch.log_softmax(pred_logits.float(), dim=-1)
        rows.append((spacy_idx, -logp[target_id].item() * LOG2E))

    predictor.on_score = record

    def score(docs, ids):
        out = []
        for tid, doc in zip(ids, docs):
            rows.clear()
            predictor(doc, window_size=w)
            out += [(tid, si, doc[si].idx, doc[si].text, doc[si].is_space, s) for si, s in rows]
        return pd.DataFrame(out, columns=["text_id", "spacy_idx", "idx", "text", "is_space", "surprisal_bits"])
    return score
