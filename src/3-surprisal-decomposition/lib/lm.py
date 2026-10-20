"""The Study 3 model, scored exactly as Study 1 scores it.

``instrument()`` wraps ``features.predictability.Predictor`` (the causal path):
the window is the (up to) WINDOW subword ids ending at the target's first
subword, and the logits at the second-to-last position predict it.
"""

import numpy as np
import torch

HF_ID = "allenai/OLMo-2-0425-1B"
REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
WINDOW = 8
LOG2E = 1.0 / np.log(2.0)


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
