"""Pilot, step 2 (GPU) — score original and corrected essays with the stage 1 pipeline.

Same model, window, batch size, dtype and ``Predictor`` path as
``1_extract.py``; only surprisal, entropy and the observed id are kept (no
top-50). ``orig`` rescored here must reproduce the stored stage 1 surprisal.

Outputs, under data/study3/pilot_corrected/:
    tokens_orig.parquet, tokens_corr.parquet
        text_id, spacy_idx, idx, text, lemma, pos, tag, is_punct, is_space,
        is_alpha, like_num, obs_id, surprisal_bits, entropy_bits
    manifest_score.json

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/pilot_corrected/2_score.py
"""

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "1-predictability-benchmark"))
from common import OUT  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402
from run_surprisal import load_predictor, window_int  # noqa: E402
from util.paths import WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

LOG2E = 1.0 / math.log(2.0)
MODEL_KEY, WINDOW, BATCH_SIZE = "olmo2-1b", "8", 32


class Recorder:
    def __init__(self):
        self.rows = []

    def __call__(self, spacy_idx, target_id, pred_logits):
        logp = torch.log_softmax(pred_logits.float(), dim=-1)
        self.rows.append((spacy_idx, target_id, -logp[target_id].item() * LOG2E,
                          -(logp.exp() * logp).sum().item() * LOG2E))


def score(predictor, rec, docs, w) -> pd.DataFrame:
    out = []
    for doc in docs:
        tid = doc.user_data["meta"]["text_id"]
        rec.rows = []
        predictor(doc, window_size=w)
        for si, oid, s, h in rec.rows:
            t = doc[si]
            out.append((tid, si, t.idx, t.text, t.lemma_, t.pos_, t.tag_, t.is_punct, t.is_space,
                        t.is_alpha, t.like_num, oid, s, h))
    return pd.DataFrame(out, columns=["text_id", "spacy_idx", "idx", "text", "lemma", "pos", "tag",
                                      "is_punct", "is_space", "is_alpha", "like_num", "obs_id",
                                      "surprisal_bits", "entropy_bits"])


def main():
    spec = MODEL_REGISTRY[MODEL_KEY]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    predictor, commit = load_predictor(spec, torch.bfloat16, "cuda", BATCH_SIZE)
    rec = Recorder()
    predictor.on_score = rec
    w = window_int(spec, WINDOW)
    timing = {}
    for name, src in [("corr", OUT / "docbins_corrected"), ("orig", WI_DOCBINS_DIR)]:
        path = OUT / f"tokens_{name}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        df = score(predictor, rec, list(load_all_docbins(src)), w)
        df.to_parquet(path, index=False)
        timing[name] = {"seconds": round(time.time() - t0, 1), "n_tokens": len(df)}
        print(name, timing[name], flush=True)
    (OUT / "manifest_score.json").write_text(json.dumps({
        "model_key": MODEL_KEY, "hf_id": spec.hf_id, "revision": commit, "window_tokens": w,
        "batch_size": BATCH_SIZE, "dtype": "bfloat16 forward, fp32 softmax", "timing": timing,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0)}, indent=2))


if __name__ == "__main__":
    main()
