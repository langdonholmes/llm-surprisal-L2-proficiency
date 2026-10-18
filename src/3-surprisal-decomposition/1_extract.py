"""Study 3, stage 1 — one forward pass per corpus, keep the top-k prediction.

Runs the Study 1 selected configuration (OLMo-2 1B base, 8-token window)
through the same ``features.predictability.Predictor`` that produced the Study 1
tables, and records for **every** scored token:

    surprisal / entropy (bits), the observed first-subword id, and the top-K
    (K = 50) predicted ids with their (un-renormalized) probabilities.

Nothing is thresholded here. Storing the top-K distribution for every token
makes the threshold, the delta arithmetic and every top-k / threshold
sensitivity a CPU step over these files (``2_deltas.py``), so the GPU pass runs
once.

Outputs, under ``data/study3/{corpus}/``:

    shards/shard_XXX.parquet   token rows (spaCy attributes + surprisal + obs id)
    shards/shard_XXX.npz       topk_ids [n, K] int32, topk_probs [n, K] float32
    essays.parquet             text_id, text, labels
    manifest.json

Shards are written every SHARD_SIZE essays and skipped on re-run, so the pass is
resumable. Batch size, dtype and windowing match Study 1, and ``check`` in
``2_deltas.py`` verifies that the per-essay mean surprisal reproduces the Study 1
table.

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/1_extract.py --corpus ellipse
    CUDA_VISIBLE_DEVICES=1 python src/3-surprisal-decomposition/1_extract.py --corpus toefl11
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1-predictability-benchmark"))
from models import MODEL_REGISTRY  # noqa: E402
import run_surprisal  # noqa: E402
from run_surprisal import load_essays, load_predictor, window_int  # noqa: E402

from util.paths import DATA_DIR, WI_DOCBINS_DIR  # noqa: E402

# W&I+LOCNESS (gold-annotated validation corpus) goes through the same pass.
run_surprisal.CORPUS_DOCBINS["wi"] = WI_DOCBINS_DIR

LOG2E = 1.0 / math.log(2.0)
MODEL_KEY = "olmo2-1b"
WINDOW = "8"
TOP_K = 50
BATCH_SIZE = 32  # Study 1's batch size, so the logits are the same ones
SHARD_SIZE = 500
OUT_ROOT = DATA_DIR / "study3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("extract")


class TopKRecorder:
    """``Predictor.on_score`` callback that keeps the top-K of each prediction."""

    def __init__(self, k: int):
        self.k = k
        self.reset()

    def reset(self):
        self.spacy_idx, self.obs_id, self.surprisal, self.entropy = [], [], [], []
        self.top_ids, self.top_probs = [], []

    def __call__(self, spacy_idx, target_id, pred_logits):
        logp = torch.log_softmax(pred_logits.float(), dim=-1)
        p = logp.exp()
        top_p, top_i = torch.topk(p, self.k)
        self.spacy_idx.append(spacy_idx)
        self.obs_id.append(target_id)
        self.surprisal.append(-logp[target_id].item() * LOG2E)
        self.entropy.append(-(p * logp).sum().item() * LOG2E)
        self.top_ids.append(top_i.to(torch.int32).cpu().numpy())
        self.top_probs.append(top_p.cpu().numpy())


def token_rows(text_id, doc, rec: TopKRecorder) -> pd.DataFrame:
    rows = []
    for si, oid, s, h in zip(rec.spacy_idx, rec.obs_id, rec.surprisal, rec.entropy):
        t = doc[si]
        rows.append({
            "text_id": text_id, "spacy_idx": si, "idx": t.idx,
            "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
            "dep": t.dep_, "head_pos": t.head.pos_,
            "is_punct": t.is_punct, "is_space": t.is_space, "is_alpha": t.is_alpha,
            "obs_id": oid, "surprisal_bits": s, "entropy_bits": h,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=["ellipse", "toefl11", "wi"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = OUT_ROOT / args.corpus
    (out / "shards").mkdir(parents=True, exist_ok=True)

    spec = MODEL_REGISTRY[MODEL_KEY]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    predictor, commit = load_predictor(spec, torch.bfloat16, "cuda", BATCH_SIZE)
    rec = TopKRecorder(TOP_K)
    predictor.on_score = rec
    w = window_int(spec, WINDOW)

    essays = load_essays(args.corpus, args.limit)
    logger.info("%d %s essays, model %s @ %s", len(essays), args.corpus, spec.hf_id, commit)
    pd.DataFrame([{**lab, "text_id": tid, "text": doc.text} for tid, doc, lab in essays]) \
        .to_parquet(out / "essays.parquet", index=False)

    t0 = time.time()
    n_shards = math.ceil(len(essays) / SHARD_SIZE)
    for s in range(n_shards):
        stem = out / "shards" / f"shard_{s:03d}"
        if stem.with_suffix(".npz").exists():
            continue
        frames, ids, probs = [], [], []
        for tid, doc, _ in essays[s * SHARD_SIZE:(s + 1) * SHARD_SIZE]:
            rec.reset()
            predictor(doc, window_size=w)
            if not rec.spacy_idx:
                continue
            frames.append(token_rows(tid, doc, rec))
            ids.append(np.stack(rec.top_ids))
            probs.append(np.stack(rec.top_probs))
        pd.concat(frames, ignore_index=True).to_parquet(stem.with_suffix(".parquet"), index=False)
        tmp = stem.with_name(stem.name + ".tmp.npz")
        np.savez(tmp, topk_ids=np.concatenate(ids), topk_probs=np.concatenate(probs))
        tmp.rename(stem.with_suffix(".npz"))
        logger.info("shard %d/%d  %.0fs elapsed", s + 1, n_shards, time.time() - t0)

    (out / "manifest.json").write_text(json.dumps({
        "model_key": MODEL_KEY, "hf_id": spec.hf_id, "revision": commit,
        "window": WINDOW, "window_tokens": w, "top_k": TOP_K, "batch_size": BATCH_SIZE,
        "dtype": "bfloat16 forward, fp32 softmax", "n_essays": len(essays),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "surprisal_units": "bits",
    }, indent=2))
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
