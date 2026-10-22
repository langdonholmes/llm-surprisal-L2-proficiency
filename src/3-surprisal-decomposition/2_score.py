"""Study 3, stage 2 (GPU) — score the original and the corrected essays.

The Study 1 instrument (``lib.lm.instrument``): OLMo-2 1B base, 8-subword
window ending at each word's first subword, ``features.predictability.Predictor``,
batch 32, bf16 forward, fp32 softmax.

Writes, under data/study3/<corpus id>/:
    tokens_orig.parquet, tokens_corr.parquet
        text_id, spacy_idx, idx, text, is_space, surprisal_bits
    manifest_score.json

A variant of W&I or FCE (``lib.corpora.VARIANTS``) has the same original texts
as its base corpus, so their scores are copied from it; so are the corrected
texts' scores when the corrected texts are the same (the ERRANT re-cut of gold).

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/2_score.py --corpus wi
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import ALL_IDS, DOCBINS, base_corpus, data_dir  # noqa: E402
from lib.lm import REVISION, instrument  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402


def reusable(corpus: str, name: str) -> bool:
    """The base corpus's scores of the same texts exist (originals always; corrected if identical)."""
    base_dir = data_dir(base_corpus(corpus))
    if not (base_dir / f"tokens_{name}.parquet").exists():
        return False
    if name == "orig":
        return True
    mine = pd.read_parquet(data_dir(corpus) / "essays.parquet", columns=["text_id", "corrected"])
    theirs = pd.read_parquet(base_dir / "essays.parquet", columns=["text_id", "corrected"])
    return mine.equals(theirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    args = ap.parse_args()
    out_dir = data_dir(args.corpus)
    base = base_corpus(args.corpus)
    score = None
    timing = {}
    for name, src in [("orig", DOCBINS[base]), ("corr", out_dir / "docbins_corrected")]:
        if base != args.corpus and reusable(args.corpus, name):
            shutil.copyfile(data_dir(base) / f"tokens_{name}.parquet", out_dir / f"tokens_{name}.parquet")
            timing[name] = {"copied_from": base}
            print(name, timing[name], flush=True)
            continue
        score = score or instrument()
        t0 = time.time()
        docs = list(load_all_docbins(src))
        tokens = score(docs, [d.user_data["meta"]["text_id"] for d in docs])
        tokens.to_parquet(out_dir / f"tokens_{name}.parquet", index=False)
        timing[name] = {"seconds": round(time.time() - t0, 1), "n_tokens": len(tokens)}
        print(name, timing[name], flush=True)
    (out_dir / "manifest_score.json").write_text(json.dumps(
        {"revision": REVISION, "timing": timing, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0) if score else None}, indent=2))


if __name__ == "__main__":
    main()
