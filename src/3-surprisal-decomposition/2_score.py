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
from lib.corpora import ALL_IDS, DOCBINS, base, out  # noqa: E402
from lib.lm import REVISION, instrument  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402


def reusable(corpus: str, name: str) -> bool:
    """The base corpus's scores of the same texts exist (originals always; corrected if identical)."""
    b = out(base(corpus))
    if not (b / f"tokens_{name}.parquet").exists():
        return False
    if name == "orig":
        return True
    mine = pd.read_parquet(out(corpus) / "essays.parquet", columns=["text_id", "corrected"])
    theirs = pd.read_parquet(b / "essays.parquet", columns=["text_id", "corrected"])
    return mine.equals(theirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    args = ap.parse_args()
    O = out(args.corpus)
    bc = base(args.corpus)
    score = None
    timing = {}
    for name, src in [("orig", DOCBINS[bc]), ("corr", O / "docbins_corrected")]:
        if bc != args.corpus and reusable(args.corpus, name):
            shutil.copyfile(out(bc) / f"tokens_{name}.parquet", O / f"tokens_{name}.parquet")
            timing[name] = {"copied_from": bc}
            print(name, timing[name], flush=True)
            continue
        score = score or instrument()
        t0 = time.time()
        docs = list(load_all_docbins(src))
        df = score(docs, [d.user_data["meta"]["text_id"] for d in docs])
        df.to_parquet(O / f"tokens_{name}.parquet", index=False)
        timing[name] = {"seconds": round(time.time() - t0, 1), "n_tokens": len(df)}
        print(name, timing[name], flush=True)
    (O / "manifest_score.json").write_text(json.dumps(
        {"revision": REVISION, "timing": timing, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0) if score else None}, indent=2))


if __name__ == "__main__":
    main()
