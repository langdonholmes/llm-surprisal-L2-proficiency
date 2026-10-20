"""Study 3, stage 2 (GPU) — score the original and the corrected essays.

The Study 1 instrument (``lib.lm.instrument``): OLMo-2 1B base, 8-subword
window ending at each word's first subword, ``features.predictability.Predictor``,
batch 32, bf16 forward, fp32 softmax.

Writes, under data/study3/main/<corpus>/:
    tokens_orig.parquet, tokens_corr.parquet
        text_id, spacy_idx, idx, text, is_space, surprisal_bits
    manifest_score.json

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/2_score.py --corpus wi
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import DOCBINS, out  # noqa: E402
from lib.lm import REVISION, instrument  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wi", "fce"], required=True)
    args = ap.parse_args()
    O = out(args.corpus)
    score = instrument()
    timing = {}
    for name, src in [("orig", DOCBINS[args.corpus]), ("corr", O / "docbins_corrected")]:
        t0 = time.time()
        docs = list(load_all_docbins(src))
        df = score(docs, [d.user_data["meta"]["text_id"] for d in docs])
        df.to_parquet(O / f"tokens_{name}.parquet", index=False)
        timing[name] = {"seconds": round(time.time() - t0, 1), "n_tokens": len(df)}
        print(name, timing[name], flush=True)
    (O / "manifest_score.json").write_text(json.dumps(
        {"revision": REVISION, "timing": timing, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0)}, indent=2))


if __name__ == "__main__":
    main()
