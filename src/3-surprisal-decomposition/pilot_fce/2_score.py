"""FCE step 2 (GPU) — score original and corrected answers with the D22 instrument.

Reuses ``pilot_corrected/2_score.py`` unchanged (``Recorder``, ``score``, the
model key, window and batch size): OLMo-2 1B, 8-subword window ending at the
first subword, ``Predictor``, batch 32, bf16 forward, fp32 softmax.

Outputs, under data/study3/pilot_fce/:
    tokens_orig.parquet, tokens_corr.parquet   (columns as in pilot_corrected)
    manifest_score.json

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/pilot_fce/2_score.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fce_common import FCE_DOCBINS_DIR, OUT, pc  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

S = pc("2_score")   # puts 1-predictability-benchmark on sys.path


def main():
    spec = S.MODEL_REGISTRY[S.MODEL_KEY]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    predictor, commit = S.load_predictor(spec, torch.bfloat16, "cuda", S.BATCH_SIZE)
    assert commit.startswith("a1847dff"), commit
    rec = S.Recorder()
    predictor.on_score = rec
    w = S.window_int(spec, S.WINDOW)
    timing = {}
    for name, src in [("corr", OUT / "docbins_corrected"), ("orig", FCE_DOCBINS_DIR)]:
        path = OUT / f"tokens_{name}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        df = S.score(predictor, rec, list(load_all_docbins(src)), w)
        df.to_parquet(path, index=False)
        timing[name] = {"seconds": round(time.time() - t0, 1), "n_tokens": len(df)}
        print(name, timing[name], flush=True)
    (OUT / "manifest_score.json").write_text(json.dumps({
        "model_key": S.MODEL_KEY, "hf_id": spec.hf_id, "revision": commit, "window_tokens": w,
        "batch_size": S.BATCH_SIZE, "dtype": "bfloat16 forward, fp32 softmax", "timing": timing,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0)}, indent=2))


if __name__ == "__main__":
    main()
