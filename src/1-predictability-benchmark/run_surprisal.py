"""Study 1 — compute mean token surprisal for every model × window × corpus.

Drives the existing surprisal engine (``features.predictability.Predictor``)
over the Study 1 model matrix (``models.py``) and writes one idempotent,
resumable table per model per corpus:

    data/predictability/{corpus}/{model_key}/surprisal.parquet   # per essay × window
    data/predictability/{corpus}/{model_key}/manifest.json       # provenance / run log
    data/predictability/{corpus}/{model_key}/tokens.parquet      # only with --dump-tokens

Each surprisal row is one essay scored at one context window, carrying the
essay's labels (ELLIPSE trait scores / TOEFL l1+level) so the §3 analysis has
everything in one place. Surprisal is reported in **bits** (log base 2).

Idempotent at (text_id, window) granularity: re-running skips work already on
disk and writes are flushed atomically every FLUSH_EVERY essays, so a crash or
a newly-added model/window only computes what is missing.

Examples
--------
    # Smoke test: two small models, one corpus, 5 essays, on CPU
    python src/1-predictability-benchmark/run_surprisal.py \
        --models gpt2 bert-base --corpora ellipse --windows 64 --limit 5 --device cpu

    # Full matrix (all models × all windows × both corpora)
    python src/1-predictability-benchmark/run_surprisal.py

    # One newly-added model, both corpora, all windows
    python src/1-predictability-benchmark/run_surprisal.py --models olmo2-1b
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from features.predictability import Predictor
from util.paths import ELLIPSE_DOCBINS_DIR, PREDICTABILITY_DIR, TOEFL_DOCBINS_DIR
from util.process_docs import load_all_docbins

# Local import (package dir name starts with a digit, so load by path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODEL_REGISTRY  # noqa: E402

LOG2E = 1.0 / math.log(2.0)  # nats -> bits
WINDOW_LABELS = ["8", "64", "full"]
CORPUS_DOCBINS = {"ellipse": ELLIPSE_DOCBINS_DIR, "toefl11": TOEFL_DOCBINS_DIR}
FLUSH_EVERY = 200  # essays between atomic parquet flushes
SEED = 42


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_essays(corpus: str, limit: int | None):
    """Return [(text_id, doc, label_dict), ...] from a corpus's DocBins."""
    docbin_dir = CORPUS_DOCBINS[corpus]
    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"No DocBins for {corpus} at {docbin_dir}. Run essays/ingest.py first."
        )
    essays = []
    for doc in load_all_docbins(docbin_dir):
        meta = doc.user_data.get("meta")
        if isinstance(meta, dict):
            text_id = meta.get("text_id")
            labels = {k: v for k, v in meta.items() if k != "corpus"}
        else:
            text_id = meta
            labels = {"text_id": meta}
        essays.append((str(text_id), doc, labels))
        if limit and len(essays) >= limit:
            break
    return essays


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def resolve_dtype(name: str, device: str):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if device == "cuda" else torch.float32  # auto


def load_predictor(spec, dtype, device, batch_size):
    tok = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model_cls = AutoModelForMaskedLM if spec.arch == "masked" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        spec.hf_id, torch_dtype=dtype, trust_remote_code=spec.trust_remote_code
    )
    predictor = Predictor(
        tokenizer=tok, model=model, model_type=spec.arch,
        batch_size=batch_size, device=device,
    )
    commit = getattr(model.config, "_commit_hash", None)
    return predictor, commit


def window_int(spec, label: str) -> int:
    """Transformer-token window to request from the engine for this model.

    'full' = the model's max context. Masked models reserve 2 positions for
    [CLS]/[SEP]; causal models use the full budget. Explicit 8/64 are capped the
    same way (only matters for tiny-context models, none here).
    """
    cap = spec.max_ctx - 2 if spec.arch == "masked" else spec.max_ctx
    return cap if label == "full" else min(int(label), cap)


# --------------------------------------------------------------------------- #
# Storage (idempotent, atomic)
# --------------------------------------------------------------------------- #
def save_parquet_atomic(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_existing(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records") if path.exists() else []


def update_manifest(out_dir: Path, spec, commit, dtype, device, entry: dict):
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "revision": commit,
        "arch": spec.arch,
        "params": spec.params,
        "max_ctx": spec.max_ctx,
        "instruct": spec.instruct,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "seed": SEED,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "surprisal_units": "bits",
    })
    manifest.setdefault("runs", {})
    manifest["runs"][entry.pop("key")] = entry
    path.write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def run_model_corpus(spec, predictor, commit, corpus, essays, windows,
                     dtype, device, force, dump_tokens):
    out_dir = PREDICTABILITY_DIR / corpus / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = out_dir / "surprisal.parquet"

    target = set(windows)
    rows = load_existing(surprisal_path)
    if force:  # recompute the targeted windows: drop their stale rows
        rows = [r for r in rows if str(r["window"]) not in target]
    done = {(str(r["text_id"]), str(r["window"])) for r in rows}

    token_rows = []
    for window in windows:
        w_int = window_int(spec, window)
        todo = [(tid, doc, lab) for tid, doc, lab in essays
                if (tid, window) not in done]
        if not todo:
            continue

        t0 = time.time()
        n_new = n_trunc = 0
        desc = f"{spec.key}/{corpus}/w{window}"
        for tid, doc, labels in tqdm(todo, desc=desc, unit="essay"):
            n_sub = len(predictor.tokenizer(doc.text, add_special_tokens=False).input_ids)
            truncated = n_sub > spec.max_ctx
            dp = predictor(doc, window_size=w_int)
            if len(dp) == 0:
                continue
            rows.append({
                **labels,
                "text_id": tid,
                "window": window,
                "mean_surprisal_bits": dp.mean_loss * LOG2E,
                "mean_entropy_bits": dp.mean_entropy * LOG2E,
                "var_surprisal_bits2": dp.var_loss * (LOG2E ** 2),
                "n_tokens_scored": len(dp),
                "n_subwords": n_sub,
                "truncated": truncated,
            })
            n_new += 1
            n_trunc += int(truncated)
            if dump_tokens:
                for t in dp:
                    token_rows.append({
                        "text_id": tid, "window": window,
                        "spacy_idx": t.spacy_idx, "text": t.text,
                        "surprisal_bits": t.mean_loss * LOG2E,
                        "entropy_bits": t.mean_entropy * LOG2E,
                        "num_subwords": len(t),
                    })
            if n_new % FLUSH_EVERY == 0:
                save_parquet_atomic(surprisal_path, pd.DataFrame(rows))

        save_parquet_atomic(surprisal_path, pd.DataFrame(rows))
        update_manifest(out_dir, spec, commit, dtype, device, {
            "key": f"{corpus}/{window}",
            "window_tokens": w_int,
            "n_essays": sum(1 for r in rows if str(r["window"]) == window),
            "n_new_this_run": n_new,
            "n_truncated": n_trunc,
            "truncation_rate": round(n_trunc / n_new, 4) if n_new else 0.0,
            "runtime_seconds": round(time.time() - t0, 1),
        })
        print(f"  {desc}: {n_new} new, {n_trunc} truncated, "
              f"{time.time() - t0:.0f}s")

    if dump_tokens and token_rows:
        save_parquet_atomic(out_dir / "tokens.parquet", pd.DataFrame(token_rows))
        print(f"  wrote {len(token_rows)} per-token rows")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model keys (see models.py) or 'all'")
    p.add_argument("--corpora", nargs="+", default=["all"],
                   choices=["ellipse", "toefl11", "all"])
    p.add_argument("--windows", nargs="+", default=["all"],
                   choices=WINDOW_LABELS + ["all"])
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N essays per corpus (smoke testing)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dump-tokens", action="store_true",
                   help="also write per-token records (recomputes targeted configs)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if results already exist")
    args = p.parse_args()

    models = list(MODEL_REGISTRY) if "all" in args.models else args.models
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        p.error(f"unknown model(s): {unknown}. choices: {list(MODEL_REGISTRY)}")
    corpora = ["ellipse", "toefl11"] if "all" in args.corpora else args.corpora
    windows = WINDOW_LABELS if "all" in args.windows else args.windows

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device={device} dtype={str(dtype).replace('torch.', '')} "
          f"models={models} corpora={corpora} windows={windows}"
          + (f" limit={args.limit}" if args.limit else ""))

    essay_cache: dict[str, list] = {}
    force = args.overwrite or args.dump_tokens

    for model_key in models:
        spec = MODEL_REGISTRY[model_key]
        print(f"\n=== {spec.key} ({spec.hf_id}, {spec.arch}, {spec.params}) ===")
        predictor, commit = load_predictor(spec, dtype, device, args.batch_size)
        for corpus in corpora:
            if corpus not in essay_cache:
                essay_cache[corpus] = load_essays(corpus, args.limit)
                print(f"  loaded {len(essay_cache[corpus])} {corpus} essays")
            run_model_corpus(spec, predictor, commit, corpus, essay_cache[corpus],
                             windows, dtype, device, force, args.dump_tokens)
        del predictor
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
