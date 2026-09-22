"""Bring existing masked-model tables onto the first-subword policy cheaply.

`features.predictability` scores a word by its first subword. A table written
under any other aggregation over a word's subwords differs only on words the
tokenizer split into more than one piece, because a single-subword word has the
same score under every such rule. This rescores only those words and folds their
exact contribution into the stored essay-level aggregates.

That is worth doing here because the masked configurations are expensive: a full
rescore of the three masked models over both corpora and all three windows is
about 36 GPU-hours, and roughly 2-4% of words are split, so this route costs a
few per cent of that.

**Exactness.** The stored row carries ``n_tokens_scored``, ``mean_surprisal_bits``
and ``var_surprisal_bits2``, which determine the sums Sx and Sxx over the essay's
per-word scores. The word set does not change, so applying
``sum(first - old)`` and ``sum(first^2 - old^2)`` over the split words alone
reproduces the aggregates the full rescore would write, to within the arithmetic
noise of rescoring in the same precision.

The alignment, window and masking logic below mirrors ``Predictor._process_masked``
so the recomputed words are the same words, in the same windows, as the table
being updated.

Run from the repo root:

    python src/1-predictability-benchmark/recompute_masked_first_subword.py --dry-run
    python src/1-predictability-benchmark/recompute_masked_first_subword.py
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from features.predictability import get_centered_window
from util.paths import ELLIPSE_DOCBINS_DIR, PREDICTABILITY_DIR, TOEFL_DOCBINS_DIR
from util.process_docs import load_all_docbins

from models import MODEL_REGISTRY
from run_surprisal import WINDOW_LABELS, window_int

LOG2E = 1.0 / math.log(2.0)
CORPUS_DOCBINS = {"ellipse": ELLIPSE_DOCBINS_DIR, "toefl11": TOEFL_DOCBINS_DIR}
MASKED_MODELS = [k for k, s in MODEL_REGISTRY.items() if s.arch == "masked"]
TOKEN_BUDGET = 131072  # per forward pass; sets the batch size from the window
WORD_SCORE = "first subword"  # recorded in the manifest; also the rerun guard
MAX_BATCH = 512

crossloss = torch.nn.CrossEntropyLoss(reduction="none")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("recompute")


def split_word_jobs(tokenizer, doc, window_size):
    """[(spacy_idx, win_start, win_end, tok_start, tok_end), ...] for split words.

    Indices follow ``Predictor._process_masked``: ``tok_start``/``tok_end`` are
    positions inside the window, already clipped to it, and a word is returned
    only when at least two of its subwords survive that clipping.
    """
    encoding = tokenizer(doc.text, add_special_tokens=False, return_tensors="np",
                         return_offsets_mapping=True)
    trf_tok_ids = encoding.input_ids[0]
    offset_mapping = encoding["offset_mapping"][0]

    adjusted_starts = []
    for start, end in offset_mapping:
        token_text = doc.text[start:end]
        adjusted_starts.append(start + len(token_text) - len(token_text.lstrip()))
    trf_starts = np.array(adjusted_starts)
    spacy_starts = np.array([t.idx for t in doc])
    _, common2spacy, common2trf = np.intersect1d(
        spacy_starts, trf_starts, return_indices=True)

    bounds = np.append(common2trf, common2trf[-1] + 1) if len(common2trf) else common2trf
    seq_len = len(trf_tok_ids)

    jobs = []
    for n, spacy_idx in enumerate(common2spacy):
        subword_start, subword_end = int(bounds[n]), int(bounds[n + 1])
        win_start, win_end = get_centered_window(seq_len, subword_start, window_size)
        window_len = win_end - win_start
        tok_start = max(0, subword_start - win_start)
        tok_end = min(window_len, subword_end - win_start)
        if tok_end - tok_start < 2:
            continue
        jobs.append((int(spacy_idx), win_start, win_end, tok_start, tok_end))
    return jobs, trf_tok_ids


@torch.no_grad()
def score_jobs(model, tokenizer, batch, device):
    """(first_loss, mean_loss, first_entropy, mean_entropy) per job, in nats."""
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0
    mask_id = tokenizer.mask_token_id

    seqs = []
    for ids, _spacy_idx, ws, we, a, b in batch:
        window = ids[ws:we].copy()
        window[a:b] = mask_id
        seqs.append([cls_id] + window.tolist() + [sep_id])

    max_len = max(len(s) for s in seqs)
    input_ids = torch.tensor([s + [pad_id] * (max_len - len(s)) for s in seqs],
                             device=device)
    attn = torch.tensor([[1] * len(s) + [0] * (max_len - len(s)) for s in seqs],
                        device=device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits

    out = []
    for j, (ids, _spacy_idx, ws, we, a, b) in enumerate(batch):
        positions = torch.arange(a + 1, b + 1, device=device)
        targets = torch.tensor([int(t) for t in ids[ws + a : ws + b]], device=device)
        sel = logits[j, positions].float()
        losses = crossloss(sel, targets)
        probs = torch.softmax(sel, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        out.append((losses[0].item(), losses.mean().item(),
                    entropies[0].item(), entropies.mean().item()))
    return out


def essay_deltas(model, tokenizer, essays, window_size, device):
    """{text_id: (d_sum, d_sumsq, d_entropy)} in nats, over split words only."""
    pending, deltas = [], {}
    batch_size = max(1, min(MAX_BATCH, TOKEN_BUDGET // max(1, window_size + 2)))

    def flush():
        if not pending:
            return
        for (tid, _job), (first, mean, first_e, mean_e) in zip(
                pending, score_jobs(model, tokenizer, [p[1] for p in pending], device)):
            d_sum, d_sq, d_ent = deltas.get(tid, (0.0, 0.0, 0.0))
            deltas[tid] = (d_sum + (first - mean),
                           d_sq + (first * first - mean * mean),
                           d_ent + (first_e - mean_e))
        pending.clear()

    for tid, doc in essays:
        jobs, ids = split_word_jobs(tokenizer, doc, window_size)
        for spacy_idx, ws, we, a, b in jobs:
            pending.append((tid, (ids, spacy_idx, ws, we, a, b)))
            if len(pending) >= batch_size:
                flush()
    flush()
    return deltas


def apply_deltas(rows: pd.DataFrame, deltas: dict) -> tuple[pd.DataFrame, dict]:
    """Rewrite the essay-level aggregates from the per-word deltas."""
    stats = {"n_rows": len(rows), "n_changed": 0, "max_abs_bits": 0.0,
             "mean_abs_bits": 0.0}
    changes = []
    out = rows.copy()
    for i, row in out.iterrows():
        d = deltas.get(str(row["text_id"]))
        if not d or (d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0):
            continue
        d_sum, d_sq, d_ent = d
        n = int(row["n_tokens_scored"])
        if n < 1:
            continue
        mean_nats = row["mean_surprisal_bits"] / LOG2E
        var_nats = row["var_surprisal_bits2"] / (LOG2E ** 2)
        ent_nats = row["mean_entropy_bits"] / LOG2E

        sx = n * mean_nats
        sxx = (n - 1) * var_nats + n * mean_nats ** 2 if n > 1 else n * mean_nats ** 2
        se = n * ent_nats

        new_mean = (sx + d_sum) / n
        new_sxx = sxx + d_sq
        new_var = max(0.0, (new_sxx - n * new_mean ** 2) / (n - 1)) if n > 1 else 0.0
        new_ent = (se + d_ent) / n

        changes.append(abs(new_mean * LOG2E - row["mean_surprisal_bits"]))
        out.at[i, "mean_surprisal_bits"] = new_mean * LOG2E
        out.at[i, "var_surprisal_bits2"] = new_var * (LOG2E ** 2)
        out.at[i, "mean_entropy_bits"] = new_ent * LOG2E
    if changes:
        stats["n_changed"] = len(changes)
        stats["max_abs_bits"] = float(np.max(changes))
        stats["mean_abs_bits"] = float(np.mean(changes))
    return out, stats


def load_essays(corpus: str, limit: int | None):
    essays = []
    for doc in load_all_docbins(CORPUS_DOCBINS[corpus]):
        meta = doc.user_data.get("meta")
        tid = meta.get("text_id") if isinstance(meta, dict) else meta
        essays.append((str(tid), doc))
        if limit and len(essays) >= limit:
            break
    return essays


def save_atomic(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MASKED_MODELS)
    p.add_argument("--corpora", nargs="+", default=["ellipse", "toefl11"],
                   choices=["ellipse", "toefl11"])
    p.add_argument("--windows", nargs="+", default=WINDOW_LABELS, choices=WINDOW_LABELS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    p.add_argument("--force", action="store_true",
                   help="rerun a table whose manifest already records this policy")
    args = p.parse_args()

    bad = [m for m in args.models if MODEL_REGISTRY[m].arch != "masked"]
    if bad:
        p.error(f"not masked models: {bad}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        tok = AutoTokenizer.from_pretrained(spec.hf_id,
                                            trust_remote_code=spec.trust_remote_code)
        model = None
        for corpus in args.corpora:
            path = PREDICTABILITY_DIR / corpus / spec.key / "surprisal.parquet"
            if not path.exists():
                logger.warning("no table at %s — skipping", path)
                continue
            man_path = path.parent / "manifest.json"
            manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
            if manifest.get("word_score") == WORD_SCORE and not args.force:
                # The deltas are relative to the stored aggregates, so applying
                # them twice would double-count. Pass --force to override.
                logger.info("%s/%s already on this policy — skipping",
                            spec.key, corpus)
                continue

            table = pd.read_parquet(path)
            essays = None

            for label in args.windows:
                mask = table["window"].astype(str) == label
                if not mask.any():
                    continue
                w_int = window_int(spec, label)
                if model is None:
                    dtype = torch.bfloat16 if device == "cuda" else torch.float32
                    model = AutoModelForMaskedLM.from_pretrained(
                        spec.hf_id, dtype=dtype,
                        trust_remote_code=spec.trust_remote_code).to(device).eval()
                if essays is None:
                    essays = load_essays(corpus, args.limit)
                    logger.info("loaded %d %s essays", len(essays), corpus)

                t0 = time.time()
                deltas = essay_deltas(model, tok, essays, w_int, device)
                updated, stats = apply_deltas(table[mask], deltas)
                table.loc[mask, updated.columns] = updated
                logger.info(
                    "%s/%s/w%-4s  %d/%d essays changed  mean |Δ| %.4f bits  max %.4f  %.0fs",
                    spec.key, corpus, label, stats["n_changed"], stats["n_rows"],
                    stats["mean_abs_bits"], stats["max_abs_bits"], time.time() - t0)

            if args.dry_run:
                logger.info("dry run — %s/%s not written", spec.key, corpus)
                continue
            save_atomic(path, table)
            manifest["word_score"] = WORD_SCORE
            man_path.write_text(json.dumps(manifest, indent=2))
            logger.info("wrote %s", path)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
