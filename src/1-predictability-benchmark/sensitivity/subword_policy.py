"""Study 1 sensitivity — what changes if a word is not scored by its first subword.

`fertility.qmd` established that the tokenizer channel is real but small, and
that the reported model ranking survives holding fertility constant. What it
could not do, because it never ran a model, is say how far apart the aggregation
conventions actually are on the two-to-four per cent of words that split. This
script runs the models and answers that, at window 8 only, on a stratified
subsample of both corpora.

Policies scored, all on the *same* words in the *same* run so every comparison is
paired:

  causal   first      the current convention: the word's first subword, left context only
           sum        the chain-rule joint, -log P(word | context). Exact for a causal LM.
           mean       per-subword density, i.e. sum / k

  masked   first      first subword with the whole word masked. The cheap fix that
                      makes the masked arm consistent with the causal one, and needs
                      no joint.
           mean       what `features.predictability` does today, and a defect:
                      density rather than information content.
           sum_indep  sum of the all-masked marginals. The naive joint, which
                      ignores within-word dependence.
           sum_chain  within-word autoregressive scoring (Kauf & Ivanova 2023).
                      The masked analogue of the chain rule, and the only honest
                      joint available for an MLM.

Word spans are rebuilt from the tokenizer's offset mapping rather than reused
from `features.predictability.get_token_alignment`, which absorbs unaligned words
into their predecessor and truncates each document's last word. Those defects do
not bite under `first`, which is why they have not, and they would bite under
every summing policy here.

The `first` column is a re-implementation of the production path, so it doubles
as a check: its essay means should reproduce `mean_surprisal_bits` at window 8
from `data/predictability/`. The report verifies that before reading anything
else.

Writes one resumable table per model:

    data/predictability/sensitivity/subword_policy/{model_key}.parquet

Run from the repo root:

    python src/1-predictability-benchmark/sensitivity/subword_policy.py --device cuda
    python src/1-predictability-benchmark/sensitivity/subword_policy.py --models gpt2 --limit 20
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from features.predictability import get_centered_window
from util.paths import ELLIPSE_DOCBINS_DIR, PREDICTABILITY_DIR, TOEFL_DOCBINS_DIR
from util.process_docs import load_all_docbins

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import MODEL_REGISTRY  # noqa: E402

LOG2E = 1.0 / math.log(2.0)
WINDOW = 8  # the selected configuration's window; the only one at issue
SEED = 42
OUT_DIR = PREDICTABILITY_DIR / "sensitivity" / "subword_policy"
CORPUS_DOCBINS = {"ellipse": ELLIPSE_DOCBINS_DIR, "toefl11": TOEFL_DOCBINS_DIR}
LABEL_COLS = {"ellipse": ["overall"], "toefl11": ["l1", "level"]}

# Subsample sizes. ELLIPSE carries the validity criterion and is stratified on it;
# TOEFL 11 carries the fairness criterion and is stratified on L1 x level, which
# is what max|d| reads. Every policy sees the same essays, so the comparisons are
# paired and far more precise than these n's would suggest marginally.
N_ELLIPSE = 1500
N_TOEFL_PER_L1 = 200

crossloss = torch.nn.CrossEntropyLoss(reduction="none")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("subword-policy")


# --------------------------------------------------------------------------- #
# Word spans
# --------------------------------------------------------------------------- #
def word_spans(doc, tokenizer):
    """[(spacy_idx, start_subword, end_subword), ...] plus the subword ids.

    A subword belongs to the spaCy token holding its first non-whitespace
    character. Only words whose subwords form a contiguous run are returned, so
    every span can be summed over honestly; the dropped remainder is counted by
    the caller.
    """
    text = doc.text
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"])

    owner_of_char = np.full(len(text), -1, dtype=np.int32)
    for tok in doc:
        owner_of_char[tok.idx : tok.idx + len(tok.text)] = tok.i

    charged: dict[int, list[int]] = {}
    for i, (start, end) in enumerate(enc["offset_mapping"]):
        if end <= start:
            continue
        span = text[start:end]
        start_adj = start + len(span) - len(span.lstrip())
        if start_adj >= end - (len(span) - len(span.rstrip())):
            continue  # whitespace-only subword
        w = owner_of_char[start_adj]
        if w >= 0:
            charged.setdefault(int(w), []).append(i)

    spans, n_noncontig = [], 0
    for w, idxs in charged.items():
        if idxs[-1] - idxs[0] + 1 != len(idxs):
            n_noncontig += 1
            continue
        spans.append((w, idxs[0], idxs[-1] + 1))
    spans.sort(key=lambda s: s[1])
    return spans, ids, n_noncontig


# --------------------------------------------------------------------------- #
# Causal
# --------------------------------------------------------------------------- #
@torch.no_grad()
def causal_losses(model, ids, targets, window, device, batch_size):
    """Cross-entropy (nats) for each subword index in `targets`, window ending at it.

    Mirrors `Predictor._process_causal`: the first `window` subwords are scored
    from one pass over the opening window, everything after from an equal-length
    window whose last position is the target. Index 0 has no left context and is
    returned as NaN, as the production path leaves it unscored.
    """
    seq_len = len(ids)
    out = {}
    first_end = min(window, seq_len)

    head = [t for t in targets if t < first_end and t > 0]
    if head:
        inp = torch.tensor(ids[:first_end].astype("int64"), device=device).unsqueeze(0)
        logits = model(input_ids=inp).logits[0].float()
        tgt = torch.tensor([int(ids[t]) for t in head], device=device)
        pos = torch.tensor([t - 1 for t in head], device=device)
        out.update(zip(head, crossloss(logits[pos], tgt).tolist()))

    tail = [t for t in targets if t >= first_end]
    for i in range(0, len(tail), batch_size):
        chunk = tail[i : i + batch_size]
        windows = np.stack([ids[t - window + 1 : t + 1] for t in chunk]).astype("int64")
        inp = torch.tensor(windows, device=device)
        logits = model(input_ids=inp).logits[:, -2].float()
        tgt = torch.tensor([int(ids[t]) for t in chunk], device=device)
        out.update(zip(chunk, crossloss(logits, tgt).tolist()))

    for t in targets:
        out.setdefault(t, float("nan"))
    return out


# --------------------------------------------------------------------------- #
# Masked
# --------------------------------------------------------------------------- #
@torch.no_grad()
def masked_word_losses(model, tokenizer, ids, spans, window, device, batch_size):
    """Per-word subword losses under two maskings, for every span.

    Returns {spacy_idx: (indep, chain)} where both are lists of length k:
      indep  every subword of the word masked at once, each position scored
             against that. A product of marginals, which is what the production
             path averages.
      chain  position j scored with 0..j-1 restored to their true ids and j..k-1
             still masked. Kauf & Ivanova (2023). Summing this is a real joint.

    For k == 1 the two are identical and only one pass is made. Windows are the
    production path's centered window, extended only where a multi-subword word
    would not otherwise fit.
    """
    seq_len = len(ids)
    mask_id = tokenizer.mask_token_id
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0

    # One job per (word, unmask-prefix-length). j == 0 serves both policies.
    jobs = []
    for spacy_idx, s, e in spans:
        k = e - s
        win_start, win_end = get_centered_window(seq_len, s, window)
        if e > win_end:  # keep the whole word inside its window
            win_end = min(seq_len, e)
        if s < win_start:
            win_start = s
        for j in range(k):
            jobs.append((spacy_idx, s, e, win_start, win_end, j))

    results: dict[int, dict] = {}
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        seqs, positions, targets = [], [], []
        for _idx, s, e, ws, we, j in chunk:
            w = ids[ws:we].copy()
            w[s - ws + j : e - ws] = mask_id          # j..k-1 masked
            # 0..j-1 keep their true ids, which `w` already holds.
            seq = [cls_id] + w.tolist() + [sep_id]
            seqs.append(seq)
            positions.append(s - ws + j + 1)          # +1 for [CLS]
            targets.append(int(ids[s + j]))

        max_len = max(len(s_) for s_ in seqs)
        input_ids = torch.tensor(
            [s_ + [pad_id] * (max_len - len(s_)) for s_ in seqs], device=device)
        attn = torch.tensor(
            [[1] * len(s_) + [0] * (max_len - len(s_)) for s_ in seqs], device=device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        sel = logits[torch.arange(len(chunk), device=device),
                     torch.tensor(positions, device=device)].float()
        losses = crossloss(sel, torch.tensor(targets, device=device)).tolist()

        for (spacy_idx, s, e, _ws, _we, j), loss in zip(chunk, losses):
            r = results.setdefault(spacy_idx, {"k": e - s, "indep": {}, "chain": {}})
            r["chain"][j] = loss
            if j == 0:
                r["indep"][0] = loss

    # The remaining `indep` positions come from the all-masked pass, which is the
    # j == 0 masking. Score them in a second sweep over words with k >= 2.
    multi = [(idx, s, e) for idx, s, e in spans if e - s > 1]
    jobs2 = []
    for spacy_idx, s, e in multi:
        win_start, win_end = get_centered_window(seq_len, s, window)
        if e > win_end:
            win_end = min(seq_len, e)
        if s < win_start:
            win_start = s
        for j in range(1, e - s):
            jobs2.append((spacy_idx, s, e, win_start, win_end, j))

    for i in range(0, len(jobs2), batch_size):
        chunk = jobs2[i : i + batch_size]
        seqs, positions, targets = [], [], []
        for _idx, s, e, ws, we, j in chunk:
            w = ids[ws:we].copy()
            w[s - ws : e - ws] = mask_id              # the whole word masked
            seq = [cls_id] + w.tolist() + [sep_id]
            seqs.append(seq)
            positions.append(s - ws + j + 1)
            targets.append(int(ids[s + j]))

        max_len = max(len(s_) for s_ in seqs)
        input_ids = torch.tensor(
            [s_ + [pad_id] * (max_len - len(s_)) for s_ in seqs], device=device)
        attn = torch.tensor(
            [[1] * len(s_) + [0] * (max_len - len(s_)) for s_ in seqs], device=device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        sel = logits[torch.arange(len(chunk), device=device),
                     torch.tensor(positions, device=device)].float()
        losses = crossloss(sel, torch.tensor(targets, device=device)).tolist()
        for (spacy_idx, _s, _e, _ws, _we, j), loss in zip(chunk, losses):
            results[spacy_idx]["indep"][j] = loss

    out = {}
    for spacy_idx, r in results.items():
        k = r["k"]
        out[spacy_idx] = ([r["indep"][j] for j in range(k)],
                          [r["chain"][j] for j in range(k)])
    return out


# --------------------------------------------------------------------------- #
# Per-essay aggregation
# --------------------------------------------------------------------------- #
def summarise(word_scores: dict[int, dict]) -> dict:
    """Essay-level means per policy, over all scored words and over split words only."""
    if not word_scores:
        return {}
    policies = next(iter(word_scores.values())).keys()
    ks = np.array([1 if "k" not in v else v["k"] for v in word_scores.values()])
    row = {"n_scored": len(word_scores), "n_split": int((ks >= 2).sum())}
    split = ks >= 2
    for pol in policies:
        if pol == "k":
            continue
        vals = np.array([v[pol] for v in word_scores.values()], dtype=float)
        ok = np.isfinite(vals)
        row[f"{pol}_bits"] = float(np.mean(vals[ok])) * LOG2E if ok.any() else np.nan
        sel = ok & split
        row[f"{pol}_bits_split"] = float(np.mean(vals[sel])) * LOG2E if sel.any() else np.nan
    return row


def score_essay(spec, model, tokenizer, doc, device, batch_size) -> dict:
    spans, ids, n_noncontig = word_spans(doc, tokenizer)
    if not spans:
        return {}

    word_scores: dict[int, dict] = {}
    if spec.arch == "causal":
        targets = sorted({t for _i, s, e in spans for t in range(s, e)})
        loss = causal_losses(model, ids, targets, WINDOW, device, batch_size)
        for spacy_idx, s, e in spans:
            vals = [loss[t] for t in range(s, e)]
            if not np.isfinite(vals).all():
                continue  # the document's first subword, left unscored as in production
            word_scores[spacy_idx] = {
                "k": e - s,
                "first": vals[0],
                "sum": float(np.sum(vals)),
                "mean": float(np.mean(vals)),
            }
    else:
        per_word = masked_word_losses(model, tokenizer, ids, spans, WINDOW,
                                      device, batch_size)
        for spacy_idx, (indep, chain) in per_word.items():
            word_scores[spacy_idx] = {
                "k": len(indep),
                "first": indep[0],
                "mean": float(np.mean(indep)),
                "sum_indep": float(np.sum(indep)),
                "sum_chain": float(np.sum(chain)),
            }

    row = summarise(word_scores)
    row["n_noncontig"] = n_noncontig
    row["n_subwords"] = len(ids)
    return row


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def load_essays(corpus: str):
    essays = []
    for doc in load_all_docbins(CORPUS_DOCBINS[corpus]):
        meta = doc.user_data.get("meta") or {}
        labels = {c: meta.get(c) for c in LABEL_COLS[corpus]}
        essays.append((str(meta.get("text_id")), doc, labels))
    return essays


def subsample(corpus: str, essays, limit: int | None):
    """Stratified on the criterion the corpus supplies, with a fixed seed."""
    rng = np.random.default_rng(SEED)
    idx = pd.DataFrame([lab for _t, _d, lab in essays])
    if corpus == "ellipse":
        strata = idx["overall"].astype(str)
        target = N_ELLIPSE
        take = (strata.value_counts(normalize=True) * target).round().astype(int)
    else:
        strata = idx["l1"].astype(str) + "/" + idx["level"].astype(str)
        per_l1 = idx.groupby("l1").size()
        share = idx.groupby(["l1", "level"]).size().div(per_l1, level="l1")
        take = (share * N_TOEFL_PER_L1).round().astype(int)
        take.index = [f"{a}/{b}" for a, b in take.index]

    keep = []
    for stratum, n in take.items():
        pool = np.flatnonzero((strata == stratum).to_numpy())
        if len(pool) == 0 or n <= 0:
            continue
        keep.extend(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    keep.sort()
    if limit:
        keep = keep[:limit]
    return [essays[i] for i in keep]


# --------------------------------------------------------------------------- #
def load_model(spec, dtype, device):
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    cls = AutoModelForMaskedLM if spec.arch == "masked" else AutoModelForCausalLM
    model = cls.from_pretrained(spec.hf_id, torch_dtype=dtype,
                                trust_remote_code=spec.trust_remote_code)
    model.to(device).eval()
    return tok, model


def save_atomic(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--corpora", nargs="+", default=["all"],
                   choices=["ellipse", "toefl11", "all"])
    p.add_argument("--limit", type=int, default=None, help="first N sampled essays (smoke test)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    models = list(MODEL_REGISTRY) if "all" in args.models else args.models
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        p.error(f"unknown model(s): {unknown}")
    corpora = ["ellipse", "toefl11"] if "all" in args.corpora else args.corpora

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.dtype, torch.bfloat16 if device == "cuda" else torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = {}
    for corpus in corpora:
        allessays = load_essays(corpus)
        samples[corpus] = subsample(corpus, allessays, args.limit)
        logger.info("%s: sampled %d of %d essays", corpus, len(samples[corpus]), len(allessays))

    for model_key in models:
        spec = MODEL_REGISTRY[model_key]
        path = OUT_DIR / f"{model_key}.parquet"
        rows = [] if args.overwrite or not path.exists() else \
            pd.read_parquet(path).to_dict("records")
        done = {(r["corpus"], str(r["text_id"])) for r in rows}
        todo = [(c, e) for c in corpora for e in samples[c]
                if (c, e[0]) not in done]
        if not todo:
            logger.info("%s: already complete (%d rows)", model_key, len(rows))
            continue

        logger.info("=== %s (%s, %s) — %d essays to score ===",
                    spec.key, spec.hf_id, spec.arch, len(todo))
        tok, model = load_model(spec, dtype, device)
        t0 = time.time()
        for n, (corpus, (text_id, doc, labels)) in enumerate(todo, start=1):
            row = score_essay(spec, model, tok, doc, device, args.batch_size)
            if row:
                rows.append({"corpus": corpus, "text_id": text_id, "model": model_key,
                             "arch": spec.arch, **labels, **row})
            if n % args.log_every == 0 or n == len(todo):
                el = time.time() - t0
                logger.info("  %s %d/%d  %.1f essay/s  eta %.0fs",
                            model_key, n, len(todo), n / el, (len(todo) - n) / (n / el))
                save_atomic(path, pd.DataFrame(rows))
        save_atomic(path, pd.DataFrame(rows))
        (OUT_DIR / f"{model_key}.json").write_text(json.dumps({
            "model_key": model_key, "hf_id": spec.hf_id, "arch": spec.arch,
            "window": WINDOW, "dtype": str(dtype).replace("torch.", ""),
            "device": device, "seed": SEED, "n_rows": len(rows),
            "n_ellipse": N_ELLIPSE, "n_toefl_per_l1": N_TOEFL_PER_L1,
            "runtime_seconds": round(time.time() - t0, 1),
        }, indent=2))
        logger.info("%s done in %.0fs — %d rows", model_key, time.time() - t0, len(rows))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info("All done.")


if __name__ == "__main__":
    main()
