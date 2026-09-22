"""Study 1 sensitivity — tokenizer fertility, with no model forward passes.

Study 1 scores a word by its **first** subword under a causal LM. That makes a
word's surprisal a function of how its tokenizer cuts the word up: the more
pieces, the more continuations the first piece sums over, and the cheaper the
word looks. Two consequences matter for a benchmark whose output is a ranking
across eleven models and a fairness audit across eleven L1 groups.

1. The eleven models carry six distinct tokenizers, so the deflation differs by
   model. A cross-model ranking inherits it.
2. Fertility is not constant across writers. Misspellings, non-target morphology
   and rare vocabulary all split into more pieces, and all are more frequent in
   lower-proficiency writing, so the deflation may correlate with the very trait
   Study 1 validates against.

Neither claim needs a language model to check — only its tokenizer. This script
measures fertility over both corpora in full and writes the per-essay table the
report reads. If fertility turns out to be flat across proficiency and similar
across tokenizers, the aggregation conventions are near-affine reparameterizations
of each other at essay level and the matter closes with a footnote. If it is not,
this table says how big the exposure is and where.

Subword-to-word assignment uses the offset mapping directly: a subword belongs to
the spaCy token containing its first non-whitespace character. That is stricter
than the character-start intersection in ``features.predictability``, and the gap
between the two is itself reported (``n_straddle``, ``n_unaligned_alpha``), since
any move to summing over a word's subwords depends on the spans being right.

Writes:
    results/predictability/fertility_by_essay.csv.gz   # essay x model
    results/predictability/fertility_tokenizers.csv    # which models share one

Run from the repo root (CPU only, a few minutes):

    python src/1-predictability-benchmark/sensitivity/fertility.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from util.paths import ELLIPSE_DOCBINS_DIR, RESULTS_DIR, TOEFL_DOCBINS_DIR
from util.process_docs import load_all_docbins

# models.py lives one directory up, whose name starts with a digit and so can't
# be imported as a package. run_surprisal.py gets it for free by sitting beside
# it; from here it needs the explicit path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import MODEL_REGISTRY  # noqa: E402

CORPUS_DOCBINS = {"ellipse": ELLIPSE_DOCBINS_DIR, "toefl11": TOEFL_DOCBINS_DIR}
LABEL_COLS = {"ellipse": ["overall"], "toefl11": ["l1", "level"]}
OUT_DIR = RESULTS_DIR / "predictability"

# Probe text for the tokenizer signature. Deliberately includes a misspelling, a
# rare derivation and punctuation, so two tokenizers that agree on it agree on
# the cases this analysis is about.
PROBE = (
    "The unpredictability of teh students' writing, however, is antidisestablishmentarianism "
    "— a peoples' misstake worth 3.14 tokens."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("fertility")


def char_to_word(doc) -> np.ndarray:
    """Map every character offset in ``doc.text`` to its spaCy token index (-1 = none)."""
    owner = np.full(len(doc.text), -1, dtype=np.int32)
    for tok in doc:
        owner[tok.idx : tok.idx + len(tok.text)] = tok.i
    return owner


def essay_fertility(doc, tokenizer) -> dict:
    """Per-essay subword counts for one tokenizer.

    Each subword is charged to the spaCy token holding its first non-whitespace
    character. Whitespace-only subwords are charged to nothing and counted apart,
    so ``n_subwords_charged`` and ``n_words`` are the two sides of one ratio.
    """
    text = doc.text
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    owner_of_char = char_to_word(doc)

    pieces = np.zeros(len(doc), dtype=np.int32)  # subwords charged to each token
    n_ws = n_straddle = n_empty = 0

    for start, end in offsets:
        if end <= start:
            n_empty += 1
            continue
        span = text[start:end]
        start_adj = start + len(span) - len(span.lstrip())
        end_adj = end - (len(span) - len(span.rstrip()))
        if start_adj >= end_adj:  # subword is pure whitespace
            n_ws += 1
            continue
        first = owner_of_char[start_adj]
        last = owner_of_char[end_adj - 1]
        if first >= 0:
            pieces[first] += 1
        if first != last:
            # The subword crosses a spaCy token boundary, so no word-level
            # aggregation can recover this word's probability exactly.
            n_straddle += 1

    is_alpha = np.fromiter((t.is_alpha for t in doc), bool, len(doc))
    is_word = np.fromiter((not t.is_space for t in doc), bool, len(doc))
    alpha_pieces = pieces[is_alpha]

    return {
        "n_subwords": len(offsets),
        "n_subwords_ws": n_ws,
        "n_subwords_empty": n_empty,
        "n_subwords_charged": int(pieces.sum()),
        "n_subwords_alpha": int(alpha_pieces.sum()),
        "n_words": int(is_word.sum()),
        "n_words_alpha": int(is_alpha.sum()),
        "n_split2_alpha": int((alpha_pieces >= 2).sum()),
        "n_split3_alpha": int((alpha_pieces >= 3).sum()),
        "n_unaligned_alpha": int((alpha_pieces == 0).sum()),
        "n_straddle": n_straddle,
        "n_chars": len(text),
    }


def load_essays(corpus: str, limit: int | None):
    """[(text_id, doc, {label: value}), ...] for one corpus."""
    docbin_dir = CORPUS_DOCBINS[corpus]
    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"No DocBins for {corpus} at {docbin_dir}. Run essays/ingest.py first."
        )
    essays = []
    for doc in load_all_docbins(docbin_dir):
        meta = doc.user_data.get("meta") or {}
        labels = {c: meta.get(c) for c in LABEL_COLS[corpus]}
        essays.append((str(meta.get("text_id")), doc, labels))
        if limit and len(essays) >= limit:
            break
    return essays


def tokenizer_signature(tok) -> tuple:
    """Identify tokenizers that are interchangeable for this analysis.

    Vocabulary size alone is too coarse (it collides across unrelated
    tokenizers) and ``name_or_path`` too fine (base and instruct checkpoints
    ship the same tokenizer under different names), so the signature is the
    vocabulary size plus the probe's actual segmentation.
    """
    ids = tok(PROBE, add_special_tokens=False).input_ids
    return (tok.vocab_size, tuple(ids))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model keys (see ../models.py) or 'all'")
    p.add_argument("--corpora", nargs="+", default=["all"],
                   choices=["ellipse", "toefl11", "all"])
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N essays per corpus (smoke testing)")
    args = p.parse_args()

    models = list(MODEL_REGISTRY) if "all" in args.models else args.models
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        p.error(f"unknown model(s): {unknown}. choices: {list(MODEL_REGISTRY)}")
    corpora = ["ellipse", "toefl11"] if "all" in args.corpora else args.corpora

    # --- load tokenizers, group the ones that segment identically ------------
    tokenizers, sig_owner, groups, skipped = {}, {}, {}, []
    for key in models:
        spec = MODEL_REGISTRY[key]
        try:
            tok = AutoTokenizer.from_pretrained(
                spec.hf_id, trust_remote_code=spec.trust_remote_code
            )
        except Exception as exc:  # gated repo without a token, no network, etc.
            logger.warning("SKIP %s (%s): %s", key, spec.hf_id, exc)
            skipped.append(key)
            continue
        sig = tokenizer_signature(tok)
        if sig in sig_owner:
            groups[key] = sig_owner[sig]
            logger.info("%-22s shares a tokenizer with %s", key, sig_owner[sig])
        else:
            sig_owner[sig] = key
            groups[key] = key
            tokenizers[key] = tok
            logger.info("%-22s vocab=%d, probe=%d subwords",
                        key, tok.vocab_size, len(sig[1]))
    if not tokenizers:
        sys.exit("No tokenizers could be loaded.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": k, "hf_id": MODEL_REGISTRY[k].hf_id, "arch": MODEL_REGISTRY[k].arch,
         "tokenizer_group": g, "vocab_size": tokenizers[g].vocab_size,
         "probe_subwords": len(tokenizer_signature(tokenizers[g])[1])}
        for k, g in sorted(groups.items())
    ]).to_csv(OUT_DIR / "fertility_tokenizers.csv", index=False)

    # --- measure, once per distinct tokenizer --------------------------------
    rows = []
    for corpus in corpora:
        essays = load_essays(corpus, args.limit)
        logger.info("loaded %d %s essays", len(essays), corpus)
        for group_key, tok in tokenizers.items():
            t0 = time.time()
            counts = {}
            for text_id, doc, labels in essays:
                counts[text_id] = essay_fertility(doc, tok)
            # Replay the group's counts under every model key that shares it, so
            # the table joins straight onto results/predictability/*_long.csv.gz.
            members = [k for k, g in groups.items() if g == group_key]
            for text_id, doc, labels in essays:
                for member in members:
                    rows.append({"corpus": corpus, "text_id": text_id, "model": member,
                                 "tokenizer_group": group_key, **labels,
                                 **counts[text_id]})
            logger.info("  %-22s %s: %d essays in %.0fs (-> %s)", group_key, corpus,
                        len(essays), time.time() - t0, ", ".join(members))

    df = pd.DataFrame(rows)
    df["fertility"] = df["n_subwords_charged"] / df["n_words"]
    df["fertility_alpha"] = df["n_subwords_alpha"] / df["n_words_alpha"]

    out = OUT_DIR / "fertility_by_essay.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    logger.info("wrote %s  %s", out, df.shape)
    if skipped:
        logger.warning("models skipped (no tokenizer): %s", skipped)


if __name__ == "__main__":
    main()
