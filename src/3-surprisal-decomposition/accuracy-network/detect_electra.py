"""Accuracy network, detector arm (D54, D55): ELECTRA token-level error detection (GPU).

Runs the 25-class ELECTRA detector of Yuan et al. (2021), as released by Goto
(gotutiyan/token-ged-electra-large-25cls), over one corpus's spaCy parses. It
gives every token a label, CORRECT or an ERRANT category without its operation
prefix, and those categories go through the same groups as the corrector's edits
(``lib.errant.group_of``).

The model reads at most 128 subwords and its own code silently cuts longer inputs,
so tokens past the limit would never be labelled. Unpunctuated essays produce long
sentences, so a sentence over the limit is split into consecutive pieces instead.

Writes data/study3/<corpus>/detector_electra.parquet, one row per unit: the tokens
flagged in each ERRANT group (n_<group>) and in total (n_flagged), and the tokens
read (n_read).

    python src/3-surprisal-decomposition/accuracy-network/detect_electra.py --corpus wi fce ellipse
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.corpora import DATA, DOCBINS, data_dir  # noqa: E402
from lib.errant import GROUPS, group_of  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

MODEL = "gotutiyan/token-ged-electra-large-25cls"
REVISION = "90c6d2748f5e3bfa5ce13f1e9dba507c7f98a173"  # the release used, pinned
MAX_SUBWORDS = 126  # the model's 128 less [CLS] and [SEP]


def split_long_sentence(tokens: list[str], tokenizer) -> list[list[str]]:
    """Consecutive pieces of a sentence, each within the model's subword limit."""
    subword_counts = [len(subwords) for subwords in tokenizer(tokens, add_special_tokens=False)["input_ids"]]
    if sum(subword_counts) <= MAX_SUBWORDS:
        return [tokens]
    pieces, piece, piece_subwords = [], [], 0
    for token, count in zip(tokens, subword_counts):
        if piece and piece_subwords + count > MAX_SUBWORDS:
            pieces.append(piece)
            piece, piece_subwords = [], 0
        piece.append(token)
        piece_subwords += count
    return pieces + [piece]


@torch.inference_mode()
def label_pieces(pieces: list[list[str]], model, tokenizer, batch_size: int) -> list[list[str]]:
    """One label per token of every piece, read off each token's first subword as in training."""
    order = sorted(range(len(pieces)), key=lambda index: len(pieces[index]))  # similar lengths batch together
    labels = [None] * len(pieces)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        encoded = tokenizer([pieces[index] for index in batch_indices], is_split_into_words=True, padding=True,
                            truncation=False, return_tensors="pt")
        predictions = model(**{key: value.to(model.device) for key, value in encoded.items()}).logits.argmax(-1).cpu()
        for row, index in enumerate(batch_indices):
            first_subword = {}
            for position, word_id in enumerate(encoded.word_ids(row)):
                if word_id is not None and word_id not in first_subword:
                    first_subword[word_id] = position
            # A token with no subwords at all (a stray control character) cannot be flagged
            labels[index] = [model.config.id2label[int(predictions[row, first_subword[word]])]
                             if word in first_subword else "CORRECT" for word in range(len(pieces[index]))]
    return labels


def detect(corpus: str, model, tokenizer, batch_size: int):
    unit_of_text = pd.read_parquet(DATA / corpus / "essays.parquet", columns=["text_id", "unit"]).set_index("text_id").unit
    pieces, piece_unit, n_sentences, n_split = [], [], 0, 0
    for doc in load_all_docbins(DOCBINS[corpus]):
        unit = unit_of_text[doc.user_data["meta"]["text_id"]]
        for sentence in doc.sents:
            tokens = [token.text for token in sentence if not token.is_space]
            if not tokens:
                continue
            sentence_pieces = split_long_sentence(tokens, tokenizer)
            n_sentences += 1
            n_split += len(sentence_pieces) > 1
            pieces += sentence_pieces
            piece_unit += [unit] * len(sentence_pieces)
    print(f"{corpus}: {n_sentences} sentences, {n_split} over the limit, {len(pieces)} pieces", flush=True)
    labels = label_pieces(pieces, model, tokenizer, batch_size)
    flagged_by_unit, read_by_unit, categories = {}, Counter(), Counter()
    for unit, piece_labels in zip(piece_unit, labels):
        read_by_unit[unit] += len(piece_labels)
        counts = flagged_by_unit.setdefault(unit, Counter())
        for label in piece_labels:
            if label != "CORRECT":
                categories[label] += 1
                counts[group_of(label)] += 1
    table = pd.DataFrame({f"n_{group}": {unit: counts[group] for unit, counts in flagged_by_unit.items()} for group in GROUPS})
    table["n_flagged"] = table.sum(axis=1)
    table["n_read"] = pd.Series(read_by_unit)
    table.rename_axis("unit").to_parquet(data_dir(corpus) / "detector_electra.parquet")
    (data_dir(corpus) / "detector_electra.json").write_text(json.dumps(
        {"model": MODEL, "revision": REVISION, "n_units": len(table), "n_sentences": n_sentences,
         "n_sentences_split": n_split, "n_pieces": len(pieces), "n_read": int(table.n_read.sum()),
         "n_flagged": int(table.n_flagged.sum()), "by_category": dict(categories.most_common())}, indent=2))
    print(f"{corpus}: {table.n_flagged.sum() / table.n_read.sum():.3f} of {table.n_read.sum()} tokens flagged", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, revision=REVISION, use_safetensors=True)
    model = model.to("cuda").eval()
    print(label_pieces([["This", "are", "wrong", "sentece", "."], ["This", "is", "correct", "."]], model, tokenizer, 2))
    for corpus in args.corpus:
        detect(corpus, model, tokenizer, args.batch_size)


if __name__ == "__main__":
    main()
