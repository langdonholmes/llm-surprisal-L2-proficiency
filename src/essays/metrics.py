"""Calculate linguistic features for ELLIPSE essays and merge with predictability."""

import argparse

import pandas as pd
import numpy as np
import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm

from features import calculate_all_features, MiCalculator, n_words
from util.paths import (
    ELLIPSE_DIR,
    ELLIPSE_DOCBINS_DIR,
    DOLMA_FREQ_DIR,
    PREDICTABILITY_DIR,
)


# Which Dolma reference corpus to draw frequency / dependency norms from.
CORPUS_CHOICES = {"a": ["a"], "b": ["b"], "both": ["a", "b"]}

# Word predictability source: the Study 1 benchmark's selected configuration.
# Llama-3.1-8B (base) at the 8-token window was the validity/fairness Pareto
# selection for the base model (results/predictability/pareto_points.csv — the
# only base-Llama config on the front); the instruct variant is not used.
PRED_MODEL = "llama3.1-8b"
PRED_WINDOW = "8"


def load_predictability(model: str = PRED_MODEL, window: str = PRED_WINDOW):
    """Load per-essay word predictability from the Study 1 surprisal benchmark.

    Reads data/predictability/ellipse/{model}/surprisal.parquet, selects the
    chosen context ``window``, and renames the benchmark's bit-scale columns to
    the names the network analysis expects (mean_loss, var_loss, mean_entropy).
    """
    path = PREDICTABILITY_DIR / "ellipse" / model / "surprisal.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing surprisal parquet at {path}. Run the Study 1 benchmark "
            f"(src/1-predictability-benchmark/run_surprisal.py --models {model}) first."
        )
    cols = ["text_id", "window", "mean_surprisal_bits",
            "var_surprisal_bits2", "mean_entropy_bits"]
    p = pd.read_parquet(path, columns=cols)
    p = p[p["window"].astype(str) == str(window)].copy()
    if len(p) == 0:
        raise ValueError(f"No rows for window '{window}' in {path}.")
    p = p.rename(columns={
        "text_id": "text_id_kaggle",
        "mean_surprisal_bits": "mean_loss",
        "var_surprisal_bits2": "var_loss",
        "mean_entropy_bits": "mean_entropy",
    })
    return p[["text_id_kaggle", "mean_loss", "mean_entropy", "var_loss"]]


def load_reference_grams(corpus: str):
    """Load token-frequency (3-gram) and dependency-bigram tables from the
    collated Dolma frequency tables. ``corpus`` is one of a / b / both;
    ``both`` concatenates the two register corpora and sums shared counts.

    Replaces the pilot SlimPajama lists (data/pilot/slim_pajama_lists).
    """
    corpora = CORPUS_CHOICES[corpus]
    ngram_frames, dep_frames = [], []
    for c in corpora:
        corpus_dir = DOLMA_FREQ_DIR / f"corpus_{c}"
        ngram_path = corpus_dir / "3grams.parquet"
        dep_path = corpus_dir / "depgrams.parquet"
        if not ngram_path.exists() or not dep_path.exists():
            raise FileNotFoundError(
                f"Missing frequency tables for corpus {c} at {corpus_dir}. "
                "Run src/reference_corpus/4_collate.py --corpus "
                f"{c} first."
            )
        ngram_frames.append(pd.read_parquet(ngram_path))
        dep_frames.append(pd.read_parquet(dep_path))

    ngram_df = pd.concat(ngram_frames, ignore_index=True)
    dep_df = pd.concat(dep_frames, ignore_index=True)

    # MiCalculator keys on (head_lemma, dependent_lemma, relation, count).
    # The Dolma collate also carries POS tags (head_tag, dependent_tag);
    # collapse that granularity by summing counts over the lemma+relation key.
    dep_df = dep_df.groupby(
        ["head_lemma", "dependent_lemma", "relation"], as_index=False
    )["count"].sum()

    return ngram_df, dep_df


def main(corpus: str = "both"):
    # Load ELLIPSE public data (scores + prompt); predictability is joined below.
    df = pd.read_csv(ELLIPSE_DIR / "ELLIPSE_Final_github.csv")
    print(f"Loaded {len(df)} essays")

    # Word predictability from the Study 1 benchmark (Llama-3.1-8B base, window 8)
    pred_df = load_predictability()
    print(f"Loaded predictability for {len(pred_df)} essays "
          f"({PRED_MODEL}, window {PRED_WINDOW})")

    # Load pre-built DocBins (created by essays/ingest.py)
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    docbin_dir = ELLIPSE_DOCBINS_DIR

    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"DocBins not found at {docbin_dir}. Run essays/ingest.py --ellipse first."
        )

    print("Loading docs from DocBin files...")
    docs = []
    filenames = []
    for file_path in sorted(docbin_dir.glob("*.spacy")):
        doc_bin = DocBin().from_disk(file_path)
        for doc in doc_bin.get_docs(nlp.vocab):
            docs.append(doc)
            meta = doc.user_data.get("meta", None)
            filenames.append(meta["text_id"] if isinstance(meta, dict) else meta)

    print(f"Loaded {len(docs)} documents")
    assert len(docs) == len(df), f"Expected {len(df)} docs, got {len(docs)}"

    # Load reference data (Dolma frequency tables)
    print(f"Loading reference grams from Dolma corpus '{corpus}'...")
    ngram_df, dep_df = load_reference_grams(corpus)

    token_freq_df = ngram_df.groupby("token_2", as_index=False)["count"].sum()
    token_freq = dict(zip(token_freq_df["token_2"], token_freq_df["count"]))
    total_tokens = sum(token_freq.values())
    print(f"Loaded {len(token_freq)} unique tokens, total {total_tokens} tokens")

    print(f"Loaded {len(dep_df)} dependency bigrams")
    mi_calculator = MiCalculator(dep_df)

    # Calculate features
    results = []
    for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Calculating features"):
        features = calculate_all_features(doc, token_freq, total_tokens, mi_calculator)
        # Essay length, on the same tokenization every other index here uses.
        # Not a complexity feature -- carried as a covariate for analyses that
        # need to show an effect is not just a length effect.
        features["n_tokens"] = n_words(doc)
        features["text_id_kaggle"] = filenames[i]
        results.append(features)

    df_features = pd.DataFrame(results)
    print(f"Computed features for {len(df_features)} documents")

    # Merge scores (public ELLIPSE) + predictability (benchmark) + features
    score_cols = [
        "Overall", "Cohesion", "Syntax", "Vocabulary",
        "Phraseology", "Grammar", "Conventions",
    ]
    df_meta = df[["text_id_kaggle", "prompt"] + score_cols]
    df_meta = pd.merge(df_meta, pred_df, on="text_id_kaggle", how="left")

    df_merged = pd.merge(df_meta, df_features, on="text_id_kaggle", how="inner")
    print(f"Merged dataframe: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")

    output_path = ELLIPSE_DIR / "ellipse_metrics.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # NaN audit
    feature_cols = [
        "mtld", "lexical_density", "log_mean_token_freq", "mean_word_length",
        "clauses_per_tunit", "mean_length_of_clause",
        "complex_nominals_per_clause", "coord_phrases_per_clause",
        "deps_per_clause", "deps_per_nominal",
        "amod_mi", "dobj_mi", "advmod_mi",
        "content_word_overlap", "connective_density", "sentence_similarity",
        "mean_loss", "mean_entropy", "var_loss",
    ]
    print("\n=== NaN Audit ===")
    nan_counts = df_merged[feature_cols].isna().sum()
    nan_pct = (nan_counts / len(df_merged) * 100).round(2)
    nan_report = pd.DataFrame({"NaN count": nan_counts, "NaN %": nan_pct})
    print(nan_report[nan_report["NaN count"] > 0].to_string())
    if nan_report["NaN count"].sum() == 0:
        print("No NaN values found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate ELLIPSE linguistic features merged with predictability."
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPUS_CHOICES),
        default="both",
        help="Dolma reference corpus for frequency / dependency norms "
        "(a=edited, b=unedited, both=combined; default: both).",
    )
    args = parser.parse_args()
    main(corpus=args.corpus)
