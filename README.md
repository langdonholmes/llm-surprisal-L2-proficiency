# Word Predictability and L2 Proficiency

This project investigates where **LLM surprisal** fits within the structure of second language (L2) writing proficiency. Rather than imposing a theoretical factor structure (e.g., CAF), I use network psychometric methods to empirically discover how LLM surprisal relates to established linguistic features across lexical, syntactic, phraseological, and discourse levels.

## Data

- **ELLIPSE Corpus** (public): 6,482 L2 essays with dimensional scores (Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions) and 44 writing prompts. Source: `ELLIPSE_Final_github.csv`.
- **Dolma v1.7**: sampled into two ~1.05B-token reference corpora used for token frequency and dependency MI calculations:
  - **Corpus A (edited prose)**: wiki, pes2o, cc_news
  - **Corpus B (unedited prose)**: reddit, stackexchange, cc_blogs

## Features

Linguistic features are computed from spaCy parses and a Dolma reference corpus:

| Feature | Level | Source |
|---------|-------|--------|
| MTLD | Lexical | lexicalrichness |
| Lexical density | Lexical | spaCy POS |
| Log token frequency | Lexical | Dolma reference |
| Mean word length | Lexical | spaCy |
| Mean length of clause (MLC) | Syntactic | spaCy |
| Complex nominals per clause (CN/C) | Syntactic | spaCy |
| Coordinate phrases per clause (CP/C) | Syntactic | spaCy |
| amod MI | Phraseological | Dolma reference |
| dobj MI | Phraseological | Dolma reference |
| advmod MI | Phraseological | Dolma reference |
| Content word overlap | Cohesion | spaCy |
| Connective density | Cohesion | spaCy + word list |
| Mean surprisal (`mean_loss`) | Predictability | Study 1 benchmark (OLMo-2 1B) |
| Loss variance (`var_loss`) | Predictability | Study 1 benchmark (OLMo-2 1B) |

Word predictability is produced by the Study 1 benchmark
(`src/1-predictability-benchmark/`); OLMo-2 1B (base) at the 8-token window was
selected as the configuration that feeds Studies 2–3 — the smallest model with a
publicly released pretraining corpus in the highest-validity configuration
class. `mean_entropy` is computed but excluded from
network analysis because it largely duplicates `mean_loss` (r = 0.85 in the
selected configuration).

## Pipeline

Processing is split into a **reference-corpus** track (Dolma, built once) and a
**shared essay** track (TOEFL/ELLIPSE ingestion + feature extraction). Run all
from the repo root.

```bash
# Reference corpus: sample, preprocess, verify, dependency-parse, and
# collate Dolma into frequency tables. Parse/collate run once per corpus.
python src/reference_corpus/0_sample.py
python src/reference_corpus/1_preprocess.py
python src/reference_corpus/2_verify.py
python src/reference_corpus/3_parse.py   --corpus a   # then --corpus b
python src/reference_corpus/4_collate.py --corpus a   # then --corpus b

# Essays: ingest TOEFL 11 and/or ELLIPSE into spaCy DocBins
python src/essays/ingest.py --ellipse        # or --toefl / --all

# Essays — word predictability comes from the Study 1 benchmark; the selected
# config is OLMo-2 1B (base) at the 8-token window. See
# src/1-predictability-benchmark/.
#   Source: data/predictability/ellipse/olmo2-1b/surprisal.parquet

# Essays — linguistic features merged with scores + predictability.
#   Reference norms come from the Dolma frequency tables (--corpus a|b|both,
#   default both). metrics.py joins the OLMo-2 1B (base, window 8) surprisal
#   above as mean_loss / var_loss. Output: data/ellipse/ellipse_metrics.csv
python src/essays/metrics.py --corpus both

# Network analysis (R/Quarto)
quarto render src/2-nomological-network/network-analysis.qmd
```

The network analysis residualizes all features on writing prompt before estimation to remove systematic prompt-driven variance.

## Project Structure

`src/` is organized into shared libraries (`features/`, `util/`), the two
processing tracks (`reference_corpus/`, `essays/`), and one directory per
dissertation chapter (`1-…`, `2-…`, `3-…`).

```
src/
  features/                 # Linguistic feature calculators (shared lib)
    predictability.py       # Word predictability (masked or causal LM surprisal + variance)
    lexical.py              # MTLD, lexical density, token frequency, word length
    syntactic.py            # L2SCA battery (Lu 2010/11): MLC, CN/C, CP/C
                            #   (C/T + K&C 2018 deps/clause, deps/nominal are
                            #    computed but not entered — C/T indexes
                            #    punctuation here, MLC supersedes deps/clause)
    phraseological.py       # Dependency MI (amod, dobj, advmod)
    cohesion.py             # Content word overlap, connective density, sentence sim
  util/                     # Shared utilities
    paths.py                # Canonical path constants (DATA_DIR, DOLMA_DIR, ELLIPSE_DIR, …)
    process_docs.py         # Batch spaCy DocBin processing utilities
    collation.py            # N-gram / depgram counting and parquet writers
    text_heatmap.R          # Token-level text heat map for the R/Quarto reports
  reference_corpus/         # Dolma reference-corpus pipeline (run in order)
    dolma_config.py         # Dolma sources and per-corpus sampling targets
    dolma_stream.py         # Streaming Dolma reader/sampler
    0_sample.py             # Sample corpora A and B from Dolma v1.7
    1_preprocess.py         # Clean, length-filter, dedup -> jsonl.gz
    2_verify.py             # Token / PII / duplicate verification report
    3_parse.py              # Dependency-parse to DocBins (stride parallelism)
    4_collate.py            # Frequency tables (n-grams + dependency bigrams)
  essays/                   # Shared TOEFL/ELLIPSE essay processing
    ingest.py               # TOEFL 11 + ELLIPSE ingestion -> DocBins
    metrics.py              # ELLIPSE features merged with scores
  1-predictability-benchmark/   # Study 1: predictability benchmark (ELLIPSE, TOEFL)
  2-nomological-network/        # Study 2: network psychometric analysis (R/Quarto, ELLIPSE)
  3-vector-transformations/     # Study 3: NMF / vector transformations (ELLIPSE, TOEFL)
    pilot/                  # Archived pilot code (UMAP/HDBSCAN + NMF exploration)
data/
  ellipse/                  # ELLIPSE target corpus (studies 1, 2, 3)
    ELLIPSE_Final_github.csv  # Public ELLIPSE dataset (6,482 essays, 44 prompts)
    docbins/                # ELLIPSE essay DocBins
    *.csv / *.parquet       # Computed predictability + metrics outputs
  toefl11/                  # TOEFL 11 target corpus (studies 1, 3)
    toefl_11_txt.zip, index-training-dev.csv
    docbins/, context_routes/
  dolma/                    # Dolma reference corpus
    corpus_a/, corpus_b/    # Raw sampled source shards
    preprocessed/           # Cleaned jsonl.gz + summaries
    docbins/                # spaCy DocBins (corpus_a, corpus_b)
    frequency_tables/       # Collated n-gram / depgram parquet (corpus_a, corpus_b)
    reports/                # Sampling and verification reports (JSON)
  pilot/                    # Archived pilot artifacts (SlimPajama lists, delta vectors)
```

## Setup

Install the project as an editable package so the shared libraries (`util`,
`features`, `reference_corpus`, `essays`) import cleanly from any script — no
`sys.path` manipulation:

```bash
uv pip install -e .
```

## Requirements

Python 3.11+ with PyTorch, Transformers, spaCy, and pandas. See `pyproject.toml`
for full dependencies. R with lavaan, bootnet, qgraph, and EGAnet for network
analysis.
