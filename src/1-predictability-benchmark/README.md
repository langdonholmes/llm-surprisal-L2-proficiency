# Study 1 — Predictability Benchmark

Compute mean token surprisal for every **model × context window × corpus**, then
select the configuration that best balances validity and fairness. See
[`plan.md`](plan.md) for the full design and statistical analysis.

**Result:** Llama-3.1-8B (base) was selected as the predictability model that
feeds Studies 2–3.

This directory implements both **§2 (surprisal computation)** — `run_surprisal.py`
— and **§3 (validity / fairness / Pareto analysis)** — `assemble_analysis_data.py`
plus the `pareto-*.qmd` reports, which run on the §2 outputs.

## Files

- `models.py` — the 11-model matrix (`MODEL_REGISTRY`). Add a model by appending
  one `ModelSpec`; its `key` becomes the CLI name and output directory.
- `run_surprisal.py` — the runner. Drives `features.predictability.Predictor`
  over the matrix and writes idempotent, resumable per-model tables.
- `assemble_analysis_data.py` — exports the per-model surprisal parquets to tidy
  long CSVs under `results/predictability/` for the R/Quarto analysis.
- `pareto-analysis.qmd`, `pareto-by-proficiency.qmd` — validity/fairness Pareto
  reports (§3); `_viz_helpers.R` holds shared plotting helpers.
- `proficiency-confound-sensitivity.qmd` — bounds how much of the cross-L1
  displacement could be within-band proficiency rather than L1. TOEFL 11's
  proficiency scale has only three levels, so stratifying on it (or residualizing
  on it, which uses the same information) cannot remove within-band differences.
  Structure: a rule-of-thumb bound built on Cochran's (1968) result that three
  strata remove about 79% of a normal confounder's bias, corroborated by
  `sensemakr` (Cinelli & Hazlett 2020), with a latent-variable version kept in an
  appendix. All three agree; the two main methods rank the 33 configurations at
  Spearman 0.99 and clear all 33. Depends on `ordinal`, `truncnorm`, `sensemakr`.

## Prerequisites

1. **DocBins built** for the corpora you want (`essays/ingest.py --ellipse`,
   `--toefl`). The runner reads parsed docs + labels straight from
   `data/{ellipse,toefl11}/docbins/`.
2. **GPU** for anything past the ~125M tier. bf16 is the default on CUDA; the
   8B models fit on a single 48 GB card.
3. **Hugging Face auth for the gated Llama 3.1 configs** — accept the license on
   the model pages, then `huggingface-cli login` (or `export HF_TOKEN=...`)
   before running `llama3.1-8b` / `llama3.1-8b-instruct`. All other models need
   no token.

## Running

### 1. Smoke test first (small, fast, CPU-friendly)

```bash
python src/1-predictability-benchmark/run_surprisal.py \
    --models gpt2 bert-base --corpora ellipse --windows 64 --limit 5 --device cpu
```

`--limit N` scores only the first N essays; combine with one small masked
(`bert-base`) and one small causal (`gpt2`) model to exercise both code paths in
a couple of minutes. Inspect `data/predictability/ellipse/{gpt2,bert-base}/surprisal.parquet`,
then delete `data/predictability/` before the real run so test rows don't mix in.

### 2. Full matrix — all GPUs (recommended)

`run_all.sh` cost-balances the model list across every detected GPU and launches
one process per GPU (pinned via `CUDA_VISIBLE_DEVICES`), each writing its own
log. The work is embarrassingly parallel across models and every model fits on
one card, so this is the fastest path — no model is split across GPUs.

```bash
# All 11 models × 3 windows × 2 corpora, fanned out over all GPUs.
bash src/1-predictability-benchmark/run_all.sh

# Extra args pass through to run_surprisal.py:
bash src/1-predictability-benchmark/run_all.sh --windows 64 --corpora ellipse

# Restrict the model set / pin GPUs via env:
MODELS="qwen2.5-7b olmo2-7b llama3.1-8b" bash src/1-predictability-benchmark/run_all.sh
GPUS="0,1" bash src/1-predictability-benchmark/run_all.sh

# Long detached run:
nohup bash src/1-predictability-benchmark/run_all.sh > logs/run_all.log 2>&1 & disown
tail -f logs/surprisal_gpu*.log
```

### 3. Render the reports (R/Quarto)

The `.qmd` reports are a separate step from the Python runner. Nothing above
generates them, and `**/*.html` is gitignored, so a fresh checkout has no
rendered output until this is run. They read only the tidy CSVs from
`assemble_analysis_data.py`, not the parquets.

```bash
python src/1-predictability-benchmark/assemble_analysis_data.py   # if not already done
quarto render src/1-predictability-benchmark                      # all three reports
```

Render one at a time with `quarto render src/1-predictability-benchmark/<file>.qmd`.
A bare `quarto render` renders every `.qmd` in the project, including Study 2.
Paths inside the reports are relative to the repo root (see `_quarto.yml`).

### 4. Single process / single model

```bash
# All 11 models on one GPU, sequentially.
python src/1-predictability-benchmark/run_surprisal.py

# One model (e.g. a newly-added one), both corpora, all windows.
python src/1-predictability-benchmark/run_surprisal.py --models olmo2-1b
```

Because the run is **idempotent and resumable** (keyed on `text_id × window`,
flushed atomically every 200 essays), re-running any command skips
configurations already on disk — a crash or OOM only loses the current config's
tail. Adding a new model later is just `--models <new-key>` (and it's picked up
by `run_all.sh` automatically); adding a window is `--windows full`.

### Useful flags

| Flag | Effect |
|---|---|
| `--models KEY...` / `all` | which models (default `all`) |
| `--corpora ellipse toefl11` / `all` | which corpora (default `all`) |
| `--windows 8 64 full` / `all` | which context windows (default `all`) |
| `--limit N` | first N essays per corpus (testing) |
| `--batch-size B` | masked: words/batch; causal: windows/batch (default 32) |
| `--dtype auto\|bf16\|fp16\|fp32` | precision (default bf16 on CUDA) |
| `--device auto\|cuda\|cpu` | default auto |
| `--dump-tokens` | also write per-token `tokens.parquet` (recomputes the targeted configs) |
| `--overwrite` | recompute even if results exist |
| `--log-every N` | progress line every N essays (default 100) |

Per-token records are **opt-in** (`--dump-tokens`) — typically only for the
Study 2/3 winning configuration, since storing tokens for all 66 configs is large.

## Output

```
data/predictability/{corpus}/{model_key}/
    surprisal.parquet   # one row per essay × window
    manifest.json       # hf_id, pinned commit, dtype, lib versions, per-run stats
    tokens.parquet      # only if --dump-tokens
```

`surprisal.parquet` columns: the essay's labels (ELLIPSE: trait scores + prompt;
TOEFL: `l1`, `level`, prompt), plus `window`, `mean_surprisal_bits`,
`mean_entropy_bits`, `var_surprisal_bits2`, `n_tokens_scored`, `n_subwords`, and
`truncated` (essay longer than the model's max context — expected for ~30% of
ELLIPSE essays on BERT/GPT-2 at the full window; flagged for reporting).

## Known issues

**`modernbert-base` at the full window produces corrupt surprisal. Open.**
Mean surprisal at the 8190-token window (5.364 bits on ELLIPSE) exceeds its own
8-token window (4.736). A working model cannot do this, since the 8-token
context is a subset of the full one. `modernbert-large` is correct on the
identical code path, and windows 8 and 64 are correct for both models.
Within-essay surprisal variance is 40.3 against 8.1 for `modernbert-large/full`,
and a two-component fit implies roughly 28% of tokens scored near log2(vocab),
i.e. near-uniform over the vocabulary. It reproduces on TOEFL 11, where the
configuration reaches |rho| = 0.034 against proficiency band. Not truncation
(`n_truncated: 0`) and not the windowing code. Suspected bf16 instability at
long sequence lengths, upstream of this repo, but untested. **Studies 2 and 3
are unaffected**, since the selected configuration is `llama3.1-8b/8`.

Code that works around it is tagged `TEMPORARY(modernbert-base/full)`:

```bash
grep -rn "TEMPORARY(modernbert-base/full)" src/
```

Once the configuration is fixed or removed from the matrix, re-run
`assemble_analysis_data.py` first, then all three reports, then clear those tags.

## Method notes (kept consistent across all models)

- **Surprisal in bits** (log base 2); text-level index = mean token surprisal.
- **Masked** models mask the whole word and score it from a centered context
  window (full window reserves 2 positions for `[CLS]`/`[SEP]`).
- **Causal** models score each word's *first* subword from left context only,
  using equal-length windows that end at the target (verified against a manual
  next-token reference). The first token of an essay has no context and is left
  unscored.
- **Raw essay text is scored identically for base and instruct models** — no
  chat template, no system prompt — so the post-training contrast is valid.
