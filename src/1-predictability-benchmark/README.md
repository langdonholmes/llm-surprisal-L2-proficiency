# Study 1 — Predictability Benchmark

Compute mean token surprisal for every **model × context window × corpus**, then
(in a later pass) select the configuration that best balances validity and
fairness. See [`plan.md`](plan.md) for the full design and statistical analysis.

This directory currently implements **§2 (surprisal computation)**. The §3
statistical analysis (validity / fairness / Pareto) is the next phase and runs
on the outputs produced here.

## Files

- `models.py` — the 11-model matrix (`MODEL_REGISTRY`). Add a model by appending
  one `ModelSpec`; its `key` becomes the CLI name and output directory.
- `run_surprisal.py` — the runner. Drives `features.predictability.Predictor`
  over the matrix and writes idempotent, resumable per-model tables.

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

### 2. Full matrix

```bash
# All 11 models × 3 windows × 2 corpora. Long-running — see notes below.
python src/1-predictability-benchmark/run_surprisal.py
```

Because the run is **idempotent and resumable** (keyed on `text_id × window`,
flushed atomically every 200 essays), prefer running it model-by-model in the
background so a failure or an OOM only loses one model's tail:

```bash
for m in bert-base modernbert-base modernbert-large gpt2 gpt2-xl \
         olmo2-1b olmo2-7b qwen2.5-7b qwen2.5-7b-instruct; do
    python src/1-predictability-benchmark/run_surprisal.py --models "$m"
done
# gated — after huggingface-cli login:
python src/1-predictability-benchmark/run_surprisal.py --models llama3.1-8b llama3.1-8b-instruct
```

Re-running any command skips configurations already on disk. Adding a new model
later is just `--models <new-key>`; adding a window is `--windows full`.

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
