# Study 3 — Implementation Plan: Decomposing Prediction Errors via Delta Vectors + NMF

**Goal.** Take the high-surprisal tokens in learner essays and turn each one into a
*delta vector* (the predicted embedding minus the observed embedding). Then
decompose the delta matrix with NMF into additive "error-type" components, give
each component an interpretation with an automated pipeline that is checked by a
human, and test whether essay-level component profiles add information about
proficiency beyond mean surprisal. Extend to TOEFL 11 for an L1 analysis.

**Dissertation home.** Chapter 5 (Study 1 = Chapter 3, Study 2 = Chapter 4).
The chapter will follow the structure Scott asked for in Chapter 3: Overview,
then Method, Results by RQ, and Discussion by RQ, with the RQs kept in the same
order throughout. The `../Dissertation/Chapter 4/chapter4-study2-outline.docx`
document shows the target style for the drafting outline. It uses the markers
[NEW], [GAP], [NEEDS], [SUGGESTED], [DECIDE] and has a table of "approved
outline vs. as implemented".

**Organizing principle: reproducibility.** Every number in Chapter 5 must be
regenerable from (a) a pinned environment, (b) a frozen configuration, and (c) a
chain of stage manifests whose input hashes verify end to end. Any step that
calls an LLM must be replayable offline from a committed cache. Analytic choices
are fixed in this document *before* outcome analyses run (§9). Choices made
later are logged as deviations, not silently absorbed.

---

## 0. Sources this plan is built from (read these first)

| Source | What it contributes |
|---|---|
| `Dissertation Proposal/langdon-holmes-dissertation-proposal-llm-surprisal-2026.docx` (Study 3 section) | Approved method: 90th-pct threshold from top-tertile essays, top-k=50 renormalized e_pred, pos/neg split, NNDSVDa + CD NMF, explain-then-simulate interpretation, 4-point human validation of ~10 components, elastic net ΔR² over mean surprisal, hypotheses (ΔR² > .05) |
| `Dissertation/dissertation_outline.docx` (Ch. 5) | Post-defense revisions: **TOEFL 11 extension** and an **explicit baseline comparison** (raw n-gram incl. unigram features, "simpler model") |
| `Dissertation Proposal/Defense Notes.docx` | Andy's firm requests (TOEFL 11 for L1; compare NMF to n-gram / simpler models); the terminology rule: say "generative", not "causal", in prose |
| `src/1-predictability-benchmark/README.md`, `plan.md`, `results/predictability/{validity_table,pareto_points}.csv` | Selected configuration = `olmo2-1b/8`; the masked-path subword-averaging issue; manifest/idempotency conventions |
| `src/3-surprisal-decomposition/pilot/` (`run_pilot.py`, `nmf_features.py`, `concordance.py`, `umap-hdbscan-results.md`) | Working delta extraction (ModernBERT-base, window 64, 999 essays, 55,998 deltas); **key pilot finding: observed-token lexical identity dominates delta space**; four suggested remedies |
| `Dissertation/Chapter 4/chapter4-study2-outline.docx` | Hand-off: predictability sits with the dependency-MI community but tracks **Grammar and Conventions** ratings more than Phraseology. "What the mean cannot say is which kind of unconventionality an essay contains. That is the question Study 3 takes up." |
| `Dissertation/Chapter 3/holmes_chapter3-study1-v3.docx` | Reproducibility paragraph style; note that line ~831 says Study 3 "applies the resulting indices to the modelling of learner writing development". That is stale and should be corrected to match this plan |

---

## 1. Model and embedding space (settled)

**Decision (Langdon, 2026-10-17): the proposal's tied-embedding requirement is
withdrawn.** Study 3 uses the Study 1 selected configuration, **`olmo2-1b/8`**
(`allenai/OLMo-2-0425-1B`, generative, 8-token window). Its config has
`tie_word_embeddings: false` (hidden 2048, vocab 100,352), so the only open
question is which matrix to compute deltas in. The answer is fixed here:

- **Primary: output (unembedding) space.** e_pred = Σ p̃ᵢ · W_U[i] (top-k
  renormalized) and e_obs = W_U[obs]. Both are rows of the same matrix, so δ is
  well-defined geometrically. This is also the space where the prediction is
  made (logitᵢ = h · W_U[i]). A useful identity for Method: h · δ =
  E_p̃[logit] − logit_obs, which ties δ directly to the surprisal it
  decomposes. The rationale for dropping the tied requirement: tying only
  guarantees that the input and output spaces coincide, and the delta needs a
  single space, not two coinciding ones.
- Benefit: Studies 1 → 2 → 3 use one estimator, and the high-surprisal tokens
  are the ones whose aggregate Study 2 characterized.
- **Sensitivity checks (§8):** the same model with `embedding_space: input`
  (W_E), and optionally a tied masked model (ModernBERT-large/8, the highest
  validity tied config). The second shows that the decomposition does not depend
  on architecture and connects to the ModernBERT pilot.

The pipeline stays model-agnostic, with `model_key`, `window` and
`embedding_space` read from config, so these checks run through identical code.
Chapter 5 Method needs one paragraph explaining the change from the approved
proposal, which had a "Tied Emb." column in Table 1 and the sentence "requires a
model with tied embeddings". Log it in `DECISIONS.md` as the first entry.

## 2. Research questions and hypotheses (frozen)

| # | Question | Primary analysis |
|---|---|---|
| RQ1 | Do prediction errors at high-surprisal positions decompose into interpretable linguistic components? | NMF + automated detection scores + human validation → taxonomy (error pattern / lexeme-POS feature / artifact) |
| RQ2 | Are components differentially associated with proficiency? | Essay-level component profile × Overall (prompt-adjusted Spearman, FDR); morphosyntactic vs. lexical comparison |
| RQ3 | Do component profiles add information beyond aggregate surprisal, and beyond simpler models? | Held-out ΔR² (elastic net) over a mean-surprisal baseline, plus the baseline ladder (§7) |
| RQ4 [NEW] | Are components differentially aligned with the analytic ratings (Grammar, Conventions, Vocabulary, Phraseology, Syntax, Cohesion)? | Component profiles → each analytic score; links to Ch. 4 RQ4 (LMG). **Exploratory; label it as such** |
| RQ5 [NEW, from defense] | Do component profiles differ by L1 at matched proficiency (TOEFL 11)? | Within-stratum SMD per component per L1, same logic as Study 1 fairness |

Hypotheses (from the proposal; do not edit after Phase 4 begins):
- H1. Components will correspond to recognizable error categories (determiner, verb-form, homophone confusion) and lexical patterns (vagueness, collocational infelicity).
- H2. Morphosyntactic components will load disproportionately on low-proficiency essays. Lexical components will be more evenly distributed.
- H3. Component profiles will predict proficiency beyond mean surprisal (ΔR² > .05).

---

## 3. Pipeline overview

```
            ┌───────────────────────── config.py (frozen, hashed) ─────────────────────────┐
ELLIPSE/TOEFL DocBins ──► S0 split ──► S1 token surprisal ──► S2 threshold + token selection
                                        (reuse Study 1 tokens.parquet,       │
                                         verified)                          ▼
                                                                 S3 delta extraction (GPU)
                                                                            │
                                         S4 NMF k-sweep + stability ◄───────┘
                                                    │
                          S5 automated interpretation (LLM, cached) ──► S6 human validation
                                                    │
                     S7 essay profiles ──► S8 proficiency models + baselines (held-out)
                                                    │
                              S9 TOEFL 11 transform + L1 analysis ──► S10 reports (Quarto)
```

Every stage is a numbered script in `src/3-surprisal-decomposition/` that writes to
`data/study3/<run_id>/<stage>/` and drops a `manifest.json` (§10.2). Reusable
logic belongs in the shared library (`src/features/delta.py`, `src/features/nmf_tools.py`)
so that it can be imported and unit-tested. The chapter directory only holds
entry-point scripts and reports, following the repo convention.

Proposed layout:

```
src/features/delta.py              # e_pred / e_obs / delta math, embedding-space resolution, pos/neg split
src/features/nmf_tools.py          # fit, k-sweep, component matching (Hungarian on cosine), transform
src/3-surprisal-decomposition/
  config.py                        # frozen dataclass StudyConfig; one instance per named run
  0_split.py                       # dev/test essay split (ELLIPSE)
  1_token_surprisal.py             # verify/produce per-token surprisal (wraps Study 1 runner)
  2_select_tokens.py               # threshold + high-surprisal token table
  3_extract_deltas.py              # GPU; sharded, resumable, idempotent
  4_fit_nmf.py                     # k-sweep, stability, final fit, transform of all rows
  5_interpret.py                   # explain + detect; LLM calls through a content-addressed cache
  6_human_validation.py            # export blinded rating sheet / import ratings
  7_profiles.py                    # essay-level profiles (+ baseline feature sets)
  8_baselines.py                   # n-gram, unigram, POS-confusion, PCA, lemma-centered features
  9_toefl.py                       # transform TOEFL deltas with dev H; L1 SMDs
  run_all.sh                       # whole DAG; each stage skips itself if its manifest validates
  decomposition.qmd                # RQ1 report
  proficiency.qmd                  # RQ2–RQ4 report
  toefl-l1.qmd                     # RQ5 report
  sensitivity/                     # §8 runs, one subdir each
  pilot/                           # untouched archive
tests/study3/                      # pytest; see §10.4
```

---

## 4. Stage specifications

### S0 — Essay split (reproducibility-critical; do first)

The threshold uses the proficiency labels (top tertile), and k selection and
interpretation use the data. If both are done on the same essays used to estimate
ΔR², the estimate is optimistically biased. Therefore:

- **ELLIPSE dev / test = 70 / 30**, stratified by `prompt × Overall-score bin`, seed
  fixed in config. Write `split.csv` (`text_id, split`) and commit it to the repo
  (it is small and is the anchor for everything downstream).
- **Dev** is used for the threshold, NMF fitting, k selection, interpretation,
  human validation, and the hyperparameters for models fit by nested CV.
- **Test** is touched once, in S8, to produce the confirmatory ΔR² and RQ2
  estimates. Gate this in code: S8 refuses to read test rows unless
  `--confirmatory` is passed *and* `DECISIONS.md` contains a frozen-analysis
  entry with the config hash.
- Deltas are still extracted for test essays (S3) and are projected onto the
  dev-fitted H in S4 (`transform`), never refit.
- TOEFL 11 is entirely out-of-sample: it is never used to fit, select, or generate
  interpretations. The only use in S5 is the optional label-transfer check.

> **Resolved 2026-10-17:** the 70/30 hold-out is the design. If Scott asks for
> it, a later robustness run can use repeated grouped CV over the full corpus
> (threshold + NMF refit inside each fold, with interpretation held fixed from
> the primary run). That would be logged as a deviation, not a replacement.
> Keep the stages parameterized by split file, so the CV run needs no code
> changes.

### S1 — Token-level surprisal

- Source: `data/predictability/{corpus}/olmo2-1b/tokens.parquet` from
  `run_surprisal.py --models olmo2-1b --windows 8 --dump-tokens`. **Check on
  the server whether it exists.** If it does not, generate it with that exact
  command. Do not write a new surprisal path.
- Verify that the `manifest.json` revision/commit matches what Study 1 reported,
  and copy the manifest hash into the S1 manifest.
- Units: Study 1 stores **bits** (`surprisal_bits`). The pilot thresholded in
  **nats** (`mean_loss`). All Study 3 thresholds and reports use **bits**.
- If the optional ModernBERT sensitivity check (§8.4) is run: the README records that the masked path *averages*
  subword surprisals, and that the fix (score first subword only) is
  "identified, not yet applied". Apply that fix **before** generating ModernBERT
  tokens for Study 3, so that the threshold and the delta use the same
  first-subword quantity as the proposal requires. Record the code version.

### S2 — Threshold and token selection

- Threshold τ = 90th percentile of token surprisal among **dev** essays in the top
  Overall tertile (tertile cut-points computed on dev). Store τ, the tertile
  cut-points, and the n of reference tokens in the manifest.
- Selected tokens = all ELLIPSE tokens (dev + test) and all TOEFL tokens with
  surprisal ≥ τ. TOEFL primary uses the **ELLIPSE τ**: same model, same scale,
  and the same definition of "high surprisal". Sensitivity: a TOEFL-native τ
  computed from TOEFL `high` essays (§8).
- Exclusions, decided now: tokens with no score (the first token of an essay
  under generative scoring), whitespace/newline tokens, and spaCy `SPACE`.
  **Punctuation is kept** (the Conventions ratings are part of the RQ4 story).
  Log counts per exclusion reason.
- Output `tokens_selected.parquet`: `row_id` (stable int), `corpus, text_id,
  split, spacy_idx, token_text, lemma, pos, tag, dep, head_pos, surprisal_bits,
  entropy_bits, num_subwords, left_ctx, right_ctx` (±10 spaCy tokens), plus
  labels (Overall, six analytic scores, prompt for ELLIPSE; L1, level, prompt for
  TOEFL). `row_id` is the join key for every later stage. **Never align arrays
  by implicit row order.**
- Expected scale: the pilot gave ~56 deltas per essay at a laxer, nats-based
  threshold. Estimate the counts from tokens.parquet before S3 and record them.
  At 2048-d float32, 400k deltas ≈ 3.3 GB (6.6 GB after the pos/neg split).
  Plan for memmapped `.npy` shards.

### S3 — Delta extraction (GPU)

Implementation rules:

1. **Reuse `features.predictability.Predictor`** for tokenization alignment and
   windowing. Add an opt-in hook that returns the full next-token / masked
   distribution for requested `spacy_idx`s instead of the loss alone.
   `TokenPredictability` already has unused `hidden_state_pred/obs` fields. Do
   not fork the windowing logic, which was a source of divergence in the pilot
   (`run_pilot.py` reimplemented masked windowing).
2. Per selected token: take the distribution at the first subword. p̃ = top-k
   (k = 50) probabilities renormalized. e_pred = p̃ · E[top_k]. e_obs =
   E[first-subword id]. δ = e_pred − e_obs. E is W_U (`output`, primary) or
   W_E (`input`). For tied models, assert `data_ptr` equality as in the pilot.
3. **Self-check in the same pass:** recompute surprisal from that distribution
   and assert |Δ| < 1e-3 bits against S1 for every token. If it fails, the stage
   fails. This is the single most important guard that "the delta was computed
   at the same prediction that was thresholded".
4. Precision: forward pass in the Study 1 dtype (bf16), with softmax, top-k and
   the embedding arithmetic in **fp32**. Record whether the self-check tolerance
   held. If bf16 batching makes logits depend on batch composition, run with
   batch-invariant settings (fixed batch size, deterministic algorithms enabled:
   `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`)
   and record them.
5. Also store, per row: top-3 predicted tokens and probabilities (for KWIC and the
   LLM prompts), top-1 predicted token's POS (needed for the POS-confusion
   baseline; obtain it by re-tagging the sentence with the top-1 substituted, or
   approximate from a lexicon. **Decide and document**), ‖δ‖, entropy, and
   p(obs).
6. Sharded, resumable, idempotent (`.done` sentinels as in `reference_corpus/3_parse.py`):
   `deltas/shard_XXXX.npy` (float32) + `deltas/index.parquet` (`row_id → shard, offset`).
7. Manifest: model hf_id + resolved commit, `tie_word_embeddings`, embedding space,
   k, dtype, window, torch/transformers/CUDA versions, GPU name, determinism
   flags, self-check max error, rows written, and sha256 of every shard.

### S4 — NMF: k-sweep, stability, final fit

- Input: dev rows only. X = [max(δ,0) ‖ max(−δ,0)] (2d columns). **Pre-specify
  scaling:** none (raw), as in the proposal. Sensitivity: row-L2-normalized δ (so
  that large OOV deltas do not dominate, cf. pilot cluster 55).
- sklearn `NMF(init="nndsvda", solver="cd", beta_loss="frobenius", tol=1e-4,
  max_iter=1000, random_state=SEED)`. Log `n_iter_` and fail if it hits
  max_iter. Pin BLAS threads (`OMP_NUM_THREADS`, `threadpoolctl`) and record them.
  NNDSVDa is deterministic, but threaded BLAS is not bit-exact, so the
  reproducibility claim is "identical to 1e-5 relative Frobenius". Test that.
- **k grid:** {10, 20, 30, 40, 50, 60, 80, 100}. For each k record: relative
  reconstruction error; sparsity of W; **split-half stability** (fit on two
  random essay-level halves of dev, match components by Hungarian assignment on
  cosine of H rows, report median matched cosine; repeat over 5 seeds); and,
  from S5, the mean detection score.
- **k selection rule (frozen):** take the smallest k such that (i) median
  split-half matched cosine ≥ .80, and (ii) mean detection score is within 1 SE of
  the maximum over the grid. Report the full curve. Detection scoring across the
  whole grid is expensive. To keep cost bounded, score a random 15 components per
  k for the sweep, then all components at the chosen k.
- Final fit at the chosen k on all dev rows. `transform` test + TOEFL rows with H
  held fixed. Save `W_{dev,test,toefl}.npy` and `H.npy` (all float32), plus
  `components.parquet` (per component: top tokens, POS/dep distributions,
  proficiency tertile mix, mean surprisal, top-lemma concentration = share of
  top-200 loadings from the single most common lemma). The last measure is the
  **lexical-identity diagnostic** that the pilot says to expect.

### S5 — Automated interpretation (explain → detect)

Follows the proposal (Bills et al., 2023; Paulo et al., 2024), with these
reproducibility additions:

- **Explain:** for each component, 35 dev tokens sampled across loading quantiles
  of the top-loading set (seeded, stratified by quantile), shown as KWIC + top-3
  predictions + POS + dep. Output: a JSON `{label, description, linguistic_level:
  orthographic|morphosyntactic|lexical|phraseological|discourse|other}`. The
  `linguistic_level` field feeds H2 directly and avoids a later recoding step.
- **Detect:** 20 dev tokens **disjoint from the explain set**: 10 high-loading
  and 10 negatives. Two negative types, both scored:
  - *random negatives* (loading < component median). This is the proposal's
    score and the **primary** one.
  - *lexeme-matched negatives* (same observed lemma, low loading). This is a
    harder test, added because the pilot showed that lexical identity dominates.
    A component that passes random negatives but fails lexeme-matched ones is a
    lexeme feature, not an error type, which feeds the taxonomy.
  Balanced accuracy per type. Shuffle item order with a seed. Flag components
  with BA < .70 (the proposal left the threshold as "e.g."; **freeze .70 here**).
- **Interpreter model:** set in config (`interpreter_model`, an exact snapshot ID,
  never an alias), along with temperature and max tokens. Before writing the
  client, the implementing agent should load the `claude-api` skill for current
  model IDs and parameters. Run a **second interpreter** (an open-weight
  instruct model pinned by HF revision, run locally) on the detection step only,
  and report agreement. This addresses the proposal's limitation that the
  interpreting LLM shares the categories of the LLM being interpreted.
- **Cache:** every request is keyed by sha256(model ‖ params ‖ prompt). Each
  request/response pair is stored as JSONL in `data/study3/<run>/llm_cache/` and
  **archived with the release**. Re-running S5 with the cache present makes zero
  network calls, and `--offline` must enforce that. Prompts live as versioned
  template files (`prompts/explain_v1.txt`, `detect_v1.txt`). The template hash
  goes in the manifest.
- **Data governance (resolved 2026-10-17):** TOEFL 11 excerpts may be sent to
  the interpreter API (confirmed by Langdon). Explanations are still generated on
  ELLIPSE dev only, because H is fitted there. TOEFL tokens may be used for an
  optional **label-transfer check**: detection scoring of each ELLIPSE-derived
  description on TOEFL tokens. That check shows whether the labels hold in the
  out-of-sample corpus. Tag every cache entry with its `corpus`. Sending text to
  the API is allowed, but redistributing it is a separate question, so TOEFL
  entries stay out of the public release (§10.5).

### S6 — Human validation

- Sample ~10 components stratified by detection-score tertile (seeded). For each,
  10 activating contexts (disjoint from the explain set) are rated 1–4 against the
  generated description, per Paulo et al. (2024), giving ≈100 judgments.
- **Blinding:** the export sheet hides the detection score, the component index
  and the proficiency labels, and randomizes order. The key file is written
  separately.
- Ratings CSV committed to the repo (`human_ratings_v1.csv`) with rater ID and
  date. Report Spearman(mean human rating, detection BA) across the sampled
  components.
- Deferred (Langdon decides later; does not block any phase): a second rater on
  a 3-component subset for inter-rater agreement. Build the export so that it
  can produce a second blinded sheet from the same sample without re-sampling.
- **Final taxonomy:** each component → error pattern / lexeme-POS feature /
  artifact, using the decision table below. Write it down before seeing RQ2/RQ3
  results:
  - BA_random < .70 → artifact (unless human mean ≥ 3, in which case flag for review)
  - BA_random ≥ .70 and BA_lexeme < .60 → lexeme/POS feature
  - BA_random ≥ .70 and BA_lexeme ≥ .60 → error pattern (or a lexical pattern
    according to `linguistic_level`)

### S7 — Essay-level profiles

- Per essay: mean W row over its high-surprisal tokens (the proposal's
  definition). Also store `n_high` and `rate_high = n_high / n_tokens_scored`.
  Essays with zero high-surprisal tokens get a zero profile plus an indicator.
  Report how many there are.
- The proposal's "mean loading" discards how many errors there were. Pre-specify
  a second profile, **per-token rate** (Σ loadings / n_tokens_scored), as a
  sensitivity check (§8). Primary = proposal definition.

### S8 — Proficiency analyses (RQ2–RQ4)

- **Prompt handling:** to match Study 2, all models include prompt fixed effects
  (or residualize outcomes and predictors on prompt within the training data
  only). Essay length (n_tokens_scored) is included in **every** model, baseline
  included, because the profile and rate_high both covary with length.
- **RQ2:** per component, Spearman(profile, Overall) after prompt adjustment, on
  test, with BH-FDR q < .05. H2 test: compare |ρ| between morphosyntactic and
  lexical components (from the S5/S6 taxonomy) using a permutation test over
  component labels (10,000 permutations, seeded). Tertile distributions are
  descriptive only.
- **RQ3:** elastic net (glmnet or sklearn; pick one and pin it). α and λ are
  tuned by 10-fold CV **within dev**, grouped by prompt. Fit on dev, evaluate
  once on test. ΔR² = R²_test(baseline + profile) − R²_test(baseline). The CI
  comes from 2,000 essay-level bootstrap resamples of the test set (fixed seed),
  holding the fitted models fixed. H3 is supported if the lower CI bound > 0 and
  the point estimate > .05. Report both separately.
- **RQ4 (exploratory):** the same profile → each of the six analytic scores, with
  per-component standardized coefficients. Key comparison inherited from Ch. 4:
  do orthographic/morphosyntactic components align with Grammar/Conventions, and
  lexical/phraseological components with Vocabulary/Phraseology?

### S9 — TOEFL 11 (RQ5)

- Deltas use the S2 τ (ELLIPSE). Project with the dev-fitted H (transform only).
- **Transfer check:** relative reconstruction error on TOEFL vs. ELLIPSE test. Also
  fit a separate TOEFL NMF at the same k and report the matched-cosine
  distribution against ELLIPSE components (does the component space replicate?).
- **L1 analysis:** per component, per L1, within each proficiency level: Cohen's
  d (focal L1 vs. all others), pooled within-stratum SD, focal-size-weighted
  average across strata. This is the Study 1 fairness procedure, so **reuse its R
  code** from `pareto-analysis.qmd`. Report a component × L1 heat map. Flag
  |d| ≥ .20 as the descriptive threshold, and freeze that number now.
- Validity replication: Spearman(profile, level) per component, and the ΔR²
  analogue with an ordinal outcome (ordinal elastic net, or a proportional-odds
  model on the S8-selected components). Pre-specify one.

---

## 5. Baseline ladder (Andy's "does it earn its complexity")

All baselines are evaluated in S8 with the same split, the same CV, the same
controls (prompt + length), and the same bootstrap. Report one table:
test R² and ΔR² vs. B0 with CIs.

| ID | Feature set | Question it answers |
|---|---|---|
| B0 | mean surprisal (+ prompt, length) | Proposal baseline |
| B1 | B0 + surprisal variance + rate_high | Do distributional summaries already capture it? (Defense notes) |
| B2 | B0 + **unigram bag of observed tokens at high-surprisal positions** (TF-IDF, top-V lemmas fixed on dev) | Is the profile just "which words were surprising"? (the pilot's lexical-identity confound) |
| B3 | B0 + **observed-POS × predicted-POS confusion counts** at high-surprisal positions | The pilot's suggestion 4: does a symbolic error typology do as well with no embeddings? |
| B4 | B0 + **Dolma 3-gram / bigram features**: essay-level mean n-gram surprisal from `data/dolma/frequency_tables/corpus_{a,b}/3grams.parquet`, plus the proportion of unattested trigrams (add-k smoothing; k fixed now) | Raw n-gram model (Andy's explicit request) |
| B5 | B0 + Study 2 feature battery (15 features from `ellipse_metrics.csv`) | Does the profile add to established indices? |
| B6 | B0 + **PCA** profile of deltas at the same k | Does NMF beat a dense linear decomposition? (proposal's interpretability claim) |
| B7 | B0 + profile from **shuffled deltas** (rows permuted across essays, within POS) | Null: is the gain from structure or from dimensionality? |
| **M** | B0 + NMF profile | Primary |
| M+ | B5 + NMF profile | Incremental over everything |

---

## 6. Phasing (with parallel workstreams for sub-agents)

| Phase | Work | Can start | Gate to exit |
|---|---|---|---|
| **P0** | Log the settled decisions (§1 model, S0 hold-out split, S5 TOEFL API use); write `config.py` and `DECISIONS.md`; lock environment (§10.1) | now | Decisions logged and config hash recorded |
| **P1** | S0 split; S1 verify/produce tokens.parquet; S2 thresholds and counts | after P0 | Counts reported; τ in manifest |
| **P2** | S3 extraction (ELLIPSE, then TOEFL) | after P1 | Self-check passes on 100% of rows |
| **P3** | S4 k-sweep and final fit | **dev on pilot deltas now**; real run after P2 | k chosen by the frozen rule |
| **P4** | S5 interpretation and S6 human validation | **dev on pilot deltas now** | Taxonomy frozen and `DECISIONS.md` entry "analysis frozen" with config hash |
| **P5** | S7, S8 confirmatory test-set run; baselines | after P4 | Reports render from a clean checkout |
| **P6** | S9 TOEFL; §8 sensitivity runs | after P4 | — |
| **P7** | Reports; Chapter 5 drafting outline (Ch. 4 style) | after P5 | — |

Suggested sub-agent split (each owns files, so they do not collide):

1. **Extraction agent**: `features/delta.py`, S1–S3, the Predictor hook and its
   tests. Runs on the GPU server.
2. **Decomposition agent**: `features/nmf_tools.py`, S4. Develop against
   `data/pilot/pilot_deltas.npy` + `pilot_metadata.parquet` (768-d ModernBERT-base
   deltas) until P2 lands. Use the same code path; only the input changes.
3. **Interpretation agent**: S5/S6, prompt templates, LLM cache, rating sheet.
   Also develops against the pilot NMF. Use the pilot's documented clusters
   (copula "is" in broken syntax, the/they confusion, were/where, negation
   placement, severe misspellings) as a **sanity fixture**. A working pipeline
   should give these components high detection scores.
4. **Modeling agent**: S7, S8, `8_baselines.py`, `proficiency.qmd`. Develop
   with synthetic profiles until P4.
5. **TOEFL agent**: S9, `toefl-l1.qmd`; reuse the Study 1 fairness R code.
6. **Writing agent** (last): Chapter 5 drafting outline in the Ch. 4 format, with
   the "approved vs. as implemented" table filled from `DECISIONS.md`.

Coordination rule: config fields, file schemas (§4) and `row_id` are the
contract. A schema change requires updating this plan and bumping the schema
version in the manifests.

---

## 7. Pilot lessons to carry forward (do not re-learn these)

- Lexical identity (e_obs) dominates delta space. Coarse solutions turned into POS
  grab-bags, and error structure appeared only at fine granularity (68 HDBSCAN
  clusters). NMF at k = 50 on the pilot exists (`pilot/nmf_cache/`). Read
  `nmf_features.py summary` output before choosing the k grid.
- The pilot's proficiency gradients were judged against base rates. Do the same
  here: report each component's tertile mix against the base rate of all
  high-surprisal tokens, not against 33/33/33.
- Severe misspellings yield large, idiosyncratic deltas. Expect an "OOV"
  component. That motivates the row-normalized sensitivity check.
- Concordance tooling (`concordance.py format_kwic_line`) can be reused for KWIC
  in prompts and reports. Move it into the library rather than importing from
  `pilot/`.

---

## 8. Pre-specified sensitivity analyses

Each runs through the same scripts with a named config override. Results go
in `sensitivity/` and an appendix table. None of them replaces the primary
analysis.

1. Lemma-centered deltas: δ − mean δ[observed lemma], with means from dev
   (pilot suggestion 2).
2. Row-L2-normalized δ before the pos/neg split.
3. Prediction-only representation: e_pred alone (pilot suggestion 3).
4. Tied masked model (ModernBERT-large/8, input = output space). Optional;
   requires the masked first-subword fix (S1). Report component matching across
   models with Hungarian on essay profiles, since the embedding spaces differ.
5. OLMo-2 1B/8 with `embedding_space: input` (W_E).
6. τ at the 85th and 95th percentiles.
7. TOEFL-native τ.
8. Per-token-rate profile (S7).
9. top-k ∈ {10, 50, 200} for e_pred.

---

## 9. Freezing and deviations

- This plan's §2, §4 decision rules (k rule, BA threshold, taxonomy table,
  |d| flag), §5 ladder and §8 list are **frozen when P0 closes**.
- Before P5, append to `DECISIONS.md`: "Analysis frozen", the config hash, the
  taxonomy hash, and the date. S8 checks for this entry (§4 S0).
- Any later change is a deviation entry with *what, why, when, and whether it
  was made after seeing test results*. Chapter 5 Method has a deviations
  paragraph generated from this file.

---

## 10. Reproducibility infrastructure

### 10.1 Environment
- Python: extend `pyproject.toml` / `uv.lock`. Add `threadpoolctl` and the
  interpreter client. Pin `scikit-learn` exactly (NMF numerics change across
  versions). R: `renv.lock` (packages: glmnet if R is used, plus those reused from
  Study 1). Quarto renders through `_quarto.yml`'s `execute-dir: project`.
- Add a `study3` optional-dependency group rather than bloating the core.
- spaCy model version is pinned and recorded (`en_core_web_lg` 3.8.0, as in ingest).
- Record the GPU and driver in the S3 manifest. Results must not depend on them
  beyond the stated tolerances.

### 10.2 Manifests
Every stage writes `manifest.json`:
```json
{
  "stage": "S3_extract_deltas",
  "run_id": "olmo2-1b_w8_out_k50_v1",
  "config_hash": "sha256…",
  "code_version": "git sha or content hash of src/",
  "inputs": {"tokens_selected.parquet": "sha256…", "S2/manifest.json": "sha256…"},
  "outputs": {"deltas/shard_0000.npy": "sha256…"},
  "params": {"...": "..."},
  "env": {"python": "...", "torch": "...", "transformers": "...", "sklearn": "...", "cuda": "..."},
  "checks": {"surprisal_selfcheck_max_abs_bits": 0.0003},
  "started": "...", "finished": "..."
}
```
A stage verifies its upstream manifests' output hashes before running, and is
skipped if its own manifest validates against the current config (idempotent, as
in Study 1).

### 10.3 Seeds
One `SEED` in config. Per-purpose seeds are derived deterministically
(`np.random.SeedSequence(SEED).spawn` with fixed purpose order: split, k-sweep
halves, explain sampling, detect sampling, human-validation sampling,
bootstrap, permutation). Never call unseeded RNGs. Grep for this in a test.

### 10.4 Tests (`tests/study3/`, pytest, CPU, < 2 min)
- Delta math on a toy embedding: renormalization; e_pred equals e_obs when p is
  one-hot on the observed token (δ = 0); pos/neg split round-trips (x = pos − neg).
- h·δ identity for the output space on a tiny random model.
- Predictor hook on GPT-2 (tied) and on a tiny untied model: the distribution's
  surprisal equals Predictor's loss for the same token.
- Split determinism, `row_id` stability across re-runs, alignment asserts.
- NMF determinism within tolerance under pinned threads.
- LLM cache: offline mode raises on a cache miss and makes no network calls.
- Every LLM cache entry carries a `corpus` tag; the release exporter drops TOEFL entries.

### 10.5 Data and release
- `data/` is not versioned. Release artifacts are: `split.csv`, `config.py`, all
  manifests, `H.npy`, `components.parquet`, `llm_cache/` (ELLIPSE entries only),
  `human_ratings_v1.csv`, and essay-level profiles for ELLIPSE. Delta shards are
  large and can be regenerated. Their hashes in the manifest certify the
  regeneration. Do not release TOEFL text or token contexts.
- `run_all.sh` regenerates everything from DocBins plus the pinned models. Write
  a README for this directory in the style of Study 1's (what each file does,
  how to run, outputs, known issues), and delete this plan when the reports
  supersede it, as Study 1 did.

### 10.6 Reporting conventions
Bits throughout. Figures are saved via Study 1's `_viz_helpers.R` save helper
(vector PDF + 400 dpi PNG). Prose uses "generative" (not "causal") language model.
Results must be quoted from rendered reports, not from notebooks.

---

## 11. Deliverables

1. `decomposition.qmd`: k-sweep curves, stability, component table (label,
   description, BA_random, BA_lexeme, human mean, taxonomy class, top-lemma
   concentration, tertile mix vs. base rate), KWIC panels for exemplar
   components, human-vs-automated calibration.
2. `proficiency.qmd`: RQ2 forest plot of prompt-adjusted ρ (FDR-marked), H2
   permutation test, RQ3 baseline-ladder table with ΔR² CIs, RQ4 component ×
   analytic-score heat map.
3. `toefl-l1.qmd`: transfer reconstruction, component replication, component × L1
   SMD heat map, ordinal validity.
4. `sensitivity/` reports and one appendix summary table.
5. `DECISIONS.md` (decisions, freeze entry, deviations).
6. Chapter 5 drafting outline (`../Dissertation/Chapter 5/chapter5-study3-outline.docx`)
   in the Ch. 4 outline format, including an "approved outline vs. as implemented"
   table. Expected rows: tied-embedding requirement withdrawn (OLMo-2 1B, output space); TOEFL extension;
   baseline ladder; dev/test split; RQ4/RQ5 [NEW]; lexeme-matched detection score.
7. A note to fix the stale Study 3 description in Chapter 3 (~line 831).

---

## 12. Known risks

| Risk | Mitigation |
|---|---|
| Components are mostly lexeme features (pilot) | Lexeme-matched BA makes this measurable and reportable rather than a failure. Lemma-centered sensitivity check. B2 baseline quantifies it. A null on "error types" is still a Chapter 5 result. |
| OLMo output-space deltas behave differently from the pilot's ModernBERT deltas | Input-space and ModernBERT sensitivity checks (§8.4–5); validate the pipeline on the pilot fixture before attributing differences to the space |
| ΔR² gain is explained by length or count of errors | Length and rate_high are in B1 and in every model |
| Interpreter non-determinism or model deprecation | Snapshot IDs, committed cache, offline replay, second open-weight interpreter |
| TOEFL redistribution | API use cleared; TOEFL text, contexts and cache entries are kept out of the public release |
| Memory at full scale | Memmapped shards; if NMF on all dev rows is infeasible, fit on a seeded essay-level subsample and transform the rest. That choice must be made and logged in P3 *before* fitting, not after |
