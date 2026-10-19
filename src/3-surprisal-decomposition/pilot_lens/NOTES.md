# Pilot: layer-wise surprisal (tuned lens) and contextuality on W&I high tokens

Status: exploration, not committed to the main pipeline. Code in this folder,
caches in `data/study3/pilot_lens/` (not versioned). GPU 1 only.

## 0. Plan, written before any gold contrast was computed

### Sample

W&I+LOCNESS tokens above τ = 11.28 bits (bench `wi`, 88,761 tokens, gold from
`data/study3/wi/tokens_gold.parquet`). Stratified by level, seed 42: 3,000
tokens each from A, B and C, and all 3,015 LOCNESS (N) tokens. Tokens whose
word does not start on a subword boundary (`base_meta_wi.has_target == False`)
are dropped. Gold: err = ERR_TOK or ERR_INS, OK, POST (kept out of both
contrasts, as in `5_gold.py`).

### Measures (all for the observed first subword, bits)

**A. Layer-wise surprisal.** OLMo-2 1B, the scoring window (≤ 8 subwords ending
at the target), hidden state at the position that predicts the target, after
each block ℓ = 0 (embeddings) … 16.

- Logit lens: `s_LL[ℓ] = −log2 softmax(W_U · norm(h_ℓ))[obs]`.
- Tuned lens (Belrose et al. 2023): one affine translator per layer,
  `h_ℓ + A_ℓ h_ℓ + b_ℓ` (zero init), then the model's final norm and W_U,
  trained to minimise KL(p_final ‖ p_ℓ). Training text: ELLIPSE **dev**
  essays (no W&I tokens), random 8-subword windows, loss at every position.
- Focal summaries, fixed now:
  - `TL_late = s_TL[8] − s_TL[16]` (how much the second half of the network
    lowers surprisal; > 0 = "rescued late", < 0 = "pushed down late").
  - `TL_early = s_TL[1] − s_TL[8]`.
  - Same two for the logit lens (`LL_late`, `LL_early`), reported but
    secondary, because the raw logit lens is known to be poorly calibrated in
    early layers.
- Sanity checks: `s_TL[16] = s_LL[16]` reproduces the stored surprisal
  (target: |diff| < 0.05 bits median, bf16); tuned-lens held-out KL per layer
  below logit-lens KL; correlation of each `s_ℓ` with −log unigram frequency
  (edited Dolma, `a_log_uni`) and with `s_16`. If early layers are nothing but
  a frequency prior, the A-measures must add nothing over frequency + `s_16`
  in the tests below, and that is the reading.

**B. Contextuality (Ondov et al. 2024; adapted in Holmes et al. 2026,
AIMECon).** Ondov, Attal & Demner-Fushman (NAACL 2024) score a *blank* by the
plausibility of candidate fillers under sentence context minus their
probability under the whole passage (RoBERTa-large), summed over a candidate
pool. Holmes et al. (2026) adapt it to score the *answer word*:
`score = ζ log2(ζ/ψ)`, ζ = p(w | whole text, w masked), ψ = p(w | its sentence),
ModernBERT-large, all subwords masked, first-subword probability.

Implemented here:

- `MLM_gain = s_MLM_sent − s_MLM_full` (log2 ψ/ζ inverted: bits saved by the
  whole essay over the sentence), ModernBERT-large, as in Holmes et al.
- `MLM_ctx = ζ log2(ζ/ψ)` exactly as in Holmes et al. (secondary: for tokens
  above τ, ζ is tiny under the causal model, so this score is mostly ζ).
- Causal OLMo-2 1B views of the same token: `s_w8` (stored), `s_sent` (left
  context from the start of the spaCy sentence), `s_full` (all left context
  in the essay). `C_gain = s_w8 − s_full`, `C_sent_gain = s_sent − s_full`.
- Ondov's original slot-level quantity, causal: `pool_ctx` = mean over the
  top-32 candidates v under the 8-token window of
  `log2 p_w8(v) − log2 p_full(v)` (how strongly the essay rules out what was
  locally plausible; k = 32 as in Ondov et al.). Also KL(p_w8 ‖ p_full).

Focal B-measures: `C_gain`, `pool_ctx`, `MLM_gain`.

### Baselines

`s_16` (= stored surprisal), entropy (stored, 8-token window), log unigram
frequency in edited Dolma (`a_log_uni`), and for B, entropy under full context.

### Q1. Controlled on final surprisal, does the measure separate errors from OK?

- Tokens: err vs OK, POST excluded, all levels pooled.
- Base model B1: logistic regression on natural splines (5 knots) of `s_16`
  and entropy, plus `a_log_uni`. Measure model: B1 + the focal measure
  (linear) or the full trajectory (for A: all 17 `s_TL[ℓ]`).
- Out-of-fold predictions from 5-fold CV grouped by essay; ΔAUC with a 95%
  essay-cluster bootstrap CI (1,000 resamples of essays over the OOF
  predictions).
- Also: AUC of the raw measure within `s_16` quintiles (mean over bins).
- **Reading rule:** the measure "adds over surprisal" if ΔAUC over B1 ≥ .01
  and the CI excludes 0. For context, the old context-view classifier (D15,
  `ctx_excess_s4`) reached .757 vs surprisal .698 in GBM; a ΔAUC below .01 is
  called a null here even if it is "significant".

### Q2. Among acceptable (OK) tokens, does the profile shift with A → B → C → N?

- Tokens: gold OK only.
- Raw mean of each focal measure by level, with 95% CIs from 2,000 essay
  resamples within level.
- Residualised mean: the measure minus its OLS prediction from splines of
  `s_16`, entropy and `a_log_uni`, fit on all OK tokens pooled; by level,
  same CIs. This asks whether level shifts the measure beyond what the
  shifts in final surprisal, entropy and word frequency already imply.
- Baselines get the same table (raw by level), so the measure's shift can be
  compared with that of `s_16`, entropy and frequency themselves.
- Contrasts: C − A and N − C with essay-bootstrap CIs.
- **Reading rule:** the measure "says something about non-error surprise
  beyond the baselines" if its *residualised* C − A or N − C difference has a
  CI excluding 0 and is at least 0.1 SD of the measure (among OK tokens). A
  raw shift that vanishes after residualising is reported as "reproduces the
  baselines".

### Qualitative pass

KWIC (≤ 100 words, local only) for OK tokens in the extremes: rescued late vs
never supported (A), high vs low `C_gain` / `MLM_gain` (B). Labels by Claude
(Opus 5.5) from the excerpts, before looking at level composition.

---

## 1. What was run (after the plan above)

| step | file | output (`data/study3/pilot_lens/`) | cost |
|---|---|---|---|
| sample | `sample.py` | `sample.parquet`: 12,015 tokens, 2,844 essays (A 3,000, B 3,000, C 3,000, N 3,015) | CPU, 1 min |
| tuned lens | `lens_train.py`, `lens_common.py` | `tuned_lens.pt`, `lens_train.json` | GPU 1, 24 min (1,500 steps × 256 windows × 8 positions ≈ 3.1M positions, 4,311 ELLIPSE dev essays) |
| lens scores | `lens_score.py` | `lens_scores.npz` | GPU, 10 s |
| context views | `ctx_score.py causal mlm` | `ctx_causal.parquet`, `ctx_mlm.parquet` | GPU, 6 + 6 min |
| analysis | `analyze.py` | `features.parquet`, `results.json`, `kwic.md` | CPU, 3.5 min |
| post hoc checks | `sensitivity.py` | `sensitivity.csv` | CPU, 1 min |

Total GPU ≈ 37 min on one A6000. Nothing outside this folder and the data folder was touched.

**No cached context views existed.** `data/study3/lab/` was deleted with the
exploration code (b6a6ff5), so the views were recomputed for this sample.
They are cheap: the causal full-essay view needs one forward pass per essay.

**Ondov et al.** The paper is Ondov, Attal & Demner-Fushman (2024),
"Pedagogically aligned objectives create reliable automatic cloze tests"
(NAACL, pp. 3961–3972). Its "contextuality" (§3.3) scores a *blank* by the
first two terms of its energy function: the plausibility of the candidate
fillers under sentence-only context (RoBERTa-large), and their improbability
under the whole passage. It is a property of the slot and its alternatives,
not of the answer word. The pointwise answer-word version
`ζ log2(ζ/ψ)` (ModernBERT-large) is Holmes, Morris, Crossley & Min (2026,
AIMECon), "Source-text-conditioned cloze generation". Both are implemented
here: `pool_ctx` is the slot version (causal, top-32 pool), and
`MLM_gain`/`MLM_ctx` are the answer-word version.

**Overlap with the earlier context views (D15).** The old `ctx_*`
features scored the same token under 1B/7B × 8/64-token causal windows and a
±32-subword ModernBERT-large window. They were used only as classifier inputs
for error detection. New here: (a) full-essay left context instead of 64
subwords, and a sentence-only causal view; (b) the sentence-vs-essay contrast
(the cloze construct) rather than window size; (c) the slot-level pool
quantity, which scores the *alternatives* rather than the observed word;
(d) the target is the acceptable tokens' profile by level, not detection.

## 2. Sanity checks

- Final-layer lens = stored surprisal: |s_16 − stored| median 0.028 bits,
  p99 0.26, max 0.80. This is bf16 batch-composition noise; the stored
  scores came from Study 1's batched `Predictor`. Logit lens and tuned lens
  are identical at layer 16 (max diff 0). The causal `s_w8` reproduces the
  stored value to the same tolerance (median 0.029).
- Held-out KL to the final distribution (nats, 226 held-out ELLIPSE dev
  essays):

  | layer | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
  |---|---|---|---|---|---|---|---|---|---|
  | tuned | 1.10 | 0.64 | 0.60 | 0.55 | **0.78** | 0.65 | 0.34 | 0.18 | 0.08 |
  | logit | 11.2 | 9.5 | 11.2 | 11.0 | 10.9 | 10.0 | 8.8 | 7.2 | 4.2 |

  The raw logit lens is unusable for OLMo-2 1B before the last block (KL ≈
  10 nats; mean surprisal of the observed token ≈ 23 bits at every layer).
  The `LL_*` results below are therefore noise and are not interpreted.
  The tuned lens works, but it is flat from layer 2 to 10 and non-monotone,
  with bumps at layers 8 and 10. That suggests it is under-trained (one
  learning rate and 1,500 steps, no tuning). Layer 8 is the focal layer
  for `TL_late`, so the post hoc checks swap it for layers 6, 7, 9, 10 and 12.
  The tuned lens at layer 0 sees only the previous subword's embedding, so it
  is effectively a learned bigram model, not a unigram prior.
- **Frequency check.** Spearman ρ of s_TL[ℓ] with −log unigram frequency
  (edited Dolma) falls from .45 (layer 0) to .24 (layer 16). ρ with final
  surprisal rises from .49 to .87 (layer 15). So the early layers lean
  towards a frequency/bigram prior without being one (ρ .45). Splines of
  s_16, entropy and log frequency explain R² = .11 of `TL_late` and .006 of
  `TL_early`. The trajectory measures are therefore not a simple re-expression
  of frequency × final surprisal.
- Mean tuned-lens trajectories (bits, layers 0 → 8 → 13 → 16): errors
  15.2 → 15.2 → 15.5 → 15.1. OK tokens at A 14.2 → 14.3 → 14.4 → 14.0, and
  at N 15.3 → 15.0 → 14.7 → 14.1. **Errors do not "look fine early"**: they
  are as improbable at the bigram layer as at the output, and they rise a
  little in layers 11–14. Acceptable tokens fall in layers 13–16, and they
  fall more at C and N than at A.

## 3. Q1: errors vs OK, controlled on final surprisal

n = 11,100 (3,298 errors). OOF AUC from 5-fold essay-grouped CV: surprisal
splines alone .614; **B1** (splines of s_16 and entropy, plus log frequency)
**.664**. ΔAUC over B1, 95% essay-bootstrap CI. "Raw AUC in bins" is the
measure's own AUC within s_16 quintiles, where < .5 means higher values go
with OK.

| measure | AUC | ΔAUC [CI] | raw AUC in s_16 bins | rule (≥ .01, CI > 0) |
|---|---|---|---|---|
| TL_late | .668 | +.004 [.001, .007] | .452 | null |
| TL_early | .664 | +.001 [−.001, .002] | .519 | null |
| TL full trajectory (16 layers) | .671 | +.008 [.002, .012] | — | null |
| TL trajectory + layer entropies | .675 | +.011 [.005, .017] | — | borderline pass |
| C_gain (s_w8 − s_full) | .711 | **+.047 [.037, .056]** | .369 | pass |
| s_full alone | .711 | +.047 [.038, .056] | .637 | pass (same thing) |
| C_sent_gain | .691 | +.028 [.020, .035] | .426 | pass |
| pool_ctx (Ondov slot) | .671 | +.008 [.003, .012] | .432 | null |
| KL(p_w8 ‖ p_full) | .676 | +.012 [.006, .017] | .408 | pass, small |
| MLM_gain (Holmes et al.) | .666 | +.002 [−.000, .005] | .495 | null |
| MLM_ctx = ζ log2(ζ/ψ) | .692 | +.028 [.021, .035] | .443 | pass |
| s_mlm_full alone | .742 | **+.078 [.068, .088]** | .665 | pass |
| all context views | .761 | +.097 [.086, .109] | — | pass |
| all context + TL trajectory | .764 | +.100 [.088, .112] | — | the lens adds .003 |

Reading:
- **Layer-wise surprisal adds nothing over final surprisal for errors.** The
  raw means differ (TL_late is +0.76 bits for OK and +0.02 to +0.12 for
  errors), but splines of s_16 and entropy absorb almost all of that.
- **Contextuality "works" for errors only because it contains a better
  surprisal.** With s_w8 ≈ s_16 held fixed, C_gain is −s_full: the ΔAUC is
  identical. The best single view is the bidirectional full-essay MLM
  surprisal (+.078). The sentence → essay *contrast*, which is the cloze
  construct, adds nothing (MLM_gain +.002). Its ζ-weighted form works only
  through ζ. This is the D15 context-view result again (more context and
  right context expose errors), not something new.

## 4. Q2: acceptable (OK) tokens by level

Means (bits) with essay-bootstrap 95% CIs. `_res` = residual on splines of
s_16 and entropy, plus log frequency. SD is among OK tokens.

| measure | A | B | C | N | C − A [CI] (SD) | N − C [CI] (SD) |
|---|---|---|---|---|---|---|
| s_16 | 13.98 | 13.86 | 13.89 | 14.11 | −.09 [−.25, .06] | +.22 [.05, .41] (.10) |
| entropy | 7.36 | 7.36 | 7.40 | 7.51 | +.04 | +.11 |
| log freq | 10.37 | 10.14 | 9.91 | 10.07 | **−.46 [−.66, −.25] (−.17)** | +.16 |
| TL_late | 0.30 | 0.67 | 0.88 | 0.93 | +.57 [.36, .79] (.19) | +.05 |
| **TL_late_res** | −.35 | −.10 | .04 | .20 | **+.39 [.19, .60] (.14)** pass | +.16 [−.01, .33] (.06) |
| C_gain_res | −.38 | .00 | −.08 | .26 | +.31 [−.04, .67] (.07) | +.33 [−.04, .69] (.07) |
| pool_ctx_res | −.16 | −.11 | .05 | .10 | +.21 [.02, .39] (.097) just under | +.05 |
| MLM_gain_res | −.04 | .29 | −.11 | −.08 | −.06 | +.02 |
| MLM_ctx_res (secondary) | −.32 | −.02 | −.04 | .20 | +.28 [.07, .49] (.097) | +.23 [−.07, .55] |
| s_mlm_full_res (secondary) | .80 | .07 | −.14 | −.30 | **−.94 [−1.23, −.63] (−.24)** | −.16 [−.48, .18] |

Under the pre-registered rule only **TL_late** passes (C − A). No focal
measure separates N from C. In the secondary measures, the
bidirectional full-essay surprisal moves most: C writers' acceptable
surprises are about 0.9 bits less surprising to an MLM that sees the whole
essay than A writers' are, at matched causal 8-window surprisal.

**Post hoc checks (`sensitivity.csv`, written after seeing Q2).**

| residualised C − A | all OK | clean context (prev. 7 tokens all OK) | no PROPN/NUM |
|---|---|---|---|
| TL_late (layer 8) | +.39 [.19, .60] | +.14 [−.21, .46] | +.39 [.19, .59] |
| TL_late, layer 7 | +.39 [.19, .59] | +.31 [−.01, .62] | +.41 [.22, .61] |
| TL_late, mean of layers 6–10 | +.35 [.17, .55] | +.21 [−.10, .49] | +.36 [.18, .55] |
| C_gain | +.31 [−.04, .67] | **+.91 [.34, 1.48] (.19 SD)** | +.14 |
| pool_ctx | +.21 [.01, .39] | **+.59 [.26, .91] (.27 SD)** | +.22 |
| s_mlm_full | −.94 [−1.23, −.63] | −.71 [−1.17, −.25] | −.89 |

- The TL_late shift does not depend on layer 8 or on proper nouns. It
  roughly halves, and its CI includes 0, when the preceding window holds no
  corrected token. Clean-context shares are A .32, B .47, C .69, N .76, so
  much of "A's acceptable surprises are not rescued late" is "A's acceptable
  surprises sit next to errors". That is context quality, not a property of
  the word.
- The contextuality measures go the other way. In clean contexts, C writers'
  acceptable surprises are more essay-predictable than A writers' (+0.9
  bits, 0.19 SD), and their local alternatives are more strongly ruled out by
  the essay (pool_ctx, 0.27 SD). This is post hoc, on a level-dependent
  subset, so it is a hypothesis for a pre-registered follow-up, not a
  finding.
- **Contextuality is largely repetition.** A crude "the word occurred
  earlier in the essay" flag correlates r = .50 with C_gain among OK tokens.
  Mean C_gain is 8.4 bits for repeated words and 3.3 for first mentions,
  and the repeated share of OK tokens rises A .25, B .29, C .29, N .35. Any
  follow-up must control for previous mention (given/new) before reading
  contextuality as anything deeper.

## 5. Qualitative pass (labels by Claude, from `kwic.md`; excerpts stay local)

- **Rescued late** (TL_late top decile; 66% C/N, 31% PROPN, rare words):
  register-elevated or abstract content words that fit a semantic frame
  built over the window (*deleterious*, *dilute*, *communism*, *horrors*).
  It also includes sentence-initial topic nouns echoing the previous
  sentence (*Violence* after *violent*). Label: *frame-licensed rare words*.
- **Pushed down late** (bottom decile; higher frequency, s_16 15.3): many
  look like errors the annotator left uncorrected, or awkward learner
  phrasing (*much more advantages*, *the success of passing exams*, *feel
  convenient*). The rest are discourse-new sentence starts. Label:
  *uncorrected or marginal non-target usage*. This is the one place where
  Li et al.'s "anomaly appears late" pattern is visible, and it shows among
  the "acceptable" tokens.
- **Never supported** (s_TL ≥ 14 bits at every layer from 4 on): names,
  local institutions and domain terms (*Basisberoeps*, *municipal*,
  *rectangular*), plus a few missed errors. Label: *out-of-model lexis*.
- **High C_gain**: topic words and named entities that were already
  mentioned in the essay (*bus*, *Hong Kong*, *Candide*/*Baron*,
  *boxers*), and genre formulas (*Yours* sincerely). Label: *given
  information*.
- **Low C_gain**: one-off lexical choices that are not tied to the topic
  (*opted*, *merely*, *surplus*, *evokes*). Label: *discourse-new
  vocabulary*. This is plausibly the "lexical sophistication" kind of
  acceptable surprise.
- **High MLM_gain**: mostly sentence- or paragraph-initial words (*The*,
  *Whenever*, *Nowadays*), which the sentence-only MLM has no left context
  for. This is an artefact of the sentence/essay contrast. The rest are
  repeated names.
- **High pool_ctx**: concrete nouns in narrative and descriptive essays
  (*bus*, *drama* club, *math* homework, *slaves*). The essay rules out the
  local alternatives; the slot is topic-constrained.

## 6. Verdicts

**A. Tuned/logit lens: partially, mostly no.**
- The logit lens does not work for OLMo-2 1B before layer 16.
- The tuned lens works technically: KL .08–1.1 nats, exact at 16.
- It adds nothing to error separation beyond s_16 and entropy (ΔAUC
  ≤ .011).
- Among acceptable tokens it shows the only pre-registered level shift
  (C − A +0.39 bits, 0.14 SD): higher-level writers' acceptable surprises are
  "rescued" by late layers more. Most of that shift goes away in clean
  contexts. A cleaner, better-trained lens (more steps, a learning-rate
  sweep, more and more diverse text) would be needed before claiming it.
- The hypothesis that errors look fine early is not supported: errors are
  improbable already at the bigram layer.

**B. Contextuality: partially for errors (not new); no for the level profile
under the pre-registered rule; interesting post hoc.**
- For errors, the signal is the long- or bidirectional-context surprisal
  itself (s_mlm_full +.078 ΔAUC), which is the D15 context-view finding
  again. The faithful sentence → essay contrast (MLM_gain) is null for both
  questions and is contaminated by sentence-initial position.
- The causal and slot versions shift with level only in the post hoc
  clean-context subset (0.19–0.27 SD), and contextuality is half
  repetition.
- What it can add that error methods cannot is a given/new split of
  acceptable surprise ("topic words the essay set up" vs "discourse-new
  vocabulary"). That deserves a pre-registered test with a previous-mention
  control.

**Next, if pursued:**
1. Pre-register the clean-context × level contrast for C_gain and pool_ctx,
   with a previous-mention covariate.
2. Retrain the tuned lens properly and re-test TL_late in clean contexts.
3. Run on all 88k W&I high tokens; the GPU cost is minutes.
