# Study 3 — decisions and deviations

Entries are appended in date order. Each entry says what was decided, why, and
whether it was made before or after any outcome had been seen. `plan.md` is the
full design. This file records where the first pass departs from it.

---

## 2026-10-17 · D1. Model and embedding space (from plan §1, settled)

OLMo-2 1B base (`allenai/OLMo-2-0425-1B`, revision `a1847dff`), 8-token
window, i.e. the Study 1 selected configuration. The proposal's tied-embedding
requirement is withdrawn. Deltas are computed in the output (unembedding)
space W_U, where the prediction is made. Before any outcome.

## 2026-10-17 · D2. First pass is deliberately simple

Langdon asked for a simple first pass that spends its effort on checking the
basic premises, not on the plan's reproducibility machinery. The following
parts of `plan.md` are **deferred, not dropped**. All of them were decided
before any outcome was seen.

| Plan | First pass | Why |
|---|---|---|
| §10.2 manifest chain with input hashes per stage | One manifest for the GPU pass. Correctness is anchored by an exact reproduction check against the Study 1 table instead (per-essay mean surprisal and token count) | The reproduction check is the guarantee that matters ("the delta is computed at the prediction that was thresholded") |
| S1 tokens.parquet + S3 separate delta pass with a per-token self-check | One GPU pass stores the top-50 distribution for **every** token. The threshold and the delta arithmetic are CPU steps over that file. Surprisal is recomputed from the same logits the top-50 comes from | A single pass makes the self-check hold by construction, and threshold/top-k sensitivity checks need no further GPU time |
| S5 LLM explain→detect with cached API calls and a second open-weight interpreter | Components are labelled from **component cards** (KWIC, top observed and predicted words, POS, lexical concentration) built from dev rows only. Claude (Opus 5.5) does the labelling in this session, which stands in for the explain step. No detection scores | Detection scoring is the expensive part. The question it answers (is a component a lexeme or an error type?) is addressed more directly by the lexical-concentration diagnostic and the M_obs baseline below |
| S6 human validation | Deferred to Langdon | Needs a human |
| §8 sensitivity analyses | All deferred | First pass |
| §10.4 tests | Not written | First pass |
| B3 (POS confusion), B4 (n-gram), B7 (shuffled deltas) | Dropped from this pass | B4 needs Dolma n-gram lookups per essay. B3 needs re-tagging. B7 is superseded as a null by M_obs |

## 2026-10-17 · D3. Two premise-check representations added to the ladder

The pilot found that observed-word identity dominates delta space. In a delta
δ = e_pred − e_obs, the e_obs term is one vocabulary row, while e_pred is a
top-50 mixture that is pulled toward the centroid. On geometry alone, e_obs
should therefore carry most of δ's variance. Two comparison decompositions run
through the identical NMF code at the same k:

- **M_obs**: NMF of −e_obs (the observed word alone, no prediction).
- **M_pred**: NMF of e_pred (the prediction alone, no observed word).

Reading rule, fixed now: the claim that δ components capture *how the
prediction failed*, over and above *which word was surprising*, requires that M
beat M_obs on held-out R² (the `M - M_obs` row of the ladder, with a CI
excluding 0). If it does not, the components are reported as lexical
descriptions of the high-surprisal positions, and H1's error-category reading
is not supported.

## 2026-10-17 · D4. k selection without detection scores

Plan §4 S4 selects the smallest k with split-half stability ≥ .80 *and* a
detection score within 1 SE of the maximum. With no detection scores, the
stability rule alone would favour the coarsest grid point, which the pilot
showed collapses error patterns into POS grab-bags. Rule, fixed before the
sweep: **the largest k in {10, 20, 30, 50, 80} whose median split-half matched
cosine is ≥ .80 in both seeds** (2 seeds, essay-level halves of dev). That is
the finest decomposition that replicates. If no k qualifies, take the k with
the highest median matched cosine and report it as unstable.

## 2026-10-17 · D5. Taxonomy without detection scores

Replaces the BA-based decision table of plan §4 S6. Fixed before the cards are
read:

- top-lemma share ≥ .50 among the component's 200 highest-loading dev tokens →
  **lexeme feature** (a single word dominates).
- otherwise the component gets a class from its card: **error pattern** (a
  recognizable non-target form), **lexical/phraseological pattern** (target-like
  word choice that is merely unexpected), or **artifact** (formatting, names,
  tokenization). It also gets a `linguistic_level` of orthographic /
  morphosyntactic / lexical / phraseological / discourse / other, as in the
  plan.
- H2's permutation test compares the morphosyntactic and lexical levels. Labels
  are written to `component_labels.csv` before `6_proficiency.py
  --confirmatory` is run.

## 2026-10-17 · D6. Analysis details fixed for this pass

- Split: 70/30 dev/test, stratified by prompt × Overall tertile (tertile cuts
  from the whole corpus, used only for balancing), seed 42, committed as
  `split.csv`.
- τ: 90th percentile of non-whitespace token surprisal (bits) in dev essays
  with Overall ≥ the dev 2/3 quantile. The same τ applies to TOEFL 11.
- Prompt handling: outcomes and predictors are centred on dev prompt means,
  which are then applied to test. Every R² is within-prompt. log(n tokens
  scored) is in every model.
- Elastic net: sklearn `ElasticNetCV`, standardized inputs, l1_ratio ∈ {.1,
  .5, .9, 1}, 40 alphas, 10-fold CV grouped by prompt.
- Development estimates use 10-fold prompt-grouped CV within dev. The test set
  is read once, by `6_proficiency.py --confirmatory`, which refuses to run
  without an "Analysis frozen" entry in this file.
- H3 as in the plan: supported if the ΔR² (M − B0) point estimate is > .05
  and the bootstrap lower bound is > 0.
- RQ5 ordinal validity: the plan's "ordinal elastic net or proportional-odds
  model" is replaced by applying the ELLIPSE-dev elastic nets (B0, B1, M_obs,
  M) unchanged to TOEFL 11 and correlating their predictions with level
  (Spearman). This is out-of-sample transfer with no TOEFL fitting.
- RQ5 L1 effect sizes: Study 1's weighted within-level Cohen's d, implemented
  in Python with the same formula (`7_toefl.py::weighted_d`). Flag |d| ≥ .20.

## 2026-10-17 · D7. Are high-surprisal tokens errors? A blinded coding sample

Premise check added after stage 2, before any NMF output or proficiency
analysis had been seen. 150 dev tokens above τ (50 per Overall tertile,
shuffled, no proficiency information shown) are coded into four categories,
fixed before coding:

- **E_TOK**: the token itself is non-target: misspelling, wrong form or
  inflection, wrong word or function word.
- **E_CTX**: the token is target-like, but it is surprising because of a
  non-target form or a missing boundary/punctuation immediately before it. The
  error is in the context, not at the token.
- **OK**: a target-like token in a target-like context. The word choice is
  unexpected but acceptable (topic words, rarer words, names, a new sentence
  after a correct boundary).
- **OTHER**: placeholders (`Generic_Name`), formatting, tokenization, unclear.

The first coder is Claude (Opus 5.5), in this session (`error_coding_claude.csv`).
The sheet `coding_sample.csv` is the blinded version for a human coder, and
`coding_key.csv` holds the tertiles.

## 2026-10-17 · D8. Representation lab, and a second representation

Prompted by Langdon's request for a testing framework and more interpretable
deltas, and by the premise result that δ is close to 90% observed-word
identity. Built `lab/`, a synthetic error benchmark with lexeme-matched clean
controls, a fixed evaluation battery, and a held-out confirmation benchmark.
Success criteria were written into `lab/README.md` before any idea agent
reported. Three idea families were piloted by sub-agents (geometry, LM repair,
hidden states).

Result on the confirmation bench: the **LM repair representation** is the only
family that meets all four criteria (`rep_repair_subpos`, 29 named edit
classes: detect AUC .922, lexeme-disjoint F1 .807, NMF NMI type .419 >
word .354; δ: .678 / .613 / .318 < .467). It replaces NMF-on-δ as the
explanatory representation. δ, M_obs and M_pred stay in the ladder as
comparisons, and the δ analysis is still reported.

Decided before running repair on any real essay:

- Real-data representation: `rep_repair_subpos` (positive part of the best
  gain per named edit class, no channel cost) for profiles and the ladder, and
  the channel-cost **discrete label** (best edit after 1 bit per character,
  NONE if none pays) for token-level reading and the coded-sample check.
- Essay profile: per-token rate of each discrete label (count /
  n_tokens_scored), and the mean positive-gain vector over the essay's
  high-surprisal tokens. The ladder adds `R` = B0 + repair profile and
  `S2+R`.
- Coded-sample check (D7): E_TOK and E_CTX tokens should get a paying edit
  more often than OK tokens, and E_CTX tokens should get insertion or deletion
  edits more often than E_TOK tokens.
- **Pre-registered L1 prediction (TOEFL 11).** The rate of the
  insert-determiner label (per token scored) is higher for L1s without
  articles (CHI, HIN, JPN, KOR, TEL, TUR) than for L1s with them (ARA, FRE,
  GER, ITA, SPA), within proficiency level. The test is the level-stratified
  difference in mean rate, with a permutation test that shuffles L1 group
  labels within level (10,000 permutations). The prediction is falsified if
  the article-less group is not higher.

## 2026-10-17 · D9. k = 50 under the D4 rule

Sweep result (dev, two essay-level split-half seeds per k; median matched
cosine): k = 10 .996/.993, 20 .994/.995, 30 .995/.997, 50 .988/.994. k = 80
reached .992 on seed 0, but one half of seed 1 did not converge within
max_iter = 1000. A non-converged fit does not demonstrate replication, so k = 80
does not qualify, and D4 selects **k = 50**. The k = 80 final fits that had
been started early were discarded. Applied before any profile or outcome
analysis.

## 2026-10-17 · D10. δ component labels (k = 50), written before any outcome analysis

`component_labels.csv`, labelled from `results/study3/components_delta.md`
(dev only):

- 40 lexeme features (top-lemma share ≥ .50, the D5 rule).
- 7 error patterns: determiner expected before a noun (C0), *be* expected
  after a modal or *to* (C4), non-word misspellings (C7), have/do/make where
  the copula was expected (C8), *the* expected (C12), *to* expected (C13),
  bare *be* plus *because* misspellings (C40).
- 3 artifacts: subword continuation after a fragment (C1), punctuation
  spacing (C2), anonymization placeholders (C9).

Levels. D5 did not anticipate lexeme features for function words. A
component that is one function word (*because*, *there*, *its*, *they*, *can*
…) is given level `function`, and content-word lexemes level `lexical`. The
H2 permutation test therefore compares the 5 morphosyntactic error patterns
with the 27 content-word lexeme features. Function-word lexemes, artifacts and
mixed/orthographic components are outside that comparison. With 5 against 27,
H2 has little power, and the write-up should say so.

Observation for the write-up. Every non-lexeme, non-artifact δ component is
defined by what the model *expected* (*the*, *to*, *be*, *is*), not by what
was written. This is consistent with the premise result that the prediction
half is where error information lives once identity is set aside, and with
the lab finding that an explicit edit test (repair) recovers it best.

## 2026-10-17 · D11. Analysis frozen

Analysis frozen before reading any test row. Everything the confirmatory run uses
is fixed: split.csv (sha256 26c6541c78bd1337…), k = 50 (D9), component_labels.csv
(sha256 6d1a4baf886c4624…, D10), the repair representation and profiles (D8), and the
ladder, H2 and H3 definitions in 6_proficiency.py + 8_repair.py (sha256
e97492c83ba83b87…). The dev-CV results (results/study3/ladder_devcv.csv,
h2_permutation_devcv.json) were seen before freezing. They motivated no change.

Seen in dev before the freeze, for the record: M − M_obs = −.015, so the D3
reading rule says δ components do not describe the prediction failure beyond
the observed word. R − M_obs = +.004 [−.007, .015]. S2+R − S2 = +.017.

## 2026-10-17 · D12. Confirmatory results (test, read once)

`6_proficiency.py --confirmatory`, n = 1,945 test essays. Within-prompt R²:
B0 .421. ΔR² over B0 [95% bootstrap CI]: M .062 [.048, .076], M_obs .061
[.044, .077], R .051 [.035, .066], PCA .055, B2 .044, S2 .101. Contrasts:
M − M_obs .001 [−.008, .010], R − M_obs −.010 [−.029, .008], R − M −.011
[−.029, .006], S2+M − S2 .014 [.008, .020], S2+R − S2 .012 [.003, .022]. H2:
morphosyntactic mean |ρ| .173 vs lexical .094, permutation p = .018.

Readings under the frozen rules: H3 supported for M (point estimate > .05,
lower bound > 0). D3: M does not beat M_obs, so the δ profile's increment is
not evidence that it encodes how the prediction failed. R − M in dev (+.019)
did not replicate on test. No change to any analysis followed.

## 2026-10-17 · D13. The acceptable-surprise half, and a gold validation corpus

Langdon's direction after the repair results: the strongest angle is
decomposing surprisal into *errors* vs *acceptable-but-unexpected* words, and
the acceptable half ("what about creativity?") should be developed rather than
left as the residual NONE.

Data. Langdon approved downloading BEA-2019 W&I+LOCNESS v2.1
(`data/wi_locness/raw/`, unversioned). License: non-commercial research and
education only; published excerpts under 100 words; cite Yannakoudakis et al.
(2018); credit CECL (Louvain) for LOCNESS and send them a copy of
publications; no redistribution. First-annotator edits are used as gold.

Design, fixed before any W&I token was scored:

- Gold categories per token: ERR_TOK, ERR_INS, POST, OK (`9_wi.py`).
- The W&I **train** split (3,000 learner essays) is the development set for
  any acceptability model. W&I **dev** (300 learner essays) and **LOCNESS**
  (50 native essays, N) are held out and used once, for confirmation.
- Same model, window and τ as ELLIPSE (11.28 bits).
- First questions, answered before any modelling: (1) what share of
  above-τ tokens are gold errors at each CEFR level, and (2) how well the
  existing repair net gain separates errors from OK tokens.

## 2026-10-17 · D14. TOEFL 11 results (out of sample; nothing fit on TOEFL except the replication NMF)

- Transfer. The ELLIPSE-dev elastic nets were applied unchanged, and their
  predictions correlated with level (Spearman): B0 .563, B1 .573, M_obs .626,
  M .605, R .650. Reconstruction error with the dev-fitted H: dev .702, test
  .702, TOEFL .727.
- Replication. A TOEFL-fitted δ NMF at k = 50 matches the ELLIPSE components
  at median cosine .46. The components do not replicate across corpora, which
  is consistent with their being topic-bound lexemes.
- **Pre-registered article test (D8): supported.** Insert-determiner repairs
  per token are higher for the article-less L1s at every level (low .0011 vs
  .0006, medium .0013 vs .0005, high .0010 vs .0003). The level-stratified
  difference is 0.36 pooled within-level SD, permutation p < .0001. Across
  the 11 L1 means, Mann–Whitney p = .004. The four European article
  languages (GER, SPA, ITA, FRE) are lowest. The one inversion is CHI (.66 per
  1,000 tokens) just below ARA (.68).

## 2026-10-17 · D15. Final acceptability model and the surprisal decomposition (fixed before confirmation)

Development (W&I train, 5-fold CV grouped by essay, error vs OK AUC,
gradient boosting) gave: surprisal .698, context views alone (`ctx_excess_s4`)
.757, attestation (`att_full_s`) .826, repair (`rep_mdl_plus_pos`) .840,
repair + attestation (`comb_ra`) .865, repair + context (`comb_rc`) .855,
**repair + attestation + context (`comb_rac`) .875**. The linear probe was
lower throughout (`comb_rac` .822), so the GBM is kept.

- **Primary model: `comb_rac`, GBM** (fixed hyperparameters in `accept.py`),
  fit on all of W&I train. It outputs P(OK) for each token above τ.
- Confirmation runs once on W&I dev + LOCNESS: `comb_rac` (primary) with
  `surprisal`, `rep_mdl_plus_pos`, `att_full_s` and `ctx_excess_s4` as
  pre-declared comparisons. No further changes after that run.

Essay-level decomposition, applied unchanged to ELLIPSE and TOEFL with the
W&I-train model (no ELLIPSE or TOEFL scores are used in fitting). Per essay,
with n the scored non-space tokens and p the P(OK) of each token above τ:

    high_rate = #(above τ) / n
    err_rate  = Σ (1 − p) / n      expected errors above τ
    acc_rate  = Σ p / n            expected acceptable-but-unexpected

so high_rate = err_rate + acc_rate. ELLIPSE dev is for looking, and ELLIPSE
test is confirmatory.

- A1. err_rate correlates more negatively with Overall than high_rate does
  (Spearman). Test: bootstrap over essays of the difference in ρ.
- A2. |ρ(acc_rate, Overall)| < |ρ(err_rate, Overall)|. There is no
  directional prediction for acc_rate's own sign, which is the "creativity"
  question; it is reported with a CI.
- A3, exploratory. Partial Spearman correlations of each rate with each
  analytic score, controlling Overall. Expected: err_rate is most negative for
  Grammar and Conventions, and acc_rate is most positive for Vocabulary and
  Phraseology.
- A4. 10-fold CV R² (grouped by prompt) of Overall from high_rate alone vs
  err_rate + acc_rate. This tests whether the split carries information that
  the total does not.
- A5. On W&I dev + LOCNESS, model err_rate falls with level (A > B > C ≥ N),
  and gold-based rates are shown alongside. On TOEFL, the Spearman ρ of each
  rate with level is reported.
- Kinds of acceptable surprise, descriptive: the acceptable mass split by POS
  and by attestation type (rare attested content word, proper noun or number,
  other).

## 2026-10-17 · D16. Per-essay prior adjustment of P(OK) (post hoc, after the D15 results)

The D15 confirmation passed at the token level (W&I dev AUC .873; dev CV
.875). Essay-level checks then showed a bias. On W&I held out, the model's
essay error rate tracks gold (ρ = .89, and the level trend is kept), but its
acceptable rate does not keep the level trend: ρ with CEFR level is .04
against gold .40. The token classifier shrinks P(OK) toward the training
base rate. Mean P(OK) is .53 at A vs gold .45, and .73 at C vs gold .82.
This is the known bias of classify-and-count under prior shift (Forman
2008). Every essay has its own base rate of acceptable surprise, and the
classifier assumes the training one.

Correction, chosen in W&I train CV only: EM prior adjustment per essay
(Saerens, Latinne & Decaestecker 2002), with no shrinkage and the training
base rate .637. Within each essay it rescales the odds of P(OK) by
π_essay / π_train and iterates π_essay to the mean adjusted P(OK). The ranking
within an essay is unchanged, and no new model is fit. Results in W&I train
CV, as essay acceptable rate vs gold / ρ with level (gold .31) / mean P(OK)
at A, B, C (gold .46, .65, .83):

- unadjusted: .74 / −.04 / .54, .65, .74
- EM: .79 / .33 / .41, .66, .87
- EM shrunk toward π_train (10 pseudo-tokens): .82 / .18
- essay-context features in the GBM, then EM: an AUC of up to .884, but EM
  then over-sharpens. Not used.

Plain EM is adopted because the question is the level trend. Both versions
are reported. The D15 rows marked "_em" use the adjusted P(OK). This is the
second use of W&I dev + LOCNESS, and it is flagged as such. The D15
hypotheses A1–A5 are unchanged.

## 2026-10-17 · D17. Decomposition results (ELLIPSE test confirmatory; TOEFL out of sample)

ELLIPSE test, n = 1,945 essays, Spearman ρ with Overall, 95% bootstrap CI.

| rate | unadjusted (D15) | EM (D16) |
|---|---|---|
| high_rate | −.55 [−.58, −.52] | (same) |
| err_rate | −.65 [−.68, −.63] | −.67 [−.70, −.64] |
| acc_rate | −.19 [−.24, −.15] | **+.51 [.48, .55]** |
| acc_share (acceptable / all high) | +.62 | +.64 |
| rare attested, acceptable mass | +.16 | +.37 |

- A1 supported in both versions: Δρ −.10 [−.12, −.09] unadjusted, −.12
  [−.14, −.10] EM.
- A2 supported in both versions: |ρ acc| − |ρ err| = −.46 unadjusted, −.16 EM.
- A4. CV R² for Overall rises from .33 (high_rate) to .43 (split,
  unadjusted) and .45 (EM).
- A3, exploratory, partial ρ controlling Overall. err_rate is most negative
  for Grammar (−.29) and Conventions (−.29). acc_rate (EM) is positive for
  Conventions .17, Grammar .16 and Vocabulary .11. The rare-attested mass is
  most positive for Vocabulary (.10).
- A5. On W&I held out (second use, D16), mean rate per token by level,
  EM vs gold:
  - err: A .079 vs .067, B .046 vs .042, C .016 vs .018, N .017 vs .020
  - acc: A .059 vs .054, B .075 vs .068, C .089 vs .082, N .116 vs .108
  - Essay-level ρ with gold: err .90, acc .85 (unadjusted .79). The level
    trend in acc is .38 (gold .40; unadjusted .04).
- Independent check against the 148 blind-coded ELLIPSE dev tokens
  (`decompose_coded_check.csv`). Token AUC is .90 unadjusted and .88 EM.
  Acceptable rate per 1,000 tokens, low/mid/high tertile:
  - coded: 30 / 43 / 48
  - EM: 19 / 34 / 50
  - unadjusted: 68 / 61 / 58
  The unadjusted sign is wrong. EM is right in sign and a little steep at the
  low end.

Reading: total surprise falls with proficiency only because errors fall
faster than acceptable surprise rises. EM's acc_rate partly carries the
essay's error prior, as its Grammar/Conventions partials show. The rare
attested mass is the cleaner lexical index.
- TOEFL 11 (n = 11,000, nothing fit on TOEFL), Spearman ρ with level
  (low/medium/high):
  - high_rate −.38
  - err_rate −.57 unadjusted, −.61 EM
  - acc_rate +.04 unadjusted, **+.52 EM**
  - acc_share +.60
  - rare attested +.25 unadjusted, +.41 EM
  - EM acceptable rate per 1,000 tokens: low 22, medium 46, high 71. Errors:
    171, 110, 65.
  The ELLIPSE pattern replicates.

## 2026-10-18 · D18. Cleanup of exploration code (no analysis change)

Langdon's direction: the write-up needs the δ result (appendix) and the repair
method, not the exploration that led to it. Removed from the code, and kept in
the git history before this date: `pilot/` (the earlier ModernBERT UMAP/HDBSCAN
clustering of δ), the geometry and hidden-state representation families and
their notes, the repair variants that lost to `rep_repair_subpos` and
`rep_mdl_plus_pos`, the attestation and context sub-variants that only fed
the sweep, `comb_ra` / `comb_rc`, and `lab/representation-lab.qmd`. No result
reported in D1–D17 is changed. Every module that the real-data stages
(`8_repair.py`, `10_decompose.py`) import is kept.

## 2026-10-18 · D19. Redesign around gold labels (supersedes D7, D8's synthetic bench, D13, D15–D17)

Langdon's direction: start from the simplest design an advisor can approve,
with one scored corpus and one gold-annotated corpus.

- **Error vs acceptable surprise is read off the W&I+LOCNESS corrections
  directly** (`5_gold.py`). The acceptability classifier, its context views,
  the EM prior adjustment and the ELLIPSE/TOEFL decomposition (D15–D17) are
  dropped. With no model fit, W&I is no longer split into train and held-out
  parts, and all of it (learners A–C and LOCNESS) is described once.
- **The blind-coded ELLIPSE sample (D7) and the synthetic error benchmark (D8)
  are dropped.** The W&I corrections replace both as the check on the repair
  labels.
- **The pre-registered article test (D8, D14) is dropped.** TOEFL 11 stays in
  the pipeline for a later, descriptive L1 comparison (`6_profiles.py l1`), and
  is not in the main report.
- **Repairs, τ and the ELLIPSE split are unchanged.** The new stages reproduce
  the old benches, repair labels and appendix profiles exactly.
- **The δ analysis (D1–D6, D9–D12) moves to `appendix/`** unchanged.
- **Rare word**, for the attestation types, is now fixed at fewer than 1 per
  million words of edited Dolma, instead of a quartile of W&I-train tokens.

Decided after the earlier results had been seen, which the write-up states.
FCE (Yannakoudakis et al. 2011) was downloaded as a possible second gold corpus
and has not been used.

## 2026-10-19 · D20. Direct logit attribution pilot (null; code removed)

Question: is a gold error *suppressed* (some component pushes against the
observed word) while acceptable surprise is merely *unsupported*? The logit gap
logit(obs) − E_p[logit] (which equals H − S, so only the shape of the split is
new) was split exactly over 273 OLMo-2 1B components (embedding, 256 heads, 16
MLPs), fp32, final and post-attention RMS scales frozen. Sample: 7,419 W&I
tokens above τ, 1,000 per level × error/OK cell plus all 419 scored N errors.
Statistics and reading rule were written before the gold contrast was computed.

- Errors vs OK within surprisal quintiles: suppression share AUC .585 (bar
  .60). Top-1 probability alone gives .580. Adding all eight split statistics
  to surprisal, entropy, top-1 and word frequency gains +.006 AUC (bar +.02).
- OK tokens by level, adjusted for S and H: every ρ with level has a CI that
  includes 0, and A−C and N−C differences are under .1 SD (bar .2). Word
  frequency does shift (ρ −.10), so the design could see a shift.
- Post hoc, splitting logit(obs) alone: also null, and errors are *less* pushed
  down than OK tokens.
- Post hoc mean-ablation: per-head attributions are not causal. Removing the
  top suppressing head (L15H10) was predicted to add +4.1 nats and added +0.06,
  because the post-attention norm rescales the remaining heads (~15×). Per-head
  DLA on OLMo-2 needs norm-aware or path-patching attribution.
- The layer-14 MLP is the top suppressor for ~55% of tokens whatever the
  category. No head is specific to agreement or determiners.

Code, notes and tables are in commit 4ceb451 (`pilot_dla/`). Data in
`data/study3/pilot_dla/` (unversioned).

## 2026-10-19 · D21. Tuned-lens and contextuality pilot (mostly null; code removed)

Two ways to split surprisal, tested on 12,015 W&I tokens above τ (3,000 each
from A, B, C, and all 3,015 LOCNESS), seed 42. Statistics and reading rule
were written before the gold contrast was computed. Baseline for both
questions: surprisal + entropy + Dolma word frequency (error vs OK AUC .664).

- **Tuned lens** (Belrose et al. 2023), one untuned run of 1,500 steps on
  ELLIPSE dev windows. The final layer reproduces the stored surprisal (median
  gap .03 bits). The raw logit lens is unusable on OLMo-2 1B before the last
  block. Errors vs OK: ΔAUC +.004 (late drop, layer 8 minus final) to +.008
  (full trajectory). Errors are already improbable at early layers, so they do
  not "look fine early".
- **Contextuality.** Ondov, Attal & Demner-Fushman (NAACL 2024) score the
  blank. The word-level ζ·log2(ζ/ψ) form is Holmes et al. (2026). Both were
  implemented. Causal sentence→essay contrast +.002, Ondov's blank score
  +.008. Full-essay surprisal alone gives +.047 causal and +.078
  bidirectional, which repeats the D15 context views: a better surprisal, not
  a new construct. Contextuality correlates .50 with an earlier mention of the
  word in the essay.
- **OK tokens by level**, adjusted for the baseline. Only the lens late drop
  shifts (C − A +0.39 bits [.19, .60], .14 SD). Post hoc, it roughly halves
  and loses significance when the 7 preceding tokens contain no gold error. The
  share of OK high tokens with such clean windows rises from .32 at A to .69
  at C. No measure separates LOCNESS from C.

Consequence for the main analysis: errors within the window inflate the
surprisal of the acceptable tokens that follow them, beyond the single POST
token that 5_gold.py excludes. This is followed up with the corrected essays
(D22).

Code and notes are in commit a5926c9 (`pilot_lens/`). Data in
`data/study3/pilot_lens/` (unversioned).

## 2026-10-19 · D22. Corrected-essay pilot: the uncorrected-surprise trend is mostly counting

Terminology from here on: a high token that the gold annotator left alone is
**uncorrected**, not "acceptable". The category mixes effective choices,
awkward but not wrong phrasing, missed errors, and names or sentence starts.

The first annotator's edits were applied to every W&I+LOCNESS essay, the
corrected essays were rescored with the stage 1 pipeline, and the writer's own
tokens were aligned between versions (99.96%). The rescored originals match
stage 1 exactly on all 708,020 tokens. GPU time 34 min.

- **Context contamination (the D21 effect) is real but small.** The share of
  uncorrected high tokens that fall below τ once the context is corrected is
  .19 at A, .13 B, .06 C, .04 N, which is 10.6 per 1,000 tokens at A and 4.9
  at C.
- **The reported rise is mostly how 5_gold.py counts.** `acc_rate` leaves
  out POST tokens (the writer's own word right after a corrected token) but
  keeps every token in n. POST is common at A. Per 1,000 tokens, A / B / C / N:
  - 5_gold.py `acc_rate`: 59 / 72 / 83 / 108
  - `acc_rate + post_rate` from the same table: 77 / 83 / 87 / 113
  - post-correction, the writer's own words: 81 / 86 / 89 / 114 (C − A +8
    [5, 11], ρ .12)
  - post-correction, the whole corrected essay: 87 / 89 / 90 / 116, flat
    across learner levels
- Post hoc breakdown of the A-to-C slope against 5_gold.py: error tokens
  leaving n −8.5, correcting the context +4.4 [3.4, 5.3], putting the
  post-error words back −10.8.
- **LOCNESS still separates from C:** +25 [16, 34] per 1,000.
- Corrections are mostly unsurprising (inserted words above τ 1.6%,
  substitutions 14%, base rate 9.1%). A deleted article makes the following
  bare noun phrase surprising (.10 → .23 above τ).
- The intrinsically surprising uncorrected tokens are the same kind at every
  level: topic nouns, names, sentence-initial adverbs, some missed errors.
  Median Dolma frequency falls only from 32 to 26 per million from A to C.

Consequence: the README and results.qmd claim that acceptable surprise rises
steeply with level (59 → 83) overstates the trend. Among learners, uncorrected
surprise is nearly flat. The robust contrast is native vs learner. Not yet
acted on.

Code and notes: `pilot_corrected/` (commit 31b9dba). Data:
`data/study3/pilot_corrected/` (unversioned).

## 2026-10-19 · D23. Native vs C under topic and register controls (pilot)

After D22, the gap between native writers and C learners in uncorrected
surprise is the main evidence that surprisal tracks something other than
errors. LOCNESS (50 essays) differs from W&I in prompts and genre. This pilot
reuses the D22 corrected-essay tokens (no GPU). The rule was written before any
N − C contrast under a control was computed. Baseline N − C = +25.0 [16.0,
33.9] per 1,000 of the writer's own tokens.

N − C per 1,000 kept own tokens (N/C ratio):

- names and numbers excluded: +25.8 [17.3, 34.5] (1.32)
- lemmas used earlier in the essay excluded: +33.9 [23.4, 44.5] (1.29)
- topic words (keyness against Dolma) excluded: +21.3 [14.1, 28.5] (1.31)
- sentence and paragraph starts excluded: +21.8 [12.6, 30.7] (1.25)
- all four excluded (58% of tokens): +36.8 [26.1, 48.4] (1.38). Per 1,000
  of all own tokens, this is +12.0 [6.5, 17.9].

Matched subsets (W&I has no prompt metadata, so these come from the texts),
all four exclusions applied: expository genre +33.1 [21.0, 46.0]; the 5
topic-nearest C essays per N essay +28.3 [15.1, 41.8]; top quartile of C on
a register score (sentence length, nouns, passives) +21.3 [8.4, 33.8]. With
no exclusions the register-matched gap is +9.3 [−0.9, 19.7]. Topic and genre
matching cut the gap by 15–20%; register matching cuts most of it, although it
may partly match on the construct itself.

Mean surprisal per essay (corrected essays, all tokens): A 5.33, B 5.32,
C 5.29 bits, flat across learners. N − C = +0.38 [0.26, 0.50], +0.54 with all
exclusions, +0.19 [0.06, 0.33] against register-matched C.

All of the excess is in content words (nouns 44%). Reweighting N to C's mix of
Dolma frequency bands removes 32% of it. Reading: N > C survives topic,
repetition and position controls, and is partly a matter of register. The
bootstrap resamples essays, not the handful of LOCNESS prompts, so the real
uncertainty is larger.

Code and notes: `pilot_native_c/` (commit 3eed473). Data:
`data/study3/pilot_native_c/` (unversioned).

## 2026-10-19 · D24. Exact split of essay surprisal by the gold corrections

Requested by Langdon after D22–D23; computed on the D22 alignment, no GPU.
Per essay, mean surprisal over the n original tokens splits exactly into:

- **clean**: the writer's own aligned words, scored in the corrected essay;
- **omission**: the extra surprisal of own words with a word missing just
  before them (ERR_INS);
- **context**: the extra surprisal of other own words caused by errors
  elsewhere in the window;
- **error tokens**: the corrected tokens' excess over clean.

Because the parts add up, their covariances with level (A = 1, B = 2, C = 3)
split cov(total, level) exactly. Essay-bootstrap CIs.

Mean surprisal (bits), A / B / C / N: total 6.36 / 5.92 / 5.56 / 5.94; clean
5.25 / 5.29 / 5.28 / 5.66; omission .11 / .08 / .04 / .04; context .30 / .19
/ .09 / .08; error tokens .69 / .37 / .15 / .16.

Share of the learner proficiency signal (cov with level):

| part | mean surprisal | high-token rate |
|---|---|---|
| clean | −4% [−10, 1] | −19% [−28, −11] |
| omission | 8% [7, 9] | 11% [9, 12] |
| context | 27% [25, 29] | 24% [21, 26] |
| error tokens | 69% [65, 73] | 85% [78, 92] |

Spearman ρ with level: total −.44, clean +.04, error tokens −.62, context −.52.

- All of the learner proficiency signal in mean surprisal is in the part the
  corrections remove. The clean part is flat for the mean, and for the
  high-token rate it rises slightly against the total.
- About a quarter of the signal is errors raising the surprisal of the
  correct words *after* them, not the error tokens themselves.
- N − C in mean surprisal (+.38 [.26, .50]) is entirely in the clean part
  (+.38). N − C in high tokens (+28.6) is +25.0 clean.

Code: `pilot_corrected/5_decompose.py` (commit ce6ef92). Tables:
`data/study3/pilot_corrected/tables/decompose_*`.

## 2026-10-19 · D25. The D24 split replicates on FCE

Requested by Langdon. The same corrections, scoring, alignment and split as in
D22/D24, applied to BEA-2019 FCE v2.1 (one annotator; Yannakoudakis, Briscoe &
Medlock 2011). FCE has no CEFR level, so the criterion is the script score
(`script-s`, 0–40, the score Yannakoudakis et al. model). n = 1,237 scripts
(one score-0 script excluded), the two answers pooled. As a check, each answer
was also run against `answer-s` (1–15 ordinal, n = 2,451), bootstrapping whole
scripts. The reading rule was written before any share was computed: the
result replicates if (a) the clean share has a CI that includes 0 or
|share| < .15, and (b) the three correction parts take at least 85%.

Share of cov(total, score), with W&I (cov with level) for comparison:

| part | FCE mean | W&I mean | FCE high rate | W&I high rate |
|---|---|---|---|---|
| clean | 9% [3, 14] | −4% | −4% [−11, 3] | −19% |
| omission | 12% | 8% | 16% | 11% |
| context | 22% | 27% | 18% | 24% |
| error tokens | 58% | 69% | 70% | 85% |
| corrections, total | 91% [86, 97] | 104% | 104% [97, 111] | 119% |

Spearman with score: total −.58, clean −.08 [−.14, −.03] (mean); total −.50,
clean +.03 (high rate). Replicates on both measures; for the mean, (a) passes
on size only, since the clean CI excludes 0. The answer-level analysis gives
the same verdict. FCE part sizes are close to W&I B (total 5.93 vs 5.92 bits,
clean 5.20 vs 5.29).

Checks: exact sum holds; 99.99% of own tokens align; unchanged windows move
< .05 bits in 99.2% of cases; json texts match M2 after whitespace
normalisation, and corrected answers differ from M2 in 0.07% of characters
(moved words, sentence joins). GPU 28 min.

Code: `pilot_fce/`, `src/essays/ingest.py --fce` (commit c45226b). Tables:
`data/study3/pilot_fce/tables/`.

## 2026-10-19 · D26. Native-writing calibration (pilot)

Langdon asked roughly where native writing falls in mean surprisal compared
with C-level learners, preferably using text that is not in the training data.
Same instrument, and stored scores are reproduced exactly. Documents of
200–600 words, cut at a sentence end. Essay mean surprisal (bits), 95%
bootstrap CI:

- W&I A / B / C: 6.60 / 5.98 / 5.60 [5.56, 5.64]
- LOCNESS: 5.94 [5.82, 6.06]
- ELLIPSE Overall ≤ 2.5 / 3–3.5 / ≥ 4: 6.86 / 6.04 / 5.59 [5.56, 5.61]
- PERSUADE 2.0 (CC BY-NC-SA 4.0; not in OLMo-mix-1124, 0/900 spans): ELL
  6.13; non-ELL 5.61 [5.56, 5.66]; non-ELL scoring 5 or above 5.54 [5.50, 5.58]
- Wikipedia articles created 2018–21 (in the training mix, 97/100 docs): 5.73;
  articles created 2026 (after the cutoff): 5.77 (+.04 [−.05, .13])

Reading: native US student essays sit at the C-learner level. On the seven
prompts PERSUADE shares with ELLIPSE, non-ELL minus ELLIPSE ≥ 4 is −.04 bits.
More literate native prose (LOCNESS, Wikipedia) is *more* surprising. Above C,
mean surprisal is not a nativeness scale: once errors are rare, topic, register
and vocabulary set the level. Memorization barely changes surprisal under this
instrument. PERSUADE's non-ELL flag is a school category, not native-speaker
status.

**Data sent outside (licence incident).** The training-data check sent about
5,350 15-word spans of licensed learner text to the public infini-gram API
before it was stopped: 90 each from W&I C, LOCNESS and ELLIPSE, then about
2,376 W&I A, 2,397 W&I B and 300 W&I C. The coordinator's prompt suggested
infini-gram checks without restricting them to public sources. None of those
results are used. `contam.py` is now limited to public text. Rule from here
on: licensed corpora (W&I+LOCNESS, FCE, ELLIPSE, TOEFL 11) never go to an
external service.

Code and notes: `pilot_native/` (commit beedefd). Data: `data/study3/pilot_native/`
(unversioned; the three unused files from the stopped check are still there).

## 2026-10-19 · D27. Shapley pilot: context-word attribution fails, edit-level attribution works

Langdon asked for an explainability method for the spillover component (D24
"context") that a committee would accept. W&I+LOCNESS only, all local. The
rules were written before the results. 52,979 targets, ERRANT types for 98.8%
of edits, about 1.8 GPU-hours.

**Context words, removal baseline (the planned method): does not explain
spillover.** Exact Shapley over the 7 context words of each target, removing
words by attention masking with the original position ids kept. Efficiency
holds to 3e-14 bits, and masking matches deletion (≤ 3e-5 bits). With no
context the target costs about 23 bits, so almost every context word, error
words included, *lowers* surprisal (mean φ of error words −2.4 bits). Removal
compares a word with no word, but spillover compares an error with its
correction. Positive attribution falls on errors no more than their positions
predict (ratio 1.02 [0.97, 1.06]). It does not track the counterfactual Δ
(ρ .006). Every ERRANT group gets negative attribution. Leave-one-out gives the
same verdicts, and so does replacement with a separator token (ρ .53 with
masking). Not to be reused: `2_targets.py`, `3_shapley.py`,
`4_analyze.py`, `5_kwic.py` (`7_edit_analyze.py` imports helpers from
`4_analyze.py`).

**Edits, correction baseline (post hoc): an exact split of Δ.** The players are
the gold edits in a target's window, and the baseline is the corrected text,
so ψ splits the counterfactual Δ among the errors (Σψ vs Δ r = .9996, 5
GPU-min). 69% of windows hold one edit, where ψ is just Δ. Multi-edit windows
carry 51% of ΣΔ. There, leave-one-out picks the same top edit 95% of the time
but leaves 54% of |Δ| unassigned, so Shapley is needed for an exact split by
type.

- Spillover per error is the same at every level: A 1.57, B 1.61, C 1.60,
  N 1.62 bits. The error rate accounts for 102% of the A − C gap. The plain Δ
  over all essays agrees (1.53 vs 1.44 bits per error at A vs C; the rate takes
  95%).
- Spillover roughly halves with each word (0.89 / 0.48 / 0.24 / 0.12 bits).
  The 8-subword window caps the reach at a few words, so "gone after about 4
  words" is partly the window.
- By ERRANT group: lexical, preposition and determiner errors spread most
  (2.0–2.7 bits per error), punctuation least (0.33).

Code and notes: `pilot_shapley/` (commit d3728a8). Data:
`data/study3/pilot_shapley/`.

## 2026-10-19 · D28. Exact split of essay surprisal by ERRANT error type

Requested by Langdon. The D24 split is refined to S_orig = clean +
Σ_g (at_site_g + spill_g) + flagged + residual, over ERRANT groups g. At site
covers a replacement or unnecessary edit's own tokens, or the word after a
missing-word edit. Spill is the remaining Δ on own aligned tokens, split among
the window's edits by edit-level Shapley with the correction as baseline (D27).
Flagged is spans the annotator marked but did not correct (no type). Residual
is instrument noise (0%). The sum holds in every essay and script (< 1e-9) and
reproduces D24/D25. Error tokens downstream of an earlier error (61% W&I, 54%
FCE) keep all their excess. Reassigning the earlier errors' part (−7% / −2%)
moves no group by more than 0.9 points. About 42 GPU-min.

Share of cov(total, level) in %, at site + spill = total; W&I CEFR A–C,
FCE script score:

| group | W&I | FCE |
|---|---|---|
| clean | −4 | 9 |
| spelling/orthography | 18 + 3 = 21 | 14 + 3 = 17 |
| verb form/SVA/tense | 10 + 2 = 12 | 8 + 2 = 10 |
| determiner | 3 + 3 = 6 | 8 + 3 = 11 |
| preposition | 2 + 3 = 5 | 3 + 3 = 6 |
| punctuation | 5 + 1 = 6 | 1 + 1 = 2 |
| noun number | 3 + 1 = 4 | 2 + 0 = 3 |
| word order | 1 + 0 = 1 | 1 + 0 = 1 |
| other lexical | 11 + 5 = 15 | 9 + 4 = 13 |
| other (ERRANT OTHER rewrites) | 17 + 9 = 26 | 14 + 6 = 20 |
| flagged | 7 | 9 |

By operation (missing / replacement / unnecessary): W&I 17 / 75 / 5, FCE
23 / 56 / 4. Error frequency explains 75–110% of the low-to-high gap for most
groups. Spill per error is flat overall (replacements +.02 bits A→C) and per
group on FCE except "other". On W&I it shifts per group in ways that cancel:
it rises at C for spelling (+.80) and determiners (+.38) and falls for
prepositions, noun number and punctuation. CEFR sublevels add little (total
ρ −.46 vs −.44 with A/B/C). The 8-subword window caps spill, so spill shares
are lower bounds.

Code and notes: `pilot_types/` (commit 20eadb6). Data: `data/study3/pilot_types/`.

## 2026-10-20 · D29. Reconciling the "null" clean part with the earlier gold correlations

Langdon asked why the clean part (D24) looks null among learners when earlier
gold results had the uncorrected high-token rate at ρ = +.32 with level, and
+.24 partialling out errors. W&I learners (n = 3,300). Spearman ρ with level,
and partial ρ (Pearson on ranks after regressing out the control's ranks),
1,000 essay bootstraps:

| measure | ρ | partial ρ \| gold error rate |
|---|---|---|
| old OK-only rate (POST dropped; D15–D17 table) | .316 | .244 [.210, .279] |
| uncorrected rate, OK + POST (5_gold.py since 7bdc91a) | .140 | .182 [.146, .215] |
| clean high-token rate (D24, corrected essays) | .124 | .195 [.160, .228] |
| clean mean surprisal (D24) | .045 | .162 [.125, .197] |

The +.32 / +.24 are the old OK-only rate. Three things change between that
figure and the clean part:

1. **Counting (D22).** POST tokens are uncorrected but concentrated at A.
   Dropping them made the rate rise with level (ρ .32 → .14 when counted).
2. **Units.** The old rate counts high tokens per original token, so the
   errors' own tokens are in the denominator and spillover stays in the
   uncorrected words. The clean part is rescored in the corrected essay, per
   own word, and the spillover moves to the errors (D24, D27).
3. **Share of covariance vs partial correlation.** The share (−4% / −19%)
   reflects the raw covariance, which is near zero. But the clean part
   correlates *positively* with the error rate (ρ .13 for the mean, .05 for
   the rate), and the error rate falls steeply with level (−.61). That masks
   a positive association: at a given error rate, higher-level writers'
   corrected text is more surprising (partial +.16 to +.20). A likely reason
   is that the clean part mixes level-rising surprise (vocabulary, register)
   with what minimal corrections leave behind (awkwardness, missed errors),
   which tracks the error rate.

Controlling the *surprisal* of the corrections (omission + context + error
tokens) instead of the error count, the partial shrinks to .067 [.032, .102]
for the rate and .027 [−.010, .065] for the mean. That control also removes
essay-level predictability shared by the clean and error parts, such as topic,
so the gold count is the cleaner control for "beyond errors". Both are to be
reported. FCE replication pending (pilot_rq3).

So "the clean part is null among learners" holds for its raw covariance
only. Conditional on error rate it carries a modest positive level signal.
Computed with a scratch script over `pilot_corrected/aligned.parquet`,
`results/study3/gold_essays.csv` and `results/study3/decompose_wi_essays.csv`.
To be rebuilt in the end-to-end run.

## 2026-10-20 · D30. W&I and FCE are treated as not looked at in advance

Langdon's decision. The study and its RQs are partly exploratory, and the
committee does not expect a strictly confirmatory design. The design decisions
relevant to the write-up (counterfactual correction and rescoring, the exact
split, ERRANT grouping, edit-level Shapley with the correction as baseline,
incremental validity over error counts) could have been made without seeing
the data. The write-up therefore treats W&I+LOCNESS and FCE as not looked at
in advance. Langdon writes the hypotheses as they would have been predicted
before the pilots. The pilots (D20–D29) stay on record here and are not cited
as prior looks. A fresh held-out test set is built only if the advisor asks
for one.

## 2026-10-20 · D31. RQ3: surprisal beyond ERRANT error counts; D29 corrected

Requested by Langdon. The RQ3 baseline is the human error counts, not
aggregate surprisal. B = the per-token rate of each ERRANT group, the
flagged-span rate and log n. S compact = clean, mean at-site cost per error,
mean spill per error. S full = per-group costs and spill. Ridge, alpha by
inner CV, 10 × 10-fold CV, paired folds, 2,000 unit bootstraps. The rule was
written before fitting: S adds information if Δ R² (B + S compact) − B has a
CI that excludes 0 on W&I and is the same sign with a CI that excludes 0 on FCE.

| model | W&I R² (level) | FCE R² (script score) |
|---|---|---|
| B (error counts) | .581 | .481 |
| B + S compact | .585 (Δ +.0037 [.0007, .0068]) | .496 (Δ +.0152 [.0056, .0250]) |
| B + S full | .582 (Δ +.0007 [−.0029, .0042]) | .504 (Δ +.0230 [.0087, .0382]) |
| B + mean surprisal | .581 (Δ .0000) | .493 (Δ +.0119) |
| mean surprisal alone | .183 | .329 |
| S compact alone | .034 | .034 |

The rule is met for S compact, narrowly. The increment is 0.4 points of R² on
W&I and 1.5 on FCE, where plain mean surprisal gives most of the FCE gain.
S full fails on W&I. The increment sits in the per-error cost and spill terms.
At a fixed error count, more surprising errors go with a lower level or
score. The clean term's sign differs between the corpora.

**D29 correction.** D29's "gold error rate" was 5_gold.py's `err_rate`:
corrected tokens *above τ* per token. That depends on surprisal, so it is not
a plain error count. With the plain share of corrected tokens, the W&I
partials of the clean part are .118 [.083, .153] (mean) and .135 [.101, .167]
(high-token rate). **FCE does not replicate them.** Clean mean surprisal vs
script score: ρ −.078 [−.135, −.025], partial given the corrected-token share
−.076 [−.131, −.024]. High-token rate: ρ .032, partial −.009 [−.068, .043].
D29's claim that "at a given error rate, higher-level writers' corrected text
is more surprising" holds on W&I only and should not be reported as a
general finding.

Code and notes: `pilot_rq3/` (commit 7be4e51). Data: `data/study3/pilot_rq3/`.

## 2026-10-20 · D32. Association measures under correction: surprisal is not unique

Langdon's questions. Is "explained by correctable errors" specific to
surprisal, or shared by the phraseological and lexical measures of applied
linguistics? And does treating surprisal as an association measure change
that? W&I (level) and FCE (script score). Each measure is computed on the
original and the corrected essays. Errors are controlled by the plain
corrected-token share. The increment is over ERRANT error counts (D31
framework) for the measure computed on corrected essays. The rule was written
before the level analyses: a measure survives correction if its
corrected-essay ρ keeps at least half of the original ρ with the same sign and
its ΔR² over B has a CI excluding 0, on both corpora.

Design revision R1 (Langdon, before the level analyses):
- **Primary LLM measures:** LLM-PMI = log2 p(w|ctx) − log2 f(w) and ΔP-LLM =
  p(w|ctx) − f(w), with f the corpus-a word relative frequency, over real
  words that are one subword (98% of real words).
- **Model marginal p̄** = mean softmax over 30M 7-subword Dolma contexts. It is
  a sensitivity check: SE median .004 bits, and it agrees with the subword
  relative frequency at r .997 and with word frequency at .93–.97. The corpus
  b (unedited) p̄ differs by a median .7 bits (register). All denominators give
  the same verdicts.
- **Added variants:** attested-only (left bigram, or both bigrams, attested in
  corpus a; unattested = n/a) and a surprisal cutoff (> τ excluded; it selects
  on the outcome).

Key rows, W&I / FCE (ΔR² in points of R²):

| measure | ρ original | ρ corrected | share removed | ΔR² over B |
|---|---|---|---|---|
| surprisal | −.44 / −.58 | .04 / −.08 | 104% / 91% | n.s. |
| LLM-PMI | .40 / .56 | .03 / .22 | 93% / 75% | n.s. |
| ΔP-LLM | .43 / .53 | .14 / .24 | 73% / 62% | n.s. / 1.25 [.26, 2.29] |
| ΔP-LLM, both bigrams attested | .41 / .49 | .15 / .25 | 70% / 57% | n.s. / 1.28 [.30, 2.32] |
| bigram MI | .23 / .31 | .00 / .14 | 100% / 55% | n.s. |
| bigram ΔP | .24 / .19 | .16 / .14 | 32% / 28% | .25 [.02, .46] / .71 [−.00, 1.42] |
| content-word log frequency | −.18 / −.14 | −.19 / −.20 | none (grows) | .52 / .70 (both CIs exclude 0) |

- **Q1.** The Study 2 dependency MI measures, bigram MI and t, the MI ≥ 3 and
  t ≥ 2 shares, and the unattested-bigram share all lose 50–134% of their
  association under correction and add nothing over B. Bigram ΔP survives on
  W&I, and its FCE CI touches 0.
- **Q2.** No LLM measure survives on both corpora (content-word,
  frequency-adjusted, cutoff and attested-only variants included). ΔP-LLM
  survives on FCE only.
- Only content-word frequency (lexical sophistication) keeps its association
  and adds over error counts on both corpora.
- **Q3.** At the token level, surprisal tracks bigram ΔP (ρ −.62) and LLM-PMI
  tracks bigram MI (.55). Tokens that bigram measures exclude by construction
  carry 60% / 85% of surprisal's original-essay signal. The attested-only
  filter removes 71% / 97% of that signal, yet the attested LLM-PMI still loses
  94% / 69% to correction, through real-word errors and spillover. The W&I/FCE
  sign flip of D31 recurs for post-correction LLM-PMI (−.12 / +.08 given the
  corrected-token share).

Study 2's measures reproduce `ellipse_metrics.csv` exactly. D24, D25 and D31
reproduce. About 2.6 GPU-h for p̄.

Code and notes: `pilot_assoc/` (commit 647d9c8). Data: `data/study3/pilot_assoc/`.

## 2026-10-20 · D33. Which kinds of correction carry each measure's signal

Langdon's question: how much of the error-driven signal is plausibly
phraseological? Buckets, fixed before the run (ERRANT category):
- mechanical: SPELL, ORTH, PUNCT, VERB:FORM/SVA/INFL/TENSE, NOUN:NUM/INFL,
  MORPH, ADJ:FORM, CONTR, NOUN:POSS
- function word: DET, PREP, PRON, CONJ, PART
- content-word choice: NOUN, VERB, ADJ, ADV
- multi-word / order: OTHER, WO

**Surprisal:** the D28 split regrouped by bucket (per-edit Shapley within
windows). **Classic measures:** exact Shapley over the four buckets per essay
(16 partially corrected versions, re-parsed). Both use the correction as
baseline. The sums are exact, and D28 and D32 reproduce. CPU only.

Share of cov(original measure, level or score), %, W&I / FCE.
"Phraseological" = content + multi-word, then with function words added:

| measure | corrected | mechanical | function word | content | multi-word | phraseological |
|---|---|---|---|---|---|---|
| surprisal | −4 / 9 | 47 / 35 | 14 / 20 | 13 / 10 | 22 / 17 | 35–49 / 27–47 |
| bigram MI | −1 / 31 | 38 / 27 | 22 / 10 | 12 / 10 | 29 / 21 | 41–63 / 31–41 |
| share MI ≥ 3 | −3 / 38 | 39 / 27 | 23 / 5 | 14 / 10 | 28 / 20 | 42–64 / 30–35 |
| bigram ΔP | 66 / 70 | 9 / 7 | 8 / 13 | 5 / 4 | 12 / 6 | 17–25 / 10–23 |
| dependency MI (mean of 3) | 24 / 40 | 14 / 12 | 10 / 8 | 28 / 25 | 24 / 16 | 52–62 / 41–48 |
| content-word frequency | 104 / 135 | 1 / −27 | 0 / 1 | −4 / −5 | −2 / −5 | ≈ 0 |
| reference: edit-rate covariance | – | 50 / 44 | 21 / 29 | 10 / 12 | 16 / 15 | 27–48 / 27–56 |

Reading:
- Surprisal's signal follows the error mix itself. Its bucket shares are close
  to the edit-rate covariance.
- About a third to a half is phraseological (content-word choice and
  multi-word), about half with function words. A third to a half is
  mechanical.
- Bigram MI and the MI ≥ 3 share also carry a large mechanical part (27–39%):
  misspellings break bigrams.
- Study 2's dependency MI is the most phraseological, led by content-word
  choice, and keeps 24–40% after correction.
- Bigram ΔP keeps most of its signal (66–70%), but in D32 its increment over
  error counts was not reliable on FCE.
- Content-word frequency is untouched by correction.
- Side finding: on FCE, Study 2's lemma-frequency measure (unseen lemmas
  counted as 1) is a misspelling artifact (96% mechanical). Its W&I signal is
  null.
- Caveat: the two Shapley granularities can differ only in windows holding
  edits from more than one bucket. Not measured.

Code and notes: `pilot_buckets/` (commit baec5ff). Data: `data/study3/pilot_buckets/`.

## 2026-10-20 · D34. Scope of the write-up, and the rebuild

Langdon's decisions:
- **Classic association measures** (D32, D33) go in the write-up as a
  post-hoc follow-up, not as a new RQ, pending the advisor's view.
- **The LLM association measures** (LLM-PMI, ΔP-LLM, p̄, attested-only
  variants) are not reported.
- **LM repair** is excluded from the write-up. It stays available, in git
  history, in case the advisor wants ELLIPSE in Study 3.
- **The main corpus** is now W&I+LOCNESS, with FCE as the replication. ELLIPSE
  appears only through τ and the δ appendix. Pending the advisor.

The rebuild replaces the exploratory pipeline with one that answers the RQs
directly, and reproduces D24–D33 as acceptance checks:
- corrections (first annotator, ERRANT types, buckets and groups);
- scoring of original and corrected essays;
- alignment;
- edit-level Shapley spill (correction baseline);
- the exact decomposition;
- the increment over ERRANT error counts;
- the classic measures with bucket attribution (post hoc);
- the calibration table (PERSUADE, post-cutoff Wikipedia).

Removed from the tree once the rebuild reproduces the results, with this entry
and the commit hashes as the record: the repair and attestation stages, the
old `5_gold.py` and `6_profiles.py`, and the `pilot_*` directories.

## 2026-10-20 · D35. Rebuild accepted; pilots and old stages removed

The rebuilt pipeline (stages 0–8, commit 6057d95) reproduces D24–D33.
`REBUILD.md` holds the comparison table.
- **Exact:** τ, corrections, ERRANT types, scores, alignment, the D24, D25,
  D31 and D32 results, and the classic bucket shares.
- **Within noise:** edit-level Shapley differs by ≤ 1.2e-4 bits, because of
  fp32 batch composition. The surprisal shares move by ≤ 2.4e-9.
- **Two small differences, explained in REBUILD.md:** single-relation
  dependency MI ρ (≤ .0024; the mean of three is exact), and the D26 CIs
  (≤ .007 bits; RNG stream).

The coordinator re-ran the comparison independently before removal.

Removed from the tree; recover them from git history if needed:
- **pilots** (`pilot_assoc`, `pilot_buckets`, `pilot_corrected`, `pilot_fce`,
  `pilot_native`, `pilot_native_c`, `pilot_rq3`, `pilot_shapley`,
  `pilot_types`): last present in 6057d95.
- **LM repair and attestation**, parked per D34: `3_repair.py`,
  `4_attestation.py`, `lib/repair_candidates.py`, `lib/repair_core.py`,
  `lib/repair_labels.py`, `lib/attestation.py`. The old `5_gold.py` and
  `6_profiles.py` went too. To restore repair for an ELLIPSE extension, check
  them out from 6057d95. They also need the W&I part of `2_select.py`
  (`select_wi`, removed in this commit) and the `LM` class of `lib/lm.py`.
- Also removed: `boot_spearman` and `weighted_d` from `lib/stats.py`.

`pilot_native/fetch_wiki.py` moved to `fetch_wiki.py`, because stage 8's
Wikipedia sample depends on it. `1_extract.py` and `2_select.py` stay as the δ
appendix's inputs. Stage 0 reads `1_extract.py`'s ELLIPSE shards. Every module,
the appendix's included, compiles and imports after the removal.

Open:
- `appendix/appendix.qmd` still discusses the LM repair profile, which D34
  excludes from the write-up.
- Untracked outputs of the removed stages remain in `results/study3/`.

## 2026-10-21 · D36. LM repair dropped; ELLIPSE to be corrected by a fine-tuned corrector model

After the advisor meeting, LM repair is dropped, not parked. ELLIPSE comes
back into Study 3 a different way. A corrector model is fine-tuned on the W&I and
FCE (original, corrected) pairs to produce minimal corrections of ELLIPSE.
ERRANT types its output, so ELLIPSE gets the same edit format as the two gold
corpora and runs through the same decomposition. ELLIPSE's analytic scores
then let the components be compared with rated dimensions, not only with an
overall level. The model, the data handling and the check against gold on
held-out W&I/FCE essays are still to be decided (the repo reorganisation
plan).

Done here:
- `appendix/A5_proficiency.py` no longer builds the repair profile (R,
  S2+R and their contrasts), and `appendix.qmd` no longer reports it. The
  stored ladder still has the old R rows; the figure filters them, and the
  next A5 run drops them.
- Deleted the untracked outputs of the removed stages: the old
  `results/study3/` files (`decompose_*`, `gold_*`, `repair_*`, `profiles_*`
  apart from the appendix's, `rq4`–`rq6`, `lab/`), `data/study3/pilot_*`,
  `data/study3/repair/` (5.1 GB) and `data/study3/wi/tokens_gold.parquet`.
  The code that made them is in 6057d95. The appendix renders.

## 2026-10-21 · D37. Repo reorganised around the current plan

Restructure only; no analysis changed.
- **Appendix self-contained.** `1_extract.py` → `appendix/A0_extract.py`,
  `2_select.py` → `appendix/A0_select.py` (repair bench writing removed),
  `split.csv` → `appendix/split.csv`. Its data and results moved to
  `data/study3/appendix/` and `results/study3/appendix/`. `lib/bench.py` is
  gone.
- **Main outputs flattened.** `data/study3/main/` → `data/study3/`,
  `results/study3/main/` → `results/study3/`.
- **Stages renumbered to the final plan.** `1_correct.py` → `1_edits.py`,
  `7_classic.py` → `8_classic.py`, `8_calibration.py` → `9_calibration.py`.
  Stage 7 is reserved for the ELLIPSE analytic scores (D36).
- `REBUILD.md` removed (D35 summarises it, and it is in 62bf690).
  `notes/literature.md` lost its repair sections.

Acceptance: stages 0, 1, 3, 5, 6, 8 and `9_calibration.py table` were re-run.
All 30 non-figure outputs in `results/study3/` and the 12 rewritten unit,
edit and alignment tables are identical to the pre-move copies. Both reports
render. Stage 0 and stage 9 still read the ELLIPSE token surprisal stored by
`appendix/A0_extract.py`. When ELLIPSE runs through `2_score.py`, they will
read that instead, and τ will be taken from all ELLIPSE essays rather than
the dev split.

## 2026-10-21 · D38. The corrector: Qwen3.5-9B with LoRA, five-fold cross-fitting

- **Name.** `corrector/`, not GEC: the corrections are not only grammatical,
  and that is part of the argument.
- **Data and folds.** Whole W&I essays and FCE answers, original →
  first-annotator correction, taken from stage 1. Five folds by unit,
  stratified by corpus × level. Each fold's model corrects its own fold.
  A sixth model, trained on everything, corrects ELLIPSE. No ensembling.
- **Model.** Qwen3.5-9B-Base, with Qwen3.5-2B-Base as the small comparison;
  no 27B. Gemma 4 12B was the alternative. Qwen trains 1.8× faster
  (938 vs 525 tokens/s) and generates 1.2× faster on an A6000, and the Qwen
  9B/2B pair differs only in size.
- **Settings.** LoRA r 16, α 32, all linear layers; cosine, 10% warm-up,
  batch 16, bf16, greedy decoding. Learning rate searched over
  {5e-5, 1e-4, 2e-4} on fold 0's tuning split by ERRANT F0.5. Each value is
  sourced in `corrector/README.md` (CoCoGEC; Adapting LLMs for Minimal-edit
  GEC; LoRA Without Regret; QLoRA).
- **Epochs.** Fixed at 2 in advance. A cosine schedule ties the whole LR path
  to the epoch count, so early stopping inside a run is not clean. One
  1-epoch run and one r 64 run at the chosen LR are the checks.
- **Baseline.** The instruct model with the minimal-edit prompt, zero- and
  few-shot, corrects fold 0 before any fine-tuning. It sets the agreement to
  beat.
- **Environment.** The corrector has its own uv environment (transformers 5,
  PEFT, the flash-linear-attention and causal-conv1d kernels). The main
  environment keeps transformers < 5 for the Study 1 instrument.
- **Data handling.** Sending the corpora to model APIs (Anthropic, OpenAI) is
  private, non-commercial dissertation use and within the licences (user,
  2026-10-21). This loosens D26's rule. Redistribution is still out.

## 2026-10-21 · D39. The fine-tuning input keeps the instruction; LR 5e-5

- **Instruction kept.** Each training pair is the chat template, with the
  minimal-edit instruction plus the original text as the user turn and the
  correction as the assistant turn (`corrector/README.md`, "Input format").
  Dropping the instruction was considered, since a prompt is worth at most a
  few hundred examples (Le Scao & Rush 2021) and we train on about 4,200. It
  was not worth re-running the search for: the instruction is minimal and
  harmless, so the runs proceed as they are (user, 2026-10-21).
- **Chat format on the base model.** Qwen3.5-9B-Base has no post-training, but
  its chat tokens are trained, and the untuned base model already answers in
  the assistant turn and stops at `<|im_end|>`. The LoRA trains no embeddings,
  so delimiters the model already knows are needed.
- **Learning rate.** Tuning-split F0.5 is flat across the search: .635 for 5e-5,
  .633 for 1e-4, .633 for 2e-4, with confidence intervals about .025 wide.
  The rule fixed in advance picks 5e-5. The result is reported as
  insensitivity to the learning rate, not as an optimum.
- **Checks: a margin, set before their results (user, 2026-10-21).** The
  rank-64 and 1-epoch runs replace the default (rank 16, 2 epochs) only if
  their tuning-split F0.5 lies outside the default run's 95% bootstrap CI
  (.623–.648 at 5e-5). A smaller difference keeps the default, so the
  setting cannot flip on noise.
- **2B learning rate.** The 2B search is flat as well: .589 for 5e-5,
  .589 for 1e-4, .593 for 2e-4. The rule picks 2e-4.
- **Checks: the default stays.** Rank 64 scores .635 (.623–.648), identical
  to rank 16. One epoch scores .631 (.618–.643): higher precision (.68 vs
  .67), lower recall (.48 vs .53), fewer edits (6.8 vs 7.7 per 100 tokens).
  Both are inside the default's interval. Final 9B settings: LR 5e-5,
  r 16, 2 epochs. The 2B model uses 2e-4 and is run on fold 0 only, as the
  size comparison.

## 2026-10-21 · D40. Corrections become edits the same way for model and gold

The decomposition runs on edits (stage 4 applies every subset of them), so the
corrector's corrected texts are turned into edits: `corrector/c5_edits.py`
(`common.char_edits`) takes ERRANT 3's edits from the original to the
correction and writes them as character spans of the original. Applied
together, a text's edits give the correction exactly; stage 1 applies them
without its whitespace repair and checks that.

- **Whitespace.** ERRANT sees only non-whitespace tokens, as in the BEA-2019
  M2 files (otherwise FCE's double spaces, left where a word is missing, are
  aligned with the inserted word, and half the determiner insertions turn
  into R:OTHER). Whitespace changes join the adjacent edit, or two changes
  one token apart make one edit; any other is an R:ORTH edit. The model's
  dropped final newline is not an edit.
- **Three edit sources on W&I/FCE.** `wi`/`fce`: the annotator's json edits
  (the main analysis, unchanged). `wi_model`/`fce_model`: the cross-fitted
  corrector's. `wi_errant`/`fce_errant`: the annotator's corrected texts,
  re-cut by ERRANT like the model's. The validation compares model with
  re-cut gold, so the two differ only in the corrections; re-cut gold vs json
  gold shows what the cut alone changes. The re-cut gold has the same
  corrected texts as the json gold, and about as many edits (W&I 22.1 vs
  21.8 per text, FCE 21.2 vs 20.8).
- **ELLIPSE** (`ellipse`) is cut the same way from the final model's
  corrections. Its unit is the essay and its group the score band (D41).
- Stage 1 on `wi` and `fce` gives byte-identical outputs after the change.

## 2026-10-22 · D41. ELLIPSE score bands: ≤2.5 / 3.0 / 3.5 / ≥4.0

The frequency-vs-cost table (`rate_cost`, stage 5) compares the lowest and
highest level groups, so ELLIPSE's overall score (1–5 in halves) is cut into
bands, as FCE's 1–40 score is. The nine values allow no exact quartiles; the
closest cut is ≤2.5 / 3.0 / 3.5 / ≥4.0 (1,739 / 2,335 / 1,302 / 1,106
essays), which also falls on whole and half points. The table compares ≤2.5
with ≥4.0. The bands are the group in stage 1 and only change the per-group
tables; the shares use the continuous score.

## 2026-10-22 · D42. Rubrics: what the level scores rate, and how it is used

The three level criteria differ in what they rate. W&I's CEFR level is one
rater's holistic score with no published criteria (Yannakoudakis et al.
2018). FCE's script score uses the general impression bands (task, content,
range, accuracy incl. spelling and punctuation, organisation, register,
effect on the reader), so accuracy is one criterion of seven. ELLIPSE's
overall score rates language only, and names accuracy in every band; its
analytic scores separate grammar (morphology, word classes), conventions
(spelling, punctuation, capitalization), syntax, vocabulary and phraseology
(`ELL_Rubrics.docx`, ELLIPSE-Corpus repository).

- Results compare the ELLIPSE analytic scores with the ERRANT groups
  (e.g. conventions with spelling/orth and punctuation, grammar with verb
  form, noun number, determiners, prepositions).
- Discussion, read cautiously and not as definitive results: (1) the
  criterion partly counts errors, so part of the errors' share is built in;
  (2) FCE's lower errors' share (.81 vs W&I .97) fits a criterion that also
  rates task achievement and content.

## 2026-10-22 · D43. OpenAI prompting comparison (no fine-tuning)

Suggested by a reader of the in-progress results: how do OpenAI models do
at the same correction task? OpenAI's fine-tuning platform is closing (no new
users; jobs until 2027-01-06; only GPT-4.1 models, which shut down
2027-01-14), so the comparison is prompting only and no OpenAI model is a
corrector candidate.

- Models: `gpt-6-luna`, then `gpt-6.1-sol`; `gpt-6-astra` (Batch API) only if
  Sol's results warrant it.
- Sample: the tuning split (462 texts), for every number in the comparison,
  including Qwen's prompting baselines, which are rerun there (fine-tuned
  Qwen on these texts: .635). Wider CIs than fold 0 (about 1.6x), and the
  best setting is chosen and reported on the same texts; acceptable for a
  side comparison, and the same holds for Qwen's chosen learning rate.
- Input identical to Qwen's prompting baseline: the instruction and text as
  one user turn, the same three few-shot pairs as earlier turns.
- Matrix per model: (1) zero- vs few-shot at default settings; (2) with the
  better prompt, lowest reasoning effort vs default (Luna: none vs default;
  Sol and Astra accept no lower than low). No temperature comparison: the
  GPT-6 models accept only temperature 1 (Luna accepts 0 only with
  effort none). Defaults are not sent; the docs give medium effort.
- Chat Completions, `store=False`, 8 requests at a time
  (`corrector/c3_openai.py`). Cost is logged per run at list prices.
- 2026-10-22 update: Luna's matrix ran in full (zero .537, few .527, zero
  with effort none .545; $0.71). Few-shot helped neither Qwen nor Luna, so
  Sol runs zero-shot only, low vs default effort. Astra is not tested.
- 2026-10-22 results (F0.5, tuning split): Sol zero-shot .551 at effort low,
  .540 at default ($4.70). Best prompted settings: Qwen 9B .507, Luna .545,
  Sol .551; fine-tuned Qwen 9B .635. Reasoning helped neither OpenAI model.
  Total API cost about $5.50 (list prices). The comparison is complete.

## 2026-10-22 · D44. Stage 7: ELLIPSE analytic scores against the parts of surprisal

`7_analytic.py` implements D42's comparison on the stage 5 ELLIPSE units
(model corrections, 6,482 essays). Two views, each part being the clean
part or one ERRANT group's or bucket's at-site + spill part:
- shares: each part's share of cov(mean surprisal, score) for all seven
  scores, with the paired bootstrap difference from overall's share;
- partial: Spearman of each part with each analytic score, controlling
  overall (D35's A3), so only the score's own signal beyond overall counts.
No prompt centring, as in stage 5. 2,000 essay bootstraps.

First results (partial ρ given overall): conventions goes with spelling/orth
(−.37) and punctuation (−.18) and not with the grammar groups (verb form
.01); grammar goes with verb form (−.37), noun number (−.22), determiner
(−.20) and preposition (−.17) and not with spelling (.00) or punctuation
(.02); syntax's strongest is punctuation (−.14). The clean part's share is
highest for cohesion (.39) and lowest for conventions (.27); overall .34.

## 2026-10-22 · D45. Corrector validation with CIs; stages 6 and 8 on ELLIPSE; write-up

- **Paired validation** (`5_decompose.py --validate`, validation_shares.csv):
  the shares of the json gold, the ERRANT re-cut of gold and the model on the
  same W&I/FCE units, from the same 2,000 unit resamples. Model minus re-cut,
  errors' share: W&I −3.8 points [−5.8, −1.9], FCE −0.4 [−3.1, 2.4]. The
  model gives more to mechanical (+1.6, +2.4) and less to content-word choice
  (−3.2, −1.5) and multi-word / order (−2.5, −2.3).
- **Stages 6 and 8 take --corpus**, and both stratify the CV by the units'
  level group. For W&I and FCE these are the same folds as before (stage 6
  and 8 outputs byte-identical). Run on ellipse, and stage 6 also on
  wi_model / fce_model so that ELLIPSE's ΔR² is compared with model-based
  error counts: B + S over B adds 5.2 points [4.1, 6.4] on ELLIPSE, against
  0.5 on W&I and 1.2 on FCE. Corrected ELLIPSE surprisal adds 3.8 [2.8, 4.8];
  the classic measures at most 1.3.
- **results.qmd** gains ELLIPSE in 3.1 (with the rubric comparison of D42),
  3.9 The corrector, 4.6 (corrector agreement, OpenAI side table, Table 14
  validation), 4.7 ELLIPSE, 4.8 analytic scores, two discussion paragraphs
  (a third corpus; what the criteria rate, read cautiously per D42) and a
  limitation (model corrections).

## 2026-10-22 · D46. results/study3/ laid out by corpus

The flat `results/study3/` of D37 had grown to about 110 files named
`<analysis>_<corpus id>`. It now mirrors `data/study3/`: one folder per
corpus id (`wi/`, `fce/`, `ellipse/`) with plain file names inside, the
corrector-check variants and the paired comparison under `validation/`,
plus `calibration/`, `figures/`, `threshold.json`, and the unchanged
`corrector/` and `appendix/`. `lib.corpora.results_dir(id)` gives the folder.
Stages 5 and 6 now write one file per corpus (the combined W&I + FCE
`clean_assoc.csv`, `increment.csv` and `coefs.csv` are gone; `results.qmd`
stacks them). The existing files were moved, not recomputed; stages 5 and 6
re-run on W&I/FCE write identical files, and both reports render.
Also deleted from data/study3/ (user, 2026-10-22): the OpenAI smoke tests,
the pre-renumbering logs, the unused W&I appendix extract, and the nine
corrector adapters not used downstream (tuning runs and the 2B model; their
corrections and scores are kept).
