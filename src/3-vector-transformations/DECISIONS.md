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
