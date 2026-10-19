# Pilot: does native > C survive topic and register controls?

Follow-up to D22. In the corrected essays, uncorrected high-surprisal tokens
per 1,000 of the writer's own tokens (`corr_own`) are A 81, B 86, C 89 and
LOCNESS (N) 114, so N − C = +25 [16, 34]. Among learners the trend is nearly
flat, so native vs C is now the main evidence that surprisal carries
information beyond errors. LOCNESS (50 essays by native-speaker university
students) differs from W&I in prompts, topics and genre. This pilot asks
whether N > C survives controls for those differences.

**Terms.** As in D22: a high token (≥ τ = 11.28 bits) the annotator left alone
is *uncorrected*. Tokens of the corrected essays are *post-correction*.
*Own* tokens are the writer's own words, aligned between the original and the
corrected essay (the D22 `corr_own` set).

No new scoring. Everything reuses `data/study3/pilot_corrected/`
(`tokens_corr.parquet`, `aligned.parquet`, `docbins_corrected/`, `essays.parquet`).

## Pre-declared statistics and reading rule

Written after the token and essay features were built (`1_features.py`,
which reads no surprisal by level) and before any N − C contrast under a
control was computed.

**Token sets.** Post-correction, non-space, scored tokens, in two sets:
`own` (the D22 `corr_own` tokens) and `all` (the whole corrected essay, D22
`corr_all`).

**Outcomes, per essay.**
- `rate`: tokens ≥ τ per 1,000 *kept* tokens (the control removes tokens from
  both numerator and denominator). Primary.
- `rate_num`: tokens ≥ τ that are kept, per 1,000 of *all* tokens in the set
  (only the numerator is filtered). This decomposes the gap: the drop from
  baseline is the N − C difference carried by the excluded tokens.
- `mean_s`: mean surprisal (bits) of the kept tokens.

Essays with fewer than 20 kept tokens are dropped for that control (counts
reported).

**Controls** (token exclusions, applied to both N and C):
- **C0 baseline**: none. Must reproduce D22 (`own` N − C = +25.0).
- **C1 names and numbers**: spaCy POS PROPN or NUM, tag NNP/NNPS/CD,
  `like_num`, or any digit in the token.
- **C2 repetition**: a word-like token whose lowercased lemma occurred earlier
  in the same corrected essay (first mentions are kept).
- **C3 topic words**: content tokens (NOUN, PROPN, VERB, ADJ, ADV), by
  lowercased form, that are key for the essay or for its topic cluster
  relative to edited Dolma (corpus_a unigrams, Study 2 tables):
  - *essay*: used ≥ 2 times in the essay at ≥ 10× the Dolma rate;
  - *cluster*: in ≥ 10% (and ≥ 3) of the essays of its k-means topic cluster
    (k = 30 over content-lemma TF-IDF → 100-dim SVD, all 3,350 essays, seed
    42), at ≥ 10× the Dolma rate in the cluster.
  All occurrences are removed, including first mentions.
- **C4 sentence and paragraph starts**: spaCy `is_sent_start`, or the first
  token after a newline or at the start of the essay.
- **C5 all of C1–C4.**

**C6 matched subsets** (W&I has no prompt or genre metadata, so subsets are
built from the texts; none of the criteria uses surprisal). Each is reported
under C0 and under C5:
- **C6a genre**: expository/argumentative essays only, in both groups. Not a
  letter (opening *Dear/Hi/Hello/Hey/To:*, or a letter closing), past-tense
  share of verbs < .30 (not narrative), and *you/your* < 1% of tokens (not
  addressed to a reader). Passes 47 of 50 N and 322 of 770 C.
- **C6b topic**: the 5 nearest C essays (cosine in the SVD topic space) of
  each N essay, pooled (122 unique C essays), against all N.
- **C6c register**: C essays in the top quartile of a register propensity
  score (L2 logistic regression N vs C, C = .05, on fine POS-tag shares,
  first- and second-person pronoun rates and sentences per token; no lexical
  frequency), against all N.
- **Length**: the C0 and C5 contrasts restricted to each essay's first 250
  scored tokens (N essays are ~410 words, C ~260, and C2 depends on
  position). Sensitivity only.

**Statistic.** N − C in the mean of the per-essay outcome, with a 95%
percentile bootstrap CI (2,000 resamples of essays within group, seed 42).
Also the share of the C0 gap retained. Learner levels are reported alongside
for C0–C5 (A, B) as context, not tested.

**Reading rule** (on `rate`, `own` set).
- N > C **survives** if the C5 CI lies above 0 **and** at least two of the
  three C6 contrasts under C5 have CIs above 0.
- It **attenuates** if it survives but the C5 point estimate retains less
  than half of the C0 gap (< +12.5 per 1,000).
- It is **explained** if the C5 CI includes 0 or lies below it, or if fewer
  than two C6 × C5 contrasts have CIs above 0.
- Single controls (C1–C4) are read descriptively: the share of the gap each
  one removes.

**Mean surprisal.** No direction is predicted. N and C *differ* in mean
surprisal under a control if the CI excludes 0. Reported for the `all` set
(the user's request) and the `own` set.

**Description of the residual N excess** (descriptive only). Among high
tokens kept under C5 in the `own` set: POS shares and Dolma frequency bands
(< 1, 1–10, 10–100, ≥ 100 per million) for N vs C, and the difference in
rate per 1,000 kept tokens per category (which categories carry the excess).
A KWIC sample of 40 N and 40 C residual high tokens (seed 42), labelled by me,
written to the data directory only (excerpts under 100 words; never in this
file or the code).

---
## Method as run

1. `1_features.py` (CPU, 25 s). Token flags for C1–C4 on the corrected
   essays (spaCy parse from `docbins_corrected/`), Dolma unigram counts for
   every lowercased form, topic clusters, genre flags, topic similarity and the
   register propensity. No surprisal is read by level.
2. `2_contrasts.py` (CPU, 1 min). The pre-declared contrasts and the
   reading rule.
3. `3_describe.py` (CPU). The residual description, the KWIC sample, and two
   post hoc checks (N/C ratios, C1 vs C2 sublevels).

**Check.** C0 reproduces D22 exactly: `own` N − C = +25.0 [16.0, 33.9],
`all` +26.3 [17.3, 35.2].

**How much each control removes** (share of tokens excluded, `own` set):

| | A | B | C | N |
|---|---|---|---|---|
| C1 names, numbers | .036 | .037 | .038 | .039 |
| C2 lemma used earlier | .41 | .44 | .45 | .50 |
| C3 topic words | .20 | .19 | .18 | .18 |
| C4 sentence/paragraph start | .052 | .053 | .051 | .043 |
| C5 all | .55 | .57 | .57 | .59 |

C2 removes almost half the tokens, because it counts any repeated lemma,
including function words. C3 cluster keyness against Dolma also flags
everyday learner vocabulary (*shopping*, *go*, *find* in a shopping cluster),
so C3 removes slightly more from learners. N's topics fall mostly into three
mixed clusters (20, 21, 29), so N's topic words are caught mainly by the
essay rule (C3 cluster share .025 for N vs .062 for C).

## Results

### High-token rate, `own` set (per 1,000; 95% essay-bootstrap CIs)

`rate` counts high tokens per 1,000 *kept* tokens, and `rate_num` per 1,000
*of all* own tokens. N/C is the ratio of the group means (post hoc, added to
make `rate` and `rate_num` comparable, since they are on different scales once
half the tokens are removed).

| control | N | C | N − C `rate` | retained | N − C `rate_num` | N/C `rate` | N/C `rate_num` |
|---|---|---|---|---|---|---|---|
| C0 baseline | 114.4 | 89.3 | **+25.0** [16.0, 33.9] | 1.00 | +25.0 [16.0, 33.9] | 1.28 | 1.28 |
| C1 names, numbers | 106.8 | 81.0 | +25.8 [17.3, 34.5] | 1.03 | +24.9 [17.1, 33.1] | 1.32 | 1.32 |
| C2 repetition | 149.2 | 115.3 | +33.9 [23.4, 44.5] | 1.35 | +10.0 [3.7, 16.2] | 1.29 | 1.15 |
| C3 topic words | 90.9 | 69.6 | +21.3 [14.1, 28.5] | 0.85 | +16.6 [10.7, 22.9] | 1.31 | 1.29 |
| C4 sentence/para starts | 107.6 | 85.8 | +21.8 [12.6, 30.7] | 0.87 | +21.6 [12.8, 30.4] | 1.25 | 1.27 |
| **C5 all** | 134.9 | 98.1 | **+36.8 [26.1, 48.4]** | 1.47 | +12.0 [6.5, 17.9] | 1.38 | 1.27 |

Learner levels for context (`rate`, own): C0 A 81 / B 86 / C 89; C5 A 77 /
B 90 / C 98. The learner trend steepens a little under C5, because A's
excluded tokens (repeated and topic words) are relatively more often above τ.

### Matched subsets (`own` set)

| subset | n N / n C | control | N − C `rate` | N − C `rate_num` | N/C `rate` |
|---|---|---|---|---|---|
| C6a genre (expository, both groups) | 47 / 322 | C0 | +19.8 [9.9, 30.0] | same | 1.21 |
| | | **C5** | **+33.1 [21.0, 46.0]** | +9.4 [3.4, 16.1] | 1.32 |
| C6b topic (5 nearest C per N essay) | 50 / 122 | C0 | +21.6 [11.4, 32.1] | same | 1.23 |
| | | **C5** | **+28.3 [15.1, 41.8]** | +8.3 [1.7, 15.3] | 1.26 |
| C6c register (top-quartile propensity C) | 50 / 193 | C0 | +9.3 [−0.9, 19.7] | same | 1.09 |
| | | **C5** | **+21.3 [8.4, 33.8]** | +4.0 [−2.5, 10.7] | 1.19 |
| first 250 tokens of each essay | 50 / 770 | C0 | +22.5 [12.4, 32.2] | same | 1.25 |
| | | C5 | +28.7 [17.1, 40.0] | +12.8 [6.5, 19.1] | 1.30 |

The `all` set (whole corrected essay) gives the same picture: C0 +26.3
[17.3, 35.2], C5 +36.7 [26.2, 48.2], register-matched C0 +10.7 [0.8, 21.1]
and C5 +21.4 [9.1, 33.9] (`tables/contrasts.csv`).

**What the register-matched C essays are** (post hoc description). The top
quartile of the propensity score is impersonal, expository prose. Personal
pronouns make up 3.4% of tokens (7.7% in the other C essays, 3.0% in N),
sentences average 26.5 tokens (21.3, N 24.7), and there are more past
participles and nouns. 82% pass the genre filter (28% of the other C
essays). C1 and C2 essays are equally represented (25% each).

### Mean surprisal per essay (bits)

| control | `all` N | `all` C | N − C `all` | N − C `own` |
|---|---|---|---|---|
| C0 | 5.67 | 5.29 | **+0.38 [0.26, 0.50]** | +0.38 [0.26, 0.50] |
| C1 | 5.55 | 5.17 | +0.38 [0.27, 0.50] | +0.38 [0.27, 0.50] |
| C2 | 6.39 | 5.92 | +0.48 [0.35, 0.62] | +0.51 [0.37, 0.65] |
| C3 | 5.17 | 4.88 | +0.29 [0.18, 0.39] | +0.28 [0.18, 0.39] |
| C4 | 5.52 | 5.16 | +0.36 [0.24, 0.48] | +0.35 [0.23, 0.48] |
| C5 | 6.06 | 5.53 | **+0.54 [0.39, 0.68]** | +0.56 [0.41, 0.71] |
| C6a genre, C0 / C5 | | | +0.31 [0.18, 0.45] / +0.46 [0.30, 0.63] | +0.31 / +0.51 |
| C6b topic, C0 / C5 | | | +0.33 [0.19, 0.48] / +0.35 [0.18, 0.54] | +0.33 / +0.40 |
| C6c register, C0 / C5 | | | +0.19 [0.06, 0.33] / +0.29 [0.12, 0.46] | +0.19 / +0.32 |
| first 250, C0 / C5 | | | +0.33 [0.20, 0.46] / +0.41 [0.24, 0.56] | +0.34 / +0.43 |

By the rule, N and C **differ** in mean surprisal under every control (all
CIs exclude 0, N higher). Among learners the mean is flat: `all` set A 5.33 /
B 5.32 / C 5.29 bits (C5: 5.30 / 5.47 / 5.53).

### Reading rule

C5 `rate` N − C = +36.8 [26.1, 48.4], and all three C6 × C5 contrasts have
CIs above 0 (genre +33.1, topic +28.3, register +21.3). The C5 estimate is
above half the C0 gap. **Verdict by the pre-declared rule: N > C survives.**

### What the residual N excess is (`own` tokens kept under C5)

High tokens per 1,000 kept tokens, pooled within group (N 136.8, C 103.1):

| POS | C | N | excess | P(high) C | P(high) N |
|---|---|---|---|---|---|
| NOUN | 33.8 | 48.6 | +14.8 | .17 | .22 |
| VERB | 28.5 | 34.7 | +6.3 | .20 | .23 |
| ADJ | 22.4 | 28.6 | +6.2 | .21 | .27 |
| ADV | 8.7 | 13.7 | +5.0 | .15 | .20 |
| PUNCT | 3.5 | 4.9 | +1.4 | .015 | .021 |
| function words | 5.8 | 5.7 | 0.0 | .023 | .027 |

| Dolma frequency (per M) | C | N | excess | share of kept C / N | P(high) C | P(high) N |
|---|---|---|---|---|---|---|
| < 1 | 12.9 | 18.4 | +5.6 | .024 / .031 | .53 | .59 |
| 1–10 | 26.2 | 37.0 | +10.8 | .076 / .086 | .35 | .43 |
| 10–100 | 41.6 | 50.8 | +9.3 | .22 / .23 | .19 | .22 |
| ≥ 100 | 22.5 | 30.6 | +8.1 | .68 / .65 | .033 | .047 |

- **The excess is in content words, and none of it is in function words.**
  Nouns carry 44% of it.
- **It is only partly a vocabulary-frequency effect.** N uses somewhat
  rarer words (35% of kept tokens below 100 per M, against 32%), but
  reweighting N to C's frequency-band mix removes only 32% of the residual
  gap (23% for the POS mix). Within every band, a word written by an N writer
  is more often above τ. Even frequent words (≥ 100 per M) are above τ 4.7%
  of the time in N and 3.3% in C. The median Dolma frequency of residual high
  tokens is the same (N 18, C 20 per M).
- **KWIC labels** (40 N, 40 C residual high tokens, seed 42, my own labels;
  excerpts in `kwic_residual.tsv`, labels in `kwic_labels.tsv`, local only):

  | label | N | C |
  |---|---|---|
  | precise or literate lexical choice | 22 | 19 |
  | syntax the 8-subword window cannot anticipate (coordinated predicate after long material, fronting, mid-clause stance adverb) | 10 | 4 |
  | figurative or idiomatic phrase | 4 | 3 |
  | topic word C3 missed | 3 | 2 |
  | genre-specific (narrative, headline, colloquial, CV) | 0 | 6 |
  | residual error or awkward usage the annotator left | 1 | 6 |

  N's residue is literate vocabulary (*pervades*, *confer*, *autonomy*,
  *formidable*) and complex syntax, such as a second verb phrase coordinated
  across an intervening clause or a fronted predicate (*Dramatic it may
  be*). C's residue has as many lexical choices, but less syntax, and some
  of it is narrative or review register and uncorrected non-native usage
  (about 1 in 7).

### Post hoc

- **C2 (CEFR) vs N.** C2 essays (287) have `own` rate 93.2 (C1 87.1) at C0
  and 102.8 (C1 95.3) at C5. N (114.4 / 134.9) is above the most proficient
  learners too.
- **C2 (repetition) moves the two outcomes in opposite directions.** N
  repeats more lemmas (50% of own tokens against 45%), and repeated tokens are
  mostly low-surprisal, so `rate` among first mentions widens the gap
  (+33.9). On `rate_num`, 15 of the 25 per 1,000 are high tokens whose lemma
  had already appeared: the N excess is not mainly the first mention of a new
  word.

## Verdict

1. **N > C survives the pre-declared controls.** With names, numbers,
   repeated lemmas, topic words and sentence or paragraph starts all removed,
   N − C is +36.8 [26.1, 48.4] per 1,000 kept own tokens (N/C 1.38, against
   1.28 at baseline). It survives within expository essays (+33.1), within
   topic-matched essays (+28.3) and against register-matched C essays
   (+21.3). On the stricter numerator-only scale, C5 keeps 48% of the gap,
   +12.0 [6.5, 17.9], about what removing 59% of N's tokens would leave if
   the excess were spread evenly (N/C 1.27 vs 1.28).
2. **Register is the control that matters.** Against the quarter of C essays
   written in impersonal expository prose, the baseline gap falls from +25 to
   +9.3 [−0.9, 19.7]. The numerator-only C5 gap falls to +4.0 [−2.5, 10.7].
   Topic and genre matching reduce the gap by only 15–20%.
   Part of the N–C difference is that C writers usually write different kinds
   of texts. Whether this is a confound or part of the construct is open:
   register matching selects learners who write in the style whose surprisal
   is at issue (longer sentences, nominal and passive constructions).
3. **Mean surprisal agrees.** N is +0.38 bits [0.26, 0.50] above C in the
   whole corrected essay, +0.54 [0.39, 0.68] under C5, and +0.19 [0.06,
   0.33] against register-matched C. Learner means are flat (5.33 → 5.29).
4. **The residual is spread across content words and frequency bands,**
   from literate vocabulary and syntax that a local window cannot anticipate.
   Neither names, topic words, sentence starts nor rare-word use explains it.

**Recommendation.** State N > C as robust to topic, names, repetition and
sentence position, and report the register-matched contrast next to it, with
the "partly register" reading. Do not present the N − C gap as a pure
learner-vs-native lexical difference.

## Caveats

- **50 LOCNESS essays**, from a handful of prompts (Europe, boxing,
  computers and inventions, transport, BSE, *Candide*). The CIs resample
  essays, not prompts. With essays from the same prompt correlated, the
  real uncertainty is larger.
- **One annotator**, minimal corrections. Any errors left in the corrected
  essays are more common in C (about 6 in 40 residual C tokens against 1 in
  40 for N), which works *against* N > C, so this caveat does not explain the
  gap.
- **No prompt metadata.** The genre, topic and register subsets are built
  from the texts with heuristics, not the real W&I prompts. Probably no C
  essay shares an N prompt. The topic neighbours are only loosely similar in
  vocabulary (mean cosine to the N essays .34, against .25 for other C essays
  and about .41 between N essays).
- **The register propensity uses features that may be part of the
  construct** (sentence length, nominal style). Over-matching would bias the
  gap toward 0, so C6c is a conservative bound.
- **C3 depends on choices** (10× Dolma, ≥ 2 uses, k = 30). Cluster keyness
  against Dolma also removes everyday learner vocabulary, more often in
  learner essays.
- **`rate` vs `rate_num`.** Under heavy exclusion, the two scales give
  different absolute gaps. The ratios (1.27–1.38) agree.
- **Post hoc**: the N/C ratios, the C1/C2 split, the register-subset
  description and the band reweighting were added after the contrasts were
  seen.
- **Window.** Surprisal is from the 8-subword window of stage 1. The
  syntactic share of N's residue (coordination across a clause) is plausibly
  partly a short-window artefact, and a longer context would score it lower.

## Costs

CPU only, about 2 min in all. No GPU. Disk: 15 MB in
`data/study3/pilot_native_c/`.

## Files

- Code: `src/3-surprisal-decomposition/pilot_native_c/{1_features.py, 2_contrasts.py, 3_describe.py}`
- Data (not versioned), `data/study3/pilot_native_c/`:
  - `tokens.parquet` (post-correction tokens with control flags),
    `essays.parquet` (clusters, genre, register, subsets),
    `cluster_topic_words.csv`
  - `kwic_residual.tsv`, `kwic_labels.tsv` (excerpts, local only)
  - `tables/`: `contrasts.csv` (all N − C contrasts, both sets and three
    outcomes, with N/C ratios), `levels.csv`, `residual_pos.csv`,
    `residual_freqband.csv`, `verdict.json`
