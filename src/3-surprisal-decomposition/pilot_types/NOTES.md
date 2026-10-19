# Pilot: the D24 split by error type (W&I+LOCNESS and FCE)

Follow-up to D24, D25 and D27. D24 splits each essay's mean surprisal exactly
into clean, omission, context and error tokens. Here the correction parts are
split further, by the gold edit responsible, so that each ERRANT group gets an
at-site part and a spill part. Everything runs locally. No learner text leaves
the machine, and no excerpt is written here.

Terms as in D22. Tokens the annotator left alone are *uncorrected*. Tokens of
corrected essays are *post-correction*.

## Method

Per unit (W&I essay; FCE script with both answers pooled, as in D25), over its
n scored tokens, with O the writer's own aligned tokens and clean = mean_O(s_c):

    S_orig = clean + Σ_g (at_site_g + spill_g) + flagged + residual

- **at_site.** A replacement or unnecessary edit (json kind sub / del) gets
  Σ (s_o − clean) / n over the tokens it covers, as D24's error tokens. A
  missing-word edit (json kind ins) gets Δ = s_o − s_c of the own word right
  after the gap, as D24's omission. Two insertions at one gap share it
  equally (rare).
- **spill.** Every other own aligned token whose 8-subword window holds a gold
  edit is split among those edits by edit-level Shapley with the correction as
  baseline (the D27 method, `pilot_shapley/6_edit_shapley.py`), now on every
  token instead of a sample. Each edit's credit goes to its group.
- **flagged.** Tokens inside a detection-only span (null replacement). The
  annotator marked an error but gave no correction, so there is no ERRANT type
  and nothing to rescore. D24 counts these as error tokens. Here they get a
  row of their own.
- **residual.** Δ of own tokens with no gold edit in the window (instrument
  noise), and the excess of the few own tokens that do not align.

Groups are those of `pilot_shapley/common.py`, with "unmapped" (1.2% of W&I
edits, 0.7% of FCE) folded into "other". Second cut: the ERRANT operation
(M / R / U). An untyped edit takes its operation from the json kind. The
operation cut uses the ERRANT label, but the at-site mechanics use the json
kind. They disagree for about 5% of edits. Most of these are json
substitutions that ERRANT types M (e.g. a word rewritten as two), whose at-site
part is their own tokens' excess.

**Endpoints.** The Shapley values split the fp32 Δ. The stored instrument is
bf16, so stored Δ and fp32 Δ differ (median 0.02 bits). The difference is
added equally to each player. This is the Shapley value of the same game with
its two endpoints taken from the stored scores, and it is what makes the split
exact.

**Windowing** as in D27, with three changes needed to cover every token. (1)
The re-tokenised segment runs to the end of the word *after* the target's
word. Ending at the target's own word re-tokenised subwords such as ".\n" and
moved 1.2% of v(none) by more than 0.5 bits. After the change it is 0.02%. (2)
Tokens in the first 8 subwords of an essay get the shorter window the
instrument uses there. (3) A spaCy token that starts inside a subword is scored
at that subword. Targets with more than 7 players are skipped. Their Δ is split
equally among the players (W&I: 5 targets).

**Statistics.** Share of cov(total, level) among learners (W&I A = 1, B = 2,
C = 3; FCE script score 1–40, one score-0 script excluded). The parts add up,
so the shares add up exactly to 1. 2,000 essay (script) bootstrap resamples,
seed 42. Rate × cost (section 3) is token-pooled per level, so that pooled
part = rate × cost exactly. The low − high gap splits into a rate term and a
cost term by the midpoint rule. CIs come from 500 resamples within level.

## Checks

| | W&I | FCE |
|---|---|---|
| units (learners) | 3,300 (+50 N) | 1,237 scripts |
| exact sum per unit | holds (< 1e-9) | holds |
| own aligned tokens with an edit in the window, Shapley run | 274,923 | 203,288 |
| gap words (ERR_INS) | 15,141 | 8,086 |
| gap words vs gold ERR_INS | 14,836 agree; 28 ERR_INS have no non-null gap insertion; 305 tokens after a gap insertion are OK/POST in gold | 8,071 agree; 98 / 15 |
| v(none) vs stored s_o, median abs / share > 0.5 bits | 0.020 / 0.02% | 0.021 / 0.07% |
| v(all) vs stored s_c (own), median abs / share > 0.5 | 0.019 / 0.14% | 0.019 / 0.03% |
| endpoint adjustment, share of spill | 2.9% | 0.02% |
| Δ of tokens with no edit in the window (to residual) | +1,434 bits over 341k tokens | +11 bits |
| residual share of cov | 0.3% | 0.0% |
| ERRANT typed (non-flagged edits) | 98.8% | 99.3% (all 2,469 answers align to M2) |

Per-level part means reproduce D24 (W&I total 6.36 / 5.92 / 5.56 / 5.94, clean
5.25 / 5.29 / 5.28 / 5.66). Σ at-site + flagged equals D24 omission + error
tokens (A: 0.80), and Σ spill equals D24 context (A: 0.30). The spill shares
sum to 27% (W&I) and 22% (FCE), which are D24's and D25's context shares.

**Error tokens downstream of an earlier error.** 61% of W&I error tokens (54%
FCE) have an earlier edit in their window. So do 55% (53%) of gap words. The
Shapley share of those earlier edits is *negative*: −7.1% of the W&I error
tokens' total excess (−1.8% FCE), and −0.4% (−0.9%) of gap-word Δ. An earlier
error makes the next error slightly *less* surprising. Moving that share to
the earlier edits' spill changes no group's share of the covariance by more
than 0.9 points (W&I verb form 12.2 → 13.1, the largest). The simple rule
(all to the token's own edit) is kept.

## 1. Means by level (bits)

W&I, at-site / spill:

| group | A | B | C | N |
|---|---|---|---|---|
| clean | 5.25 | 5.29 | 5.28 | 5.66 |
| spelling/orth | .18 / .03 | .09 / .02 | .03 / .01 | .06 / .02 |
| verb form/SVA/tense | .10 / .03 | .05 / .01 | .02 / .01 | .01 / .00 |
| determiner | .04 / .04 | .02 / .03 | .01 / .01 | .01 / .01 |
| preposition | .03 / .03 | .02 / .03 | .01 / .01 | .01 / .01 |
| punctuation | .06 / .01 | .05 / .01 | .03 / .00 | .04 / .01 |
| noun number | .03 / .01 | .02 / .01 | .01 / .00 | .00 / .00 |
| word order | .02 / .00 | .01 / .00 | .01 / .00 | .00 / .00 |
| other lexical | .13 / .06 | .09 / .04 | .04 / .02 | .03 / .01 |
| other | .16 / .09 | .08 / .05 | .03 / .02 | .04 / .02 |
| flagged | .06 | .02 | .01 | .01 |

FCE by script-score band (1–23 / 24–27 / 28–31 / 32–40), at-site / spill:
spelling .17/.03, .12/.02, .09/.02, .07/.01; verb .09/.02 → .03/.01; determiner
.09/.04 → .04/.02; preposition .05/.04 → .02/.02; other lexical .14/.06 →
.07/.03; other .17/.07 → .06/.03; flagged .08 → .01. Full tables:
`tables/means_{wi,fce}.csv` (also by operation).

## 2. Share of the proficiency signal (cov with level / score), %, 95% CI

| group | W&I at-site | W&I spill | W&I total | FCE at-site | FCE spill | FCE total |
|---|---|---|---|---|---|---|
| clean | | | −4 [−10, 1] | | | 9 [3, 14] |
| spelling/orth | 18 [17, 20] | 3 [3, 3] | 21 [19, 23] | 14 [12, 16] | 3 [2, 3] | 17 [14, 19] |
| verb form/SVA/tense | 10 [9, 11] | 2 [2, 3] | 12 [11, 13] | 8 [7, 9] | 2 [1, 2] | 10 [8, 11] |
| determiner | 3 [3, 4] | 3 [2, 3] | 6 [5, 7] | 8 [6, 9] | 3 [3, 4] | 11 [9, 13] |
| preposition | 2 [2, 3] | 3 [2, 3] | 5 [4, 6] | 3 [3, 4] | 3 [2, 3] | 6 [5, 7] |
| punctuation | 5 [4, 5] | 1 [1, 2] | 6 [5, 7] | 1 [1, 2] | 1 [0, 1] | 2 [1, 3] |
| noun number | 3 [3, 4] | 1 [1, 1] | 4 [4, 5] | 2 [2, 3] | 0 [0, 1] | 3 [2, 3] |
| word order | 1 [1, 1] | 0 [0, 0] | 1 [1, 2] | 1 [1, 1] | 0 [0, 0] | 1 [1, 2] |
| other lexical | 11 [10, 12] | 5 [4, 5] | 15 [14, 17] | 9 [8, 10] | 4 [3, 5] | 13 [11, 15] |
| other | 17 [16, 19] | 9 [8, 10] | 26 [24, 28] | 14 [12, 16] | 6 [5, 7] | 20 [18, 23] |
| flagged | 7 [6, 8] | | 7 [6, 8] | 9 [7, 11] | | 9 [7, 11] |
| residual | 0 | 0 | 0 | 0 | 0 | 0 |
| *by operation* | | | | | | |
| missing | 14 [12, 15] | 4 [3, 4] | 17 [16, 19] | 18 [16, 20] | 5 [4, 6] | 23 [21, 25] |
| replacement | 54 [51, 58] | 20 [19, 22] | 75 [70, 79] | 42 [38, 46] | 14 [13, 16] | 56 [51, 60] |
| unnecessary | 2 [2, 3] | 3 [2, 3] | 5 [4, 6] | 2 [1, 2] | 2 [2, 3] | 4 [3, 5] |

"other" is almost all ERRANT OTHER, i.e. multi-word rewrites (W&I 21 of the 26
points, FCE 16 of 20). PRON is 2.5 (3.0). Spearman with W&I level: total −.44,
spelling −.33, verb −.33, other lexical −.31, other −.42, determiner −.17,
preposition −.19, punctuation −.15, word order +.01.

## 3. Rate vs cost per error

Rate = edits per 100 tokens. Cost = bits per edit (at-site or spill). Gap =
A − C (W&I) or band 1–23 − band 32–40 (FCE). "Rate share" = share of the gap
due to the rate term. Δcost = high − low, bits.

| group | W&I rate A→C | W&I spill/err A→C, Δcost | W&I rate share at-site / spill | FCE spill/err low→high, Δcost | FCE rate share at-site / spill |
|---|---|---|---|---|---|
| spelling/orth | 2.41 → 0.37 | 1.24 → 2.04, +0.80 [0.49, 1.10] | 1.12 / 1.50 | 1.60 → 1.71, +0.12 [−0.19, 0.38] | 1.01 / 1.08 |
| verb form/SVA/tense | 2.03 → 0.51 | 1.35 → 1.14, −0.21 [−0.43, −0.01] | 0.83 / 0.88 | 1.21 → 1.10, −0.10 [−0.33, 0.13] | 0.83 / 0.91 |
| determiner | 1.79 → 0.59 | 2.08 → 2.46, +0.38 [0.14, 0.62] | 0.77 / 1.20 | 2.28 → 2.14, −0.14 [−0.42, 0.15] | 0.93 / 0.93 |
| preposition | 1.42 → 0.57 | 2.43 → 2.01, −0.42 [−0.70, −0.12] | 0.74 / 0.82 | 2.30 → 2.05, −0.26 [−0.51, 0.03] | 0.80 / 0.84 |
| punctuation | 2.57 → 1.15 | 0.56 → 0.28, −0.28 [−0.43, −0.13] | 0.89 / 0.53 | 1.03 → 0.96, −0.06 [−0.31, 0.19] | 1.15 / 0.89 |
| noun number | 0.66 → 0.17 | 1.62 → 1.04, −0.58 [−0.90, −0.25] | 0.96 / 0.73 | 1.02 → 1.06, +0.05 [−0.33, 0.44] | 0.96 / 1.05 |
| word order | 0.19 → 0.10 | 1.23 → 0.79, −0.44 [−0.95, 0.00] | 0.91 / 0.60 | 0.75 → 0.86, n.s. | 0.76 / (unstable) |
| other lexical | 2.31 → 0.78 | 2.71 → 2.64, −0.07 [−0.28, 0.17] | 0.96 / 0.97 | 2.53 → 2.30, −0.23 [−0.46, 0.01] | 0.89 / 0.87 |
| other | 3.35 → 0.80 | 2.76 → 2.45, −0.31 [−0.57, −0.06] | 0.84 / 0.91 | 2.55 → 2.18, −0.37 [−0.66, −0.12] | 0.80 / 0.83 |
| *replacement (op)* | 10.8 → 2.95 | 2.19 → 2.21, +0.02 [−0.10, 0.14] | 0.87 / 1.01 | 2.10 → 2.01, −0.09 [−0.23, 0.06] | 0.84 / 0.94 |
| *missing (op)* | 4.31 → 1.60 | 0.98 → 0.69, −0.29 [−0.42, −0.17] | 0.87 / 0.72 | 1.66 → 1.38, −0.29 [−0.45, −0.11] | 0.95 / 0.81 |

At-site cost per error, W&I A → C: spelling 7.1 → 8.3 bits (+1.2 [0.6, 1.7]),
verb 5.0 → 3.9 (−1.1), other 5.0 → 4.0 (−1.1), others within about 0.5 bits.
FCE: verb −0.9, other −1.0, spelling +0.1. Full tables:
`tables/rate_cost_{group,op}_{wi,fce}.csv`.

## 4. CEFR sublevels (W&I, learners)

Counts: A1.i 239, A1.ii 346, A2.i 365, A2.ii 480, B1.i 320, B1.ii 311, B2.i
291, B2.ii 178, C1.i 230, C1.ii 253, C2.i 171, C2.ii 109, C2+ 7. Spearman ρ with
the 13-step sublevel (with A/B/C): total −.46 (−.44), clean +.05 (+.04),
spelling −.35 (−.33), verb −.33 (−.33), other −.44 (−.42), other lexical −.31
(−.31), determiner −.17, preposition −.19, punctuation −.14, noun number −.14,
word order +.03, flagged −.21. The finer scale adds almost nothing. Total falls
steadily from 6.63 (A1.i) to 5.43 (C2.ii) while clean stays at 5.2–5.3.
`tables/sublevels_wi.csv`, `tables/sublevel_means_wi.csv`.

## Reading

- The proficiency signal in mean surprisal is spread over many error types.
  No single type dominates. In both corpora the largest parts are
  multi-word rewrites (ERRANT OTHER, 20–26%), spelling (17–21%) and other
  lexical choice (13–15%), then verb form (10–12%). Determiners are 6% in W&I
  and 11% in FCE. Punctuation, noun number and word order are small.
- By operation, replacements carry most of it (75% W&I, 56% FCE), missing
  words 17–23%, unnecessary words about 5%.
- Spill is a quarter of the signal and comes mostly from lexical, determiner,
  preposition and rewrite errors. Spelling, verb form and punctuation errors
  cost mostly at their own site. Prepositions and determiners spill about as
  much as they cost at the site.
- The rate does most of the work. For most groups, 75–110% of the low − high
  gap is how often the error occurs. The range is wider for W&I spill: 53%
  for punctuation, 150% for spelling. D27's "spill per error is flat" holds in
  aggregate (replacements +0.02 bits A → C) and holds group by group in FCE
  (every CI but "other" includes 0). In W&I it does not hold per group.
  Spelling and determiner errors spill more per error at C, while
  preposition, noun number, punctuation and rewrite errors spill less. These
  shifts cancel in the total. A plausible reading is that C learners' rarer
  misspellings are of rarer words, but that is untested.
- The FCE ranking matches W&I except that determiners matter more in FCE and
  punctuation less. Both are differences in annotation as much as in
  learners: W&I annotators correct punctuation about twice as often (2.6 vs 1.3 per 100 tokens at the lowest level).

## Caveats

- **Window.** Spill is measured inside the instrument's 8-subword window, so
  it cannot reach further than about 6 words. Spill shares are lower bounds
  for an instrument with a longer context.
- Spill lands only on the writer's own aligned tokens. An error's effect on a
  later *error* token is in that token's at-site part (simple rule). The
  check shows that this share is small and negative.
- "flagged" (7–9%) has no type and stays a separate row. A missing-word edit's
  at-site part is only the word after the gap. Its effect on later words is
  spill.
- ERRANT types come from automatic M2 typing and the json-to-M2 walk-back.
  The operation label and the json kind disagree for about 5% of edits.
  "other" mixes OTHER rewrites with pronouns, conjunctions and particles.
- One annotator per corpus. The W&I level is treated as interval (A = 1 … 3)
  and the FCE score as interval. The bootstrap resamples essays, not prompts.
- W&I has a small positive bias in Δ on tokens with no edit in the window
  (+0.004 bits per token), and a 2.9% endpoint adjustment to spill, both from
  bf16 scoring of the stored corrected essays. The first is the residual row
  (0.3% of the covariance). The second is spread equally over each target's
  players. Its effect on individual group shares was not measured, but it is
  2.9% of spill, which is itself 27% of the signal.

## GPU cost

Step 2, fp32, one RTX A6000 per corpus, run in parallel: W&I 1,184,644 windows
(700,697 targets, 332,180 with edits) in 25 min of GPU time plus 2 min of CPU
window building. FCE 797,226 windows (516,837 targets) in 17 min plus 2 min.
About 42 GPU-minutes in total, 27 min wall-clock time. A 60-essay test run took
under 1 min.

## Files

Code (`src/3-surprisal-decomposition/pilot_types/`):

- `types_common.py`: paths, corpora, groups and operations. It reuses
  `pilot_shapley/common.py`. The module is named this way because
  pilot_shapley's modules import `common`.
- `1_errant_fce.py`: ERRANT types for the FCE json edits (pilot_shapley's
  `1_errant.py` functions, unchanged).
- `2_shapley.py`: GPU. Edit-level Shapley for every own aligned token and every
  error token.
- `3_decompose.py`: CPU. The split, exact-sum assertions, tables.

Data (`data/study3/pilot_types/`, git-ignored): `edit_types_fce.parquet`,
`shapley_{wi,fce}.parquet` (+ `.time.json`), `units_{wi,fce}.parquet`
(per-unit parts), `logs/`, and `tables/` (`share_table.csv`,
`means_*.csv`, `rate_cost_*.csv`, `results_*.json` with checks, both rules,
Spearman and the "other" breakdown, `sublevel*_wi.csv`,
`errant_coverage_fce.csv`).

    python src/3-surprisal-decomposition/pilot_types/1_errant_fce.py
    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/pilot_types/2_shapley.py --corpus wi
    CUDA_VISIBLE_DEVICES=1 python src/3-surprisal-decomposition/pilot_types/2_shapley.py --corpus fce
    python src/3-surprisal-decomposition/pilot_types/3_decompose.py
