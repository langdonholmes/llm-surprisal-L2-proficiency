# FCE replication of the D24 split

Question: on W&I+LOCNESS, nearly all of the learner proficiency signal in essay
mean surprisal lies in the part that the gold corrections remove (D24). Does
this also hold on FCE?

Data: BEA-2019 FCE v2.1 (Yannakoudakis, Briscoe & Medlock 2011). The licence
allows non-commercial research only, excerpts under 100 words, and no
redistribution. This file and the code contain no excerpts. Data lives in
`data/fce/` and `data/study3/pilot_fce/`, both git-ignored.

**Terms.** A token that the annotator left alone is *uncorrected*. Tokens of the
corrected text are *post-correction*.

## Pre-specified design and reading rule

Written after ingest and correction, and before any surprisal had been
computed.

**Proficiency criterion.** The json keeps both scores the readme lists.
`script-s` is the overall score of a candidate's script (two answers), on a
0–40 scale. `answer-s` is a band.sub mark per answer (1.1 … 5.3, sometimes
with a trailing `T`). Within each band, the mean script score rises with the
sub-mark, so `.3` is the top of a band. The CEFR level is not in the release.

- **Primary unit: the script, scored by `script-s`.** This is the score that
  Yannakoudakis et al. (2011) model. Its unit is the script, so each script's
  mean surprisal and its four parts pool the scored tokens of both answers
  (each answer is scored as its own document, as in the corpus). One script
  with score 0 is excluded as implausible (n = 1,237 scripts). Seven scripts
  have only one answer and are kept.
- **Secondary: the answer, scored by `answer-s`.** The mark is converted to an
  ordinal: (band − 1) × 3 + sub, giving 1–15. The `T` suffix is ignored, and
  18 answers with malformed or non-numeric marks (`S`, `0`, `.2`, …) are
  dropped. The bootstrap resamples scripts, keeping both answers of a script
  together.
- All three splits (train, dev and test) are used, since nothing is fit. L1 (16
  languages) is noted only.

**Instrument.** This is identical to D22. OLMo-2 1B at revision a1847dff,
window of 8 subwords ending at the first subword, the `Predictor` path, batch
32, bf16. τ = 11.28 bits (`results/study3/threshold.json`), with a token
counted as high when s > τ, as in `5_decompose.py`. Whitespace tokens are
excluded. The gold categories (ERR_TOK / ERR_INS / POST / OK) come from
`2_select.gold_for_doc`. The first (and only) annotator's edits are used.
Null-replacement spans are flagged and are not own.

**Split.** This is `pilot_corrected/5_decompose.split`, unchanged:
total = clean + omission + context + error tokens. The same split is applied to
the high-token rate (per 1,000). For each part, the share is
cov(part, score) / cov(total, score), with 95% percentile CIs from 2,000
script-bootstrap resamples (seed 42). Spearman ρ of each part with the score is
reported with a bootstrap CI over the same resamples. For the score-band
tables, `script-s` is split into its quartile bands: 1–23, 24–27, 28–31 and
32–40.

**Reading rule.** This was fixed by the coordinator before any shares were
computed. The W&I result *replicates* if, on the primary analysis, for both
mean surprisal and the high-token rate:

- (a) the clean part's share of cov(total, score) has a CI that includes 0,
  *or* is small in magnitude (|share| < .15); and
- (b) the three correction parts (omission + context + error tokens) together
  account for at least 85% of cov(total, score).

The point estimate is used for (b). Because the parts sum exactly,
(b) is the same as clean share ≤ .15. A clean share below −.15, as on W&I for
the high-token rate (−.19), therefore meets (b) but fails the magnitude half
of (a) unless its CI includes 0. The verdict is reported per measure.

**Checks.** (1) The exact-sum assertion in `split`. (2) Alignment coverage,
and |Δ| ≈ 0 for aligned tokens whose 8-id window is identical in both
versions. (3) The json edit offsets against the M2 files. (4) FCE part sizes
next to W&I B-level part sizes.

---

## Method as run

0. `src/essays/ingest.py --fce` (CPU, 25 s) produced 2,469 answers from 1,238
   scripts (train 2,116, dev 159, test 194), parsed with en_core_web_lg (NER
   off), into `data/fce/docbins/`. `text_id` is `<script id>_<q>`. Each json
   edit has four fields (start, end, correction, FCE error code). The triples
   go in `meta["edits"]` and the codes in `meta["edit_types"]`. 16 L1s.
1. `1_correct.py` (CPU, 35 s). `apply_edits` applied 51,296 edits, and 1,962
   null-correction spans were flagged (53,258 in total, so none was lost). The
   json nests many correction edits inside a flagged detection span (376
   cases). The flagged span makes its tokens not own and the nested correction
   is applied, which is the same rule as on W&I. No overlapping corrections
   had to be skipped. The whitespace repairs were 3,917 swallowed spaces at
   deletions, 298 spaces removed before closing punctuation, 23 replacement
   spaces stripped and 6 spaces added at insertions. Gold categories come from
   `2_select.gold_for_doc`.
2. `2_score.py` (GPU 0, RTX A6000). Calls `pilot_corrected/2_score.py`'s
   `score` / `Recorder` / constants, and asserts revision a1847dff. Corrected:
   529,233 tokens in 14.2 min. Original: 531,284 in 14.1 min. **GPU cost
   28 min.**
3. `3_align.py` (CPU, 1 min). `pilot_corrected/3_analyze.align`, unchanged.
4. `4_decompose.py` (CPU, 25 s). `pilot_corrected/5_decompose.split`,
   unchanged, with continuous-score shares and a cluster bootstrap.

## Checks

- **Exact sum.** The `split` assertion (parts sum to the total to 1e-9) passed
  for both measures and both units.
- **Edit offsets against M2.** With whitespace removed, the original json texts
  are identical to the M2 source sentences in all three splits. The only
  difference is the M2 normalising 3 backtick apostrophes, which is undone
  before comparing. The corrected texts built from the json offsets match the
  M2 corrections (annotator 0) exactly for 86% of answers. Over the corpus,
  0.06–0.08% of corrected characters differ. Inspected on dev and test, the
  differences are in the M2 conversion: a word moved by an edit is placed
  differently (paired delete/insert), and edits that join two sentences
  (comma → full stop plus capital) are applied differently at the sentence
  boundary. No offset error was found.
- **Coverage.** Own tokens are 88.2% of the 524,163 scored original tokens
  (W&I: 82–95% by level). 99.99% of own tokens align, leaving 55 unmatched,
  almost all punctuation or abbreviations split differently at an edit
  boundary. Own and not-ERR_TOK disagree on 31 tokens. 87.4% of post-correction
  tokens have an original counterpart. Gold categories: OK 417,897 (uncorrected),
  ERR_TOK 61,893, POST 36,142, ERR_INS 8,231.
- **Identical windows.** 255,945 aligned tokens (55%) have the same 8-id window
  in both versions. 95.8% of them have Δ exactly 0, and 99.2% have |Δ| < .05
  bits, with a maximum of .45 bits. This is bf16 batch noise, as on W&I.
- No stage 1 FCE surprisal exists to reproduce. The instrument is the same code
  that reproduced stage 1 bit for bit on W&I (D22).

## Results (primary: script, `script-s`, n = 1,237)

Score = script total (higher is better). A positive share means the part moves
with the total. Shares sum to 1. 95% script-bootstrap CIs.

### Part means by score band

Mean surprisal (bits):

| score band | n | total | clean | omission | context | error tokens |
|---|---|---|---|---|---|---|
| 1–23 | 256 | 6.38 | 5.24 | .157 | .282 | .698 |
| 24–27 | 353 | 6.01 | 5.23 | .111 | .205 | .464 |
| 28–31 | 301 | 5.81 | 5.19 | .093 | .168 | .361 |
| 32–40 | 327 | 5.61 | 5.17 | .070 | .118 | .257 |
| all | 1,237 | 5.93 | 5.20 | .105 | .189 | .433 |

High tokens per 1,000:

| score band | total | clean | omission | context | error tokens |
|---|---|---|---|---|---|
| 1–23 | 142.7 | 73.5 | 11.3 | 12.4 | 45.6 |
| 24–27 | 120.7 | 74.5 | 7.7 | 9.5 | 29.1 |
| 28–31 | 109.9 | 73.4 | 6.4 | 7.3 | 22.8 |
| 32–40 | 101.2 | 74.7 | 4.9 | 5.2 | 16.4 |

### Share of cov(total, score), FCE vs W&I (D24, cov with A/B/C level)

| part | FCE mean | W&I mean | FCE high rate | W&I high rate |
|---|---|---|---|---|
| clean | 9% [3, 14] | −4% [−10, 1] | −4% [−11, 3] | −19% [−28, −11] |
| omission | 12% [10, 13] | 8% | 16% [14, 18] | 11% |
| context | 22% [20, 24] | 27% | 18% [15, 20] | 24% |
| error tokens | 58% [54, 63] | 69% | 70% [65, 76] | 85% |
| three corrections | 91% [86, 97] | 104% | 104% [97, 111] | 119% |

### Spearman ρ with the score [95% CI]

| | total | clean | omission | context | error tokens |
|---|---|---|---|---|---|
| mean | −.58 [−.62, −.54] | −.08 [−.14, −.03] | −.42 | −.58 | −.60 |
| high rate | −.50 [−.54, −.46] | +.03 [−.02, .09] | −.39 | −.34 | −.56 |

(W&I, ρ with level: total −.44, clean +.04, error tokens −.62, context −.52.)

### Secondary: answers, `answer-s` ordinal (n = 2,451 answers, 1,235 scripts)

The same picture. Mean: clean 10% [4, 15], corrections 90% [85, 96], ρ(clean)
−.07 [−.11, −.03]. High rate: clean −3% [−10, 4], corrections 103%, ρ(clean)
+.01 [−.03, .05].

### Level-free summaries (scripts)

- Mean surprisal 5.93 bits. Clean makes up 88% of the mean, omission 2%,
  context 3% and error tokens 7%. For the high-token rate (117 per 1,000),
  clean is 63% (74 per 1,000) and error tokens 23%.
- Spread: the SD of the total is .48 bits and the SD of clean is .31. The
  10–90% range is 5.38–6.53 for the total and 4.83–5.61 for clean. Clean and
  total correlate r = .59. Clean accounts for 38% of the variance of the total
  across scripts (as cov(clean, total) / var(total)), but almost none of its
  covariance with the score.
- L1 (descriptive): the mean clean part ranges from 5.14 (es, n = 200) to 5.36
  (ko, n = 86) among the six largest L1 groups, with similar mean scores.

### FCE vs W&I part sizes

W&I B (the closest level) against all of FCE. Mean, bits: total 5.92 vs
5.93, clean 5.29 vs 5.20, omission .079 vs .105, context .189 vs .189,
error tokens .368 vs .433. High rate per 1,000: total 122 vs 117, clean 86
vs 74, error tokens 24 vs 28. FCE has a slightly lower clean part, which
may reflect short exam answers on set tasks. It also has slightly more
correction mass: ERR_TOK is 11.8% of tokens against 10.5% at W&I B. ERR_INS is
rarer (1.6% vs 2.3%), but the omission part is larger, so each missing word
costs more. Overall the sizes are close to W&I B.
Full table: `tables/wi_vs_fce.csv`.

## Verdict

**Replicates, for both measures** (and also on the answer-level secondary).

- Mean surprisal: the clean share is +9% [3, 14]. Its CI excludes 0, but it is
  below .15 in magnitude, so (a) holds. The corrections account for 91%, so
  (b) holds.
- High-token rate: the clean share is −4% [−11, 3], with a CI that includes 0,
  and the corrections account for 104%.

Differences from W&I worth stating:

- For mean surprisal, the clean part *falls* slightly with score (ρ −.08),
  where on W&I it was flat (+.04). A small part of the FCE proficiency signal
  is in the writer's own words even after correction.
- The negative clean share for the high rate (W&I −19%) is weaker on FCE
  (−4%, n.s.).
- Error tokens carry less of the signal (58% vs 69%), and omissions carry
  more (12% vs 8%).

FCE learners span a narrower range (roughly B1–B2) than W&I A–C, but the
total still correlates −.58 with the score, so the test had power.
