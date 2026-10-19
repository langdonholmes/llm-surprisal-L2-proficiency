# Pilot: corrected essays (removing errors from the context)

Follow-up to D21. Errors anywhere in the 8-subword window inflate the surprisal
of uncorrected tokens after them, and errors are densest at A, so the
uncorrected-surprise rates in `results/study3/gold_by_level.csv` (59 → 83 per
1,000, A → C) are contaminated. Here the gold corrections are applied to every
W&I+LOCNESS essay, the corrected essays are scored with the identical pipeline,
and tokens are aligned between the two versions.

**Terms.** A high token the annotator left alone is *uncorrected*, not
"acceptable". The class holds effective word choices, awkward but not wrong
phrasing (the corrections are minimal), errors the annotator missed, and names
or sentence starts. Tokens of the corrected essays are *post-correction*. (The
terms changed on the coordinator's instruction during the run; the plan below
did not change.)

## Pre-specified statistics and reading rules

Written after the corrected texts were built and before any surprisal of a
corrected essay had been seen.

**Units.** Scored tokens are non-space spaCy tokens scored by the stage 1 path
(OLMo-2 1B, 8 subwords ending at the first subword, `Predictor`, batch 32,
bf16). τ = 11.28 bits from `results/study3/threshold.json`. Essay-bootstrap
95% CIs: 2,000 resamples of essays within level, seed 42. Learner trends are
Spearman ρ with level (A < B < C) over essays with the bootstrap CI of
`lib/stats.boot_spearman`.

**Alignment.** An original token is *own* (written by the writer, untouched by
the annotator) when none of its characters lies in an applied or flagged edit
span. It is *aligned* when it is own and the corrected doc has a token that
starts at its mapped offset with identical text, and both are scored.
Insertions next to a token do not make it un-own.

**Q1 Contamination.** Set: original uncorrected tokens (stage 2 `gold == OK`)
above τ that are aligned. Δ = s_orig − s_corr. Per level: mean and median Δ,
share with s_corr < τ ("contaminated"), and the rest "intrinsically
surprising". Also the uncorrected rate decomposed per 1,000 original tokens
(contaminated + intrinsic = uncorrected rate restricted to aligned tokens).
Reading rule: contamination is *material* if the contaminated share is ≥ .10
at A; it is *level-dependent* if the A − C difference in the essay-level
contaminated share has a bootstrap CI excluding 0.

**Q2 Post-correction surprise.** Per essay, three rates per 1,000:

- `corr_all`  post-correction tokens above τ / scored post-correction tokens;
- `corr_own`  aligned own tokens above τ in the corrected essay / aligned own tokens;
- `orig_own`  the same aligned own tokens above τ in the original essay / the same denominator.

`orig_own` vs `corr_own` is the matched contrast (same tokens, only the
context differs). The reference is the contaminated uncorrected rate
(`acc_rate` in `gold_essays.csv`, called `uncorr_rate` here). Primary
statistic: the C − A difference in mean rate, for `corr_own` and for
`uncorr_rate`, and the difference of those differences with a paired essay
bootstrap (the same essays are resampled for both rates). Reading rule:
word-choice surprise rises *more steeply* once errors leave the context if
that difference-of-differences CI is above 0; *less steeply* if below 0;
*unchanged* if it includes 0. ρ with level is reported alongside. LOCNESS
*still separates* from C if N − C for `corr_own` has a CI excluding 0.

**Q3 Corrections.** For each applied substitution or insertion whose
corrected span contains a scored token, the surprisal of its first scored
token. Share above τ by level, and among the edits whose original token was
above τ. Sanity rule: corrections are "mostly unsurprising" if the share above
τ is below .15 (the base rate of post-correction tokens above τ is the
yardstick).

**Q4** Descriptive only: POS, Dolma frequency (edited-prose unigram, the
stage 4 attestation table), and whether the lowercased word occurs earlier in
the essay, for intrinsic vs contaminated uncorrected tokens by level. KWIC
excerpts for qualitative labels stay in the data directory.

**Checks.** (a) Original essays rescored here reproduce the stored stage 1
surprisal (`data/study3/wi/shards`) token for token. (b) Aligned tokens whose
8-id window is identical in both versions must have |Δ| ≈ 0. (c) Alignment
coverage and edge cases.

---

## Method as run

1. `1_correct.py` (CPU, 30 s). Applies the 73,137 first-annotator edits with a
   replacement. The 1,831 null-replacement spans are flagged, left as written
   and treated as not own. It repairs whitespace at edit junctions only:
   2,645 deletions between two spaces swallow one space (none when another edit
   starts there), 105 spaces left before closing punctuation are removed, 170
   leading or trailing replacement spaces are dropped next to existing
   whitespace, and 42 insertions that would glue two words get a space. The
   only overlaps are same-point insertions, applied in order. It then parses
   with en_core_web_lg (NER off), as `ingest.py` does.
2. `2_score.py` (GPU 0, A6000). The stage 1 `Predictor` path with the same
   model and revision, window 8, batch 32 and bf16. It keeps surprisal,
   entropy and the observed id, with no top-50. Corrected: 720,452 non-space
   tokens in 19.3 min. Original: 705,442 in 15.1 min.
3. `3_analyze.py` (CPU, 30 s). Checks, alignment and Q1–Q4 (pre-specified).
4. `4_posthoc.py` (CPU). The decomposition below (post hoc) and KWIC files.

## Checks

- **(a) Reproduction.** The original essays rescored here match the stored
  stage 1 surprisal for all 708,020 tokens, with a maximum difference of 0.0
  bits (bit-identical). No token crosses τ.
- **(b) Identical windows.** 348,348 aligned tokens have the same 8-id window
  in both versions. 95.4% have Δ exactly 0 and 99.3% have |Δ| < .05 bits. The
  maximum is .86 bits, and 20 tokens cross τ. This is bf16 noise from
  different batch neighbours, and from phase 1 (first 8 subwords) vs phase 2
  batching. It is negligible next to the effects below.
- **(c) Coverage.** Own tokens make up 82.1% (A), 89.5% (B), 95.4% (C) and
  96.6% (N) of the original scored tokens. Own and not-ERR_TOK coincide except
  for one token in a 150-essay A sample. 99.96% of own tokens align: only 284
  own tokens do not, almost all retokenization at an edit boundary (`.`, `"`,
  `it`, `notes(50`, contraction pieces). 99.93% of the original uncorrected
  high tokens align. 12.4% of post-correction tokens have no original
  counterpart (annotator-written or retokenized).
- **Edge case worth knowing.** A missing space after punctuation (`us.In`)
  lets the tokenizer merge the period with the next word, so the period's
  first subword is a rare merged id. 74% of the contaminated uncorrected
  punctuation tokens (416) are of this kind. The space insertion is gold-coded
  ERR_INS on the *next* word, so the inflated period counts as uncorrected.

## Results

### Q1 Contamination (uncorrected high tokens, aligned)

| level | n | mean Δ (bits) | Δ > 1 bit | contaminated (s_corr < τ) | same window | contaminated, window changed | mean Δ, window changed |
|---|---|---|---|---|---|---|---|
| A | 11,849 | 1.02 | .26 | **.19** | .39 | .31 | 1.67 |
| B | 18,748 | 0.57 | .17 | **.13** | .54 | .27 | 1.25 |
| C | 18,996 | 0.25 | .08 | **.06** | .75 | .23 | 0.97 |
| N | 2,479 | 0.19 | .07 | **.04** | .80 | .21 | 0.93 |

Per 1,000 original tokens, the uncorrected rate splits into a contaminated
part and an intrinsic part: A 10.6 + 48.8, B 8.9 + 62.6, C 4.9 + 77.7,
N 4.8 + 102.5. The essay-level contaminated share is C − A = −.13
[−.14, −.12], with ρ with level −.27. **Material and level-dependent** by the
rule (≥ .10 at A, CI excludes 0). Where the window did change, about a quarter
to a third of tokens drop below τ at every level. The level gradient comes
mostly from how often an error sits in the window (61% of A windows, 25% of C).

Correcting also *raises* some tokens above τ: induced rate A 7.7, C 3.7 per
1,000 (mostly bare nouns after a deleted article, see Q3).

### Q2 Post-correction surprise (per 1,000, 95% essay-bootstrap CIs)

| level | `uncorr_rate` (contaminated, stage 5) | `orig_own` | `corr_own` | `corr_all` |
|---|---|---|---|---|
| A | 59.5 [57.9, 61.1] | 107.5 [105.0, 109.9] | **81.2** [79.0, 83.4] | 87.4 [85.3, 89.5] |
| B | 71.6 [70.0, 73.2] | 100.8 [98.7, 103.0] | **85.5** [83.6, 87.5] | 88.6 [86.8, 90.6] |
| C | 82.6 [80.5, 84.8] | 95.7 [93.3, 98.1] | **89.3** [87.1, 91.6] | 90.0 [87.8, 92.2] |
| N | 107.5 [98.9, 116.8] | 121.3 [113.1, 130.4] | **114.4** [105.8, 123.5] | 116.3 [107.9, 125.5] |

| | C − A | ρ with level (learners) | N − C |
|---|---|---|---|
| `uncorr_rate` | +23.2 [20.4, 25.8] | .32 [.28, .35] | +24.9 [16.3, 33.7] |
| `corr_own` | +8.2 [5.0, 11.4] | .12 [.09, .16] | **+25.0 [16.0, 33.9]** |
| `corr_all` | +2.6 [−0.6, 5.6] | .06 [.02, .09] | +26.3 [17.3, 35.2] |
| annotator-written tokens above τ (`corr_new`, per post-correction token) | −17.2 | −.43 | +1.7 [−0.1, 3.4] |

**Primary (pre-specified).** (C − A) for `corr_own` minus (C − A) for
`uncorr_rate` = **−15.0 [−16.4, −13.6]**. By the reading rule, uncorrected
word-choice surprise rises **less steeply**, not more, once errors are out of
the context. **LOCNESS still separates from C**: +25.0 [16.0, 33.9] per 1,000
on `corr_own`, the same gap as before (N − C difference of differences
+0.2 [−1.8, 2.1]).

**Why the primary result runs against D21 (post hoc, `4_posthoc.py`).**
`uncorr_rate` and `corr_own` differ in three ways, not one. The C − A
difference moves as follows at each step, all on the same essays:

| step | rate | A | C | C − A | change in C − A |
|---|---|---|---|---|---|
| start | uncorrected high / all original tokens | 59.5 | 82.6 | 23.2 | |
| 1 denominator | uncorrected high / own tokens (errors out of the denominator) | 71.9 | 86.5 | 14.6 | −8.5 [−9.1, −8.0] |
| 2 context | same tokens, scored in the corrected essay | 65.3 | 84.3 | 19.0 | **+4.4 [3.4, 5.3]** |
| 3 adjacent tokens | + own tokens that were ERR_INS/POST, scored after correction | 81.2 | 89.3 | 8.2 | −10.8 [−11.7, −9.9] |

- **Step 2 is the contamination D21 pointed to.** On its own it does steepen
  the rise, but only by 4 per 1,000.
- **Steps 1 and 3 are bookkeeping in stage 5.** Its uncorrected rate keeps
  error tokens in the denominator and leaves out of the numerator the
  writer's own words next to an error (POST, and the word after a missing
  word). Both are concentrated at A, 17% of own tokens against 5% at C. After
  correction those adjacent words are above τ at roughly the rate of other
  own words (about 9–10%), so leaving them out deflated A.

Net effect: once errors are out of both the context and the counts, surprise
on the writer's own words rises only modestly from A to C (81 → 89, ρ .12).
Over the whole corrected essay it is flat (87 → 90, CI includes 0), because
corrections at A are themselves often surprising (annotator-written tokens
above τ: 23 per 1,000 at A, 6 at C).

### Q3 Corrections

| edit | n (with a scored token) | correction's first token above τ | original token above τ | correction above τ, when the original was |
|---|---|---|---|---|
| insertion | 16,879 | **.016** | .36 (word after the gap) | .023 |
| substitution | 47,824 | **.140** | .37 | .29 |
| deletion (next token) | 7,403 | .23 | .10 | .19 |

Base rate of post-correction tokens above τ: .091.

- **Insertions are almost never surprising.** By the sanity rule, corrections
  are "mostly unsurprising" (both insertions and substitutions are under .15).
- **Substitutions are 1.5× the base rate.** In the KWIC (`kwic_corrections.txt`),
  high corrected substitutions are rarer or longer forms (*salesman*,
  *psychopathic*, *palace*), sentence starts created by punctuation fixes,
  and rephrasings dropped into otherwise learner-written context. Rates are
  similar across levels (.135 A, .144 C, .21 N).
- **Deletions raise the next word: .10 → .23.** 58% of these cases are a
  deleted article. A bare noun phrase (*like [the] public transport*,
  *[the] taxis may be*) is surprising to the 8-token model even when correct
  (.37 above τ after an article deletion, .15 otherwise). This is the main
  source of the "induced" tokens in Q1.

### Q4 What the intrinsic uncorrected tokens are

| | median Dolma freq (per M) | rare (< 1 per M) | function word | punctuation | proper noun or number | earlier mention |
|---|---|---|---|---|---|---|
| contaminated, A / C | 83 / 65 | .03 / .04 | .12 / .09 | .09 / .07 | .05 / .07 | .30 / .31 |
| intrinsic, A / B / C / N | 32 / 28 / 26 / 27 | .10 / .11 / .12 / .12 | .07 / .07 / .06 / .08 | .02 | .14 / .13 / .12 / .11 | .23 / .25 / .26 / .32 |

- **Contaminated tokens look like ordinary words.** They are frequent words
  (and periods glued to the next word) made improbable by a broken neighbour.
- **Intrinsic tokens are the same kind at every level.** They are content
  words of moderate-to-low frequency, introducing a topic, plus names. The
  shifts with level are small and monotone: slightly rarer words (median 32
  → 26 per M, rare-and-attested share .019 → .030), fewer proper nouns (.14
  → .12), and more earlier mentions in LOCNESS (.32).
- **Qualitative labels.** From KWIC: 25 per level, my own labels, local file
  `kwic_uncorrected.txt`. The intrinsic class at every level mixes:
  - topic-introducing nouns (*documentaries*, *maintenance*, *pharmacy*);
  - names (*NVIDIA*, *Ghana*, *Reforma*);
  - sentence-initial discourse adverbs (*Secondly*, *Finally*, *Tomorrow*);
  - deliberate lexical choices (*topple*, *bone-chiseling wind*);
  - a residue the annotator did not correct, roughly 3–4 in 25 at A and 2–3
    at C (*Before I [want] to pay*, *the use [for] urban*, *contents and [be]
    able*).

  The labels suggest that level changes the vocabulary more than the kind of
  token.

## Verdict

1. **Contamination is real but small in aggregate.** About 19% of A's
   uncorrected high tokens (6% at C, 4% at N) fall below τ once the context is
   corrected. That is 10.6 per 1,000 tokens at A and 4.9 at C.
2. **The headline "uncorrected surprise rises 59 → 83" is mostly an artefact
   of how stage 5 counts.** Errors stay in the denominator and the writer's
   own words next to an error are excluded. Measured on the writer's own words
   in corrected essays, it rises 81 → 89 (C − A +8 [5, 11], ρ .12). The
   pre-specified contrast reads *less steep* (−15 [−16, −14]). Removing
   context contamination alone steepens the rise by +4 [3, 5].
3. **The native–C gap survives intact** (+25 [16, 34] per 1,000). Whatever
   separates LOCNESS is neither nearby errors nor the counting.
4. **Corrections are mostly unsurprising.** Inserted words are above τ 1.6% of
   the time and substitutions 14% (base rate 9%). Deleting an article leaves a
   bare noun phrase that is often above τ, a model property and not an error.

**Recommendation for the main analysis.** Report uncorrected surprise per
writer-own token in the corrected essays (`corr_own`) next to the stage 5
decomposition, or at least put error tokens out of the denominator and the
POST/ERR_INS words back in the numerator. The "acceptable surprise rises
steeply with level" sentence in the README and `results.qmd` overstates the
learner trend. The native > C claim is safe.

## Caveats

- **Minimal corrections.** The corrected essays keep awkward-but-not-wrong
  phrasing, so post-correction surprise still includes learner-specific
  phrasing (and some missed errors; see Q4). It is an upper bound on "clean"
  word-choice surprise, more so at A.
- **One annotator.** A second annotator would correct differently. Flagged
  spans (1,831) stay uncorrected.
- **Corrected essays differ in length.** They have more scored tokens (+3.4% A, +2.1% B, +1.1% C
  and N). Rates are per token, and `corr_own` uses the same denominator in
  both versions.
- **Window.** With 8 subwords, contamination is local by construction. A
  longer window would carry errors further, so the contamination share here
  is a lower bound for longer-context models.
- **Annotator text in context.** Corrections change the context of the next
  words in ways a writer would not (bare noun phrases after an article
  deletion, capitalized sentence starts). The induced tokens (3.7–7.7 per
  1,000) are partly this.
- **Post hoc.** The decomposition was done after the primary result. The
  pre-specified reading is "less steeply".

## Costs

GPU: 34 min on one A6000 (19.3 min corrected, 15.1 min original rescore; the
rescore could be skipped now that it is shown identical). CPU: about 2 min in
all. Disk: 86 MB in `data/study3/pilot_corrected/`.

## Files

- Code: `src/3-surprisal-decomposition/pilot_corrected/{common.py, 1_correct.py, 2_score.py, 3_analyze.py, 4_posthoc.py}`
- Data (not versioned), `data/study3/pilot_corrected/`:
  - `essays.parquet` (original and corrected text, char map)
  - `edits.parquet`, `flagged.parquet`
  - `docbins_corrected/`
  - `tokens_{orig,corr}.parquet`
  - `aligned.parquet`, `q3_edits.parquet`, `q4_tokens.parquet`
  - `kwic_uncorrected.txt`, `kwic_corrections.txt` (excerpts, local only)
  - `tables/`: `coverage.csv`, `q1_contamination.csv`, `q2_rates_by_level.csv`,
    `essays.csv`, `q3_corrections.csv`, `q4_kinds.csv`, `q4_pos.csv`,
    `q4_ttype.csv`, `posthoc_decomposition.csv`, `posthoc_tests.json`,
    `report.json`
