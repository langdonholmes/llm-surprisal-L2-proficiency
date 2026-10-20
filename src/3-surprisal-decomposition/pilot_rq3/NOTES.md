# Pilot: do surprisal components predict proficiency beyond error counts? (RQ3)

RQ3 asks whether component profiles add diagnostic information beyond aggregate
surprisal. Comparing components with mean surprisal is close to circular,
because the components are built from the gold corrections. Here the baseline
is what a human annotation gives you anyway: ERRANT error counts by type. The
question is whether the parts of surprisal that an error count does not contain
(bits per error at the site, spill per error, and the clean part) predict
proficiency once the counts are known.

Everything runs locally on the D28 per-unit split
(`data/study3/pilot_types/units_{wi,fce}.parquet`) and the first annotator's
edit tables. No model is run, and no learner text is read or written here.
Tokens the annotator left alone are *uncorrected*.

## Design (written before any model was fit)

**Units and outcome.** W&I+LOCNESS: 3,300 learner essays (A 1,430, B 1,100,
C 770), level A = 1, B = 2, C = 3, treated as linear. LOCNESS is not used.
FCE: 1,237 scripts (both answers pooled, as in D25), outcome the script score
`script-s` (7–40 here; the one score-0 script is excluded).

**Features** (per unit, n = scored tokens, groups g = the nine D28 ERRANT
groups, "unmapped" folded into "other"):

- **B, error counts (annotation only):** `rate_g` = edits of group g / n
  (9), `rate_flagged` = detection-only spans / n, and log n. 11 features.
- **S full (surprisal):** `clean` (mean surprisal of the writer's own words in
  the corrected essay); `cost_g` = at-site bits per error =
  at_site_g · n / count_g (9); `spill_g` = spill bits per error (9);
  `cost_flagged` = excess bits per flagged span. 20 features.
- **S compact:** `clean`, `cost_mean` = Σ_g at_site_g · n / Σ_g count_g
  (pooled at-site bits per typed error), `spill_mean` (same for spill).
- A per-error term is undefined when the unit has no error of that kind
  (e.g. 35% of W&I essays have no spelling edit, 75% no word-order edit). It is
  imputed with the training-fold mean. No absence indicator is added: absence
  is already in B as a rate of 0. In S-alone models this means S does not see
  absence, which is the intended "surprisal only" reading.

The per-error terms are ratios of a surprisal sum to a count, so they are not
mechanically a rescaled count. They can still correlate with the counts (for
example, if a writer with many errors makes cheaper ones), and B + S tests what
remains after the counts are in the model.

**Models.** B; B + S compact; B + S full. For context: S compact alone, S full
alone, mean surprisal alone, and B + mean surprisal.

**Learner.** Mean imputation → standardisation → ridge regression, all fit
inside the training fold. Alpha from {1e-3, …, 1e5} (9 values, log-spaced) by
inner 5-fold CV on the training fold. Outer CV: 10 repeats of 10-fold CV,
stratified by level (W&I) or by the D25 score band 1–23 / 24–27 / 28–31 /
32–40 (FCE). All models use the same folds, so the comparison is paired.
Seed 42.

**Metrics.** Per repeat, out-of-fold R² (1 − SSE/SST over all units) and the
Spearman ρ of the out-of-fold predictions with the outcome. Reported as the
mean over the 10 repeats with the min–max range.

**Increment.** Δ = R²(B + S) − R²(B). Point estimate: mean over repeats. CI:
2,000 bootstrap resamples of units. In each resample, ΔR² is recomputed from
the stored out-of-fold predictions in every repeat and averaged over repeats
(this carries the sampling of units; the repeat range carries the fold
assignment). Also reported: the share of repeats with Δ > 0, and Δρ.

**Reading rule.** Surprisal adds diagnostic information beyond error counts if
Δ = R²(B + S compact) − R²(B) has a 95% CI that excludes 0 on W&I, and on FCE
has the same sign with a CI that also excludes 0. The full-S version is
reported alongside under the same rule, but the verdict is the compact one.

**Which S terms.** Only if there is an increment: drop-one-term (for compact:
each of clean, cost_mean, spill_mean dropped from B + S compact; for full: the
clean term, the cost block, the spill block, cost_flagged), and standardized
ridge coefficients refit on all units with alpha by 5-fold CV.

**Reconciliation (FCE, for the D24/D25 write-up).** Spearman ρ of the clean
part with script score, for mean surprisal and for the high-token rate (D24
split with s replaced by the indicator s > τ), and the partial ρ controlling
(a) the gold error rate and (b) the correction part of surprisal (omission +
context + error tokens = total − clean, same measure). Partial = Pearson
correlation of the residuals of rank(clean) and rank(score) after a linear
regression on rank(control). 2,000 script-bootstrap resamples, seed 42.
"Gold error rate" is the one used for the W&I numbers: 5_gold.py's
`err_rate`, the share of tokens that are above τ *and* corrected (ERR_TOK or
ERR_INS). The step reproduces the W&I values first to confirm the definitions.
As a sensitivity, the plain share of corrected tokens (ERR_TOK or ERR_INS,
any surprisal) is also reported.

## Results

Everything below was computed once with the design above. Nothing was changed
after the results were seen.

### 1. Out-of-fold fit (mean over 10 repeats; R² range over repeats)

| model | W&I R² | W&I ρ | FCE R² | FCE ρ |
|---|---|---|---|---|
| B (error counts) | .581 [.580, .582] | .822 | .481 [.478, .483] | .695 |
| B + S compact | .585 [.583, .586] | .823 | .496 [.494, .499] | .711 |
| B + S full | .582 [.581, .583] | .821 | .504 [.501, .508] | .716 |
| S compact alone | .034 | .203 | .034 | .214 |
| S full alone | .040 | .212 | .058 | .231 |
| mean surprisal alone | .183 | .438 | .329 | .579 |
| B + mean surprisal | .581 | .822 | .493 | .709 |

The chosen alpha was 100 in most folds (grid 1e-3 … 1e5), well inside the grid.

### 2. Increment over B

ΔR² mean over repeats, 95% unit-bootstrap CI, repeat range, share of repeats
with Δ > 0, and Δρ with its CI:

| | W&I ΔR² | W&I Δρ | FCE ΔR² | FCE Δρ |
|---|---|---|---|---|
| **B + S compact** | **+.0037 [.0007, .0068]**; repeats .0034–.0040; 10/10 > 0 | +.0007 [−.0012, .0025] | **+.0152 [.0056, .0250]**; .0143–.0160; 10/10 | +.0155 [.0075, .0246] |
| B + S full | +.0007 [−.0029, .0042]; .0002–.0012; 10/10 | −.0012 [−.0032, .0013] | +.0230 [.0087, .0382]; .0202–.0254; 10/10 | +.0201 [.0088, .0325] |
| B + mean surprisal (context) | +.0000 [−.0008, .0008]; 5/10 | .0000 | +.0119 [.0038, .0210]; 10/10 | +.0137 |

**Verdict under the rule: met for S compact, narrowly.** The W&I CI excludes 0
and the FCE CI excludes 0 on the same side. Under the same rule S full is not
met, because its W&I CI includes 0.

### 3. Which S terms carry it

Loss in R² when a term (or block) is dropped from B + S, with 95% CI:

| dropped | W&I | FCE |
|---|---|---|
| compact − clean | .0002 [−.0008, .0012] | .0056 [.0001, .0119] |
| compact − cost_mean | .0006 [−.0007, .0019] | .0058 [.0000, .0117] |
| compact − spill_mean | .0022 [.0000, .0045] | .0062 [−.0005, .0127] |
| full − clean | .0003 [−.0009, .0014] | .0045 [−.0005, .0103] |
| full − cost block (9 groups + flagged) | −.0009 [−.0028, .0009] | .0142 [.0035, .0252] |
| full − spill block | .0004 [−.0019, .0029] | .0058 [−.0032, .0145] |
| full − cost_flagged | .0001 [−.0008, .0009] | .0104 [.0021, .0199] |

Standardized ridge coefficients of the compact terms, refit on all units (the
outcome is in level or score units per SD of the feature):

| | W&I (level) | FCE (score points) |
|---|---|---|
| clean | +.019 | −.41 |
| cost_mean | −.025 | −.45 |
| spill_mean | −.039 | −.44 |

For scale, the largest B coefficients are log n (+.32 W&I, +.99 FCE) and the
verb-form rate (−.12, −1.08). In S full on FCE the largest S terms are
cost_flagged (−.55), clean (−.38), cost_other and spill_other (−.35, −.36).
In-sample R² of each compact S term on the 11 B features: clean .05 / .04,
cost_mean .14 / .18, spill_mean .06 / .04 (W&I / FCE). So the S terms are
mostly not a rescaling of the counts. They also carry little proficiency
signal on their own (S alone R² .03–.06).

### 4. Reconciliation: the clean part on FCE

Spearman ρ of the clean part with the criterion, and partial ρ, 95%
unit-bootstrap CI. The W&I rows reproduce the coordinator's numbers exactly
(point estimates. CIs differ by at most .001).

| | W&I mean surprisal | FCE mean surprisal | W&I high-token rate | FCE high-token rate |
|---|---|---|---|---|
| ρ | .045 [.009, .079] | **−.078 [−.135, −.025]** | .124 [.089, .158] | **.032 [−.024, .085]** |
| partial \| gold error rate (`err_rate`) | .162 [.126, .196] | **−.065 [−.120, −.013]** | .195 [.160, .228] | **.015 [−.041, .067]** |
| partial \| correction part | .027 [−.011, .064] | **−.161 [−.216, −.110]** | .067 [.032, .102] | **−.082 [−.138, −.031]** |
| partial \| corrected-token share (sensitivity) | .118 [.083, .153] | −.076 [−.131, −.024] | .135 [.101, .167] | −.009 [−.068, .043] |

"Gold error rate" is 5_gold.py's `err_rate`: tokens above τ that the annotator
corrected (ERR_TOK or ERR_INS), divided by n. This is the definition that
reproduces .162 / .195 on W&I. The plain share of corrected tokens gives
.118 / .135. The coordinator's write-up should state which one it uses.

**FCE does not agree with W&I.** On W&I the clean part rises weakly with level,
and more so once the error rate is held fixed. On FCE the clean mean falls
weakly with score (ρ −.08), and holding the error rate fixed does not turn it
positive (−.07). Holding the correction part fixed makes it more negative
(−.16), where on W&I that partial is about 0. For the high-token rate, FCE is
flat (ρ .03, partial | error rate .02), not positive, and the partial on the
correction part is negative (−.08 vs +.07). The W&I pattern "once errors are
held fixed, better writers use slightly more surprising words" does not hold
on FCE. If anything, FCE shows the reverse.

## Reading

- **Surprisal does carry some information that error counts do not, but very
  little.** Error counts by type (plus length) explain most of what is
  predictable: R² .58 on W&I level and .48 on FCE score. The surprisal terms
  add 0.4 points of R² on W&I and 1.5 points on FCE (compact version). Both
  CIs exclude 0, so the rule is met, but on W&I it is only just met. The
  lower bound is .0007, Δρ is zero within its CI, and the full version with
  20 S terms adds nothing that survives (+.0007 [−.003, .004]).
- **The increment is not the same thing in the two corpora.** On W&I it comes
  mostly from spill per error (drop-one loss .0022). Essays whose errors
  disrupt the following words more are rated lower at the same error rate.
  On FCE all three compact terms share it about equally, and in the full
  model it sits in the at-site costs, especially the cost of flagged spans.
  The two per-error terms have the same sign in both corpora: at a fixed count, more surprising errors
  go with a lower score. The clean term has opposite signs (W&I +, FCE −),
  which matches the reconciliation.
- **On FCE, plain mean surprisal already adds most of it.** B + mean
  surprisal gains .012 (compact components .015, full .023). On W&I, mean
  surprisal adds nothing over the counts (.0000). The extra precision of the
  components shows only on W&I, where the increment is tiny.
- **For the chapter:** the "surprisal is an error counter" description fits
  the data at the level of proficiency prediction. Once ERRANT counts are
  known, surprisal components improve level or score prediction by about
  0.4–2 points of R². The surprisal-specific signal that exists is how
  costly the errors are (and on FCE the clean part, with a negative sign).
  The clean part, which is the only error-free part, is not a consistent
  proficiency signal: it is weakly positive on W&I and weakly negative on
  FCE.

## Caveats

- CEFR A/B/C is coarse and treated as linear. The FCE score is one examiner
  total over two answers.
- W&I has no prompt metadata, so topic is not controlled. FCE has prompts,
  but they are not controlled here either. The clean part is the term most
  exposed to topic, and its sign flip between corpora may partly come from
  this.
- The counts, the per-error costs and the "corrected" status all come from
  one annotator per corpus. Errors the annotator missed sit in the clean part.
  Edits that differ only in annotation style (e.g. W&I's punctuation
  corrections, about twice FCE's rate) change both B and S.
- Per-error costs are noisy for units with few errors of a type. They are
  mean-imputed when a type is absent (W&I: 35% of essays have no spelling
  edit, 75% none for word order). Ridge shrinks these terms. That may be why
  the full S underperforms the compact S on W&I.
- The CI is a unit bootstrap of fixed out-of-fold predictions. It does not
  include refitting variance, but the repeat range shows that is small
  (≤ .005).
- The spill terms inherit the 8-subword window of D27–D28.

## Files

Code (`src/3-surprisal-decomposition/pilot_rq3/`):

- `rq3_common.py`: paths, feature sets (B, S full, S compact), CV settings.
- `1_features.py`: per-unit features from `pilot_types/units_*.parquet` and
  the edit tables. Checks that no part is non-zero where its count is 0.
- `2_increment.py`: repeated CV, increments, drop-one, coefficients.
- `3_reconcile.py`: ρ and partial ρ of the clean part (W&I check, FCE).

Data (`data/study3/pilot_rq3/`, git-ignored): `features_{wi,fce}.parquet`,
`oof_{wi,fce}.npz` (out-of-fold predictions, all models and repeats), and
`tables/` (`models_*.csv`, `increment_*.csv`, `dropone_*.csv`, `coefs_*.csv`,
`results.json`, `reconcile.csv/.json`).

    python src/3-surprisal-decomposition/pilot_rq3/1_features.py
    OMP_NUM_THREADS=4 python src/3-surprisal-decomposition/pilot_rq3/2_increment.py   # ~5 min CPU
    python src/3-surprisal-decomposition/pilot_rq3/3_reconcile.py
