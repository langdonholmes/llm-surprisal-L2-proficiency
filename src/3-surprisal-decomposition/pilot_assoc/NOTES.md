# Pilot: is surprisal unique? Classic and LLM association measures before and after correction

Exploratory. Everything runs locally. No learner text, span or n-gram goes to
any external service. Dolma here is the local v1.7 sample (`data/dolma/`).
Tokens the annotator left alone are *uncorrected*. Tokens of corrected essays
are *post-correction*.

Background: D24/D25/D28 show that among learners, almost all of surprisal's
proficiency association sits in what the gold corrections remove. D31 shows
that surprisal adds only 0.4–1.5 points of R² beyond ERRANT error counts.

- **Q1.** Do classic phraseological and lexical measures show the same pattern?
- **Q2.** Do LLM-based association measures (surprisal recast as PMI or ΔP)
  keep a proficiency association after correction?
- **Q3.** How do surprisal and LLM-PMI relate to the classic measures, and how
  much of the surprisal signal sits in tokens the classic measures drop?

## Plan (written before any level or score analysis was computed)

### Units, criterion, versions

- **W&I+LOCNESS**: the 3,300 learner essays, level A = 1, B = 2, C = 3.
  LOCNESS is not used.
- **FCE**: the 1,237 scripts with score > 0, criterion `script-s`. Both answers
  are pooled per script (D25): token measures average over the script's
  tokens, and bigram/dependency measures average over the script's pairs.
- **Original** = the essay as written. **Corrected** = the essay with the first
  annotator's edits applied (`pilot_corrected/`, `pilot_fce/`).
  - Token-mean measures: the corrected value is the D24 *clean* part, i.e. the
    writer's own aligned words scored in the corrected essay.
  - Pair and type measures (dependency MI, bigram measures, frequency): the
    corrected value is the measure on the whole corrected essay, re-parsed by
    spaCy (`docbins_corrected`).

### Measures

All Dolma counts are lowercased spaCy tokens (3-gram tables, D-tables of
Study 2). N = total token count of the table. Unigram and bigram counts are
derived as in `lib/attestation.py` and are slight undercounts.

**Classic, Study 2** (exactly as `src/essays/metrics.py`, which Study 2 ran with
`--corpus both`, i.e. corpora a + b summed):

| name | definition |
|---|---|
| `amod_mi`, `dobj_mi`, `advmod_mi` | `MiCalculator`: per relation, mean over the essay's attested (head lemma, dependent lemma, relation) pairs of log2 P(h,d) / (P(h)P(d)), with marginals over all relations. Unattested pairs are skipped. |
| `log_freq_s2` | `log_mean_token_freq`: mean over alpha tokens of log10(count(lemma) / N), unseen lemmas counted as 1 (so non-words are kept, at the floor). |

**Classic, new** (reference: corpus a, edited prose):

| name | definition |
|---|---|
| `bigram_mi` | adjacent spaCy token pairs where both are alphabetic (so punctuation and whitespace tokens break a bigram), lowercased. MI = log2(c(ab) N / (c(a) c(b))). Mean over attested bigrams (c(ab) > 0). |
| `bigram_dp` | ΔP forward = c(ab)/c(a) − (c(b) − c(ab)) / (N − c(a)), mean over attested bigrams |
| `bigram_t` | t = (c(ab) − c(a)c(b)/N) / sqrt(c(ab)), mean over attested bigrams |
| `share_mi3` | share of attested bigrams with MI ≥ 3 (Durrant & Schmitt 2009; Bestgen & Granger 2014) |
| `share_t2` | share of attested bigrams with t ≥ 2 |
| `share_unatt` | share of all alphabetic bigrams with c(ab) = 0 |
| `content_freq` | mean log10(count(form) / N) over content words (alphabetic NOUN, VERB, ADJ, ADV) attested in corpus a |

**Surprisal (reference)**: `surprisal` = mean first-subword surprisal (bits)
over all scored non-space tokens, the D24 quantity. `surprisal_real` = the same
over real words only (definition below), to separate the effect of the token
filter from the effect of the PMI denominator.

**LLM association measures.** p(w | ctx) is the stored first-subword
probability (OLMo-2 1B, 8-subword window). p̄(v) is the model's own marginal
over the first-subword id v: the mean softmax over about 10M 7-subword contexts
from Dolma corpus a (step 1, details below). A sensitivity version uses
contexts from corpus b.

| name | tokens | definition |
|---|---|---|
| `llm_pmi` | all scored non-space tokens | mean of log2 p(w\|ctx) − log2 p̄(w) |
| `llm_dp` | all | mean of p(w\|ctx) − p̄(w) |
| `llm_pmi_real` | real words | as `llm_pmi` |
| `llm_dp_real` | real words | as `llm_dp` |
| `llm_pmi_content` | real content words | as `llm_pmi` |
| `llm_pmi_fadj` | real words | token PMI minus its OLS prediction from log10 corpus-a frequency (coefficients fit once on the corpus's original-essay real words, applied unchanged to the corrected values) |
| `llm_pmi_ge3` | real words | share with token PMI ≥ 3 bits |
| `high_s_content` | real content words | share with surprisal > τ (the Study 3 threshold) |
| `llm_pmi_uni` (check) | real words that are a single subword | mean of log2 p(w\|ctx) − log2(count(w)/N), corpus-a unigram (lowercased, so the denominator ignores case) |

**Real word** = alphabetic, not number-like, not PROPN, and the lowercased form
is attested in corpus a. Non-words (forms unseen in Dolma), punctuation,
numbers and proper nouns are excluded. Token subsets are fixed by the original
token's attributes, so the D24 split stays exact.

### Statistics, per measure and corpus

1. **ρ orig, ρ corr**: Spearman with level (W&I) or script score (FCE), on the
   original and on the corrected version; 95% CIs from 2,000 unit bootstraps,
   seed 42.
2. **Share removed.**
   - Token-mean measures: the exact D24 split (`pilot_corrected/5_decompose.py`,
     `split`) of the measure into clean + omission + context + error tokens.
     Share removed = 1 − cov(clean, y) / cov(total, y), with bootstrap CI.
   - Other measures: the relative drop 1 − ρ corr / ρ orig.
3. **Partial ρ | corrected-token share**: Pearson of the residuals of rank(x)
   and rank(y) after linear regression on rank(c), with c = share of the
   unit's non-space original tokens that are corrected (gold ERR_TOK or
   ERR_INS, any surprisal; as in `pilot_rq3/3_reconcile.py` `corr_tok`). Not
   5_gold.py's `err_rate` (D31). Reported for the original and the corrected
   version.
4. **ΔR² over B**: the pilot_rq3 framework unchanged (`2_increment.py`: ridge,
   in-fold mean imputation and scaling, alpha by inner 5-fold CV, 10 × 10-fold
   CV stratified by level or D25 score band, the same folds for every model,
   2,000-unit bootstrap of ΔR² from the stored out-of-fold predictions).
   B = ERRANT group rates + flagged rate + log n. Model = B + the measure on
   the *corrected* version. Also reported, for context, B + the original
   version.

### Reading rule

A measure **survives correction** on a corpus if (a) ρ orig has a CI that
excludes 0; (b) ρ corr has the same sign and |ρ corr| ≥ ½ |ρ orig|; and (c)
ΔR²(B + corrected measure) − R²(B) has a 95% CI above 0. It survives overall
if it survives on both corpora. A measure whose ρ orig CI includes 0 on a
corpus has no association to lose there and is reported as "no signal".

- **Q1**: surprisal is *not unique* in this respect if the classic measures
  with a signal also fail (a)–(c), i.e. their proficiency association also
  largely disappears once the gold corrections are applied. It *is* distinctive
  if most classic measures with a signal survive while surprisal does not.
- **Q2**: an LLM association measure answers the question if it survives on
  both corpora. If none does, surprisal recast as association behaves like
  surprisal.

### Q3

- Token level, original essays, attested real-word bigrams (w1 w2 both real
  words, adjacent, c(w1 w2) > 0): Spearman of w2's surprisal and LLM-PMI with
  the bigram's MI, ΔP and t. Also on the corrected essays.
- Essay level: Spearman matrix among all measures (original versions).
- **Exclusion by construction.** Original-essay mean surprisal is split
  exactly by token class: (i) w2 of an attested alphabetic bigram (what
  `bigram_*` uses), (ii) w2 of an unattested alphabetic bigram, (iii)
  alphabetic token with no alphabetic left neighbour (sentence start, after
  punctuation), (iv) non-alphabetic tokens (punctuation, numbers, symbols).
  Each class's contribution is Σ_class s / n, so their covariances with the
  criterion add up to cov(mean surprisal, y). Reported: each class's token
  share and share of the covariance, and, inside each class, the part on
  corrected tokens. The same for the real-word filter (real vs non-word vs
  other).

### Revision R1 (from Langdon via the coordinator, before any level analysis was read)

A test run of steps 3–4 had been started with a 300k-context trial p̄ to check
the code. It was stopped before its output was read, and its tables were
deleted. The trial `pbar_summary.json` the coordinator saw was from a smoke
test: the "b" file was a copy of the trial "a" file, which is why a and b
agreed at r = 1.0. The planned runs (10M corpus-a contexts, 5M corpus-b
contexts) were already running at the time. The changes:

1. **Primary denominator = relative frequency.** The primary LLM-PMI is
   log2 p(w | ctx) − log2 f(w), with f(w) = count_a(w) / N_a, the corpus-a
   word unigram (lowercased form). It is computed over real words that are a
   **single subword** (decoded first-subword id = the token text; 98% of real
   words), so that numerator and denominator are the same unit. ΔP-LLM =
   p(w | ctx) − f(w). The content-word, frequency-adjusted and cutoff variants
   all use this denominator and token set. Reason: this is the
   applied-linguistics convention and the easiest to defend.
2. **The model marginal p̄ becomes a sensitivity check.** p̄ from corpus a is
   pooled over three independent runs (seeds 42, 7, 11; position offsets 0,
   2, 1; up to 30M contexts), with p̄ from corpus b (5M) as a second check.
   Precision is reported for the subword ids the essays use, as the SE of
   log2 p̄ across the runs' half-samples. A third denominator is the observed
   relative frequency of the subword at the same corpus-a positions
   (`obs`, add-0.5 smoothing). The agreement of the three on the essay tokens
   is reported.
3. **Attested-only variants**, as classic MI is n/a for unattested pairs:
   `llm_pmi_att1` / `llm_dp_att1`, tokens whose left bigram (previous spaCy
   token, w) is attested in corpus a; `llm_pmi_att2` / `llm_dp_att2`, tokens
   with both the left and the right bigram attested. The previous and next
   token are whatever spaCy token is adjacent (punctuation and whitespace
   tokens included, as the 3-gram tables count them; '#' pad at the essay
   start and end). Also `surprisal_att2` for comparison. For every filter,
   step 5 reports the share of tokens and of the original-essay surprisal
   signal (cov with the criterion) that it removes, overall and by corrected
   vs uncorrected.
4. **Surprisal cutoff (secondary).** `llm_pmi_lowS` and `surprisal_lowS`
   keep only tokens with original surprisal ≤ τ, the LLM-native analogue of
   dropping unattested pairs. **It selects on the outcome**: high-surprisal
   tokens are removed because they are surprising, so a weakened association
   is partly built in.

The old `llm_pmi_real`, `llm_dp_real`, `llm_pmi_uni` and `llm_pmi_b_real`
names are gone: `llm_pmi` now is the single-subword relative-frequency version,
and the p̄ versions are `llm_pmi_pbar` (corpus a), `llm_pmi_pbar_b` and
`llm_pmi_subfreq`. `surprisal_real` is replaced by `surprisal_sw` (the same
single-subword real-word tokens). The reading rule, the statistics and the
error control (plain corrected-token share) are unchanged. The rule is applied
to the primary measures. The p̄ and cutoff versions are reported but carry no
verdict of their own.

### Caveats fixed in advance

First-subword units (p̄ is over first subwords, so a word's PMI is that of its
first piece); Dolma as the reference for both the counts and the p̄ contexts;
the 8-subword window (p̄ uses 7-subword contexts, and the first 7 tokens of an
essay have shorter ones); one annotator per corpus; minimal corrections leave
awkwardness in place.

## Running it

```
CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/pilot_assoc/1_pbar.py --corpus a --n 10000000
CUDA_VISIBLE_DEVICES=1 python src/3-surprisal-decomposition/pilot_assoc/1_pbar.py --corpus b --n 5000000
CUDA_VISIBLE_DEVICES=1 python src/3-surprisal-decomposition/pilot_assoc/1_pbar.py --corpus a --n 10000000 --seed 7 --offset 2 --tag _2
CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/pilot_assoc/1_pbar.py --corpus a --n 10000000 --seed 11 --offset 1 --tag _3
python src/3-surprisal-decomposition/pilot_assoc/2_classic.py --check   # reproduces Study 2 on ELLIPSE
python src/3-surprisal-decomposition/pilot_assoc/2_classic.py
python src/3-surprisal-decomposition/pilot_assoc/3_tokens.py
python src/3-surprisal-decomposition/pilot_assoc/4_levels.py
python src/3-surprisal-decomposition/pilot_assoc/5_q3.py
```

Data and tables: `data/study3/pilot_assoc/` (`tables/levels_{wi,fce}.csv`,
`levels.json`, `q3_*.csv`, `pbar_summary.json`).

## Checks

- **Study 2 reproduced.** `2_classic.py --check` recomputes `amod_mi`,
  `dobj_mi`, `advmod_mi` and `log_mean_token_freq` for the first 300 ELLIPSE
  essays from the Dolma tables with polars lookups. The max |difference| from
  `data/ellipse/ellipse_metrics.csv` is 2e-15, and the missing values agree
  exactly. Study 2 used corpora a + b (`--corpus both`), so these four do too.
- **Earlier pilots reproduced.** Surprisal's share removed is 104% (W&I) and
  91% (FCE), i.e. a clean share of −4% and 9% (D24, D25). R²(B) = .581 / .481
  and ΔR²(B + original mean surprisal) = .0000 / .0119 (D31).
- Bigram counts: the 3-gram tables prune trigrams seen once, so a derived
  unigram count can fall below its bigram count. Marginals are floored at the
  joint count (otherwise MI is infinite for a few pairs).
- 99.8% (W&I) and 99.9% (FCE) of own aligned tokens have the same first
  subword in both versions.

## p̄ estimation (step 1)

- Contexts: 7 subwords (the context a first subword gets under the 8-subword
  window), from the local Dolma v1.7 sample. Documents in file order, kept with
  probability 1/4, cut at 2,048 subwords, every 4th position, source quotas by
  gzip size (corpus a: wiki 47%, pes2o 30%, cc_news 24%; corpus b: reddit 60%,
  stackexchange 21%, cc_blogs 19%). The mean softmax over the full vocabulary
  (100,352 logits) is accumulated in float64 on the GPU.
- **Corpus a: 30M contexts** in three independent runs (seeds 42 / 7 / 11,
  position offsets 0 / 2 / 1), 10M each. **Corpus b: 5M.**
- **Cost.** About 3,700 contexts/s on one RTX A6000 (bf16 forward, fp32
  softmax). 45 min per 10M, about 2.6 GPU-hours in all (two GPUs in
  parallel, about 1.5 h wall-clock).
- **Precision**, on the subword ids the essays actually use (single-subword
  real words, token-weighted). The SE of log2 p̄_a across the six half-samples
  is median .004 bits, 95th percentile .021, max .84 (W&I; FCE the same).
  That is negligible next to the differences between denominators. The 2-bit
  split-half figure the coordinator saw came from the 300k smoke test.
- **Agreement of the three denominators** on the essay tokens (W&I / FCE):
  - log2 f(w) vs log2 p̄_a: Pearson .97 / .93, median |diff| .36 bits, mean
    p̄_a − f = −.42 / −.54 bits;
  - log2 f(w) vs log2 subword frequency: .97 / .93, median |diff| .38 bits;
  - log2 p̄_a vs log2 subword frequency: .997, median |diff| .13–.15 bits;
  - log2 p̄_a vs log2 p̄_b: median |diff| .66 / .79 bits, 95th percentile
    2.6 / 2.9 bits. This is register (edited vs unedited prose), not noise.
  - Essay-level ρ with the criterion is nearly the same with f, p̄_a or the
    subword frequency (W&I .40 / .40 / .38; FCE .56 / .57 / .55). So the
    choice of denominator does not matter, except p̄_b, which keeps a bit more
    (below).

## Results, Q1 and Q2

Spearman ρ with level (W&I, n = 3,300) or script score (FCE, n = 1,237),
95% CIs from 2,000 unit bootstraps. **removed** = 1 − cov(clean, y) /
cov(total, y) for token-mean measures (exact D24 split), and "drop" = 1 − ρ
corr / ρ orig for the rest. **partial** = partial ρ given the plain
corrected-token share. **ΔR²** = R²(B + measure) − R²(B), ×100 (percentage
points of R²), with the measure from the corrected (corr) or original (orig)
version. B alone: R² .581 (W&I), .481 (FCE). Verdict under the rule (a)–(c).
The first 18 rows are LLM-based and the last 11 classic. The p̄ and
cutoff rows are sensitivity checks with no verdict of their own (R1).
### W&I+LOCNESS learners (level A = 1, B = 2, C = 3)

| measure | ρ orig | ρ corr | removed | partial orig | partial corr | ΔR² corr (×100) | ΔR² orig (×100) | verdict |
|---|---|---|---|---|---|---|---|---|
| `surprisal` | −.44 [−.47, −.41] | .04 [.01, .08] | 104% [99, 110] | .04 | .12 [.08, .15] | .08 [−.07, .24] | .00 [−.08, .08] | fails (rho+dR2) |
| `surprisal_sw` | −.39 [−.42, −.36] | .05 [.01, .08] | 107% [99, 114] | .05 | .13 [.09, .16] | .12 [−.06, .29] | −.01 [−.08, .06] | fails (rho+dR2) |
| `llm_pmi` | .40 [.37, .43] | .03 [−.01, .06] | 93% [87, 99] | −.05 | −.12 [−.16, −.09] | −.03 [−.04, −.02] | .01 [−.08, .11] | fails (rho+dR2) |
| `llm_dp` | .43 [.40, .46] | .14 [.11, .18] | 73% [67, 79] | .07 | .02 [−.02, .05] | .07 [−.08, .21] | .34 [.07, .62] | fails (rho+dR2) |
| `llm_pmi_content` | .38 [.35, .41] | .05 [.02, .09] | 87% [81, 93] | −.02 | −.09 [−.12, −.05] | −.03 [−.05, −.02] | .00 [−.09, .10] | fails (rho+dR2) |
| `llm_pmi_fadj` | .41 [.38, .44] | −.00 [−.04, .03] | 98% [92, 104] | −.05 | −.13 [−.17, −.10] | −.01 [−.09, .07] | −.03 [−.07, .01] | fails (rho+dR2) |
| `llm_pmi_ge3` | .32 [.29, .35] | −.04 [−.07, −.00] | 105% [96, 114] | −.10 | −.15 [−.19, −.12] | .13 [−.04, .30] | −.00 [−.06, .06] | fails (rho+dR2) |
| `high_s_content` | −.24 [−.27, −.20] | .14 [.10, .17] | 137% [122, 154] | .09 | .15 [.12, .19] | .24 [.00, .46] | .12 [−.04, .28] | fails (rho) |
| `llm_pmi_att1` | .36 [.34, .39] | .02 [−.02, .05] | 95% [88, 102] | −.06 | −.12 [−.16, −.09] | −.03 [−.04, −.02] | .01 [−.08, .11] | fails (rho+dR2) |
| `llm_pmi_att2` | .36 [.33, .39] | .02 [−.02, .05] | 94% [87, 101] | −.07 | −.13 [−.16, −.10] | −.02 [−.07, .02] | −.02 [−.08, .04] | fails (rho+dR2) |
| `llm_dp_att1` | .41 [.38, .45] | .14 [.11, .18] | 71% [65, 78] | .07 | .02 [−.01, .06] | .12 [−.06, .29] | .42 [.12, .72] | fails (rho+dR2) |
| `llm_dp_att2` | .41 [.38, .44] | .15 [.11, .18] | 70% [64, 77] | .07 | .03 [−.01, .06] | .16 [−.03, .34] | .47 [.15, .78] | fails (rho+dR2) |
| `surprisal_att2` | −.38 [−.41, −.35] | .04 [−.00, .07] | 103% [96, 110] | .04 | .11 [.08, .15] | .04 [−.08, .15] | −.03 [−.04, −.02] | fails (rho+dR2) |
| `llm_pmi_lowS` | .31 [.28, .34] | .07 [.04, .10] | 78% [71, 86] | −.06 | −.10 [−.13, −.07] | −.01 [−.06, .05] | .08 [−.06, .22] | fails (rho+dR2) |
| `surprisal_lowS` | −.40 [−.43, −.37] | −.03 [−.06, .01] | 92% [86, 100] | .00 | .06 [.02, .09] | −.04 [−.06, −.02] | .00 [−.11, .11] | fails (rho+dR2) |
| `llm_pmi_pbar` | .40 [.37, .43] | .04 [.00, .07] | 92% [86, 98] | −.06 | −.14 [−.17, −.10] | −.03 [−.07, .02] | −.02 [−.07, .04] | fails (rho+dR2) |
| `llm_pmi_pbar_b` | .49 [.46, .52] | .14 [.11, .18] | 80% [76, 84] | .05 | −.03 [−.07, .00] | .17 [−.01, .36] | .38 [.10, .67] | fails (rho+dR2) |
| `llm_pmi_subfreq` | .38 [.35, .41] | .02 [−.01, .06] | 94% [87, 100] | −.06 | −.14 [−.17, −.11] | −.01 [−.08, .05] | −.03 [−.06, .01] | fails (rho+dR2) |
| `amod_mi` | .09 [.06, .13] | .02 [−.02, .05] | drop 82% [57, 132] | .03 | .01 [−.02, .05] | .05 [−.08, .18] | .06 [−.08, .20] | fails (rho+dR2) |
| `dobj_mi` | .15 [.11, .18] | .06 [.03, .10] | drop 58% [44, 78] | .01 | .00 [−.03, .04] | −.03 [−.04, −.01] | −.03 [−.06, .01] | fails (rho+dR2) |
| `advmod_mi` | .08 [.04, .11] | .03 [−.00, .06] | drop 61% [38, 108] | −.01 | −.01 [−.04, .03] | −.02 [−.06, .02] | −.02 [−.07, .03] | fails (rho+dR2) |
| `log_freq_s2` | −.02 [−.05, .02] | −.12 [−.16, −.09] | drop -607% [-7425, 4821] | −.02 | −.05 [−.08, −.02] | .35 [.07, .61] | .38 [.09, .66] | no signal |
| `content_freq` | −.18 [−.22, −.15] | −.19 [−.23, −.16] | drop -7% [-14, -1] | −.08 | −.09 [−.13, −.06] | .52 [.19, .85] | .44 [.12, .75] | survives |
| `bigram_mi` | .23 [.20, .27] | .00 [−.03, .03] | drop 100% [87, 116] | −.05 | −.09 [−.12, −.05] | −.02 [−.07, .02] | −.03 [−.07, .01] | fails (rho+dR2) |
| `bigram_dp` | .24 [.20, .27] | .16 [.12, .19] | drop 32% [25, 40] | .15 | .13 [.09, .16] | .25 [.02, .46] | .28 [.04, .52] | survives |
| `bigram_t` | .11 [.07, .14] | .02 [−.01, .06] | drop 80% [58, 114] | .06 | .03 [−.00, .07] | .00 [−.05, .05] | −.03 [−.07, .01] | fails (rho+dR2) |
| `share_mi3` | .17 [.14, .20] | .00 [−.03, .04] | drop 98% [81, 122] | −.04 | −.06 [−.10, −.03] | −.00 [−.08, .07] | −.01 [−.07, .04] | fails (rho+dR2) |
| `share_t2` | .20 [.16, .23] | −.04 [−.08, −.01] | drop 123% [105, 147] | −.03 | −.05 [−.08, −.01] | −.03 [−.04, −.01] | −.01 [−.07, .04] | fails (rho+dR2) |
| `share_unatt` | −.20 [−.24, −.17] | .07 [.03, .10] | drop 134% [115, 156] | −.02 | .03 [−.01, .06] | .00 [−.08, .08] | .02 [−.07, .10] | fails (rho+dR2) |

### FCE (script score)

| measure | ρ orig | ρ corr | removed | partial orig | partial corr | ΔR² corr (×100) | ΔR² orig (×100) | verdict |
|---|---|---|---|---|---|---|---|---|
| `surprisal` | −.58 [−.62, −.54] | −.08 [−.13, −.02] | 91% [86, 97] | −.24 | −.08 [−.13, −.02] | .29 [−.15, .76] | 1.19 [.38, 2.10] | fails (rho+dR2) |
| `surprisal_sw` | −.55 [−.59, −.50] | −.09 [−.14, −.03] | 90% [83, 97] | −.21 | −.07 [−.12, −.01] | .26 [−.19, .72] | 1.19 [.34, 2.06] | fails (rho+dR2) |
| `llm_pmi` | .56 [.52, .60] | .22 [.17, .28] | 75% [69, 81] | .21 | .08 [.02, .13] | .12 [−.17, .44] | .89 [.20, 1.62] | fails (rho+dR2) |
| `llm_dp` | .53 [.48, .57] | .24 [.19, .30] | 62% [54, 69] | .25 | .17 [.11, .23] | 1.25 [.26, 2.29] | 2.28 [1.05, 3.61] | fails (rho) |
| `llm_pmi_content` | .54 [.50, .58] | .20 [.15, .25] | 76% [71, 82] | .20 | .08 [.03, .14] | .31 [−.14, .80] | 1.17 [.38, 2.00] | fails (rho+dR2) |
| `llm_pmi_fadj` | .58 [.54, .62] | .18 [.13, .24] | 81% [76, 87] | .22 | .08 [.02, .13] | .23 [−.17, .64] | 1.29 [.47, 2.15] | fails (rho+dR2) |
| `llm_pmi_ge3` | .48 [.44, .52] | .12 [.07, .18] | 81% [74, 90] | .12 | .01 [−.05, .07] | −.07 [−.10, −.03] | .17 [−.22, .58] | fails (rho+dR2) |
| `high_s_content` | −.41 [−.45, −.36] | .05 [−.01, .11] | 109% [98, 121] | −.10 | .03 [−.03, .08] | −.05 [−.08, −.02] | .13 [−.25, .53] | fails (rho+dR2) |
| `llm_pmi_att1` | .51 [.46, .55] | .23 [.17, .28] | 68% [61, 75] | .16 | .08 [.02, .14] | .11 [−.18, .42] | .57 [−.03, 1.20] | fails (rho+dR2) |
| `llm_pmi_att2` | .50 [.45, .54] | .22 [.17, .27] | 69% [62, 76] | .16 | .08 [.02, .13] | .13 [−.17, .44] | .57 [−.01, 1.19] | fails (rho+dR2) |
| `llm_dp_att1` | .50 [.45, .55] | .25 [.20, .31] | 58% [51, 66] | .23 | .17 [.12, .23] | 1.20 [.23, 2.22] | 2.15 [.96, 3.44] | survives |
| `llm_dp_att2` | .49 [.45, .54] | .25 [.20, .31] | 57% [50, 65] | .23 | .17 [.12, .23] | 1.28 [.30, 2.32] | 2.15 [.94, 3.42] | survives |
| `surprisal_att2` | −.50 [−.55, −.46] | −.13 [−.18, −.07] | 83% [75, 91] | −.18 | −.09 [−.14, −.03] | .46 [−.17, 1.04] | 1.20 [.31, 2.10] | fails (rho+dR2) |
| `llm_pmi_lowS` | .49 [.45, .53] | .26 [.21, .31] | 58% [51, 65] | .16 | .09 [.03, .14] | .13 [−.17, .45] | .46 [−.08, 1.00] | fails (dR2) |
| `surprisal_lowS` | −.48 [−.53, −.44] | −.11 [−.16, −.05] | 85% [76, 95] | −.19 | −.09 [−.15, −.04] | .31 [−.16, .81] | .78 [.02, 1.60] | fails (rho+dR2) |
| `llm_pmi_pbar` | .57 [.52, .61] | .21 [.15, .26] | 75% [66, 85] | .20 | .05 [−.00, .11] | −.03 [−.10, .03] | .02 [−.22, .24] | fails (rho+dR2) |
| `llm_pmi_pbar_b` | .60 [.56, .64] | .26 [.21, .31] | 72% [65, 80] | .25 | .09 [.04, .15] | .01 [−.16, .16] | .13 [−.24, .46] | fails (rho+dR2) |
| `llm_pmi_subfreq` | .55 [.51, .60] | .19 [.14, .25] | 75% [66, 86] | .19 | .05 [−.01, .11] | −.03 [−.09, .02] | .01 [−.21, .21] | fails (rho+dR2) |
| `amod_mi` | .10 [.04, .15] | .05 [−.00, .11] | drop 49% [17, 110] | .04 | .04 [−.01, .10] | .06 [−.21, .33] | −.00 [−.23, .23] | fails (dR2) |
| `dobj_mi` | .20 [.15, .26] | .09 [.04, .15] | drop 56% [38, 78] | .05 | .02 [−.03, .07] | −.06 [−.14, .02] | −.03 [−.21, .16] | fails (rho+dR2) |
| `advmod_mi` | .14 [.08, .20] | .06 [.00, .12] | drop 56% [33, 96] | .06 | .04 [−.02, .10] | .18 [−.18, .57] | .34 [−.14, .86] | fails (rho+dR2) |
| `log_freq_s2` | .15 [.09, .20] | −.08 [−.14, −.02] | drop 155% [111, 243] | .06 | −.02 [−.08, .03] | .09 [−.25, .42] | −.05 [−.20, .09] | fails (rho+dR2) |
| `content_freq` | −.14 [−.19, −.08] | −.20 [−.25, −.14] | drop -46% [-95, -22] | −.07 | −.11 [−.16, −.05] | .70 [.05, 1.40] | .52 [−.06, 1.12] | survives |
| `bigram_mi` | .31 [.26, .36] | .14 [.09, .20] | drop 55% [44, 69] | .06 | .02 [−.03, .08] | −.08 [−.11, −.04] | −.05 [−.24, .15] | fails (rho+dR2) |
| `bigram_dp` | .19 [.14, .24] | .14 [.08, .19] | drop 28% [15, 46] | .13 | .13 [.07, .19] | .71 [−.00, 1.42] | .78 [−.01, 1.53] | fails (dR2) |
| `bigram_t` | .04 [−.02, .09] | −.06 [−.12, −.01] | drop 263% [-1994, 2090] | .05 | .00 [−.05, .06] | −.04 [−.15, .06] | .11 [−.18, .41] | no signal |
| `share_mi3` | .21 [.16, .27] | .10 [.05, .16] | drop 51% [36, 71] | .03 | .02 [−.03, .08] | −.06 [−.10, −.02] | −.09 [−.15, −.02] | fails (rho+dR2) |
| `share_t2` | .25 [.19, .30] | .05 [−.01, .10] | drop 81% [65, 105] | .03 | −.02 [−.07, .04] | −.05 [−.12, .01] | −.06 [−.21, .08] | fails (rho+dR2) |
| `share_unatt` | −.37 [−.42, −.32] | −.02 [−.08, .04] | drop 96% [81, 112] | −.16 | −.04 [−.10, .01] | −.02 [−.19, .17] | .22 [−.23, .68] | fails (rho+dR2) |

### Verdicts over both corpora (primary measures)

| measure | W&I | FCE | overall |
|---|---|---|---|
| `surprisal` (reference) | fails (ρ, ΔR²) | fails (ρ, ΔR²) | does not survive |
| `llm_pmi` | fails (ρ, ΔR²) | fails (ρ, ΔR²) | does not survive |
| `llm_dp` | fails (ρ, ΔR²) | fails (ρ) | does not survive |
| `llm_pmi_content`, `llm_pmi_fadj`, `llm_pmi_ge3` | fail | fail | do not survive |
| `high_s_content` | fails (ρ) | fails (ρ, ΔR²) | does not survive |
| `llm_pmi_att1`, `llm_pmi_att2` | fail | fail | do not survive |
| `llm_dp_att1`, `llm_dp_att2` | fail (ρ, ΔR²) | **survive** | do not survive |
| `amod_mi`, `dobj_mi`, `advmod_mi` | fail | fail | do not survive |
| `log_freq_s2` | no signal | fails | does not survive |
| `content_freq` | **survives** | **survives** | **survives** |
| `bigram_mi`, `bigram_t`, `share_mi3`, `share_t2`, `share_unatt` | fail | fail (`bigram_t`: no signal) | do not survive |
| `bigram_dp` | **survives** | fails (ΔR², CI [−.00, 1.42]) | does not survive, narrowly |

## Results, Q3

**Token level** (Spearman, pooled tokens, w2 of attested bigrams of two real
words, w2 a single subword; W&I n = 469,582, FCE 343,830 in the original
essays):

| | bigram MI | bigram ΔP | bigram t | log c(w2) |
|---|---|---|---|---|
| surprisal, W&I / FCE | −.30 / −.28 | −.63 / −.61 | −.43 / −.42 | −.53 / −.52 |
| LLM-PMI, W&I / FCE | .55 / .54 | .10 / .09 | .16 / .14 | −.46 / −.47 |

Post-correction (own words, corrected bigrams) the values are within .05 of
these, except LLM-PMI with ΔP and t, which fall to .01 and .07–.08.
Surprisal is closest to forward ΔP, i.e. to how probable w2 is after w1, and
LLM-PMI is closest to MI. That is what the definitions predict: both ΔP and
surprisal are conditional probabilities of w2, and both MI and LLM-PMI divide
out w2's frequency. The model's 7-subword context makes the agreement
moderate rather than high.

**Essay level** (Spearman, original versions; full matrices in
`tables/q3_matrix_{wi,fce}.csv`). Surprisal with LLM-PMI −.82 / −.84, with
ΔP-LLM −.70 / −.76. Surprisal with bigram MI −.48 / −.51, share MI ≥ 3 −.40 /
−.39, share t ≥ 2 −.45 / −.47, share unattested .55 / .60, dependency MI
−.14 to −.28, content-word frequency −.12 / −.08. LLM-PMI with bigram MI
.60 / .62. The corrected-token share correlates .67 / .69 with surprisal and
−.62 / −.70 with LLM-PMI, but |ρ| ≤ .43 with every classic measure (bigram MI
−.39 / −.43, `share_unatt` .27 / .39, the rest ≤ .35). So the LLM measures sit much closer to the
error rate than the classic ones do.

**Exclusion by construction.** Original-essay mean surprisal split exactly
by token class. Token share, then share of cov(mean surprisal, y) with 95% CI
(W&I / FCE):

| class | tokens | share of signal |
|---|---|---|
| w2 of an attested alphabetic bigram (what `bigram_*` averages over) | 73% / 70% | 40% [35, 45] / 15% [9, 21] |
| w2 of an unattested alphabetic bigram | 5% / 5% | 35% [32, 38] / 41% [36, 47] |
| alphabetic, no alphabetic left neighbour | 11% / 13% | 7% [5, 10] / 35% [30, 39] |
| non-alphabetic (punctuation, numbers, symbols) | 11% / 12% | 17% [15, 20] / 9% [7, 12] |

So 60% (W&I) and 85% (FCE) of surprisal's proficiency signal sits in tokens
that bigram association measures leave out by construction. By word type,
non-words are 0.5% / 0.9% of tokens but carry 12% / 20% of the signal.

The R1 token filters of the LLM measures, share removed (W&I / FCE), with the
removed signal split into corrected and uncorrected tokens:

| filter (kept tokens) | tokens removed | signal removed | of which on corrected tokens | on uncorrected |
|---|---|---|---|---|
| real words | 15% / 16% | 32% / 35% | 38% / 32% | −6% / 3% |
| + single subword (primary set) | 17% / 18% | 31% / 38% | 45% / 39% | −14% / −2% |
| + left bigram attested (att1) | 21% / 23% | 57% / 78% | 70% / 78% | −13% / 0% |
| + both bigrams attested (att2) | 25% / 27% | 71% / 97% | 80% / 88% | −10% / 9% |
| + s ≤ τ (cutoff; selects on the outcome) | 27% / 27% | 91% / 95% | 115% / 101% | −25% / −6% |

These shares are of the essay *sum* Σ s / n, so a class's share mixes how
many of its tokens an essay has with how surprising they are. The negative
uncorrected shares are the count effect: better essays have more
uncorrected tokens.

An important point for reading the attested-only LLM measures: the att2
filter removes 71–97% of surprisal's signal from the essay sum, yet LLM-PMI
over the remaining tokens still has ρ .36 / .50 with the criterion, and 94% /
69% of that is still removed by correction. The error signal reaches attested
tokens too: through wrong-but-real words (substitutions that form attested
bigrams) and through spillover onto the correct words after an error
(D24 context, D27). Attestation filtering takes out non-words and broken
bigrams. It does not take out the error signal.

## Reading

- **Q1: surprisal is not unique.** Every classic phraseological measure with a
  signal loses most of it once the corrections are applied: dependency MI
  (drops of 49–82%), bigram MI (100% / 55%), t-score, the MI ≥ 3 and t ≥ 2
  shares (51–123%), and the unattested-bigram share (134% / 96%). None of
  them adds to the error counts once computed on the corrected essays (all
  ΔR² CIs include 0 or are negative). They also start from much weaker
  associations (|ρ| .08–.37) than surprisal (.44 / .58). So the pattern of
  D24/D25, an association that lives in what the corrections remove, is
  shared by the classic association measures. It is not a peculiarity of the
  LLM.
- **Two measures do keep a proficiency association after correction.**
  - Content-word frequency survives on both corpora. Lower-frequency content
    words go with higher level or score (ρ −.18 / −.14, after correction
    −.19 / −.20, ΔR² +.5 / +.7 points). Its signal *grows* after
    correction, because corrected misspellings become attested words. This
    is lexical sophistication, not association.
  - Forward bigram ΔP keeps about 70% of its ρ (.24 → .16 W&I, .19 → .14
    FCE) and has the highest partial ρ given errors of any measure (.13 on
    both). It survives on W&I and misses on FCE only because the ΔR² CI
    touches 0 (+.71 [−.00, 1.42]).
- **Q2: surprisal recast as association behaves like surprisal.** LLM-PMI in
  every primary form loses most of its association: 93% / 75% of the
  covariance is removed, ρ falls from .40 to .03 (W&I) and from .56 to .22
  (FCE), and the corrected version adds nothing over B. Restricting to
  content words, adjusting for frequency, counting PMI ≥ 3, or keeping only
  attested bigrams changes none of this. The denominator does not matter
  either (f, p̄ or subword frequency give the same result).
- **The probability-scale version holds up better.** ΔP-LLM keeps ρ .14 /
  .24 after correction (62–73% removed), and given the error share its
  partial is .17 on FCE. With the attestation filters it survives on FCE
  (ΔR² +1.2 points [.2, 2.2]) but not on W&I (ρ keeps only a third, ΔR² CI
  includes 0). So no LLM measure survives on both corpora. What ΔP-LLM and
  bigram ΔP share is that the probability scale is dominated by the
  high-probability (mostly function-word) continuations, where log measures
  are dominated by rare, surprising ones. That part of predictability is
  less tied to errors.
- **The W&I/FCE sign difference of D31 reappears.** Given the corrected-token
  share, post-correction LLM-PMI is *negatively* related to level on W&I
  (−.12) and positively to score on FCE (+.08). That is the clean-surprisal
  partial (+.12 W&I, −.08 FCE) with the sign flipped. It should not be
  reported as a general finding.
- **Q3.** Surprisal tracks forward ΔP most closely at the token level (ρ
  −.62), and LLM-PMI tracks bigram MI (.55). At the essay level, LLM measures
  correlate with the error rate far more than classic measures do. A large
  part of why is construction: 60% / 85% of surprisal's original-essay signal
  sits in tokens that bigram measures exclude (unattested pairs, words after
  punctuation, punctuation itself, non-words).

## Caveats

- **First-subword units.** The primary LLM-PMI is restricted to single-subword
  real words (98% of real words, 82–83% of tokens), so numerator and
  denominator are the same unit. The p̄ versions apply to any first subword
  but also only to these tokens here. f(w) is lowercased while the subword is
  cased and carries its leading space, so sentence-initial capitals get a
  mismatched denominator.
- **Dolma as reference.** Counts and p̄ contexts come from the local Dolma
  v1.7 sample (about 1B tokens per corpus). The 3-gram tables prune trigrams
  seen once, so unigram and bigram counts are slight undercounts, and
  "unattested" means "fewer than about 2 occurrences". p̄ differs by register
  (a vs b, median .7 bits), and the corpus-b version keeps a little more of
  the association after correction (ρ corr .14 / .26 vs .04 / .21).
- **The 8-subword window.** p̄ uses 7-subword contexts. The first 7 subwords
  of an essay are scored with shorter contexts in the stored surprisal.
  Spillover is capped at a few words (D27).
- **One annotator per corpus**, minimal corrections. Awkward but uncorrected
  phrasing stays in the "clean" part.
- **The surprisal cutoff selects on the outcome** (R1). Its smaller
  "removed" (78% / 58% for LLM-PMI) is partly built in.
- **Classic "drop"** is a ratio of correlations. Its CI is wide when ρ orig
  is small (dependency MI, `bigram_t`, `log_freq_s2`).
- **Units differ by version.** Token measures are compared on the writer's own
  words, and pair measures on the whole corrected essay. So "removed" and
  "drop" are not the same statistic.
- Exploratory. W&I and FCE are treated as not looked at in advance (D30), but
  this pilot has now looked.
