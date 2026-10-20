# Pilot: which kinds of correction carry each measure's proficiency signal?

Exploratory. Everything ran locally on CPU, and no learner text, span or n-gram
left the machine. Tokens the annotator left alone are *uncorrected*. Errors are
controlled only through plain edit counts (no 5_gold.py `err_rate`, per D31).

D24/D28 (surprisal) and D32 (classic measures) show that correction removes
most of each measure's proficiency association. This pilot asks which kinds
of correction carry that association. The question is whether it is
phraseological (word choice, multi-word rewrites, word order) or mechanical
(spelling, punctuation, inflection).

## Buckets

ERRANT category with the operation prefix (M:/R:/U:) stripped:

| bucket | categories |
|---|---|
| mechanical | SPELL ORTH PUNCT VERB:FORM VERB:SVA VERB:INFL VERB:TENSE NOUN:NUM NOUN:INFL MORPH ADJ:FORM CONTR NOUN:POSS |
| function word | DET PREP PRON CONJ PART |
| content-word choice | NOUN VERB ADJ ADV |
| multi-word / order | OTHER WO |
| unclassified | UNK, plus edits ERRANT could not type (`unmapped`) |

Unclassified edits make up 1.2% (W&I) and 0.7% (FCE) of edits, and they are
treated as a fifth player throughout. Flagged-but-uncorrected spans have no
correction, so they stay as written in every version. "Phraseological" is
reported twice: content-word choice + multi-word / order, without and with
function words.

Units and criteria are as in D28/D32. W&I has 3,300 learner essays, level
A/B/C = 1/2/3, with 72,191 edits. FCE has 1,237 scripts with score > 0,
criterion `script-s`, both answers pooled, with 51,220 edits. All CIs are
2,000 unit bootstraps, seed 42.

## Methods

**Surprisal (`1_surprisal.py`).** This step rebuilds the D28 credit table
with `pilot_types/3_decompose.credits` (primary rule) and sums credits by
bucket instead of by ERRANT group:

mean surprisal = clean + Σ_bucket (at_site + spill) + flagged + residual,

which holds exactly per unit. At site is the edit's own tokens' excess over
clean, or Δ on the word after a missing-word edit. Spill is Δ on other own
tokens in the 8-subword window, split among the window's edits by edit-level
Shapley with the correction as baseline. Each part's share of
cov(total, level) is reported.

**Classic measures (`2_versions.py`, `3_shapley.py`).** The players are the
buckets present among a unit's non-null edits. For every subset A of present
players, each answer is rebuilt with only the edits of buckets in A applied
(`pilot_corrected/common.apply_edits`). Each version is parsed with
en_core_web_lg with NER disabled, the pipeline of `1_correct.py` and
`ingest.py`. Items come from `pilot_assoc/2_classic.doc_items`, and Dolma
lookups use the same `Ref`/`annotate` code, so the counts and formulas are
D32's. Measures are computed per (unit, version) from pooled item sums, so an
FCE script pools both answers.

Let v(S) be the measure with the buckets in S left uncorrected and all other
buckets applied. Then v(∅) = corrected and v(all) = original. The exact
Shapley value is

ψ_b = Σ_{S∌b} |S|!(p−|S|−1)!/p! · [v(S∪b) − v(S)],

so measure_orig = measure_corr + Σ_b ψ_b, and each term's share of
cov(measure_orig, level) is reported.

The measures are:

- bigram MI, ΔP, t and the MI ≥ 3 share (Dolma corpus a)
- Study 2 dependency MI for amod, dobj and advmod (a + b), each separately and
  as the mean of the available three
- content-word log frequency, as a contrast
- Study 2's lemma log frequency, for completeness

**Summary (`4_summary.py`).** Combined columns are summed per unit before
bootstrapping, so their CIs are exact. Reference rows give each bucket's share
of all edits, and its share of cov(edit rate, level), where edit rate is
edits per scored token.

### Why the two splits are comparable, and the caveat

Both splits use the correction as the baseline. Both are exact additive
splits of original minus corrected, and covariance shares are linear, so each
bucket's share is "the part of the measure's proficiency covariance that
disappears when that kind of error is corrected". Both splits also credit
interactions between error kinds symmetrically.

The units of the game differ:

- **Surprisal** uses edit-level Shapley inside each token's window, summed by
  bucket. At-site credit is assigned by rule, not by a game: an error token's
  whole excess goes to its own edit.
- **Classic measures** use bucket-level Shapley over the whole essay.

For an additive token measure, an essay-level bucket game equals the sum of
per-window bucket games. Within one window, summing members' edit-level
Shapley values is not identical to a bucket-level Shapley value. The two
differ only in windows that hold interacting edits from two or more buckets.
69% of windows hold a single edit (D27), so the difference should be small,
but it was not measured, because that would need GPU rescoring of 2^p
versions.

Three further differences:

- The window caps surprisal spill (spill shares are lower bounds), whereas the
  classic measures see every effect of a correction on the parse.
- Surprisal has a separate flagged + residual part (7% / 9%). For the classic
  measures, flagged spans stay in every version, so they sit inside the
  corrected share. The like-for-like surprisal "corrected" share is clean +
  flagged + residual = 3% (W&I) and 18% (FCE).
- The classic measures are ratios with item counts that change under
  correction (attested bigrams, dependency pairs). Their ψ therefore includes
  denominator effects. Examples: a spelling fix turns an unattested bigram
  into an attested one, and a punctuation fix breaks or joins alphabetic
  bigrams.

## Checks

- **Exact sums.**
  - Surprisal: max |clean + parts − total| = 1.8e-15 on both corpora. Clean
    shares match D28 (−4% / 9%).
  - Classic: max |corr + Σψ − orig| ≤ 1.1e-13 for every measure.
- **D32 reproduces.**
  - Every original and corrected classic value matches
    `pilot_assoc/classic_*_{orig,corr}` for 100% of units (max diff 6e-14,
    NaN pattern identical).
  - ρ orig / corr match D32's `levels_*.csv`.
  - Re-parsing the unedited text with the version pipeline gives the ingest
    parses.
- **Edit typing.** Every non-null json edit has an ERRANT row, and no offsets
  disagree.
- **Overlaps and interactions in partial subsets.**
  - Overlaps: `apply_edits` never had to skip an overlapping edit in any
    version (0 of 50,419 W&I and 46,852 FCE versions).
  - Adjacent cross-bucket edits: 5,880 (W&I) and 4,569 (FCE) pairs of edits
    from different buckets touch or overlap within one character, where the
    junction whitespace repair could depend on the subset.
  - Edits applied differently from the full correction: 954 of 642,858 edit
    applications in W&I (0.15%; 124 distinct edits) and 244 of 509,480 in
    FCE (0.05%; 30 distinct edits). The difference is always junction
    whitespace: a space added or stripped from the replacement (98 / 21 edits)
    or a swallowed space (26 / 9). The edit's content is never changed.
  - Handling: `apply_edits` is left unchanged, and its local whitespace repair
    resolves these cases. The full-subset version is byte-identical to D32's
    corrected text.
- **Undefined measures.** A unit with no items in some version is dropped for
  that measure only. W&I drops amod 59, dobj 41 and advmod 61 units (the
  mean of the three: 0). FCE drops amod 1. Nothing else is dropped.
- **Cost (24 spaCy processes).**

  | | units | answers | versions (unique texts) | applying edits | parsing | measures + Shapley |
  |---|---|---|---|---|---|---|
  | W&I | 3,300 | – | 50,419 (49,943) | 11 min | 3.4 min (244 docs/s) | 2 min |
  | FCE | 1,237 | 2,467 | 46,852 (39,632) | 3.8 min | 2.5 min | 1.3 min |

  Present players per W&I unit: 0: 5, 1: 45, 2: 243, 3: 651, 4: 1,953,
  5: 403.

## Results

Each cell is the % share of cov(original measure, level/score), with a 95% CI.
"Corrected" is the measure on the fully corrected essay (for surprisal, the
clean part). The parts of each row sum to 100 (flagged + residual counts for
surprisal only). Bucket columns: mech = mechanical, FW = function word,
content = content-word choice, MW/order = multi-word / order, uncl. =
unclassified. ρ = Spearman with the criterion for the original / corrected
measure.

### W&I (learners, level)

| measure | ρ orig / corr | corrected | mech | FW | content | MW/order | uncl. | phraseo. excl. FW | phraseo. incl. FW |
|---|---|---|---|---|---|---|---|---|---|
| mean surprisal | −.44 / .04 | −4 [−10, 1] | 47 [44, 50] | 14 [13, 16] | 13 [11, 14] | 22 [20, 24] | 1 [1, 2] | 35 [32, 37] | 49 [46, 52] |
| bigram MI | .23 / .00 | −1 [−15, 10] | 38 [33, 44] | 22 [19, 26] | 12 [10, 15] | 29 [25, 33] | 0 | 41 [36, 47] | 63 [56, 72] |
| bigram ΔP | .24 / .16 | 66 [58, 73] | 9 [6, 12] | 8 [5, 12] | 5 [3, 8] | 12 [8, 16] | 0 | 17 [13, 21] | 25 [20, 32] |
| bigram t | .11 / .02 | 5 [−43, 33] | 15 [7, 28] | 39 [12, 67] | 3 [−7, 12] | 38 [25, 64] | 1 | 40 [25, 69] | 79 [53, 120] |
| share MI ≥ 3 | .17 / .00 | −3 [−28, 14] | 39 [31, 50] | 23 [18, 30] | 14 [10, 19] | 28 [23, 36] | 0 | 42 [34, 52] | 64 [53, 80] |
| amod MI | .09 / .02 | 18 [−27, 43] | 21 [10, 38] | 7 [2, 14] | 29 [18, 48] | 24 [13, 41] | 0 | 53 [36, 83] | 60 [42, 94] |
| dobj MI | .15 / .06 | 31 [9, 46] | 8 [2, 14] | 14 [9, 21] | 27 [18, 38] | 21 [14, 30] | 0 | 47 [35, 64] | 61 [47, 81] |
| advmod MI | .08 / .03 | 23 [−60, 54] | 11 [−3, 32] | 12 [4, 28] | 23 [11, 50] | 30 [13, 67] | 0 | 53 [30, 113] | 65 [39, 134] |
| dependency MI (mean of 3) | .17 / .06 | 24 [5, 38] | 14 [8, 21] | 10 [6, 14] | 28 [21, 37] | 24 [17, 33] | 0 | 52 [41, 67] | 62 [50, 78] |
| content-word log freq | −.18 / −.19 | 104 [99, 110] | 1 [−3, 5] | 0 [−1, 1] | −4 [−7, −1] | −2 [−5, 0] | 0 | −6 [−10, −2] | −6 [−10, −2] |
| Study 2 lemma log freq | −.02 / −.12 | no signal (ρ orig CI includes 0) | | | | | | | |
| *ref: edit rate (cov share)* | | – | 50 [49, 52] | 21 [20, 23] | 10 [10, 11] | 16 [15, 17] | 2 [1, 2] | 27 [25, 28] | 48 [46, 49] |
| *ref: % of all edits* | | – | 48 | 25 | 13 | 14 | 1 | 26 | 51 |

Surprisal, at site + spill: mech 38 + 9, FW 8 + 7, content 8 + 4, MW/order
15 + 7, uncl. 1 + 0, flagged + residual 7.

### FCE (scripts, script score)

| measure | ρ orig / corr | corrected | mech | FW | content | MW/order | uncl. | phraseo. excl. FW | phraseo. incl. FW |
|---|---|---|---|---|---|---|---|---|---|
| mean surprisal | −.58 / −.08 | 9 [3, 14] | 35 [31, 38] | 20 [18, 23] | 10 [8, 12] | 17 [15, 19] | 1 [0, 1] | 27 [24, 30] | 47 [43, 51] |
| bigram MI | .31 / .14 | 31 [18, 41] | 27 [22, 34] | 10 [6, 14] | 10 [7, 13] | 21 [17, 27] | 0 | 31 [25, 38] | 41 [34, 49] |
| bigram ΔP | .19 / .14 | 70 [53, 82] | 7 [1, 13] | 13 [7, 22] | 4 [−1, 9] | 6 [0, 14] | 0 | 10 [2, 20] | 23 [13, 38] |
| bigram t | .04 / −.06 | no signal (ρ orig CI includes 0) | | | | | | | |
| share MI ≥ 3 | .21 / .10 | 38 [15, 53] | 27 [20, 39] | 5 [−2, 11] | 10 [6, 16] | 20 [14, 28] | 0 | 30 [22, 42] | 35 [25, 49] |
| amod MI | .10 / .05 | 50 [1, 79] | 11 [−8, 31] | 3 [−5, 14] | 18 [4, 46] | 18 [−1, 42] | 0 | 37 [13, 78] | 40 [16, 85] |
| dobj MI | .20 / .09 | 35 [11, 52] | 6 [−1, 13] | 15 [7, 25] | 36 [25, 52] | 10 [1, 18] | −1 | 45 [31, 64] | 60 [44, 83] |
| advmod MI | .14 / .06 | 37 [−1, 59] | 21 [10, 39] | 2 [−8, 12] | 15 [5, 30] | 24 [10, 44] | 1 | 39 [23, 66] | 41 [23, 69] |
| dependency MI (mean of 3) | .22 / .10 | 40 [23, 53] | 12 [5, 19] | 8 [3, 13] | 25 [18, 34] | 16 [9, 24] | 0 | 41 [31, 54] | 48 [37, 64] |
| content-word log freq | −.14 / −.20 | 135 [114, 177] | −27 [−55, −12] | 1 [−2, 4] | −5 [−14, 2] | −5 [−14, 2] | 0 | −9 [−27, 1] | −8 [−26, 3] |
| Study 2 lemma log freq | .15 / −.08 | −38 [−90, −10] | 96 [75, 136] | 20 [13, 30] | 5 [2, 9] | 16 [11, 23] | 1 [0, 2] | 21 [15, 30] | 41 [31, 58] |
| *ref: edit rate (cov share)* | | – | 44 [41, 46] | 29 [27, 31] | 12 [10, 13] | 15 [14, 16] | 1 [0, 1] | 27 [25, 29] | 56 [53, 58] |
| *ref: % of all edits* | | – | 40 | 29 | 15 | 15 | 1 | 30 | 59 |

Surprisal, at site + spill: mech 29 + 6, FW 13 + 7, content 7 + 3, MW/order
12 + 5, uncl. 0 + 0, flagged + residual 9.

The full tables, including the no-signal rows, are in
`data/study3/pilot_buckets/tables/summary_{wi,fce}.csv`, `summary.md` and
`surprisal_at_site_spill_*.csv`.

## Reading

- **Surprisal.** The signal follows the error mix. Mechanical errors carry
  about half the W&I signal (47%) and a third of the FCE signal (35%).
  Phraseological corrections carry 35% / 27% without function words and
  49% / 47% with them. These shares are close to the buckets' shares of the
  edit-rate covariance. Relative to edit counts, surprisal gives somewhat more
  to multi-word / order (22 vs 16 on W&I) and less to function words (14 vs
  21 on W&I; 20 vs 29 on FCE). The difference comes from cost per error:
  rewrites and content words are costlier at site and spill more, and
  function-word fixes cost less. Spill is a third or less of every bucket,
  and it is largest relative to at site for function words.
- **Bigram MI and the MI ≥ 3 share.** These look like surprisal. On W&I
  correction removes all of their signal. Mechanical corrections carry
  38–39%, function words 22–23%, multi-word / order 28–29% and content words
  12–14%. On FCE a third of the signal survives correction, and the rest
  splits the same way (mechanical 27, multi-word 20–21, content 10). So
  roughly 40% of these "phraseological" measures' proficiency signal is
  spelling, punctuation and inflection. Spelling and punctuation fixes change
  which bigrams exist and which are attested. With function words counted,
  the phraseological part is 63–64% (W&I) and 35–41% (FCE).
- **Dependency MI.** This measure behaves most like a phraseological measure.
  Content-word choice is the largest correction bucket (28% / 25%),
  multi-word / order follows (24% / 16%), and mechanical is small
  (14% / 12%). The relations differ:
  - dobj is mostly content-word choice (27% / 36%);
  - advmod has a larger mechanical and multi-word part;
  - amod is noisy (ρ ≈ .09–.10, wide CIs).
  The corrected essay keeps 24% (W&I) and 40% (FCE).
- **Bigram ΔP.** Most of its signal stays in the corrected essay (66% / 70%).
  The part removed is spread thinly across buckets.
- **Bigram t.** The W&I signal is weak (ρ .11). Its corrections are mainly
  function words and multi-word / order, with CIs from about 12% to 67%. FCE
  has no signal.
- **Content-word frequency (contrast).** The corrected essay carries all of
  it (104% / 135%). On FCE, mechanical corrections work against it (−27%):
  fixing them moves the frequency measure away from the level gradient.
- **Study 2 lemma frequency.** This measure has no signal on W&I. On FCE its
  signal is a misspelling artifact: unseen lemmas sit at the count floor, the
  mechanical bucket carries 96%, and the corrected-essay sign flips (−38%).

Overall, no classic measure locates its correctable proficiency signal mainly
in phraseological corrections in the narrow sense (content-word choice +
multi-word / order). Among measures where correction removes most of the
signal, the narrow phraseological part is 27–53%. It is largest for
dependency MI and smallest for surprisal, bigram MI and the MI ≥ 3 share,
where mechanical errors carry the largest single share. Bigram ΔP is the
exception: correction removes little of its signal in the first place, so
its phraseological part is only 10–17%. Counting function words lifts the
phraseological total to 47–65% for surprisal, W&I bigram MI and MI ≥ 3, and
dependency MI. For the FCE bigram measures it stays at 35–41%.

## Files

- Code (`src/3-surprisal-decomposition/pilot_buckets/`): `bucket_common.py`,
  `1_surprisal.py`, `2_versions.py`, `3_shapley.py`, `4_summary.py`.
- Data (`data/study3/pilot_buckets/`, git-ignored):
  - `surprisal_units_*`, `versions_*`, `texts_*`, `items_*`, `measures_*`,
    `shapley_*` (parquet)
  - `tables/` (`summary_*.csv`, `summary.md`, `surprisal_*.json`,
    `shapley_*.json`, `versions_*.json`)
  - `logs/`

To run: `1_surprisal.py`; `2_versions.py` (about 20 min); `3_shapley.py`;
`4_summary.py`.
