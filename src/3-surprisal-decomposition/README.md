# Study 3. What the surprisal of learner writing is made of

Studies 1 and 2 showed that an essay's mean surprisal tracks proficiency.
Study 3 asks what that signal is made of. Each essay in two expert-corrected
learner corpora (W&I+LOCNESS, the main corpus; FCE, the replication) is
corrected, rescored with the same instrument, and the difference is split
exactly over the corrections and their ERRANT error types. A component is the
part of surprisal that would disappear if one kind of error were corrected.

**Result in one paragraph.** Among learners, the proficiency signal in mean
surprisal is the errors. The corrections remove all of it on W&I (the clean
part's share of the covariance with level is −4% [−10, 1]) and 91% on FCE.
About a quarter of the signal is errors raising the surprisal of the correct
words after them. Every kind of error tracks proficiency mainly through how
often it occurs (79–98% of the low-to-high gap), and each kind's share is
close to its share of the plain error-rate signal. Over the human error
counts, surprisal adds 0.4 points of R² on W&I and 1.5 on FCE. Native student
essays sit at the C-learner level, and more literate native prose is more
surprising, so lower is not better once errors are rare. As a post-hoc
follow-up, bigram MI and dependency MI also lose most of their proficiency
association under correction.

The proposal's δ-vector method is the appendix (`appendix/`). `DECISIONS.md`
records the design decisions in date order. `REBUILD.md` records how this
pipeline reproduces the pilots it replaced.

## Files

| Stage | File | What it does |
|---|---|---|
| 0 | `0_threshold.py` | τ from the stored ELLIPSE token surprisal (`1_extract.py`) and `split.csv`. |
| 1 | `1_correct.py --corpus wi\|fce` | Applies the first annotator's corrections, types every edit with ERRANT (group, bucket), records each token's gold category, parses the corrected essays. |
| 2 | `2_score.py --corpus` (GPU) | Scores the original and the corrected essays with the Study 1 instrument. |
| 3 | `3_align.py --corpus` | Aligns the writer's own tokens between the two versions. |
| 4 | `4_spill.py --corpus` (GPU) | Edit-level Shapley split of each own word's surprisal change over the corrections in its window (correction as baseline). |
| 5 | `5_decompose.py` | The exact splits per unit (four parts; by kind of correction and ERRANT group), shares of the proficiency signal, frequency vs cost, spill by distance, the clean part across corpora, RQ3 features. |
| 6 | `6_increment.py` | RQ3: ridge ΔR² of the surprisal terms over the error counts. |
| 7 | `7_classic.py` | Post hoc. Classic measures on every partially corrected version, Shapley by kind of correction, ρ, share removed, ΔR². |
| 8 | `8_calibration.py prep\|score\|table` | Mean surprisal of PERSUADE, Wikipedia, W&I, LOCNESS and ELLIPSE documents of 200–600 words. |
| — | `lib/` | `corpora.py` (paths, units), `corrections.py` (applying edits, gold categories), `errant.py` (ERRANT types, groups, buckets), `shapley.py`, `classic.py`, `lm.py` (the instrument), `stats.py`. `bench.py` serves `2_select.py` and the appendix. |
| — | `1_extract.py`, `2_select.py` | The appendix's GPU pass and selection. `1_extract.py --corpus ellipse` also stores the ELLIPSE token surprisal that stage 0 reads. They keep their numbers because the appendix imports their outputs. |
| A1–A6 | `appendix/` | The δ-vector analysis. |

Reports: `results.qmd` (main) and `appendix/appendix.qmd`.

## Running

```bash
S=src/3-surprisal-decomposition
python src/essays/ingest.py --wi --fce         # docbins from the BEA-2019 releases
CUDA_VISIBLE_DEVICES=0 python $S/1_extract.py --corpus ellipse   # if not already stored
python $S/0_threshold.py
for c in wi fce; do python $S/1_correct.py --corpus $c; done
CUDA_VISIBLE_DEVICES=0 python $S/2_score.py --corpus wi &      # 31 min on an A6000
CUDA_VISIBLE_DEVICES=1 python $S/2_score.py --corpus fce; wait # 25 min
for c in wi fce; do python $S/3_align.py --corpus $c; done
CUDA_VISIBLE_DEVICES=0 python $S/4_spill.py --corpus wi &      # 22 min
CUDA_VISIBLE_DEVICES=1 python $S/4_spill.py --corpus fce; wait # 15 min
python $S/5_decompose.py
python $S/6_increment.py
python $S/7_classic.py                          # CPU, about 15 min with 24 processes
python -I $S/8_calibration.py prep              # reads the public downloads
CUDA_VISIBLE_DEVICES=1 python $S/8_calibration.py score   # 18 min
python $S/8_calibration.py table
quarto render $S/results.qmd
```

Outputs go to `data/study3/main/{wi,fce,calibration}/` (large, not versioned)
and `results/study3/main/` (tables and figures for `results.qmd`). They sit
in `main/` subfolders because `data/study3/wi/` and `results/study3/` already
hold the appendix's files (`1_extract.py` writes `data/study3/wi/essays.parquet`,
and the appendix reads `2_select.py`'s fuller `results/study3/threshold.json`).

The calibration downloads are public: PERSUADE 2.0 (`persuade_train.bin`, the
train CSV from github.com/scrosseye/persuade_corpus_2.0, CC BY-NC-SA 4.0) in
`data/study3/main/calibration/dl_persuade_train/`, and two random samples of
English Wikipedia (CC BY-SA 4.0) in `dl_wiki/`. The Wikipedia samples were
drawn once by `fetch_wiki.py` and are reused, because a new draw would give
different articles.

## Known issues

- The 8-subword window limits spill to the next few words, so the spill
  shares are lower bounds.
- The corrections are one annotator's minimal corrections. They mark errors,
  not quality. Tokens the annotator left alone are *uncorrected*, which
  includes effective choices, awkward but not wrong phrasing and missed
  errors.
- 1.2% (W&I) and 0.7% (FCE) of corrections get no ERRANT type and count as
  unclassified. Flagged spans have no correction and are a separate part.
- The partial versions in stage 4 are scored in fp32 from re-tokenised text,
  the essays in bf16. Each word's Shapley values are shifted to add up to its
  stored change (median gap .02 bits). Five W&I words with more than 7
  corrections in the window split their change equally.
- `4_spill.py` scores own words only. The pilot also scored error tokens, for
  a robustness check that is not reported, so its fp32 values differ from the
  pilot's by up to 1e-4 bits (batch composition).
- For the single dependency relations (amod, dobj, advmod), `7_classic.py`
  uses the units where the measure is defined in every partial version.
- Licences: W&I+LOCNESS and FCE are for non-commercial research only, without
  redistribution, and excerpts must be short. Cite Yannakoudakis et al.
  (2018) and credit CECL for LOCNESS; cite Yannakoudakis, Briscoe & Medlock
  (2011) for FCE. Never send their text, or any span of it, to an external
  service.
