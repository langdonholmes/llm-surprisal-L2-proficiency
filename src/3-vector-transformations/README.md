# Study 3 — Decomposing high-surprisal tokens

What kind of unconventionality do the tokens above the surprisal threshold
contain? Study 3 answers this twice. The first answer is the proposal's method:
delta vectors δ = e_pred − e_obs, decomposed by NMF. The second is an LM
repair representation, which the premise checks and a piloting lab showed to be
needed. `plan.md` is the full design. `DECISIONS.md` records where this first
pass departs from it, the frozen choices, and the confirmatory results.

**Result in one paragraph.**

- **δ is mostly word identity.** The surprisal is exactly Study 1's (OLMo-2
  1B, 8-token window). Close to 90% of δ's variance is the identity of the
  observed word, by construction, so its NMF (k = 50, stable) recovers 40
  single-word components out of 50.
- **The repair profile names errors at no loss of prediction.** It names an
  edit at each token, usually the correction itself, and separates errors from
  acceptable words on blind-coded real tokens. All of its labels fall with
  proficiency.
- **No profile predicts better than another.** On the held-out test set, the δ
  profile, the observed-word profile and the repair profile each add about
  .05–.06 R² over mean surprisal and length.
- **The threshold is blind to function-word substitutions.** τ misses most
  determiner and preposition substitutions.
- **High surprisal is two opposite signals.** A classifier trained on expert
  edits (W&I+LOCNESS) splits tokens above τ into errors and acceptable but
  unexpected words (held-out AUC .87). With proficiency, errors fall steeply
  and acceptable surprise rises. Native writers have the most acceptable
  surprise. The split raises the CV R² for Overall from .33 to .45.

## Files

| Stage | File | What it does |
|---|---|---|
| 1 | `1_extract.py` | One GPU pass per corpus through Study 1's `Predictor` (via its `on_score` hook). Stores surprisal, entropy, observed id and the top-50 distribution for every scored token. |
| 2 | `2_deltas.py` | Dev/test split (`split.csv`), τ, token selection, δ in W_U. Checks per-essay agreement with the Study 1 table. |
| 3 | `3_premises.py` | Premise checks: composition, δ variance decomposition, identity R², essay level, and the blinded coding sample (`coding_sample.csv`, `coding_key.csv`, `error_coding_claude.csv`). |
| 4 | `4_nmf.py` | k sweep (split-half stability) and final NMF for δ, −e_obs and e_pred. |
| 5 | `5_components.py` | Component summaries and KWIC cards (dev only). Labels: `component_labels.csv`. |
| 6 | `6_proficiency.py` | Essay profiles, the baseline ladder (RQ3), per-component correlations (RQ2/RQ4), H2. Dev-CV by default; `--confirmatory` reads test once and requires the freeze entry in `DECISIONS.md`. |
| 7 | `7_toefl.py` | TOEFL 11 for δ: transfer, replication NMF, Study 1's L1 effect sizes. |
| 8 | `8_repair.py` | The repair representation on real essays: `collect` (from lab caches), `check` (coded sample, label cards), `l1` (the pre-registered article test). |
| 9 | `9_wi.py` | W&I+LOCNESS gold categories per token (ERR_TOK, ERR_INS, POST, OK) and the gold lab bench; base rates by CEFR level. |
| 10 | `10_decompose.py` | Fits the acceptability model (`lab/reps/comb_rac`, GBM) on W&I train and applies it to ELLIPSE and TOEFL. Per-essay error and acceptable rates, the EM prior adjustment (D16), and the D15 tests. |
| — | `lab/` | The representation lab (synthetic benchmark, evaluation battery, confirmation bench) and the repair, attestation and context code the real-data stages use. See `lab/README.md`. |

Reports: `premises.qmd`, `results.qmd`. The lab's results are in `lab/README.md`.

## Running

```bash
# GPU pass (one per corpus, ~1.5-2 h each on an A6000)
CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/1_extract.py --corpus ellipse
CUDA_VISIBLE_DEVICES=1 python src/3-vector-transformations/1_extract.py --corpus toefl11
python src/3-vector-transformations/2_deltas.py
python src/3-vector-transformations/3_premises.py
python src/3-vector-transformations/4_nmf.py sweep --k-grid 10   # one process per k
python src/3-vector-transformations/4_nmf.py fit --k 50 --rep delta   # and obs, pred
python src/3-vector-transformations/5_components.py --rep delta

# repair representation on real essays (per bench; ~13 ms GPU per token)
python src/3-vector-transformations/lab/real.py --corpus ellipse --split dev   # and test; toefl11 --part 0/2, 1/2
LAB_BENCH=real_ellipse_dev LAB_LAYERS= python src/3-vector-transformations/lab/extract.py
LAB_BENCH=real_ellipse_dev python src/3-vector-transformations/lab/repair_build.py
python src/3-vector-transformations/8_repair.py collect --benches real_ellipse_dev real_ellipse_test real_toefl11_p0 real_toefl11_p1
python src/3-vector-transformations/8_repair.py check

python src/3-vector-transformations/6_proficiency.py            # dev-CV
python src/3-vector-transformations/6_proficiency.py --confirmatory
python src/3-vector-transformations/7_toefl.py
python src/3-vector-transformations/8_repair.py l1

# gold validation and the error / acceptable-surprise split (D13, D15-D17)
python src/essays/ingest.py --wi            # data/wi_locness/raw from the BEA-2019 release
CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/1_extract.py --corpus wi
python src/3-vector-transformations/9_wi.py select && python src/3-vector-transformations/9_wi.py base
# per bench (real_wi, real_ellipse_dev/test, real_toefl11_p0/p1): repair, attestation, context views
LAB_BENCH=real_wi python src/3-vector-transformations/lab/att_build.py   # att_slot.py runs on demand
LAB_BENCH=real_wi CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/lab/ctx_score.py \
    olmo1b_w8 olmo1b_w64 mlm olmo7b_w8 olmo7b_w64              # ~16 ms GPU per token
python src/3-vector-transformations/lab/accept.py --gbm comb_rac            # W&I train CV
python src/3-vector-transformations/lab/accept.py --confirm --gbm comb_rac  # once
python src/3-vector-transformations/10_decompose.py fit
python src/3-vector-transformations/10_decompose.py essays

quarto render src/3-vector-transformations/results.qmd
```

Outputs go to `data/study3/` (large, not versioned) and `results/study3/`.

## Known issues

- The first pass leaves out much of `plan.md`: LLM explain/detect scoring,
  human validation, the §8 sensitivity analyses, B3/B4/B7, the manifest chain,
  and tests (`DECISIONS.md` D2).
- The blinded coding (150 tokens) and the δ component labels have one coder,
  Claude. Both are set up for a human second pass.
- The lab's synthetic error types overlap the repair inventory by design. The
  real-data checks are the stronger evidence.
- The 8-token window limits both representations to local errors.
- Summed P(acceptable) is biased toward the training base rate, and the
  unadjusted acceptable rate gets the proficiency trend wrong. The EM
  adjustment (D16) fixes this on W&I and on the blind-coded ELLIPSE tokens,
  but it is a little steep for weak essays. It was chosen after the
  confirmation run, which is the second use of the held-out data.
- W&I gold marks errors, not quality. "Acceptable" is not evidence of
  effective writing.
- W&I+LOCNESS licence: non-commercial research only, published excerpts
  under 100 words, cite Yannakoudakis et al. (2018), credit CECL for
  LOCNESS. The data are not redistributed.
