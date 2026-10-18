# Study 3 — What the high-surprisal tokens are

Studies 1 and 2 showed that the share of high-surprisal tokens tracks
proficiency. Study 3 asks what those tokens are.

- **How much is error.** On W&I+LOCNESS, which has expert corrections, every
  token above τ is either corrected (an error) or left alone (acceptable). No
  model is involved.
- **What acceptable surprise looks like.** Whether the word and its bigrams are
  attested in the Dolma reference corpora from Study 2.
- **What the error is.** LM repairs: the same model scores about 23 minimal
  edits at each token, and the best edit that pays for itself names the problem.
  Checked against the W&I corrections, then read as essay profiles on ELLIPSE.

The proposal's δ-vector method, and why it describes words rather than errors,
is the appendix (`appendix/`). `DECISIONS.md` records the design decisions in
date order.

**Result in one paragraph.** On W&I+LOCNESS, errors fall steeply with
proficiency (69 → 17 per 1,000 tokens from A to C) while acceptable surprise
rises (59 → 83), so total surprise understates the error signal. Native writers
have the most acceptable surprise, at a C-level error rate. Rare words in
combinations attested in edited prose are the most reliably acceptable kind of
high token. Grammatical repair labels (inflection, spelling, an inserted or
deleted function word) lean towards corrected tokens, and the insertion,
inflection and nonword labels are corrected 80–90% of the time; content-word
substitutions and NONE are mostly acceptable. On ELLIPSE every frequent repair
label but inserted punctuation falls with Overall.

## Files

| Stage | File | What it does |
|---|---|---|
| 1 | `1_extract.py` | One GPU pass per corpus through Study 1's `Predictor`. Stores surprisal, entropy, observed id and the top-50 distribution for every scored token. |
| 2 | `2_select.py` | ELLIPSE dev/test split (`split.csv`), τ, the tokens above τ in every corpus, W&I gold categories, and one bench per corpus part for the repair pass. Checks per-essay agreement with the Study 1 table. |
| 3 | `3_repair.py` | LM repairs per bench: `base` and `cands` (GPU), `labels` (CPU). |
| 4 | `4_attestation.py` | Dolma attestation per bench (CPU). |
| 5 | `5_gold.py` | W&I+LOCNESS: errors vs acceptable surprise by level, attestation types, and the repair labels against the corrections. |
| 6 | `6_profiles.py` | Repair label rates as essay profiles: ELLIPSE vs Overall, W&I by level (`proficiency`); KWIC cards (`cards`); TOEFL 11 by L1 (`l1`, not in the report). |
| — | `lib/` | Shared code: the model (`lm.py`), benches (`bench.py`), repair candidates and labels, attestation, statistics. |
| A1–A6 | `appendix/` | The δ-vector analysis: deltas, premise checks, NMF, component cards, the proficiency ladder, TOEFL 11 transfer. |

Reports: `results.qmd` (main) and `appendix/appendix.qmd`.

## Running

```bash
# GPU pass, one per corpus (~1.5-2 h each on an A6000)
CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/1_extract.py --corpus ellipse
python src/essays/ingest.py --wi            # data/wi_locness/raw from the BEA-2019 release
CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/1_extract.py --corpus wi
python src/3-surprisal-decomposition/2_select.py

# repairs (~13 ms GPU per token) and attestation, per bench
for b in ellipse_dev ellipse_test wi; do
  CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/3_repair.py base $b
  CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/3_repair.py cands $b
done
python src/3-surprisal-decomposition/3_repair.py labels ellipse_dev ellipse_test wi
python src/3-surprisal-decomposition/4_attestation.py wi

python src/3-surprisal-decomposition/5_gold.py
python src/3-surprisal-decomposition/6_profiles.py proficiency
python src/3-surprisal-decomposition/6_profiles.py cards
quarto render src/3-surprisal-decomposition/results.qmd

# appendix (delta vectors)
python src/3-surprisal-decomposition/appendix/A1_deltas.py
python src/3-surprisal-decomposition/appendix/A2_premises.py
python src/3-surprisal-decomposition/appendix/A3_nmf.py sweep --k-grid 10   # one process per k
python src/3-surprisal-decomposition/appendix/A3_nmf.py fit --k 50 --rep delta   # and obs, pred
python src/3-surprisal-decomposition/appendix/A4_components.py --rep delta
python src/3-surprisal-decomposition/appendix/A5_proficiency.py --confirmatory
python src/3-surprisal-decomposition/appendix/A6_toefl.py   # needs the TOEFL 11 pass and benches
quarto render src/3-surprisal-decomposition/appendix/appendix.qmd
```

Outputs go to `data/study3/` (large, not versioned) and `results/study3/`.
TOEFL 11 (`--corpus toefl11`, benches `toefl11_p0`, `toefl11_p1`) runs through
the same stages when it is extracted; the main report does not use it.

## Known issues

- The 8-token window limits both surprisal and repairs to local errors.
- τ misses most determiner and preposition *substitutions*, which rarely make a
  word very surprising. Omissions, agreement, verb form and spelling are caught.
- W&I gold is one annotator's minimal corrections. It marks errors, not
  quality, so "acceptable" is not evidence of effective writing.
- The repair edit inventory is single words. Word order and phrasing errors get
  a content-word repair or NONE.
- W&I+LOCNESS licence: non-commercial research only, published excerpts under
  100 words, cite Yannakoudakis et al. (2018), credit CECL for LOCNESS. The data
  are not redistributed. FCE (`data/fce/raw/`, downloaded, not yet used) has
  the same terms and cites Yannakoudakis, Briscoe & Medlock (2011).
