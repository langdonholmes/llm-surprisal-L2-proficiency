# Rebuild of the Study 3 pipeline (D34): acceptance checks

The rebuilt stages 0–8 were run end to end on 2026-10-20 and compared with the
pilot outputs they replace. "Exact" means a maximum absolute difference below
1e-15 (floating-point summation order). The comparison script read the pilot
folders under `data/study3/pilot_*`, so it is not kept in the tree.

GPU time: 2_score 31 min (W&I) + 25 min (FCE), 4_spill 22 + 15 min,
8_calibration 18 min. About 1.9 GPU-hours in all, run on two A6000s in
parallel.

## Acceptance table

| Check | Decision | Target (pilot) | Rebuilt | Match |
|---|---|---|---|---|
| τ | D19, `2_select.py` | 11.275990 bits | 11.275990 bits | exact |
| Corrected texts | D22, D25 | 3,350 W&I, 2,469 FCE texts | identical strings | exact |
| ERRANT type per edit | D27, D28 | 74,968 W&I, 53,258 FCE edits | identical types | exact |
| Gold category per token | D22, D25 | `tokens_gold`, `pilot_fce/gold` | identical | exact |
| Token surprisal, original and corrected | D22, D25 | 708,020 + 722,734 W&I, 531,284 + 529,233 FCE tokens | max diff 0.0 bits | exact |
| Alignment (own, aligned, s_corr, same window) | D22, D25 | `aligned.parquet` | identical; 99.96% / 99.99% of own tokens align | exact |
| Edit-level Shapley (own words) | D27, D28 | statuses, players, v, ψ | statuses and players identical; v and ψ within 1.2e-4 bits (fp32) | yes, see note 1 |
| D24 part means, A/B/C/N, both measures | D24 | e.g. total 6.36/5.92/5.56/5.94, clean 5.25/5.29/5.28/5.66 | same | exact |
| D24 shares of cov, with CIs and ρ | D24 | clean −4 [−10, 1], omission 8, context 27, error tokens 69; high rate −19/11/24/85 | same (CIs too) | exact |
| D24 native − C | D24 | +.38 [.26, .50] total, +.38 clean | same | exact |
| D25 part means by score band | D25 | e.g. total 6.38 … 5.61 | same | exact |
| D25 shares of cov, with CIs and ρ | D25 | clean 9 [3, 14], omission 12, context 22, error tokens 58; high rate −4/16/18/70 | same (CIs too) | exact |
| D28 shares by ERRANT group (at site, spill, total; 34 columns per corpus) | D28 | `pilot_types/tables/results_*.json` | max diff 8e-10 (point), 2.4e-9 (CI) | yes, see note 1 |
| D33 surprisal shares by bucket | D33 | W&I mech 47, FW 14, content 13, multi-word 22; FCE 35/20/10/17 | max diff 1.3e-9 | yes, see note 1 |
| D33 edit-rate reference shares | D33 | W&I 50/21/10/16; FCE 44/29/12/15 | same | exact |
| D33 classic shares by bucket (all 9 measures) | D33 | `pilot_buckets/tables/shapley_*.json` | max diff 3.6e-15 | exact |
| D32 ρ original / corrected | D32 | bigram MI .23/.00, .31/.14; ΔP .24/.16, .19/.14; content freq −.18/−.19, −.14/−.20; dep. MI, MI ≥ 3, lemma freq | same | exact, except note 2 |
| D32 ΔR² over B, corrected measure | D32 | e.g. bigram ΔP .25 [.02, .46] / .71 [−.00, 1.42]; content freq .52 / .70 | same | exact |
| D31 R² and ΔR² (B + S compact) | D31 | W&I .581 → .585, Δ +.0037 [.0007, .0068]; FCE .481 → .496, Δ +.0152 [.0056, .0250] | same | exact |
| D31 other models | D31 | B + mean surprisal Δ .0000 / +.0119; mean surprisal .183 / .329; S compact .034 / .034 | same | exact |
| D31 clean part vs criterion | D31 | W&I partial .118 [.083, .153] (mean), .135 (rate); FCE ρ −.078 [−.135, −.025], partial −.076, −.009 | same | exact |
| D26 calibration means and n | D26 | W&I 6.60/5.98/5.60, LOCNESS 5.94, ELLIPSE 6.86/6.04/5.59, PERSUADE 6.13/5.61/5.54, Wikipedia 5.73/5.77 | same | exact |
| D26 calibration CIs | D26 | e.g. W&I C [5.56, 5.64] | within .007 bits | yes, see note 3 |
| D26 Wikipedia 2026 − 2018–21 | D26 | +.04 [−.05, .13] | +.04 [−.06, .14] | yes, see note 3 |

## Notes

1. **Shapley values (fp32).** `4_spill.py` scores only the writer's own words.
   The pilot (`pilot_types/2_shapley.py`) also scored the error tokens, for a
   downstream-reassignment check that the write-up does not use. Dropping
   those windows changes the composition of the fp32 batches, so the scored
   windows differ from the pilot's by at most 1.2e-4 bits (median 0). The
   players, statuses and gap assignments are identical. The shares that use
   them (D28, D33 surprisal) move by at most 2.4e-9.
2. **Single dependency relations on W&I.** ρ for amod, dobj and advmod MI
   differs from D32 by up to .0024, and the share removed by up to .03. The
   rebuild computes every statistic on the units where the measure is defined
   in all partial versions (the sample the bucket Shapley needs; 41–61 W&I
   units have no item of a relation in some version). D32 used, for each
   version, the units where that version is defined. The mean of the three
   relations (the reported dependency MI) has no missing units and matches
   exactly, as does every FCE value.
3. **Calibration CIs.** The pilot bootstrapped two statistics per group from
   one random stream (mean surprisal and high-token share). The rebuild
   bootstraps the mean only, so later groups draw different resamples. The
   means and n are identical. The contrasts' CIs are new in the rebuild's
   table and move by the same amount.

## What changed relative to the pilots

- One pipeline replaces nine pilot folders. Corrections, ERRANT typing and
  gold categories are computed once per corpus (stage 1) instead of in three
  places.
- The four buckets and the nine ERRANT groups come from one credit table
  (stage 5), so the bucket split (D33) and the group split (D28) are the same
  run.
- Not ported, because the write-up does not use them. They are the D21/D22 Q1–Q4
  analyses, context-word Shapley (D27), the downstream reassignment check
  (D28), the CEFR sublevels, the operation cut (missing / replacement /
  unnecessary), S full and drop-one models (D31), the LLM association
  measures and p̄ (D32), bigram t and the t ≥ 2 and unattested shares (D32),
  the Wikipedia fetcher and the contamination check (D26), and the native vs C
  controls (D23).
