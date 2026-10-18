# Study 3 representation lab

A testing and piloting framework for the question the first pass raised: the
plain delta vector δ = e_pred − e_obs is close to 90% observed-word identity
(see `../premises.qmd`), so its NMF components describe *which word* was
surprising. This lab measures, against ground truth, whether another token
representation describes *what kind of error* occurred, and does so well
enough to be decomposed and read.

## Pieces

| File | Role |
|---|---|
| `bench.py` | Builds the synthetic benchmark. One controlled edit per instance (8 error types) in strong dev essays, plus surprising-but-unedited tokens (`CLEAN_SURP`) and a **lexeme-matched clean control** for every error (the same word form, used correctly, in another essay). |
| `extract.py` | One GPU pass with Study 1's windowing. Stores top-50 distribution, surprisal, entropy and hidden states (layers 8 and 16, at the prediction position and after the observed token) for the target and the next word. |
| `lm.py` | Shared model access (`LM.locate`, `LM.window`, `LM.score`) for representations that need extra forward passes. |
| `harness.py` | `Ctx` (what a representation may read) and `evaluate` (the battery). |
| `run.py` | `run.py <rep> [...]` featurizes, caches, evaluates, appends to the scoreboard. |
| `reps/` | One module per representation. Baselines: `surprisal`, `e_obs`, `delta`, `e_pred`. Repair: `rep_repair_subpos` (essay profiles), `rep_mdl_plus_pos` (acceptability model). Acceptability: `att_full_s`, `ctx_excess(_s4)`, `comb_rac`. |
| `notes/` | Logs for the repair and attestation families, and the literature check. |

Two benchmarks share the code. The development bench (default) is what ideas
are iterated on. The **confirmation bench** (`LAB_BENCH=conf`) uses different
host essays (dev, Overall = 3.5) and another seed, and is only run on
finalists, so that iterating against the scoreboard cannot pass for
generalization.

## The battery

The headline is **detect_auc**: the AUC for telling an error from a correct use
of the same word form, cross-validated by essay. Word identity scores .5 by
construction. Then come **lexdis_f1** (error type on held-out words),
**nmf_nmi_type** vs **nmf_nmi_word** (do unsupervised components align with
error type or with the observed word?), **word_r2** (lexical leakage) and
**type_f1** / **type_f1_tau** (9-way type, all instances / above τ). See the
`harness.py` docstring for the exact definitions.

## Success criteria (fixed 2026-10-17, before any idea agent reported)

A representation counts as an effective explainability representation if, on
the **confirmation** bench, it:

1. detects at least as well as the four surprisal scalars (detect_auc ≥ .845),
   since a representation that knows less than surprisal about *whether* a
   token is an error cannot explain *which* error it is;
2. generalizes type across words better than word identity (lexdis_f1 > .616);
3. decomposes into components that track error type more than the observed
   word (nmf_nmi_type > nmf_nmi_word), and more than δ does (> .318);
4. is computable at corpus scale from one or a few forward passes per token.

It then has to carry over to real essays. Its components are labelled from
cards, and its essay profiles enter the proficiency ladder against B0 and M_obs.

## Baselines

| rep | bench | detect_auc | type_f1 | lexdis_f1 | nmf_nmi_type | nmf_nmi_word | word_r2 |
|---|---|---|---|---|---|---|---|
| surprisal | dev | .871 | .350 | .270 | .068 | .223 | .387 |
| e_obs | dev | .423 | .802 | .641 | .315 | .493 | .973 |
| delta | dev | .691 | .790 | .612 | .339 | .468 | .851 |
| e_pred | dev | .729 | .464 | .532 | .267 | .354 | .329 |
| surprisal | conf | .845 | .342 | .278 | .069 | .230 | .387 |
| e_obs | conf | .421 | .798 | .616 | .316 | .480 | .971 |
| delta | conf | .678 | .785 | .613 | .318 | .467 | .852 |
| e_pred | conf | .705 | .462 | .526 | .255 | .349 | .314 |

Two findings from building the benchmark itself. τ (90th percentile) catches
83–90% of injected agreement, verb-form and spelling errors, but only 3% of
determiner substitutions and 26% of preposition swaps. The selected set is
therefore structurally blind to most function-word substitution errors. And
the four surprisal scalars beat δ at detection by a wide margin.

## Confirmation results (held-out bench, finalists of each family)

| rep | family | dim | detect_auc ≥ .845 | lexdis_f1 > .616 | nmi type > word, and > .318 | passes |
|---|---|---|---|---|---|---|
| `rep_repair_subpos` | repair | 29 | .922 | .807 | .419 > .354 | **all four** |
| `rep_mdl_subpos_pos` | repair | 31 | .896 | .763 | .390 > .313 | **all four** |
| `rep_mdl_plus_pos` | repair | 99 | .938 | .839 | .333 ≈ .332 | 1, 2 |
| `geo_rel3` | geometry | 100 | .913 | .775 | .343 < .357 | 1, 2 |
| `hid_obs8c` | hidden | 2057 | .883 | .754 | .379 < .403 | 1, 2 |
| `delta` (baseline) | — | 2048 | .678 | .613 | .318 < .467 | none |

Every family improves on δ when a classifier is trained on top of it. Only the
repair representations keep that improvement in an *unsupervised*
decomposition, where components follow error type more than the observed word.
They are also the only ones whose dimensions arrive with names.

The geometry family (relations among the stored top-k vectors) and the
hidden-state family (probes on layer 2–16 residual states) were removed from
the code on 2026-10-18 and remain in the git history before that date, with
their notes. Their rows above are kept as the record of the comparison.
