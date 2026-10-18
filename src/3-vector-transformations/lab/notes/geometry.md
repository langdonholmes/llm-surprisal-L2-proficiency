# Geometry agent notes (2026-10-17)

Statistics: `lab/geo_stats.py` (per-observed-subword mean top-50 prediction,
surprisal mean/sd, from all 2.09M non-space ELLIPSE dev tokens in the stage-1
shards); `lab/geo_wcpca.py` (64-PC basis of word-centered e_pred from 100k real
dev selected rows). Nothing estimated on the benchmark.

| rep | dim | detect_auc | type_f1 | type_f1_tau | lexdis_f1 | nmf_nmi_type | nmf_nmi_word | word_r2 |
|---|---|---|---|---|---|---|---|---|
| (baseline) surprisal | 4 | .871 | .350 | .239 | .270 | .068 | .223 | .387 |
| (baseline) delta | 2048 | .691 | .790 | .770 | .612 | .339 | .468 | .851 |
| geo_scal (+ word-relative surprisal z) | 6 | .886 | .431 | .389 | .321 | .079 | .263 | .328 |
| geo_wc (word-centered delta) | 2048 | .730 | .494 | .495 | .527 | .240 | .335 | .239 |
| geo_wc_s (wc + scalars) | 2054 | .730 | .494 | .495 | .527 | .240 | .335 | .239 |
| geo_top1 (W_U[top1]-W_U[obs]) | 2048 | .681 | .736 | .680 | .596 | .243 | .366 | .614 |
| geo_wc16 (16 PCs of wc) | 16 | .719 | .436 | .445 | .517 | .217 | .340 | .230 |
| geo_rel (relational, named) | 39 | .923 | .802 | .783 | .725 | .293 | .334 | .542 |
| geo_rel_noobs | 31 | .923 | .777 | .753 | .687 | .186 | .329 | .439 |
| geo_rel2 (+mismatch, margin) | 42 | .923 | .803 | .785 | .727 | .306 | .338 | .529 |
| geo_rel2_wc16 | 58 | .924 | .835 | .823 | .760 | .306 | .353 | .445 |
| geo_rel3 (+morph transitions, UPOS) | 100 | .928 | .854 | .836 | .790 | .301 | .299 | .490 |
| geo_rel3_nowc | 84 | .927 | .839 | .817 | .768 | .249 | .294 | .541 |
| geo_rel3_dense (no one-hots) | 58 | .927 | .835 | .814 | .779 | .262 | .322 | .382 |

Findings
- Delta geometry is the wrong place to look. Word-centering removes lexical
  leakage (word_r2 .85 -> .24) but also removes type information; the error
  signal left in e_pred is weak (detect .73).
- The detectable signal is relational: how the observed word relates to what
  was predicted (class of the expected word, same-stem inflection the model
  wanted, orthographic near-miss, surprisal relative to the word's usual
  surprisal). A 39-dim named vector beats delta on every supervised metric.
- Concatenating 6 scalars to a 2048-dim block does nothing in this harness
  (standardize -> 128 PCs swamps them); low-dim blocks are needed.
- Unsupervised NMF alignment stays ~.30 under the harness preprocessing.
  Exploratory check (labels used only for scoring): NMF directly on the raw
  nonnegative named blocks (masses, one-hots, morph, positive surprisal z)
  gives NMI(type) .41 at k = 9-12 vs .30 through the harness's
  standardize -> posneg path. Standardizing sparse indicators inflates rare
  columns; for named nonneg features the decomposition should run on raw
  values.

Cost: relational features ~1.2 ms/token on one core (string comparisons over
the top 10); UPOS comes free in the real pipeline (spaCy parse exists); wc16
needs one 50x2048 gather. All inputs are in the stage-1 shards (top-50 at
target and next word, obs id, surprisal), so it runs on the full corpora with
no new GPU pass.
