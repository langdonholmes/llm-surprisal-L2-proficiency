# Pilot: direct logit attribution of high-surprisal tokens (W&I+LOCNESS)

Status: pilot, not committed. Agent-run on 2026-10-19, GPU 0.

## 1. What is computed

Model and scoring as in Study 3 (D1): OLMo-2 1B base, revision `a1847dff`,
8-subword window ending at the target's first subword, logits at the
second-to-last position. The pilot runs the model in **fp32** (stage 1 was bf16).

OLMo-2 is post-norm per sublayer. Each decoder layer adds
`post_attention_layernorm(o_proj(z))` and then `post_feedforward_layernorm(mlp(x))`
to the residual. The final hidden state at the prediction position is

    h_L = e  +  Σ_l [ a_l + m_l ],      a_l = w_pa,l ⊙ W_O,l z_l / r_a,l
                                       m_l = output of post_feedforward_layernorm
    r_a,l = sqrt(mean((W_O,l z_l)^2) + eps)

With r_a,l frozen at its actual value, a_l splits exactly over the 16 heads:
a_l,h = w_pa,l ⊙ W_O,l[:, block h] z_l,h / r_a,l. Logits are
W_U (w_f ⊙ h_L / r_L), with the final rms r_L frozen. Let
d = W_U[obs] − Σ_v p_v W_U[v]. The contribution of component c is

    contrib_c = d · (w_f ⊙ write_c / r_L)

over 273 components: embedding (1), heads (16 × 16), MLPs (16). By construction

    Σ_c contrib_c = logit(obs) − E_p[logit] = H − S   (nats)

because lse − E_p[logit] = −E_p[log p] = H. So the total is fully fixed by
surprisal and entropy; the pilot asks whether **how** it is split adds anything.

## 2. Pre-declared statistics (written before any gold contrast was computed)

Per token, on the 273 contributions c (nats):

| name | definition |
|---|---|
| `N` | Σ min(0, c) (≤ 0), negative mass |
| `P` | Σ max(0, c), positive mass |
| `supp` | \|N\| / (\|N\| + P), suppression share — **primary** |
| `max_neg` | −min_c c, largest single push against obs — **primary** |
| `top5_neg` | share of \|N\| in the 5 most negative components |
| `negH` | entropy of the distribution of negative mass over components, / log(273) |
| `neg_layer` | \|neg\|-weighted mean layer (embedding = 0, layer l = 1..16, attn and MLP of layer l both l) |
| `neg_mlp` | share of \|N\| from MLPs |

Hypothesis: errors are **suppressed** (higher `supp`, higher `max_neg`), acceptable
surprise is **unsupported** (lower `supp`, small `max_neg`).

Baselines: S (fp32 surprisal), H (entropy), log p(top-1), log unigram frequency
(`a_log_uni`, edited Dolma, from `attestation_wi.parquet`).

## 3. Sample

From the `wi` bench (88,761 tokens above τ = 11.28 bits; scored ones only),
POST excluded. Stratified by level (A, B, C, N) × class (ERR = ERR_TOK or ERR_INS,
OK), up to 1,000 tokens per cell, seed 20261019. N has fewer than 1,000 errors
so all are taken.

## 4. Reading rule (fixed before computing the contrast)

(a) **Errors vs acceptable.** Bin the pooled sample into S quintiles. Within
each bin, AUC of `supp` and of `max_neg` for ERR vs OK (ERR scored positive).
Also essay-grouped 5-fold CV logistic: baseline = S, H, log top-1 p, log unigram
freq; full = baseline + the 8 DLA stats (`N`, `P`, `supp`, `max_neg`, `top5_neg`,
`negH`, `neg_layer`, `neg_mlp`). Passes if the mean within-bin AUC of `supp` or
`max_neg` is ≥ 0.60 in the hypothesised direction AND ΔAUC (full − baseline)
≥ 0.02.

(b) **Acceptable tokens by level (the main target).** OK tokens only. Each stat
is residualised on S and H (cubic polynomial in each, fit on the pooled OK tokens).
Essay means of the residual; by-level means with 95% essay-bootstrap CIs
(2,000 resamples). Passes if, for `supp` or `max_neg`, the Spearman ρ of essay
residual with level (A < B < C, learners) has a bootstrap CI excluding 0 and the
A − C difference is ≥ 0.2 token-level SD, OR the N − C difference has a CI
excluding 0 and is ≥ 0.2 SD. Unadjusted means are reported too.

Verdict: **works** if (a) and (b) pass; **partially** if one does; **doesn't** if
neither. The secondary stats are descriptive only.

Sanity checks: Σ contrib = logit(obs) − E_p[logit] within fp32 tolerance (max
abs error < 1e-3 nats); embedding + Σ writes = h_L; fp32 surprisal vs the stored
stage-1 (bf16) surprisal for the same tokens.

---

## 5. Results

### Verdict: doesn't work

Neither arm of the reading rule passes. Errors and acceptable tokens share the
same attribution profile once surprisal is controlled. Among acceptable tokens
the profile does not move with level at all, while a trivial baseline (unigram
frequency) does. A post hoc variant also fails, so does the patching check. Under
OLMo-2's post-norm, per-head linear attributions are not causally meaningful.

### Sanity checks (all pass)

| check | result |
|---|---|
| Σ contrib − (logit(obs) − E_p logit) | max 8.6e-4 nats (gaps are ~5–15 nats) |
| embedding + Σ writes − h_L | max relative error 1.2e-6 |
| gap − (H − S) | max 1.1e-5 nats (the identity in §1) |
| fp32 S vs stored bench S (bf16) | mean \|Δ\| 0.040 bits, max 0.74, r = .9997 |
| stored bench S vs stage-1 `tokens_gold` S | max \|Δ\| 0.82 bits (bf16 batch noise between the two existing passes, not new) |

Sample: 7,419 tokens (1,000 per level × class, plus all 419 scored N errors) from
the `wi` bench. The OK tokens come from 1,800 essays (A 642, B 602, C 506, N 50).

### (a) Errors vs acceptable, within surprisal quintiles (AUC, ERR positive)

| stat | bin1 | bin2 | bin3 | bin4 | bin5 | mean within | pooled |
|---|---|---|---|---|---|---|---|
| **supp** | .617 | .578 | .585 | .572 | .572 | **.585** | .633 |
| **max_neg** | .535 | .543 | .538 | .511 | .502 | **.526** | .573 |
| neg_mlp | .583 | .564 | .578 | .550 | .538 | .563 | .587 |
| top5_neg | .530 | .543 | .569 | .531 | .497 | .534 | .572 |
| neg_layer | .537 | .532 | .526 | .506 | .520 | .524 | .578 |
| *baseline* log p(top-1) | .617 | .588 | .590 | .562 | .542 | .580 | .588 |
| *baseline* H (low = ERR) | .365 | .396 | .383 | .406 | .423 | .395 (= .605 reversed) | .387 |
| *baseline* log unigram freq | .577 | .575 | .576 | .537 | .488 | .551 | .519 |

`supp` is in the hypothesised direction but at .585, below .60, and no better
than the model's top-1 probability or (reversed) entropy.

Essay-grouped 5-fold CV logistic AUC:

| model | AUC | mean AUC within level |
|---|---|---|
| S | .621 | .612 |
| S + H | .662 | .657 |
| baseline (S, H, log top-1, log freq) | .664 | .674 |
| baseline + supp + max_neg | .665 | .674 |
| baseline + all 8 DLA stats | .670 | .679 |
| DLA stats only | .668 | .661 |

ΔAUC = **+.006** (rule: ≥ .02). **Fails.** The DLA stats are not just S and H in disguise:
R² of a cubic in S and H is .49 for `supp` and ≤ .16 for the others. But the
variance they add is not related to errors.

### (b) Acceptable tokens by level (the main target)

Essay means, 95% essay-bootstrap CIs. "resid" = residual on cubic S and H.

| stat | kind | A | B | C | N |
|---|---|---|---|---|---|
| supp | raw | .642 [.638, .647] | .642 [.637, .646] | .645 [.640, .650] | .642 [.637, .647] |
| supp | resid | −.001 [−.004, .003] | .001 [−.003, .004] | .000 [−.004, .004] | −.001 [−.004, .003] |
| max_neg | raw | 2.39 [2.24, 2.54] | 2.43 [2.27, 2.59] | 2.44 [2.28, 2.61] | 2.55 [2.37, 2.74] |
| max_neg | resid | −.03 [−.17, .11] | .03 [−.12, .19] | −.01 [−.15, .16] | .09 [−.07, .26] |
| *log unigram freq* | resid | .28 [.11, .46] | .03 [−.15, .20] | −.20 [−.38, −.01] | .18 [−.05, .40] |

Tests (resid; differences in token-level SD units):

| stat | ρ with level (A<B<C) | A − C | N − C |
|---|---|---|---|
| supp | .019 [−.027, .065] | −.01 [−.08, .06] | −.01 [−.09, .07] |
| max_neg | −.007 [−.053, .037] | −.01 [−.10, .08] | .04 [−.06, .14] |
| (all 8 DLA stats) | all CIs include 0 | all \|·\| < .09 | only top5_neg N − C = .12 [.02, .22], under .2 |
| *log unigram freq* | −.104 [−.153, −.058] | .18 [.08, .27] | .14 [.04, .25] |

**Fails.** The design can detect a level shift: the frequency baseline shows one,
at about 0.2 SD with CIs of ±0.1 SD. The attribution profile of acceptable surprise
shows nothing at that resolution. CIs exclude shifts larger than ~0.1 SD.

### Post hoc variant: attributing the observed word's own logit

Added after (a) and (b) had failed. In the declared attribution,
d = W_U[obs] − E_p W_U, so a component that promotes the predicted word counts
as "suppressing" obs. The variant uses d = W_U[obs] − mean_v W_U, which splits
logit(obs) − mean logit, so "negative" means pushing obs itself down (sum check:
max error 4e-5). Result: still null. Within-bin AUC for `supp` is .519 and for
`max_neg` .536; ΔAUC is +.007. Errors have **less** single-component push-down
than acceptable tokens (max_neg 1.14 vs 1.31) and less positive support
(P 15.7 vs 16.8). If anything that is the "unsupported" profile on the error side,
the reverse of the hypothesis. Among OK tokens, the centred obs logit (a new scalar,
not a split) rises with level after S/H adjustment: ρ = .109 [.062, .156],
A − C = −.09 SD [−.13, −.05], under the .2 SD bar. Files: `out/obs_*.csv`.

### Where the negative mass sits (main attribution)

The same components dominate for errors and acceptable tokens. L14 MLP is the
single largest suppressor for 56% of errors and 52% of OK tokens, then L12/L13/L15
MLPs and head L15H10 (`out/top_components.csv`). By repair-label group
(grammatical / lexical / none), errors have the same top suppressor (L14 MLP for
~55–60%). **No head is specific to agreement, inflection or any other error kind.**
Mean `supp` is .686/.675/.657 for grammatical/lexical/none errors against
.660/.643/.638 for OK tokens with the same labels: a small, uniform offset.

### Activation patching (post hoc): the per-head attribution is not causal

Mean-ablating one component at the prediction position: the actual Δlog p(obs)
against the DLA prediction (frozen norms), all 7,419 tokens (`out/patch.csv`):

| component | Spearman(pred, actual) | slope | tokens it most suppresses (bottom 10%): predicted / actual Δlog p |
|---|---|---|---|
| L15H10 | .29 | .03 | +4.14 / +0.06 |
| L14H5 | .14 | .07 | +2.14 / +0.19 |
| L14 MLP | .55 | .03 | +4.62 / +0.04 |
| L13 MLP | .50 | .28 | +2.56 / +0.92 |
| L12 MLP | .45 | .37 | +2.22 / +0.92 |

The rank order is partly right, but the magnitudes are wrong by 3–30×. The cause, for the
heads, is OLMo-2's `post_attention_layernorm`. It normalises the **sum** of the
heads, so each layer's attention write has a fixed RMS profile. On the tokens L15H10
"suppresses", that head is 91% of the layer's pre-norm attention output (cosine
.999 with the sum). Removing it shrinks the RMS to 7% and the norm scales the
remaining heads back up by ~15×. The per-head split with a frozen scale
is exact as arithmetic, but it measures a head's share of a direction whose length
the norm fixes. That says nothing about what the head causes. The late MLPs are similarly
compensated downstream (post-FF norm, then the final norm). Any per-head or
per-layer DLA on OLMo-2 would need path patching or a norm-aware attribution
(e.g. attributing the normalised direction, or AtP with gradients through the
norms) before being interpreted.

### Qualitative (agent-written; excerpts kept locally in `data/study3/pilot_dla/kwic.md`, not for publication)

- Highest-`supp` OK tokens are very high surprisal (S ≈ 18–23 bits): sentence
  starts after a full stop ("There", "It", "And"), proper names and rare
  content words. Lowest-`supp` OK tokens sit at S ≈ 11.5 bits, mostly the noun
  after "the" in a high-entropy slot. So `supp` largely reads off S and H.
- Highest-`supp` errors look the same: wrong-category words after "to",
  a missing determiner before an adjective, agreement after a plural subject.
  The suppressor is the late MLP stack in each case, not a recognisable
  agreement or determiner head.

### Costs

- GPU (A6000, GPU 0, fp32): DLA pass 10.6 s for 7,419 tokens; patching
  5 components × 3 passes ≈ 1 min; model load ~10 s per script. Total GPU time is
  under 5 minutes. Analysis CPU < 1 min.
- Disk: `data/study3/pilot_dla/` 19 MB.

### Files

- `dla.py`: sample, fp32 forward, exact DLA (main + post hoc obs variant), sanity checks.
- `analyze.py`: pre-declared stats, (a) and (b), top components; `analyze.py obs` runs the variant.
- `patch.py`: mean-ablation check.
- `out/*.csv`: all tables above.
- `data/study3/pilot_dla/`: `sample.parquet`, `contrib.npy`, `contrib_obs.npy`,
  `components.json`, `sample_stats.parquet`, `obs_sample_stats.parquet`, `kwic.md`.

Run: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python src/3-surprisal-decomposition/pilot_dla/{dla,analyze,patch}.py`.

### Recommendation

Drop the DLA idea for the write-up. If a mechanistic angle is wanted later, the
one live signal is not a split but a scalar: the observed word's centred logit
among acceptable tokens rises slightly with level beyond S and H. That is small,
and it needs a pre-registered test before it is anything.
