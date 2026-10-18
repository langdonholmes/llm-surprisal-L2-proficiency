# Hidden-state representations (lab notes)

Owner: hidden-state idea agent. Files: `reps/hid_*.py`, `hid_states.py`,
`hid_feats.py`, `hid_ident.py`; caches in `data/study3/lab/hidden/` (bench-aligned
caches carry the LAB_BENCH suffix).

## Round 1 — states already in base.npz (layers 8, 16)

| rep | detect_auc | type_f1 | type_f1_tau | lexdis_f1 | nmi_type | nmi_word | purity | word_r2 |
|---|---|---|---|---|---|---|---|---|
| surprisal (baseline) | .871 | .350 | .239 | .270 | .068 | .223 | .262 | .387 |
| delta (baseline) | .691 | .790 | .770 | .612 | .339 | .468 | .492 | .851 |
| hid_pred8 | .709 | .442 | .407 | .508 | .081 | .274 | .289 | .186 |
| hid_obs8 | .885 | .849 | .825 | .770 | .422 | .412 | .607 | .402 |
| hid_obs16 | .864 | .812 | .806 | .706 | .256 | .390 | .443 | .470 |
| hid_nobs8 | .884 | .773 | .772 | .690 | .141 | .326 | .354 | .222 |
| hid_dobs8 | .879 | .842 | .811 | .773 | .376 | .344 | .511 | .359 |

Reading: the state *after reading* the target at a middle layer beats both
baselines on every axis. The prediction-side state (h_pred) knows little: an
error lives in the interaction of the word with its context, which only exists
once the word has been read.

## Round 2 — all layers, counterfactual, consequence, identity removal

`hid_states.py` (GPU, 4 forward passes of 8 tokens per instance, about 6 ms per
token on one A6000, so ~1.7 h per million tokens) caches all-layer h_obs, the
same with the model's top-1 prediction substituted (h_cf), and next-word
consequence scalars. `hid_ident.py` estimates the identity component of h_obs
on 692k ordinary tokens from 1,500 dev essays with Overall < 4.0 (none hosts
a bench instance). Per-type means are shrunk toward a ridge map from W_E.

| rep | detect_auc | type_f1 | type_f1_tau | lexdis_f1 | nmi_type | nmi_word | purity | word_r2 |
|---|---|---|---|---|---|---|---|---|
| hid_conseq (surprisal + consequence scalars) | .892 | .405 | .304 | .339 | .073 | .261 | .243 | .305 |
| hid_obsL2 | .860 | .842 | .821 | .733 | .373 | .467 | .501 | .767 |
| hid_obsL4 | .876 | .849 | .828 | .767 | .387 | .439 | .553 | .561 |
| hid_obsL6 | .877 | .840 | .817 | .752 | .383 | .402 | .520 | .442 |
| hid_obs8 | .885 | .849 | .825 | .770 | .422 | .412 | .607 | .402 |
| hid_obsL10 | .886 | .848 | .826 | .771 | .451 | .447 | .626 | .429 |
| hid_obsL12 | .885 | .843 | .834 | .746 | .400 | .454 | .552 | .539 |
| hid_obs16 | .864 | .812 | .806 | .706 | .256 | .390 | .443 | .470 |
| hid_obsmid (mean L6-12) | .887 | .850 | .830 | .761 | .452 | .456 | .607 | .498 |
| hid_cfL4 (h_obs - h_cf) | .866 | .803 | .773 | .741 | .452 | .423 | .611 | .454 |
| hid_cfL8 | .877 | .805 | .770 | .744 | .373 | .391 | .501 | .340 |
| hid_cfL12 | .878 | .796 | .769 | .718 | .338 | .402 | .505 | .451 |
| hid_cfL16 | .859 | .746 | .723 | .657 | .183 | .327 | .369 | .414 |
| hid_resL4 (identity removed) | .868 | .689 | .664 | .683 | .344 | .322 | .494 | .246 |
| hid_resL8 | .876 | .678 | .656 | .684 | .235 | .324 | .398 | .182 |
| hid_resL12 | .879 | .704 | .688 | .682 | .252 | .336 | .433 | .221 |
| hid_res8c | .894 | .701 | .682 | .692 | .264 | .292 | .423 | .182 |
| **hid_obs8c** (h_obs L8 + 9 scalars) | **.904** | .853 | .838 | .774 | .411 | .402 | .563 | .401 |

Findings
- Layer profile: type information peaks at layers 8-10 and falls at the final
  layer, which is specialised for next-token prediction. Word leakage falls
  from layer 2 (.77) to layer 8 (.40).
- The counterfactual contrast does not beat plain h_obs. Subtracting the
  expected-token state removes context that the error type depends on.
- Identity removal works as intended on leakage (word_r2 .18) but costs type
  information (lexdis .77 -> .68). Error types are partly properties of word
  forms (an inflected *wants*, a non-word *thier*), so fully decorrelating
  from the word throws signal away.
- The consequence scalars (next word scored after the writer's word vs the
  expected word) give the best single detection signal and add to h_obs.
- Unsupervised NMF alignment is the weak point for every representation
  (nmi_type at most .45, about the same as nmi_word). Probes can read type
  off these vectors, but the dominant axes of variation are not type axes.
