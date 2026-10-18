# Attestation & lexis evidence (acceptable-surprise half)

These features are computed on the CPU from the Dolma reference tables
(corpus a = edited prose, b = unedited) and from the existing spaCy DocBins. No
LM quantities are used and no gold labels are read. Build scripts:
`att_build.py` (n-grams, dependency arcs, lexis, semantic fit),
`att_slot.py` (slot / forward / backward attestation) and `att_common.py`.

The 3-gram tables count lowercased tokens with two start pads per document, so
unigram(w) is the sum over token_2 == w and bigram(a, b) is the sum over
(token_1, token_2). Both are slight undercounts, since trigrams with count 1
were pruned.

Cost on 88,761 tokens: about 75 s for the main build (57 s of it is the
DocBin and vector context) and about 70 s for slot features. That comes to
about 15–25 min per million tokens on 8 threads.

## accept_scoreboard (dev mode: 5-fold CV grouped by essay, inside W&I train; OK base rate .637)

| rep | dim | auc_err_ok | auc_ins_ok | prec_ok@50 | A / B / C |
|---|---|---|---|---|---|
| surprisal (baseline) | 4 | .696 | .747 | .769 | .685 / .689 / .700 |
| att_ngram | 26 | .718 | .742 | .782 | .716 / .707 / .701 |
| att_dep | 17 | .608 | .557 | .707 | |
| att_lex | 28 | .674 | .621 | .748 | |
| att_sem | 5 | .579 | .592 | .687 | |
| att_slot | 16 | .629 | .700 | .723 | |
| att_core (21 named) | 21 | .721 | .699 | .786 | |
| att_all | 72 | .743 | .731 | .801 | .739 / .736 / .728 |
| att_full (all + slot) | 88 | .748 | .742 | .805 | |
| att_core_s | 28 | .768 | .773 | .818 | |
| att_core2_s | 31 | .772 | .775 | .821 | |
| att_all_s | 76 | .784 | .791 | .830 | |
| **att_full_s** | 92 | **.787** | **.792** | **.832** | .779 / .775 / .775 |

Nonlinear ceiling. Gradient boosting with the same grouped CV gives
att_full .799 and att_full_s .824, against .748 and .787 for the harness's
logistic probe on 64 PCs. Part of the signal sits in thresholds and
interactions.

## What describes kinds of acceptable surprise (W&I train, above τ)

| category | share in OK | share in errors | P(OK \| category) |
|---|---|---|---|
| proper noun / number | .125 | .063 | .78 |
| rare content word, attested on both sides | .055 | .023 | .81 |
| common content word, attested on both sides | .510 | .346 | .72 |
| formal register (edited ≫ unedited frequency) | .230 | .145 | .74 |
| repeated in essay (topic word, ≥ 3×) | .230 | .214 | .65 |
| common content word, unattested combination | .147 | .201 | .56 |
| rare content word, unattested combination | .068 | .155 | .43 |
| function word | .070 | .145 | .46 |
| punctuation | .025 | .065 | .40 |
| nonword (unseen in Dolma) | .009 | .091 | .15 |

Base P(OK) is .64.

Readings:

- **Attested on both sides** is the strongest generic sign of acceptability.
- **Rare and attested** is the purest acceptable category (P(OK) .81). It is
  the best candidate for "sophisticated / effective" surprise.
- **Formal register** words (more frequent in edited than unedited prose)
  lean acceptable.
- **Unattested combinations** and **function words or punctuation** lean
  error.
- **Topic repetition** is close to neutral.
- **Univariate signals are weak.** The best single attestation feature has
  AUC .58, and surprisal is .63 inverted. The evidence is cumulative.

## Caveats

- Attestation reads lowercased surface n-grams of a 1B-token sample. Rarer
  but correct combinations are unseen (sparsity), and corpus b contains
  learner-like errors.
- Gold is the first annotator's minimal edits. Stylistic infelicities left
  uncorrected count as OK.
- W&I dev and LOCNESS were not touched.
