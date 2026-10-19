# Pilot: Shapley values over the context window (explaining the context part)

Follow-up to D24/D25. Among learners, 27% of the covariance between mean
surprisal and level is *context*: errors raise the surprisal of the correct words
after them, within the 8-subword window. D24 measures this by correcting
the essay and rescoring it. That counterfactual says *how much* each target moved.
It does not say *which* context word moved it when a window holds several
errors, or whether the model's own dependence on context points to the error.
The question here is whether an attribution method can answer that, and whether
it is the simplest one that does.

Method: exact Shapley values of the target's surprisal over its context words.
Everything runs locally. No text, span or n-gram of the learner corpora leaves
the machine. KWIC excerpts are only in `data/study3/pilot_shapley/`, not here.

Terms as in D22: tokens the annotator left alone are *uncorrected*. Tokens of
corrected essays are *post-correction*.

## Design (decided before any attribution was computed)

**Instrument.** OLMo-2 1B base, rev a1847dff, 8-subword window ending at the
target's first subword, logits at the second-to-last position
(`features.predictability.Predictor`, phase 2). Only targets with subword index
k ≥ 8 are used, so every window has exactly 7 context subwords and no padding.

**Value function.** v(S) is the surprisal (bits) of the target given only the
context players in S. φ_i > 0 means player i makes the target *more*
surprising. Efficiency: s(full) = v(∅) + Σ_i φ_i.

**Players are words (spaCy tokens), not subwords.** Each context subword belongs
to the spaCy token in which its first non-space character lies, so a
whitespace-only subword joins the word after it. Consecutive subwords of one
token form one player. The first player may be the tail of a word that the
window cuts. Reasons: (1) the gold edits and ERR_TOK are defined on spaCy
tokens, so an edited player is exactly a corrected word. (2) Removing half a
word creates a string no writer produced. (3) Fewer subsets. In practice
OLMo-2's vocabulary keeps most words whole: 76% of sampled targets have 7
players, mean 6.67, so 2^p is usually 128. Sensitivity: one player per subword
on up to 2,000 targets with fewer than 7 players.

**Removal: deletion (primary).** v(S) feeds the kept subwords contiguously at
positions 0..L−1, then the target, and reads the logits at the last kept
subword. Why this scheme:

- It is the instrument itself applied to a shorter window. The stored windows
  already start at an arbitrary subword with no BOS, so a window that starts
  later has nothing new in it.
- The last context subword is the query position, whose logits predict the target.
  Under deletion, removing that word just makes the previous kept word the
  query. Masking and replacement cannot do that: the query position has to
  hold *some* token, so "removing" the last word means putting a placeholder
  there and predicting what follows the placeholder.
- v(full) is the stored input, unchanged.
- v(∅) has no input position, so it is defined on an anchor: the target after a
  lone `<|endoftext|>` (OLMo-2's document separator), which is the model's
  document-start prediction. The anchor is only used for ∅, so v(full) stays
  equal to the instrument. v(∅) affects only the size of Σφ (efficiency), not
  how it is shared out among players.

**Alternatives (sensitivity on 1,500 candidates + 1,500 controls).**

- *mask*: attention-mask the removed words and keep the original position ids.
  It is implemented as deletion with gapped position ids. With RoPE the two
  are the same computation, which is checked directly: a 4D attention mask on the
  full window and gapped position ids agree to < 3e-5 bits in fp32 on 300
  random subsets, and HF Olmo2 honours custom `position_ids` (gapped vs
  contiguous ids differ by a median .23 bits; a constant shift of all ids
  changes nothing, < 1e-4). When the last word is removed, the query becomes
  the last kept word at its own position, so under masking too the "removed
  last word" case is deletion.
- *replace*: removed subwords replaced by `<|endoftext|>` at their positions;
  v(∅) = 7 separators, then the target. This is the common fixed-token
  baseline. Reviewers may prefer it because the input length is fixed. Its
  weakness is that the separator is a document boundary: replacing the word
  right before the target turns the target into a document-initial word. That
  inflates the last player's attribution, and the replace scheme is rejected
  as primary for this reason.

**Precision.** fp32 with TF32 off. bf16 values depend on batch composition:
the same window gives median .02 and up to .5 bits different surprisal at
batch 32 vs 4,000. Shapley values are differences of 128 such values, so that
noise is the size of a small attribution. In fp32, batch size changes values by
≤ 1.3e-4 bits.

**Exact Shapley, no sampling.** φ = W v with the closed-form weights over all
2^p subsets. Leave-one-out LOO_i = v(full) − v(full ∖ i) comes from the same
table, so it costs nothing extra.

**Sample** (`2_targets.py`, seed 42). Essays drawn by level: 80 each of A, B,
C, all 50 LOCNESS (N). In those essays, every eligible target of two kinds:

- *cand* (spillover candidates): own aligned token, gold OK or POST, at least
  one edited player in the window, at all surprisal levels. Targets that are
  ERR_INS (a word missing just before the target) are excluded, because that is
  the omission part of D24, not context. No insertion point is a player.
- *err*: ERR_TOK tokens (corrected tokens themselves).

A player is *edited* if its token overlaps a substitution or deletion
(ERR_TOK) or comes right after an insertion point (ERR_INS).

*ctrl* (matched controls) are own tokens with gold OK and no edited player,
drawn 1:1 to cand, matched on level × surprisal decile × number of players.
They come from the same essays first and then from other essays of the same
level, because A has too few clean windows. Excluded everywhere: windows with a
flagged (detection-only) span, targets whose first subword is whitespace (the
paragraph-initial newline alignment), and windows where a context subword maps
to the target's own token.

Sizes: cand 23,800, ctrl 23,800, err 5,379 (A 5,749 / 7,907 / 5,444 / 4,700
cand; B is largest because B essays are longer and have many edits).

**ERRANT types** (`1_errant.py`). The M2 files were produced from the json by
`json_to_m2.py`. M2 tokens are walked back onto the json characters, and every
M2 edit gets a character span. Each first-annotator json edit takes the type of
the M2 edit it overlaps. For insertions the tiers are tried in order: M2
insertion at the point, then a span containing it, then a span ending just
before it. All 3,350 essays align token for token with the M2 sentences.
Whitespace-only substitutions have no M2 edit (ERRANT compares tokens) and are
typed R:ORTH, ERRANT's whitespace/case category. Groups: spelling/orth (SPELL,
ORTH), verb form/SVA/tense (VERB:FORM, :SVA, :TENSE, :INFL), determiner,
preposition, punctuation, noun number (NOUN:NUM, :INFL), word order, other
lexical (NOUN, VERB, ADJ, ADV, ADJ:FORM, MORPH), other (PRON, CONJ, PART,
CONTR, NOUN:POSS, OTHER, UNK).

## Pre-specified statistics and reading rules

Written after the sample and the engine checks, before any attribution was
computed. Units: bits. CIs: 2,000 essay-bootstrap resamples (cand and ctrl
resampled by essay within level), seed 42.

Notation per target: players i = 1..p, edited set E, φ_i⁺ = max(φ_i, 0),
*slot* j = position of a player counted back from the target (j = 1 is the
word just before it). Δ = s_orig − s_corr from `aligned.parquet` (the gold
counterfactual).

**Q1. Does attribution find the error?** (cand)

- π_E = Σ_{i∈E} φ_i⁺ / Σ_i φ_i⁺, the share of positive attribution on edited
  players (targets with Σφ⁺ > 0).
- Position baseline from the controls: c_p(j) = mean over ctrl with p players
  of φ_j⁺/Σφ⁺, and π̂_E = Σ_{j∈E} c_p(j). This is the share the edited slots
  would get in clean windows of the same size, level and surprisal. Also
  reported: the naive count share |E|/p.
- Top-1 hit: argmax_i φ_i ∈ E, against the control expectation
  Σ_{j∈E} q_p(j), with q_p(j) the share of controls whose top player is
  slot j.
- Reported overall, by level, and by Δ stratum: Δ < −0.5, [−0.5, 0.5],
  (0.5, 2], > 2.
- **Reading rule.** Attribution *finds the error* if, overall, the ratio
  R = mean π_E / mean π̂_E ≥ 1.5 with a CI above 1, **and** in the Δ > 2
  stratum the top-1 hit rate is ≥ .5 and ≥ 1.5× its position expectation. If
  only the Δ > 2 part passes: it finds the error where the error matters.

**Q2. Agreement with the counterfactual.** (cand)

- φ_E = Σ_{i∈E} φ_i against Δ: Pearson and Spearman, essay-bootstrap CIs.
- Because Δ scales with the target's own surprisal, also the partial Spearman
  of φ_E and Δ given s_orig, and the Spearman of s_orig alone with Δ as the
  bar.
- Also reported for single-edit windows, split by edit kind (token vs
  insertion point). Deleting the error word is not the same intervention as
  correcting it, and an insertion's missing word is not a player at all, so
  agreement should be lower for insertions.
- **Reading rule.** ρ ≥ .5 agrees, .3–.5 partly agrees, < .3 does not. The
  attribution adds something beyond "surprising targets move more" only if
  the partial ρ given s_orig is ≥ .3.

**Q3. Spillover profile.** (cand)

- Unit: (target, edited player) pairs whose player has a single ERRANT group.
  Mean φ_i by group × distance in words (d = 1, 2, 3, 4, 5+; spaCy non-space
  tokens back from the target), with CIs.
- Per error (edited player), *spillover* = Σ φ_i over all cand targets whose
  window contains it. In the sampled essays every eligible target is
  included, so this is complete except for targets that are themselves errors,
  omissions, flagged or paragraph-initial. Reported by group.
- Counterfactual counterpart, no attribution needed: mean Δ by group ×
  distance among cand targets with exactly one edited player.
- **Reading rule.** *Reach* of a group = the largest d at which mean φ ≥ .1
  bits with a CI above 0. Groups are ranked by spillover per error. The
  Shapley profile agrees with the counterfactual one if the group ranks
  correlate ≥ .7 (Spearman, groups with ≥ 100 pairs).

**Q4. Level.** Per essay (sampled essays): context_φ = Σ_cand φ_E / n tokens,
errors per token r (edited players in at least one cand window / n), spill per
error s = Σ_cand φ_E / #edited players. Same with Δ in place of φ_E, computed
on *all* W&I essays, since Δ is available for every token.

- s by level with CIs; Spearman of s with level among learners.
- Shift-share of the A − C gap in context: (r_A − r_C)·s̄ + (s_A − s_C)·r̄,
  with bars the A/C means.
- **Reading rule.** The level effect is *just the error count* if the spill per
  error differs by less than 20% between A and C or its CI includes 0, and the
  rate term takes ≥ 80% of the shift-share.

**Q5. Simplicity check.** Q1–Q3 again with LOO_i in place of φ_i. LOO gives
the same conclusions if every Q1/Q2 reading-rule verdict is unchanged and the
Q3 group ranking correlates ≥ .8 with the Shapley ranking. Also reported: the
Spearman of φ_i with LOO_i on edited and on unedited players.

**Removal-scheme and player sensitivity.** On the 3,000-target subsample:
Spearman of φ_i between delete and mask, and between delete and replace, plus
Q1 R and top-1 hit per scheme. For subword players: the sum of subword φ within
a word, against the word φ.

**Error tokens (err).** Descriptive only: share of positive attribution on
other edited players in the window (errors explained by other errors) and on
slot 1, by group.

**Qualitative.** Among cand, the unedited players that are the target's top
positive player with φ_i ≥ 2 bits. 60 drawn (15 per level), KWIC (< 100
words) written to `data/study3/pilot_shapley/kwic_strong_unedited.txt`, and
labelled by the agent (me) reading them locally. Labels: missed error /
awkward but not wrong / sentence or clause boundary / name, number or rare
word / ordinary word, no visible reason / mixed with an adjacent error. Only
counts are reported here.

## Deviations from the plan, and what was seen when

- The analysis code was tested end to end on 24 essays (6 per level) after the
  rules above were written. That test showed the main pattern reported below:
  the removal game gives large *negative* attributions to every word, error
  words included. The pre-specified analysis was not changed. Everything under
  "Post hoc" was added after that test.
- Code fixes during the run, with no rule changes: (1) Q4 first took its essay
  set from every essay that supplied a target, which wrongly included essays
  that only supplied controls. It now uses the 290 sampled essays. (2) One
  candidate has a single player and no control of that size, so its baseline
  is the uniform share 1/p. (3) The Q2 bootstrap uses 500 resamples, not 2,000.
- The control pool for A ran short within the sampled essays, so 16% of controls
  come from other essays of the same level, as the sample section says.

## Checks

| check | result |
|---|---|
| efficiency, max \|Σφ − (v(full) − v(∅))\| | 2.8e-14 bits (all 52,979 targets) |
| bf16 rescoring (batch 32) vs stored surprisal, 4,000 windows | median 1e-7, 91% < 1e-3, max .33 (batch composition) |
| fp32 v(full) vs stored bf16 surprisal | median .020, p99 .14, 86% < .05 bits |
| fp32 batch 32 vs batch 2,048 | ≤ 1.3e-4 bits |
| 4D attention mask vs gapped position ids (fp32, 300 subsets) | max 3.1e-5 bits: masking = deletion with original positions |
| gapped vs contiguous position ids | median .23, max 6.4 bits: HF Olmo2 honours `position_ids` |
| constant shift of all position ids by 5 (fp32) | max 3.8e-5 bits (RoPE is relative) |
| sens_delete rerun vs main rows | max 2.3e-5 bits |
| ERRANT types | all 3,350 essays align token for token with M2; 98.8% of applied edits typed (sub 99.1%, del 99.9%, ins 97.5%) |
| eligible cand targets' share of the learners' D24 context Δ | 94% (controls 1%, excluded windows 5%) |

v(full) is the stored instrument's input. The values differ from the stored
ones only by bf16 rounding, and the game uses fp32 throughout.

## Results (pre-specified)

**What the game looks like.** v(∅), the target after a lone separator, has a
median of 23.4 bits, and the targets' mean surprisal is 5.7. So Σφ ≈ −17.8 bits
per target, and nearly all of it is negative. Context words lower surprisal, and
each word mostly *helps*. The positive mass is small: 0.41 bits per
candidate and 0.46 per control. An error word is still a word. Against "no
word at all", a wrong preposition or an unagreeing verb still tells the model a
lot about what comes next. Mean φ of an edited player is −2.4 bits. So the
removal game asks "does this word help, compared with nothing?" The spillover
question is "does this word hurt, compared with its correction?"

**Q1. Does attribution find the error?** Share of positive attribution on
edited players (π_E) against the position baseline from the matched controls
(π̂_E):

| cell | n | π_E | π̂_E | R [95% CI] | top-1 hit | expected | ratio |
|---|---|---|---|---|---|---|---|
| all | 23,800 | .233 | .229 | 1.02 [0.97, 1.06] | .238 | .237 | 1.00 |
| Δ < −0.5 | 4,326 | .220 | .240 | 0.92 [0.83, 1.01] | .227 | .246 | 0.92 |
| Δ in [−0.5, 0.5] | 12,146 | .170 | .209 | 0.82 [0.76, 0.88] | .207 | .225 | 0.92 |
| Δ in (0.5, 2] | 4,738 | .271 | .242 | 1.12 [1.04, 1.20] | .264 | .250 | 1.06 |
| Δ > 2 | 2,590 | .385 | .263 | 1.46 [1.36, 1.56] | .353 | .255 | 1.38 |

By level R is .98 (A), 1.00 (B), 1.06 (C), 1.07 (N), all with CIs that include 1.
**Verdict: does not find the error.** Overall, positive attribution falls on
the error exactly as often as its position predicts. Only when the correction
moves the target by more than 2 bits does the error get more than its share, and
even then it is the top player for only 35% of targets. Where the error barely
matters (half the candidates), it gets *less* than its share.

**Q2. Agreement with the counterfactual.** φ_E against Δ, all candidates:
Spearman .006 [−.007, .019], Pearson .025. Partial Spearman given s_orig is −.024.
Surprisal alone gives .199. Single-edit windows: token edits ρ .082, insertion
points ρ −.038. **Verdict: does not agree, and adds nothing beyond surprisal.**

**Q3. Spillover profile.** Mean φ on an edited player is negative at every
distance (all groups: d = 1 −6.36, d = 2 −2.69, d = 3 −1.91, d = 4 −1.70,
d ≥ 5 −1.52 bits). By the rule, reach is 0 words for every group. Spill per
error is negative for every group (−8.9 to −14.5 bits). The removal profile
mostly measures how informative a word is, which falls with distance, and not
how much it hurts. Its group ranking agrees only partly with the counterfactual
ranking in single-edit windows (Spearman .60, bar .70). The counterfactual
profile itself needs no attribution. In single-edit windows, mean Δ is 1.00 bit
at d = 1, .53 at 2, .27 at 3, .12 at 4 and .09 at 5+.

**Q4. Level.** The rule mechanically returns "just the error count" for the
removal game (A − C spill per error −16%, rate term 120% of the gap). The
numbers it rests on have the wrong sign: per token, context is −2.09 (A)
to −0.78 (C) bits, against D24's +.30 to +.09. That verdict is not read. The
counterfactual version needs no attribution. On all W&I essays, Δ per edited
player in candidate windows is A 1.53 [1.49, 1.57], B 1.49, C 1.44
[1.37, 1.51], N 1.49 bits, and Spearman with level among learners is −.03.
The error-rate term takes 95% [90, 99] of the A − C gap in context per token
(.21 bits). **The level effect in the context part is the error count. Spillover
per error is the same at every level.**

**Q5. Simplicity check.** LOO gives the same verdicts on Q1 and Q2, and the
same Q3 group ranking (Spearman .88 with Shapley). It is somewhat better at Q1
(overall R 1.18 [1.15, 1.20]; Δ > 2 top-1 .43, 1.74× expected, still under the
.5 bar) and Q2 (ρ .15). φ and LOO agree only moderately on single players
(Spearman .59 edited, .60 unedited) and pick the same top player for 32% of
candidates. **Same conclusions, so by the user's rule LOO would be enough.
Neither answers the question.**

**Removal scheme and players** (1,500 cand + 1,500 ctrl).

| scheme | Spearman φ with delete | mean φ, word just before | Q1 R, all | Q1 Δ > 2 top-1 (× expected) | verdict |
|---|---|---|---|---|---|
| delete (primary) | 1 | −7.42 | 1.09 [0.94, 1.25] | .37 (1.62) | does not find |
| mask (original positions) | .986 | −7.45 | 1.10 [0.94, 1.28] | .37 (1.61) | does not find |
| replace (separator token) | .53 | −12.56 | 1.15 [1.05, 1.25] | .32 (1.34) | does not find |

Masking and deletion are practically the same game. Replacement with the
separator differs a lot, as predicted: removing the last word makes the target
document-initial, so the last player's φ swings to −12.6. The conclusion does
not depend on the scheme. Word vs subword players (2,000 targets with
multi-subword words): word φ against summed subword φ, Pearson .97 (.95 for the
multi-subword words themselves), same top player 81%.

**Error tokens (err).** In error tokens' windows, positive attribution on
*other* edited players (when there are any, 63% of windows) is .27 of the
positive mass against .26 expected from position. Errors are not attributed to
neighbouring errors. Word-order errors are the exception (.49 vs .32).

**Qualitative.** Of 23,800 candidates, 729 (3.1%) have an unedited top player
with φ ≥ 2 bits. 60 were read (15 per level):

| label | n |
|---|---|
| sentence or clause boundary (the player is `.`, `?`, `!` or a comma, and the target starts the next sentence or clause) | 30 |
| ordinary word, no visible reason | 17 (11 of them sit just before a sentence or paragraph break that lies between player and target) |
| mixed with an adjacent error (the player touches an edit) | 11 |
| missed error | 1 |
| name, number or rare word | 1 |

Strong positive attributions to unedited words are almost never missed errors.
They are the model's sentence-boundary behaviour: with the full stop present,
the next word starts a new sentence and is less predictable than if the clause
ran on, which is what deleting the stop produces. This is the removal game
working as designed, not a signal about errors.

## Post hoc: Shapley over the edits (the game that matches the question)

Added after the test run (`6_edit_shapley.py`, `7_edit_analyze.py`). The
players are the gold edits that touch the target's window. Each has two
states, as written and corrected. v(S) is the target's surprisal with only the
edits in S applied, the text retokenised, and the stored window rebuilt. u(S) =
s_orig − v(S), so the Shapley values ψ_i split Δ = s_orig − s_corr exactly over
the individual errors. This is the D24 context part attributed error by
error, with the gold correction as the baseline in place of an empty context.

Checks: 23,762 candidates scored (38 skipped). The no-edit window equals the
stored window for 98.6%, and the all-edits window equals the corrected essay's
window for 98.4%. Only rows where both match are used (98.4%). Efficiency is
exact to 1.8e-15. v(no edits) equals the main run's v(full) to 2.5e-5. v(all
edits) vs stored s_corr: median .018 bits (fp32 vs bf16). Σψ vs Δ: r = .9996.
GPU 328 s for 75,930 windows.

- **The split is trivial for most targets.** 69% of candidate windows hold one
  edit, so ψ = Δ and no method is needed. 23% hold two, 6% three, 2% four or
  more. The multi-edit windows carry 51% of Σ Δ (43% of Σ |Δ|).
- **In multi-edit windows, LOO ranks errors like Shapley but does not add up.**
  ψ against "fix only edit i" has Pearson .92, and the top edit is the same in
  95% of windows. But Σ LOO misses Δ by 0.75 bits on average (|Δ| 1.40), so
  54% of |Δ| is interaction that LOO leaves unassigned. Shapley is the
  simplest split that adds up to D24's context part. LOO would do for ranking
  alone.
- **Spill per error by ERRANT group** (bits summed over the following targets;
  bootstrap CIs):

| group | errors | spill per error [95% CI] | share of spill | share of errors |
|---|---|---|---|---|
| other lexical | 849 | 2.66 [2.37, 2.93] | .24 | .14 |
| other | 944 | 2.45 [2.15, 2.74] | .24 | .16 |
| preposition | 550 | 2.13 [1.84, 2.41] | .12 | .09 |
| determiner | 614 | 1.97 [1.67, 2.26] | .13 | .10 |
| spelling/orth | 778 | 1.41 [1.17, 1.67] | .11 | .13 |
| noun number | 187 | 1.35 [0.96, 1.77] | .03 | .03 |
| verb form/SVA/tense | 641 | 1.10 [0.84, 1.36] | .07 | .11 |
| word order | 92 | 1.06 [0.43, 1.68] | .01 | .02 |
| punctuation | 1,275 | 0.33 [0.16, 0.51] | .04 | .21 |
| all | 6,002 | 1.60 [1.50, 1.69] | 1 | 1 |

  By kind: substitutions 2.01, deletions 1.81, insertion points 0.47 bits.
  Lexical, preposition and determiner errors spread most. Punctuation errors are
  a fifth of the errors and barely spread. Reach (mean ψ ≥ .1 with CI above 0):
  4 words overall, 1 for punctuation, 4–5 for the rest. ψ by distance, all
  groups: .89 (d = 1), .48, .24, .12, .07 bits (d ≥ 5). Determiner and
  preposition errors hurt the second word as much as the first (.86 and .83
  at d = 2), consistent with an error that disturbs the whole following
  phrase.
- **Level.** Spill per error is A 1.57 [1.40, 1.75], B 1.61, C 1.60
  [1.38, 1.81], N 1.62 bits, with Spearman .006 among learners. The rate term
  takes 102% [87, 122] of the A − C gap. Reweighting C to A's error-type mix
  changes C's spill per error from 1.63 to 1.70. **Same answer as the
  counterfactual alone: the context part tracks level only through the number
  of errors.**

## Verdict

**Shapley over context words, with removal as the baseline, is not a defensible
explanation of the spillover component.** The test is exact and checked
(efficiency to 1e-14, the full window reproduces the instrument, and masking is
shown to equal deletion). It is robust to the removal scheme and to subword vs
word players. It does not find the error beyond what position predicts (R 1.02).
Its attribution to the errors does not track the gold counterfactual
(ρ .006). Its strong non-error attributions are sentence boundaries. The
reason is conceptual, not numerical. Removal compares each word with *no word*,
and against that baseline every word, wrong or not, mostly helps. The spillover
component is defined against the *correction*, and no removal scheme supplies
that contrast. LOO reaches the same conclusions, so even within this family
Shapley is not needed.

What answers the question is simpler. (1) For which error types spread, how
far, and whether level matters beyond error count, the gold counterfactual Δ
in single-edit windows answers directly, with no attribution. (2) To split the
context part exactly over errors when a window holds several, Shapley over the
*edits* (correction as baseline) is exact, cheap (5 minutes of GPU), and adds
up to D24. It is needed only for the 31% of windows with more than one edit,
where LOO ranks the errors the same way but leaves 54% of |Δ| unassigned.

For the committee: report the counterfactual profile as the main result,
present edit-level Shapley as the exact allocation in multi-error windows, and
cite the removal-baseline result as the reason context-token attribution was
not used. Findings for the dissertation: spillover per error is about 1.5–1.6
bits at every level including native writers. It decays by half per word and is
gone after about 4 words. Lexical, preposition and determiner errors spread
most, punctuation errors least.

## GPU cost

Two RTX A6000, fp32. Main removal game, 5.84M forward passes: 2,801 s + 2,345 s
(47 min wall, 1.43 GPU-h). Sensitivity (mask, replace, delete rerun, subword):
1,128 s. Engine checks: about 2 min. Edit-level Shapley: 328 s, including 110 s
of CPU window building. Total about 1.8 GPU-hours. Not counted: the smoke
tests, about 15 GPU-min.

## Files

Code (`src/3-surprisal-decomposition/pilot_shapley/`): `common.py`,
`engine.py` (value function, exact Shapley), `1_errant.py`, `2_targets.py`,
`3_shapley.py` (GPU), `4_analyze.py`, `5_kwic.py`, `6_edit_shapley.py` (GPU,
post hoc), `7_edit_analyze.py`.
Data (`data/study3/pilot_shapley/`, unversioned): `edit_types.parquet`,
`targets_all.parquet`, `sample.parquet`, `shapley_main_{0,1}.parquet`
(full v tables), `shapley_sens_*.parquet`, `edit_shapley.parquet`,
`checks_engine.json`, `tables/` (`results.json`, `edit_results.json`, q1–q4
CSVs, `errant_coverage.csv`, `edit_*.csv`), and the local-only
`kwic_strong_unedited.txt`, `kwic_labels.csv`.
