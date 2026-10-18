# Repair representations (lab notes)

## Idea
Ask the LM which minimal edit would make the local text expected. For the
observed word w (left context = Study 1's 7+1 window, right context R = next 5
subwords), score candidate edits by

    gain = log2 p(X_edit + R | L) - log2 p(X_obs + R | L)   (bits)

split into gx (edited material) and gr (right context). Candidates (≈23 per
token): MORPH (rule-based inflections + closed irregular table), FORM (vocab
words within OSA distance ≤2, ≤1 for short words; contractions list), PRED
(top-8 predictions as replacement), DEL, INS (top-5 predictions inserted
before w). Subclasses come from closed grammatical lists (DET, PREP, PRON,
AUX, CONJ, PUNCT, CONTENT) and string relations (S/ED/ING/IRREG, nonword vs.
real word vs. contraction), plus observed POS (spaCy) to split −s into noun
(number) vs. verb (agreement).

Channel cost: net gain gm = gain − 1 bit × characters edited (Shannon's
≈1 bit/char; fixed a priori, not tuned). This stops "delete or replace the
surprising word" from winning on any surprising word.

No featurizer reads labels. Files: `repair_core.py` (vocab, morphology,
scorer), `repair_build.py` (candidate table, cached at
`data/study3/lab/repair/cands{SUFFIX}.parquet`), `repair_feats.py`.

## Scoreboard (development bench)

| rep | dim | detect_auc | type_f1 | type_f1_tau | lexdis_f1 | nmf_nmi_type | nmf_nmi_word | nmf_purity |
|---|---|---|---|---|---|---|---|---|
| surprisal (baseline) | 4 | .871 | .350 | .239 | .270 | .068 | .223 | .262 |
| delta (baseline) | 2048 | .691 | .790 | .770 | .612 | .339 | .468 | .492 |
| rep_repair5 | 5 | .950 | .627 | .577 | .642 | .240 | .292 | .349 |
| rep_repair15 | 15 | .956 | .743 | .687 | .683 | .258 | .305 | .380 |
| rep_repair_plus | 25 | .958 | .804 | .772 | .758 | .338 | .317 | .456 |
| rep_repair_sub | 29 | .940 | .877 | .856 | .805 | .414 | .400 | .519 |
| rep_repair_subpos | 29 | .942 | .846 | .829 | .826 | .481 | .354 | .578 |
| rep_mdl_subpos | 29 | .917 | .787 | .779 | .768 | .516 | .346 | .635 |
| rep_mdl_subpos_pos | 31 | .917 | .798 | .792 | .784 | .527 | .337 | .578 |
| rep_mdl_plus | 97 | .955 | .911 | .902 | .854 | .429 | .344 | .493 |
| rep_mdl_plus_pos | 99 | .955 | .912 | .905 | .851 | .419 | .369 | .531 |
| rep_mdl_label | 29 | .914 | .683 | .647 | .657 | .043* | .050* | .186* |

\* The harness NMF standardizes a sparse one-hot matrix and fails. Read
directly (argmax edit after channel cost, NONE if no edit pays), the discrete
label on the above-τ error + CLEAN_SURP set has NMI(type) .536, NMI(word) .464,
purity .717. 92% of injected errors get a paying edit, against 22% of
lexeme-matched correct words and 48% of CLEAN_SURP (which is noisy: host essays
≥ 4.5 still contain natural errors).

Discrete label × type (selected rows): MORPH_S_N → NUM 503 / AGR 1;
MORPH_S_V → AGR 307 / NUM 2; MORPH_ING → VFORM 455; FORM_NONWORD → SPELL 777;
FORM_REALWORD → HOMO 508; INS_DET → DET_DEL 243; PRED_PREP → PREP 237;
NONE → CLEAN_SURP 625.

## Cost
GPU scoring 14.5 ms/token (A6000, bf16, ≈23 sequences of ~15 subwords each);
candidate generation ≈7–10 ms/token single-threaded Python (parallelizable).
1M tokens ≈ 4 h GPU + CPU candidate building. Candidate count and the
right-context length are the levers if this needs to be cheaper.

## Caveats
- The benchmark is synthetic. The edit inventory mirrors the injected types
  (morphology, typos, determiners, prepositions, homophones), so these numbers
  show the method recovers *those* edits. Real learner errors include word
  order, multi-word and lexical/collocational problems, which fall into
  PRED_CONTENT / DEL_CONTENT ("unexpected content word") or NONE.
- DET_SUB rarely exceeds τ (3%), so above-τ metrics barely test it.
- MORPH_IRREG mixes agreement (is/are) and tense (was/is). A tense-vs-person
  split of the irregular table would separate them.
- Right context is only 5 subwords and the left window is Study 1's 8 tokens,
  so long-range errors are invisible.
- AGR/NUM separation relies on spaCy POS of the possibly erroneous word.
