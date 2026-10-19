# Pilot: native calibration of mean surprisal

**Question.** Under the project's surprisal instrument, how does the mean
surprisal of known-native writing compare with advanced (C-level) learner
writing? This is a descriptive calibration, not a hypothesis test, so no
reading rules were set in advance.

**Short answer.** Native student essays on the same argumentative prompts sit
at **5.5–5.6 bits**. That is the same level as W&I C (5.60) and ELLIPSE Overall
≥ 4 (5.59). On this metric the advanced learners are already at the native
student level. The learner range runs about 1 bit above it, up to W&I A (6.60)
and ELLIPSE ≤ 2.5 (6.86). More sophisticated native prose scores *higher* than
the native student essays: LOCNESS university essays 5.94, Wikipedia 5.73–5.77.
Lower mean surprisal therefore does not simply mean "more native". Once errors
are rare, topic, register and lexical choice set the level. Memorization moves
the statistic only a little. Wikipedia text that is verifiably in OLMo-2's
training mix scores 0.04 bits *below* post-cutoff Wikipedia (CI −0.05 to
+0.13).

## Instrument

OLMo-2 1B base (`allenai/OLMo-2-0425-1B` @ `a1847dff`), 8-subword window ending
at each word's first subword, Study 1's `Predictor` (causal path), batch 32,
bf16 forward with fp32 softmax. spaCy `en_core_web_lg` with NER disabled, as
in `src/essays/ingest.py`. Per-document statistic: mean surprisal (bits) over
scored tokens. Scored tokens are the spaCy tokens whose start lines up with a
subword start. Whitespace tokens are never scored: 0 of the 12,916 scored
ELLIPSE tokens in the check were whitespace. This is exactly Study 1's
`mean_surprisal_bits`. High-token share is the share of scored tokens with
surprisal ≥ τ = 11.276 bits (`results/study3/threshold.json`).

**Check (`verify.json`).** I re-parsed 25 random W&I and 25 random ELLIPSE
essays from raw text and rescored them with `score.py`. All 5,230 W&I tokens
and all 12,916 ELLIPSE tokens matched the stage 1 shards exactly (max |Δ| = 0).
The ELLIPSE essay means match the Study 1 table to 9e-16 bits. The model
revision is asserted at load time.

## Sources

| Group | Source | Licence / access | Dates | In OLMo-mix-1124? (infini-gram) |
|---|---|---|---|---|
| W&I A/B/C, LOCNESS (N) | BEA-2019 W&I+LOCNESS v2.1, stage 1 shards in `data/study3/wi` | non-commercial research, no redistribution (already held) | W&I c. 2014–18; LOCNESS 1990s | not checked (licence; see below). Public GEC releases have circulated since 2019, so presence in web data is plausible |
| ELLIPSE bands | ELLIPSE, Study 1 / stage 1 (`data/study3/ellipse`) | as held by the project | essays 2010s, released 2022–23 | not checked (licence). Distributed as Kaggle/GitHub CSV, like PERSUADE (which is absent) |
| persuade_native, persuade_native_hi, persuade_ell | PERSUADE 2.0 train CSV, <https://github.com/scrosseye/persuade_corpus_2.0> (public Google Drive link in the README, no registration) | CC BY-NC-SA 4.0 | essays 2010s, released 2021 (Kaggle) and 2023–24 | **no: 0/900 spans** (100 docs per group) |
| wiki_old | English Wikipedia, articles created Sep 2018–Dec 2021, text of the last revision before 2023-01-01 | CC BY-SA 4.0 | revisions 2019–2022 | **yes: 97/100 docs, 79% of spans** |
| wiki_new | English Wikipedia, articles created Jan–Sep 2026, current revision (fetched 2026-10-19) | CC BY-SA 4.0 | 2026 | essentially no: 3/100 docs, 1% of spans (stock phrases such as "the council can override it with a two-thirds") |

**Contamination check (`contam.py`).** For each sampled document, three
random 15-word spans (from within one paragraph) were counted in infini-gram's
`v4_olmo-mix-1124_llama` index. That index is OLMo 2's pretraining mix: DCLM
web, Wikipedia, peS2o, arXiv, StarCoder and so on. Free prose essentially
never repeats a 15-word span by chance, so a count ≥ 1 means the text, or a
copy of it, is in the mix. 36 of the 1,500 external queries failed after
retries and count as misses.

PERSUADE is absent: it is distributed as a CSV on Kaggle and Google Drive,
which web crawls do not pick up. Wikipedia behaves as expected.

**Licensed learner corpora were not checked, and a check was stopped.** I
started the same spot check on W&I+LOCNESS and ELLIPSE essays and then stopped
it on the coordinator's instruction. These corpora are licensed with
no-redistribution terms, and sending their text to an external service was
not approved. About 5,350 15-word spans went to the infini-gram API before the
stop:

- an initial 30-doc sample: 90 spans each from W&I C, LOCNESS and ELLIPSE;
- a full-W&I run: about 2,376 spans from W&I A, about 2,397 from W&I B, and
  roughly 300 from W&I C before the job was killed (14:00 on 2026-10-19). That
  run's output was never written.

None of those results are used here. The query inputs and the 30-doc results
(`learners_for_contam.parquet`, `wi_for_contam.parquet`,
`contam_learners.csv` in the data directory) are left in place but unused, and
should be deleted if the user prefers. Whether LOCNESS, W&I or ELLIPSE are in
OLMo-2's training data is therefore not established in this pilot.

**Not used, and why.** Each of these needs registration or licence acceptance,
which I did not do on the user's behalf:

- ICNALE: native-speaker essays on the same two prompts as the Asian learners;
  ideal for prompt control. Registration required.
- BAWE: Oxford Text Archive licence.
- MICUSP: free to browse online, but the terms restrict bulk download.
- LOCNESS full: CECL request. Only the 50 BEA essays are held.

News or opinion text from 2026 would be a better "edited argumentative native"
anchor than Wikipedia. The Guardian Open Platform needs an API key, and most
outlets do not licence reuse, so I used post-cutoff Wikipedia instead.

## Preprocessing

- **PERSUADE.** Train CSV, one row per essay, **independent task only**:
  source-based essays copy source text. CRLF converted to LF, nothing else
  changed.
  - `persuade_native`: `ell_status == "No"`.
  - `persuade_native_hi`: the same with holistic score ≥ 5 of 6.
  - `persuade_ell`: `ell_status == "Yes"`. 210 of these 300 are also ELLIPSE
    essays (ELLIPSE was drawn from this pool). The PERSUADE spot check sent
    spans of the public PERSUADE release, so some of those spans are also
    ELLIPSE text.
  - 19 essays fall in both native groups.
  - Grades: native mostly 8 and 11, native_hi mostly 11, ELL spread over 8,
    11 and 12.
- **Wikipedia (`fetch_wiki.py`).** Random page ids inside each creation
  window. Page ids are assigned in creation order, and the window edges were
  located with the creation log. Pages are kept if they are current mainspace
  non-redirects. For `wiki_old` the text is the last revision before
  2023-01-01; for `wiki_new` it is the current text. The prose is the text of
  `<p>` elements only: no lists, tables, infoboxes, headings or captions, and
  reference markers removed. Pages are dropped if they have fewer than 200
  words of prose or sit in a disambiguation, AI-generated or machine-translation
  cleanup category.
  - Keep rates: 300 of 792 candidates (old), 300 of 668 (new).
  - Artifact: 33 `wiki_old` docs contain unexpanded `Template:Lang-xx` text,
    because templates were renamed after 2022. About 48 occurrences in total;
    see the sensitivity row below.
- **Length rule (all groups, `prep.py`).** Keep documents with at least 200
  whitespace words. Truncate anything longer than 600 words at the last
  sentence end that leaves ≤ 600 words. Sample 300 per group, seed 42.
- **Learner groups.** The same rule is applied to the stored token rows
  (`analyze.py`): essays under 200 words are dropped, and tokens past the
  truncation point are dropped. This is the "len" subset, which is the
  headline. Because the window is 8 subwords, truncation does not change any
  token's surprisal; it only changes which tokens are counted.
  - "All essays" rows are in `group_table.csv`.
  - W&I A keeps only 186 of 1,430 essays under this rule; most A essays are
    under 200 words.

## Results (length-matched, 200–600 words)

Values are per-document mean surprisal (bits) and the share of tokens ≥ τ.
The 95% CIs come from 2,000 bootstrap resamples of documents, seed 42.

| Group | n | median words | mean | SD | median | 95% CI | high share | 95% CI |
|---|---|---|---|---|---|---|---|---|
| W&I A | 186 | 235 | 6.60 | 0.59 | 6.58 | 6.52–6.69 | .159 | .153–.164 |
| W&I B | 473 | 254 | 5.98 | 0.52 | 5.92 | 5.93–6.03 | .125 | .122–.129 |
| W&I C | 507 | 268 | **5.60** | 0.48 | 5.57 | 5.56–5.64 | .106 | .104–.109 |
| LOCNESS (N) | 49 | 425 | 5.94 | 0.44 | 5.84 | 5.82–6.06 | .131 | .122–.140 |
| ELLIPSE ≤ 2.5 | 1490 | 371 | 6.86 | 0.81 | 6.75 | 6.82–6.90 | .166 | .164–.169 |
| ELLIPSE 3–3.5 | 3428 | 411 | 6.04 | 0.54 | 5.99 | 6.02–6.06 | .118 | .117–.119 |
| ELLIPSE ≥ 4 | 1090 | 468 | **5.59** | 0.40 | 5.56 | 5.56–5.61 | .093 | .091–.094 |
| PERSUADE ELL | 300 | 409 | 6.13 | 0.64 | 6.03 | 6.06–6.20 | .119 | .114–.124 |
| PERSUADE native | 300 | 457 | **5.61** | 0.45 | 5.58 | 5.56–5.66 | .094 | .091–.098 |
| PERSUADE native, score ≥ 5 | 300 | 586 | **5.54** | 0.36 | 5.52 | 5.50–5.58 | .093 | .091–.096 |
| Wikipedia 2018–21 (in training) | 300 | 408 | 5.73 | 0.65 | 5.77 | 5.66–5.81 | .139 | .134–.143 |
| Wikipedia 2026 (post-cutoff) | 300 | 392 | 5.77 | 0.57 | 5.81 | 5.71–5.84 | .145 | .141–.148 |

![group means](../../../data/study3/pilot_native/out/group_means.png)

The figure is at `data/study3/pilot_native/out/group_means.png`.

**Contrasts** (difference of means, bootstrap CI; `out/contrasts.csv`):

| Contrast | Δ mean (bits) | 95% CI | Δ high share | 95% CI |
|---|---|---|---|---|
| PERSUADE native − W&I C | +0.01 | −0.05, +0.08 | −.012 | −.016, −.008 |
| PERSUADE native − ELLIPSE ≥ 4 | +0.03 | −0.03, +0.08 | +.002 | −.002, +.005 |
| PERSUADE native ≥ 5 − ELLIPSE ≥ 4 | −0.04 | −0.09, +0.00 | .000 | −.003, +.003 |
| PERSUADE native − PERSUADE ELL | −0.52 | −0.61, −0.43 | −.025 | −.030, −.019 |
| LOCNESS − W&I C | +0.34 | +0.22, +0.47 | +.025 | +.016, +.035 |
| LOCNESS − PERSUADE native | +0.33 | +0.20, +0.46 | +.037 | +.027, +.047 |
| Wikipedia 2026 − Wikipedia 2018–21 | +0.04 | −0.05, +0.13 | +.006 | +.000, +.012 |
| the same, without the 33 Template: docs | +0.06 | −0.05, +0.16 | +.007 | +.001, +.013 |
| Wikipedia 2026 − PERSUADE native | +0.16 | +0.08, +0.24 | +.050 | +.045, +.055 |

**Prompt-matched (`out/prompt_matched.csv`).** PERSUADE's seven independent
prompts all occur in ELLIPSE. Restricted to those prompts, ELLIPSE ≥ 4 is 5.66
(n = 121), PERSUADE native 5.61 and native ≥ 5 5.54. Averaged within prompt
(6 prompts with ≥ 5 essays per side), native − ELLIPSE ≥ 4 = −0.04 bits.
Prompt does not explain the overlap.

## Reading

1. **Student essays.** Within this genre, top learners and native writers are
   indistinguishable on mean surprisal. The comparison is cleanest within
   PERSUADE and ELLIPSE: same prompts, same US school population, neither in
   the training data. There, native essays (5.61) and ELLIPSE ≥ 4 (5.59)
   coincide. The best-scored native essays are at most about 0.05 bits lower,
   with the CI touching 0. The proficiency gradient is a learner-side
   gradient: about 1–1.3 bits from the lowest band to C/≥ 4 level, and native
   writing does not extend it further. This fits the idea that most of the
   learner excess comes from errors and non-target collocations. Once those
   are gone, the 8-subword window has little left to reward.
2. **High-token share** follows the same pattern. Native students .094 vs
   ELLIPSE ≥ 4 .093. W&I C is slightly higher (.106) than native students,
   about one extra high token per 80 tokens, but the two come from different
   corpora with different prompts.
3. **More skilled or more informative native prose is *more* surprising.**
   LOCNESS university essays (+0.33 bits over native teens) and Wikipedia
   (+0.16, with a high share of .14–.15 driven by names, numbers and dates)
   are both higher. Mean surprisal is not a monotone "nativeness" scale above
   the C level. It mixes error-driven surprisal, which falls with proficiency,
   with content and lexical-sophistication surprisal, which rises with
   register and information density. LOCNESS (N) above W&I C in Study 3 is
   this effect, not a defect.
4. **Memorization is a small effect at this window.** Wikipedia that the
   model trained on (97% of sampled docs hit) is only 0.04–0.06 bits lower than
   2026 Wikipedia, with CIs including 0. With an 8-subword context the model
   cannot recall a specific document. If any of the learner or LOCNESS
   corpora are in the training data (unchecked, see above), that should bias
   their means by roughly 0.15 bits at most, the upper CI bound. The two Wikipedia samples also differ in topic mix (2026
   articles cover recent events and people), so the gap mixes memorization
   with topic and is not a clean estimate of either.

## Caveats

- **"Non-ELL" is not "native speaker".** PERSUADE's flag is a school
  classification. Non-ELL includes former ELLs and bilingual heritage
  speakers. LOCNESS is the only source that is native by design.
- **Genre and register.** Only PERSUADE matches the argumentative,
  prompt-driven genre. Wikipedia is expository and encyclopedic, and LOCNESS
  is university and A-level writing on different topics. Topic is probably as
  large an effect as nativeness: compare Wikipedia vs PERSUADE (+0.16) with
  PERSUADE ELL vs native (0.52).
- **Age and grade.** PERSUADE writers are US grades 8–12. ELLIPSE writers are
  grade 8–12 ELLs in the same programs. W&I writers are adult Cambridge
  learners. LOCNESS writers are 18+.
- **Length rule.** It drops most W&I A essays (186 of 1,430 kept), so the "len"
  A row is the longer, probably better, A essays. All-essay rows are in
  `group_table.csv`; the ordering is unchanged.
- **8-subword window.** The statistic is local predictability. It cannot see
  discourse-level coherence or memorized long-range context, which is also
  why memorization barely registers. A full-context model would likely show
  larger memorization and genre effects.
- **Wikipedia 2026.** It could contain LLM-assisted or translated text, which
  might be unusually predictable. Pages in the AI-generated and
  machine-translation cleanup categories were dropped, but unflagged cases
  remain. Some 2026 articles also reuse older text (splits, stock phrasing):
  3 of 100 docs had a span in the mix.
- **Contamination check limits.** It is a spot check (3 spans per document).
  A miss does not prove absence, and paraphrased or partial copies are not
  detected.
- **Licences.** The downloads are kept only in the git-ignored
  `data/study3/pilot_native/`. Do not redistribute: PERSUADE is BY-NC-SA, and
  Wikipedia is BY-SA with attribution to the article histories.

## Files

Code (`src/3-surprisal-decomposition/pilot_native/`):

- `fetch_wiki.py`: Wikipedia samples.
- `prep.py`: sampling and length rule.
- `score.py`: scoring, plus `verify` against stage 1.
- `contam.py`: infini-gram spot check. Use it only on public text; never
  on the licensed learner corpora.
- `analyze.py`: per-document means and the group table.
- `contrasts.py`: contrasts, prompt-matched comparison, contamination split
  and the figure.

Data (`data/study3/pilot_native/`, git-ignored):

- `dl_persuade_train/`, `dl_wiki/`: raw downloads.
- `docs.parquet`: the 1,500 external documents as scored.
- `tokens.parquet`: token rows.
- `verify.json`.
- `contam_external.csv`: PERSUADE and Wikipedia only.
- `out/`: `group_table.csv`, `doc_means.csv`, `contrasts.csv`,
  `prompt_matched.csv`, `contam_split.csv`, `group_means.png`.

Rerun:

```
D=data/study3/pilot_native; P=src/3-surprisal-decomposition/pilot_native
python -I $P/fetch_wiki.py $D/dl_wiki
python -I $P/prep.py $D/dl_persuade_train/persuade_train.bin $D/dl_wiki $D/docs.parquet
CUDA_VISIBLE_DEVICES=1 python $P/score.py verify
CUDA_VISIBLE_DEVICES=1 python -I $P/score.py docs $D/docs.parquet $D/tokens.parquet
python -I $P/contam.py $D/docs.parquet $D/contam_external.csv 100
python -I $P/analyze.py $D/docs.parquet $D/tokens.parquet $D/out
python -I $P/contrasts.py $D/docs.parquet $D/out
```

(In this run, PERSUADE and Wikipedia were scored as two separate parquets and
concatenated; the PERSUADE sample was checked to be identical across the two
`prep.py` runs.)
