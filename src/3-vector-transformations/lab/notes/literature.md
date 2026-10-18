# Novelty review (2026-10-17, sub-agent report, abstracts verified, full texts mostly not read)

## Closest prior art for the repair representation
- Bryant & Briscoe 2018, LM-based GEC without annotated data, BEA. https://aclanthology.org/W18-0529/
  Confusion sets (spelling, inflection, determiners, prepositions) scored by an
  LM, with an edit applied only above a threshold. The generator, the scorer and
  the NONE logic are all here.
- Yasunaga, Leskovec & Liang 2021, LM-Critic, EMNLP. https://aclanthology.org/2021.emnlp-main.611
  A sentence counts as grammatical when the LM prefers it to every nearby edit.
  This is the "no edit pays" test.
- Park & Levy 2011 (ACL, P11-1094) and Flachs, Lacroix & Søgaard 2019 (BEA,
  W19-4420), noisy-channel GEC with edit costs.
- Alikaniotis & Raheja 2019 (BEA, W19-4412), transformer LMs for the same.
- ERRANT, Bryant, Felice & Briscoe 2017 (ACL, P17-1074). Map the 29 repair
  classes onto ERRANT types.
- GECToR, Omelianchuk et al. 2020. Token-level edit tags, supervised.
- Graliński et al. 2024, "Oddballness" (arXiv 2409.03046). Raw low probability
  is a poor error detector.

## Explaining predictions / δ
- Yin & Neubig 2022, contrastive explanations (EMNLP).
- Ross et al. 2021, MiCE: minimal contrastive edits as explanations.
- Vylomova et al. 2016, classifying embedding offset vectors (ACL).
- Linzen 2016, offset vectors are dominated by the individual words'
  neighbourhoods (RepEval). This is the published precedent for "δ is mostly
  observed-word identity".
- Murphy et al. 2012, NNSE: non-negative sparse factorisation of embeddings.
- Bills et al. 2023 and Paulo et al. 2024 for the explain/simulate pipeline.
- Analytic point: with the full softmax, ∂loss/∂h = W_Uᵀ(p − y) =
  E_p[e] − e_obs. δ is the last-layer loss gradient truncated to the top 50,
  so dominance by e_obs is expected, since an expectation over many embeddings
  has a smaller norm than any single one. Gradient embeddings: BADGE, Ash et
  al. 2020.
- No prior NMF of e_pred − e_obs for error types was found.

## Surprisal and L2 proficiency, error profiles, NLI
- Cong 2025 (CSL 89:101700); Hu & Cong 2025 (CMCL); Tang & van Hell 2024
  (EMNLP 2024.emnlp-main.715).
- Muñoz Sánchez, Dobnik & Volodina 2024 (BEA 2024.bea-1.34): perplexity vs
  CEFR, and original vs normalised essays. Some perplexity spikes are not
  errors. This is the closest work on what drives surprisal.
- NLI from errors: Koppel et al. 2005; Wong & Dras 2009; Malmasi & Dras 2014;
  Tetreault et al. 2013 (TOEFL11 shared task); Rozovskaya & Roth 2011
  (L1-specific article and preposition priors).
- Synthetic errors and minimal pairs: Felice & Yuan 2014; Marvin & Linzen 2018;
  BLiMP 2020. No error-detector benchmark with lexeme-matched correct controls
  was found.

## Assessment
- **δ-NMF.** The components are prior art and the combination is new but
  weak. Report it as an explained negative result (gradient argument, Linzen
  2016).
- **Repair as GEC.** Proposing and scoring edits, noisy-channel costs, the
  threshold/NONE class and edit tags are all prior art.
- **Plausibly new:**
  - using the measure's own model to explain its own surprisal (faithful
    explanation);
  - splitting high-surprisal tokens into repairable vs. acceptable;
  - unsupervised, surprisal-anchored essay-level edit profiles as proficiency
    and L1 features.
- **Framing:** "unsupervised LM-based correction repurposed as an
  interpretation of surprisal", not a new correction method.
- **Must cite and position against:** Bryant & Briscoe 2018; LM-Critic 2021;
  Muñoz Sánchez et al. 2024 (plus Cong 2025); ERRANT for the taxonomy.
