# Study 1 — Implementation Plan: Optimal LLM-Surprisal Configurations

**Goal.** For each learner essay, compute mean token surprisal under a matrix of language models × context windows, then select the configuration that best balances *validity* (correlation with holistic proficiency) and *fairness* (cross-L1 score displacement). The winning configuration feeds Studies 2–3.

**Scope of this handoff.** Surprisal computation already exists. This plan specifies (a) the model matrix and (b) the statistical analysis. Implement the analysis pipeline and the model-loading/inference glue around the existing surprisal code.

---

## 1. Model Matrix

Eleven models. Surprisal is computed at three context windows (8, 64, full document) for each, on both ELLIPSE and TOEFL 11 → **33 configurations × 2 corpora**.

| Model | HF identifier | Arch. | Params | Max ctx | Training data | Post-trained |
|---|---|---|---|---|---|---|
| BERT-base (uncased) | `google-bert/bert-base-uncased` | Masked | 110M | 512 | BooksCorpus, Wikipedia | No |
| ModernBERT-base | `answerdotai/ModernBERT-base` | Masked | 149M | 8192 | Web, code, scientific | No |
| ModernBERT-large | `answerdotai/ModernBERT-large` | Masked | 395M | 8192 | Web, code, scientific | No |
| GPT-2 | `openai-community/gpt2` | Generative | 124M | 1024 | WebText | No |
| GPT-2 XL | `openai-community/gpt2-xl` | Generative | 1.5B | 1024 | WebText | No |
| OLMo-2 1B | `allenai/OLMo-2-0425-1B` | Generative | 1B | 4096 | Dolma / OLMo-mix-1124 | No |
| OLMo-2 7B | `allenai/OLMo-2-1124-7B` | Generative | 7B | 4096 | Dolma / OLMo-mix-1124 | No |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | Generative | 7B | 32768 | Proprietary (undisclosed) | No |
| **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | Generative | 7B | 32768 | Proprietary + SFT/RLHF | **Yes** |
| **Llama 3.1 8B** | `meta-llama/Llama-3.1-8B` | Generative | 8B | 128000 | Proprietary (undisclosed) | No |
| **Llama 3.1 8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | Generative | 8B | 128000 | Proprietary + SFT/RLHF | **Yes** |

**Informative overlaps.** ~125M tier (BERT-base, ModernBERT-base, GPT-2): matched capacity, different architecture/data. ~7–8B tier (OLMo-2 7B, Qwen2.5-7B, Llama 3.1 8B): matched scale, different training data and tokenizers.

**Post-training contrast (new).** The proposal restricted Study 1 to pure pretrained LMs so that surprisal reflects statistical language knowledge rather than instruction-following optimization. SFT/RLHF sharpens and miscalibrates a model's output distribution, so instruct-model surprisal no longer cleanly estimates "how probable is this text in natural English." The two base/instruct pairs (Qwen2.5-7B, Llama 3.1 8B) are therefore included as a **deliberate manipulation**, not as additional validity candidates: holding pretraining fixed, does post-training degrade or alter the surprisal–proficiency relationship? Treat instruct configurations as a contrast condition in reporting, not as eligible defaults for Studies 2–3.

> *Optional, free contrast:* `allenai/OLMo-2-1124-7B-Instruct` gives a third base/instruct pair on a fully-open model whose pretraining data matches the Study 2 reference corpus. Worth adding if the post-training question becomes a focus.

---

## 2. Surprisal Computation (constraints for the existing code)

- **Score raw essay text identically across all models.** For the instruct models, do **not** apply the chat template and do **not** prepend a system prompt — feed the plaintext exactly as for base models. Any chat wrapping invalidates the base-vs-instruct comparison.
- **First sub-token convention** for word-level probability (mask whole word for masked models; use the first sub-token position for generative models).
- **Surprisal in bits** (log base 2). Text-level index = **mean token surprisal**.
- **Generative models** use left context only; the first token of each essay has no context.
- **Masked models** require one forward pass per masked word — batch aggressively.
- **Full-document window** is capped at each model's max context. Note BERT (512) and GPT-2 (1024) truncate roughly 30% of ELLIPSE essays; flag truncation in output so it can be reported.
- Half/bf16 precision is fine; the largest model (8B) fits on a single 48 GB GPU.

---

## 3. Statistical Analysis

### 3.1 Validity (concurrent)
- Criterion = human holistic proficiency.
- **ELLIPSE** (continuous scale): Pearson *r* between mean surprisal and holistic score, per configuration.
- **TOEFL 11** (ordinal: low/medium/high): Spearman ρ instead.
- Bootstrap 95% CIs (2,000 iterations, essay-level resampling) for every configuration.
- Compare configurations pairwise with **Zou (2007) intervals for dependent overlapping correlations** (all configs share the same essays, so correlations are dependent).
- **Output:** tidy table (model × window × corpus → r/ρ, CI, n, truncation rate) and a heatmap (model × context window, faceted by corpus).

### 3.2 Fairness (TOEFL 11 only — it carries L1 labels)
For each of the 33 configurations:
1. Within each proficiency stratum (low/medium/high), compute Cohen's *d* in surprisal between each focal L1 group and all remaining learners. Stratifying conditions on ability, isolating group displacement from true proficiency differences (DIF logic).
2. Standardize by the pooled within-stratum SD so effect sizes are comparable across configurations on different surprisal scales.
3. Combine the three stratum-level *d*'s into a focal-group weighted-average *d* (weight by focal-group stratum size).
4. **Primary fairness summary = max |d| across all 11 L1 groups** (minimax / worst-case).
5. Flag any configuration where any group exceeds **|d| ≥ 0.10** (Williamson et al., 2012).
6. **Supplementary:** max |d| across all group × stratum cells (more conservative bound).

Use **standardized mean difference**, not Jensen-Shannon divergence: mean surprisal is approximately normally distributed and SMD is the standard fairness metric in the automated-scoring literature.

### 3.3 Pareto Front (selection)
- Plane: validity (Pearson *r*, ELLIPSE) on the y-axis vs. fairness (max |d|, TOEFL 11) on the x-axis.
- The **non-dominated set** (no other configuration is simultaneously higher *r* and lower max |d|) forms the front.
- Bootstrap confidence regions for the front: 2,000 iterations, essay-level resampling, stratified by proficiency level for the TOEFL 11 effect sizes.
- **Default for Studies 2–3 = highest-*r* point on the front** among the *pure pretrained* models. If Study 3's tied-embedding requirement applies, instead take the highest-*r* tied-embedding model on the front (confirm tied-embedding status at that point).
- **Output:** scatterplot of all 33 configurations with the front and bootstrap envelopes highlighted; instruct configurations marked distinctly.

### 3.4 Targeted comparisons
- **Architecture:** masked vs. generative at the ~125M tier (Zou intervals).
- **Context window:** 8 vs. 64 vs. full, within each model.
- **Post-training:** base vs. instruct within each pair (Qwen2.5-7B, Llama 3.1 8B) — paired comparison of both *r* (Zou) and max |d|.

---

## 4. Expected Directions (for sanity-checking output)
- Masked ≥ comparable generative on validity, sharpest at ~125M.
- Larger context windows → stronger correlations (diminishing past ~64 tokens).
- Within-family scaling → diminishing returns, not monotonic gains.
- Small, configuration-dependent L1 moderation; configuration *rankings* stable across L1s.
- Approximately flat Pareto front (fairness costs little validity).
- Instruct models: equal-or-weaker validity and/or worse calibration than their base counterparts.

## 5. Deliverables
1. Per-configuration results table (validity + fairness + truncation), both corpora.
2. Validity heatmap and Pareto scatterplot.
3. Selected default configuration (pure-pretrained) and selected tied-embedding configuration, each with justification.
4. Reproducibility: pinned model revisions, fixed bootstrap seeds, logged transformers/torch versions.