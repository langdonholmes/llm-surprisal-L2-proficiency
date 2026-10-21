# Corrector: minimal corrections for ELLIPSE

ELLIPSE has no human corrections. A model fine-tuned on the W&I+LOCNESS and
FCE (original, first-annotator correction) pairs corrects it, ERRANT types the
edits, and ELLIPSE then runs through the same decomposition as the two gold
corpora (DECISIONS D36). It is called the corrector, not a GEC system, because
the corrections it learns are not only grammatical: word choice, collocation
and phrasing are corrected too, and that is part of the argument.

The corrector has its own environment (`pyproject.toml`, `uv.lock`), because
the candidate models need transformers 5 and the main environment pins
transformers < 5 for the Study 1 instrument.

```bash
uv sync --project src/3-surprisal-decomposition/corrector
uv run --project src/3-surprisal-decomposition/corrector python <script>
```

## Design

- **Cross-fitting.** Five folds by unit (W&I essay, FCE script), stratified by
  corpus × level (`c0_folds.py`). Each fold's model is trained on the other
  four and corrects its own fold, so every W&I/FCE text gets a model correction
  from a model that never saw it. The decomposition is then run with model
  corrections and with gold corrections on the same texts. That comparison is
  the validation.
- **Final model.** A sixth model, trained on all W&I and FCE pairs with the same
  settings, corrects ELLIPSE. No ensembling, so there is nothing to reconcile.
  The validation describes models trained on 80% of the data, so if anything
  it understates the final model.
- **Unit.** Whole texts in and out (W&I essay, FCE answer), so the model has
  the context the annotator had. ERRANT aligns output with input, as for the
  gold corrections.
- **Prompting baseline.** Before fine-tuning, the instruct checkpoint of the
  same model corrects fold 0 with a minimal-edit prompt (zero- and few-shot).
  This sets the agreement the fine-tuned model has to beat.
- **Model size.** The main model is compared with a 1–4B model of the same
  family.

## Model choice

Qwen3.5-9B-Base (February 2026, Apache 2.0) is the corrector, and Qwen3.5-2B-Base
the small comparison. In October 2026 Qwen3.5 was the newest Qwen generation
with a dense model between 6B and 13B (Qwen3.6 and 3.8 released only 27B
dense and larger), and 9B and 2B share a tokenizer, architecture and
training recipe, so model size is the only difference between them.

The other candidate was Gemma 4 12B (June 2026), the most-downloaded open
family on Hugging Face. Its small siblings (E2B, E4B) use a different
architecture from the 12B (Gemma4 vs Gemma4Unified), so a size comparison
within Gemma would also change the architecture. A speed check (`c1_bench.py`,
one A6000, bf16, LoRA r 16, `results/study3/corrector/bench_*.json`) decided it:

| | Qwen3.5-9B-Base | Gemma 4 12B |
|---|---|---|
| LoRA training (tokens/s) | 938 | 525 |
| Peak memory, training (GB) | 24.0 | 31.6 |
| Generation, batch 32 (tokens/s) | 315 | 259 |

Qwen3.5 mixes linear-attention (Gated DeltaNet) and attention layers. It is
fast only with the `flash-linear-attention` and `causal-conv1d` kernels
installed (both in `pyproject.toml`); without `causal-conv1d`, transformers
falls back to a much slower PyTorch path.

OLMo 3 7B (November 2025) was also considered: its pre-training data is open,
so W&I/FCE contamination could be checked directly. It was not chosen because
it is older; contamination is instead a stated limitation, and the prompting
baseline shows what the model does before fine-tuning.

## Training settings

Neither model release gives a fine-tuning recipe, so each value has a source.

| Setting | Value | Source |
|---|---|---|
| Adapter | LoRA on all linear layers | LoRA Without Regret: attention-only LoRA underperforms; MLP layers matter most. CoCoGEC: all modules. |
| Rank, α | r = 16, α = 32 | CoCoGEC (r = 16). LoRA Without Regret: the optimal LR changes < 2× from rank 4 to 512, and α = 32 is the standard setting they could not improve on. Rank is therefore not searched; one r = 64 run at the chosen LR checks it. |
| Learning rate | searched: {5e-5, 1e-4, 2e-4} | Low end: CoCoGEC (5e-5, Qwen3-8B, LoRA, GEC), which is also 10× the 5e-6 full fine-tuning rate of Adapting LLMs for Minimal-edit GEC (BEA 2025), and LoRA Without Regret finds LoRA's optimal LR ≈ 10× full fine-tuning (fitted 9.8). High end: 2e-4, the common LoRA default (QLoRA; Unsloth guide), and the optimum in a Llama-2-7B LoRA LR sweep. |
| Schedule | linear warm-up 10%, cosine decay | CoCoGEC |
| Epochs | 2, fixed in advance; one 1-epoch run at the chosen LR | CoCoGEC (2); Adapting LLMs for Minimal-edit GEC (1). See "Epochs and the cosine schedule". |
| Batch | effective 16 (2 × 8 accumulation) | CoCoGEC and Adapting LLMs for Minimal-edit GEC (both 16). LoRA Without Regret: LoRA prefers small batches. |
| Precision | bf16 | CoCoGEC |
| Loss | on the correction only (prompt tokens masked) | Standard supervised fine-tuning |
| Decoding | greedy | Deterministic and minimal |
| Selection | ERRANT F0.5 against the first annotator on the tuning split (10% of fold 0's training units), final checkpoint only | The standard GEC metric (BEA-2019) |

The search runs on fold 0 only. The chosen settings are then frozen for all
five folds and the final model. The small model gets its own LR search over the
same three values, because the optimal LR depends on model size (LoRA Without
Regret's fit).

### Epochs and the cosine schedule

A cosine schedule decays to zero at the step count it was given, so the number
of epochs fixes the whole learning-rate trajectory. A checkpoint taken halfway
through a 2-epoch run is not the same as a 1-epoch run, so early stopping inside
a cosine run is not a clean way to choose epochs. Epochs are therefore fixed in
advance at 2 (CoCoGEC's setting for the same task, model family and adapter),
and only the final checkpoint is evaluated. The one 1-epoch run at the chosen
LR is the check: if it beats the 2-epoch run on the tuning split, 1 epoch is
used. With about 4,200 training texts and batch 16, 2 epochs is about 525
optimizer steps.

## References

- Schulman, J. & Thinking Machines Lab (2025). LoRA Without Regret.
  https://thinkingmachines.ai/blog/lora/ (TRL reproduction:
  https://huggingface.co/docs/trl/main/en/lora_without_regret)
- CoCoGEC: Counterfactual Generation for Robust Grammatical Error Correction
  (2026). arXiv 2606.15069. Table 6: LoRA configuration for Qwen3-8B (r 16, all
  modules, LR 5e-5, 2 epochs, batch 2 × 8, warm-up 0.1, cosine, max length 1024,
  bf16; LLaMA-Factory).
- Adapting LLMs for Minimal-edit Grammatical Error Correction (BEA 2025).
  https://aclanthology.org/2025.bea-1.9 (arXiv 2506.13148). Appendix A,
  Table 10: full fine-tuning, LR 5e-6, linear schedule, 100 warm-up steps,
  1 epoch, batch 4 × 4.
- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
  NeurIPS. (LR 2e-4 for 7–13B models.)
- Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
  ICLR.
- Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning (2026).
  LoRA accuracy on Llama-2-7B peaks near LR 2e-4.
- Unsloth LoRA hyperparameters guide:
  https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- Bryant, C., Felice, M. & Briscoe, T. (2017). Automatic annotation and
  evaluation of error types for grammatical error correction (ERRANT). ACL.
- Bryant, C. et al. (2019). The BEA-2019 shared task on grammatical error
  correction. BEA.
- Qwen3.5 model cards: https://huggingface.co/Qwen/Qwen3.5-9B-Base,
  https://huggingface.co/Qwen/Qwen3.5-2B-Base
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4

## Files

| Stage | File | What it does |
|---|---|---|
| c0 | `c0_folds.py` | Training pairs and the five folds (+ the tuning split) → `data/study3/corrector/pairs.parquet` |
| c1 | `c1_bench.py --model` (GPU) | Speed check: LoRA training and generation throughput of a candidate model |
| c2 | `c2_train.py --model --fold [--tune] --lr [--r --epochs]` (GPU) | LoRA fine-tuning → `data/study3/corrector/adapters/<run>/` |
| c3 | `c3_correct.py --model (--adapter <run> \| --prompt zero\|few) --texts` (GPU) | Greedy corrections → `data/study3/corrector/out/` |
| c4 | `c4_score.py <out file>` | ERRANT agreement with the first annotator → `results/study3/corrector/score_*.json`, edits → `data/study3/corrector/edits/` |
| — | `common.py` | Candidate models, the prompt and its chat formatting, the LoRA settings |

## Running

```bash
C=src/3-surprisal-decomposition/corrector
R="uv run --project $C python"
python $C/c0_folds.py                                   # main env is fine (pandas only)
# prompting baseline, fold 0
for p in zero few; do
  CUDA_VISIBLE_DEVICES=0 $R $C/c3_correct.py --model qwen9b --prompt $p --texts fold:0
  $R $C/c4_score.py data/study3/corrector/out/qwen9b_prompt-${p}__fold0.parquet
done
# learning-rate search, fold 0 (about 2 h per run on an A6000)
for lr in 5e-05 0.0001 0.0002; do
  run=qwen9b_f0_tune_lr${lr}_r16_e2
  CUDA_VISIBLE_DEVICES=1 $R $C/c2_train.py --model qwen9b --fold 0 --tune --lr $lr
  CUDA_VISIBLE_DEVICES=1 $R $C/c3_correct.py --model qwen9b --adapter $run --texts tune
  $R $C/c4_score.py data/study3/corrector/out/${run}__tune.parquet
done
```

The chat formatting is built in `common.py` so that training and inference
see the same text: the generation prompt, then the correction and the
template's end-of-turn marker. Qwen3.5's thinking is switched off. Gemma 4's
pretrained checkpoint has no chat template, so the instruct tokenizer's is
used.

## Input format

Each training pair is written in Qwen's chat template (shipped with the base
checkpoint's tokenizer), with the instruction from `common.py` and the original
text as the user turn and the correction as the assistant turn. The loss is on
the correction and `<|im_end|>` only:

```
<|im_start|>user
{INSTRUCTION}

{original}<|im_end|>
<|im_start|>assistant
<think>

</think>

{corrected}<|im_end|>
```

The empty `<think>` block is what the template writes with thinking off; it is
kept so that training and generation see the same text.

**Why the chat format on a base model.** Qwen3.5-9B-Base is not post-trained,
but its pre-training evidently included chat-formatted text, so the chat
delimiters already mean something to it:

- The embedding and output rows of `<|im_start|>`, `<|im_end|>`, `<think>` and
  `</think>` in the base checkpoint have the norms of ordinary trained tokens
  (0.79–1.16, against medians of 0.93 and 1.01), nearly the same as in the
  instruct checkpoint, not the near-zero norms of untrained rows.
- Given the template with the instruction, the untuned base model (checked
  on Qwen3.5-2B-Base, 2026-10-21) answers in the assistant turn, corrects
  ("Yesterday I go … my freinds and we was" → "went … friends … were") and
  stops at `<|im_end|>`.

This matters for LoRA in particular: the adapter trains the linear layers only,
not the embeddings or the output layer, so it could not repair untrained
delimiter tokens. The chat format therefore gives the model a delimiter between
input and correction, and a stop token, that it already knows, with no
made-up separator. It also matches the instruct checkpoint used for the
prompting baseline.

**The instruction.** Whether the instruction is kept in the fine-tuning input is
open. With about 4,200 training pairs it is expected to make no difference: a
prompt is worth on the order of a hundred to a few hundred training examples
under full fine-tuning (Le Scao & Rush 2021, How Many Data Points is a Prompt
Worth?, NAACL), and its advantage shrinks as the data grows. If it is dropped,
the user turn is the original text alone and the chat delimiters stay. The
learning-rate search (2026-10-21) was run with the instruction.
