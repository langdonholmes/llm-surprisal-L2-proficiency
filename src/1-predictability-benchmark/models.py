"""Model matrix for Study 1 (optimal LLM-surprisal configurations).

Eleven language models spanning two architectures, three capacity tiers, and a
base-vs-instruct post-training contrast. The tiers are chosen for their
overlaps: the ~125M tier matches capacity across architectures, and the ~7-8B
tier matches scale across training corpora and tokenizers. Each spec is keyed by
a short slug used both on the command line and as the output directory name, so
adding a model later is a one-line append here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str  # CLI slug + output dir name
    hf_id: str  # Hugging Face identifier
    arch: str  # "masked" | "causal"
    max_ctx: int  # tokenizer/model max context (caps the "full" window)
    params: str  # human-readable parameter count
    gated: bool = False  # requires HF license acceptance + token
    instruct: bool = False  # post-trained (SFT/RLHF) — contrast condition, not a default
    trust_remote_code: bool = False


_SPECS = [
    # ~125M tier — matched capacity, different architecture / training data
    ModelSpec("bert-base", "google-bert/bert-base-uncased", "masked", 512, "110M"),
    ModelSpec("modernbert-base", "answerdotai/ModernBERT-base", "masked", 8192, "149M"),
    ModelSpec("modernbert-large", "answerdotai/ModernBERT-large", "masked", 8192, "395M"),
    ModelSpec("gpt2", "openai-community/gpt2", "causal", 1024, "124M"),
    ModelSpec("gpt2-xl", "openai-community/gpt2-xl", "causal", 1024, "1.5B"),
    # ~1B and 7-8B tiers — matched scale, different training data / tokenizers
    ModelSpec("olmo2-1b", "allenai/OLMo-2-0425-1B", "causal", 4096, "1B"),
    ModelSpec("olmo2-7b", "allenai/OLMo-2-1124-7B", "causal", 4096, "7B"),
    ModelSpec("qwen2.5-7b", "Qwen/Qwen2.5-7B", "causal", 32768, "7B"),
    ModelSpec("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", "causal", 32768, "7B", instruct=True),
    ModelSpec("llama3.1-8b", "meta-llama/Llama-3.1-8B", "causal", 128000, "8B", gated=True),
    ModelSpec("llama3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "causal", 128000, "8B", gated=True, instruct=True),
]

MODEL_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}
