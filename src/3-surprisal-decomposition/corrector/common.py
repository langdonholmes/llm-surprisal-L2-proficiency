"""Shared pieces of the corrector: candidate models, the prompt, the LoRA setup.

The hyperparameters and their sources are in README.md ("Training settings").
"""

from pathlib import Path

import torch

from util.paths import DATA_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
DATA = DATA_DIR / "study3" / "corrector"
RES = RESULTS_DIR / "study3" / "corrector"

# base checkpoint for fine-tuning, instruct checkpoint for the prompting baseline
MODELS = {
    "qwen9b": {"base": "Qwen/Qwen3.5-9B-Base", "instruct": "Qwen/Qwen3.5-9B"},
    "qwen2b": {"base": "Qwen/Qwen3.5-2B-Base", "instruct": "Qwen/Qwen3.5-2B"},
    "gemma12b": {"base": "google/gemma-4-12B", "instruct": "google/gemma-4-12B-it"},
}

INSTRUCTION = (
    "Correct the errors in the following text written by a learner of English. "
    "Make only the changes needed to make it correct, and keep everything else "
    "(wording, paragraphing, punctuation style) exactly as written. "
    "Return only the corrected text."
)

# LoRA (README, "Training settings")
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET = "all-linear"


def messages(text: str, corrected: str | None = None) -> list[dict]:
    m = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
    if corrected is not None:
        m.append({"role": "assistant", "content": corrected})
    return m


def chat_kwargs(model_key: str) -> dict:
    # Qwen3.5 instruct defaults to thinking; the corrector never thinks
    return {"enable_thinking": False} if model_key.startswith("qwen") else {}


def prompt_text(tok, key: str, text: str, shots=()) -> str:
    """The generation prompt (few-shot pairs, if any, as earlier turns)."""
    m = [x for t, c in shots for x in messages(t, c)] + messages(text)
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kwargs(key))


def encode_pair(tok, key: str, text: str, corrected: str):
    """Token ids and labels (-100 on the prompt) for one training pair.

    The target is exactly what follows the generation prompt at inference: the
    correction and the template's end-of-turn marker. (Rendering the finished
    conversation instead would differ for Gemma 4, whose generation prompt has
    an empty thought block that the rendered history drops.)
    """
    prompt = prompt_text(tok, key, text)
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(corrected + end_of_turn(tok, key), add_special_tokens=False).input_ids
    return p + c, [-100] * len(p) + c


def end_of_turn(tok, key: str) -> str:
    """What the template writes after an assistant message (found with a dummy message,
    because some templates trim the content)."""
    mark = "ZZEND"
    ref = tok.apply_chat_template(messages("x", mark), tokenize=False, **chat_kwargs(key))
    return ref[ref.rindex(mark) + len(mark):]


def load_lm(name: str, device: str = "cuda:0"):
    """Text-only causal LM in bf16 (the vision towers are not loaded)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.chat_template is None:  # e.g. Gemma pretrained: use the instruct tokenizer's template
        instruct = next(m["instruct"] for m in MODELS.values() if m["base"] == name)
        tok.chat_template = AutoTokenizer.from_pretrained(instruct).chat_template
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": device})
    return tok, model


def lora_config(r: int = LORA_R):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET,
                      task_type="CAUSAL_LM")
