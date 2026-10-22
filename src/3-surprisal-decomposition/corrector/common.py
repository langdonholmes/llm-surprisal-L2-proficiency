"""Shared pieces of the corrector: candidate models, the prompt, the LoRA setup.

The hyperparameters and their sources are in README.md ("Training settings").
"""

import re
from pathlib import Path

import numpy as np
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

# the chosen settings (DECISIONS D39): the c2 run name of fold k's model, or "all"
FINAL_MODEL, FINAL_LR = "qwen9b", 5e-5


def final_run(fold) -> str:
    return f"{FINAL_MODEL}_f{fold}_lr{FINAL_LR:g}_r{LORA_R}_e2"


# TOEFL 11 is distributed tokenized ("a good way . But , I do n't"). The corrector, trained on
# naturally spaced text, sees it detokenized, and its edits are mapped back (D50).
BEFORE = re.compile(r"[ \t]+(?=[.,?!:;)\]}%]|n't\b|'(?:s|m|d|re|ve|ll)\b|'(?=\s|$))", re.I)
AFTER = re.compile(r"(?<=[(\[{$`])[ \t]+")


def detokenize(text: str) -> tuple[str, np.ndarray]:
    """The text with tokenization spaces removed, and for each of its characters the index of
    that character in ``text`` (plus len(text) at the end). Only spaces and tabs are removed:
    before closing punctuation and clitics, after opening brackets and the opening quote `, and inside straight double
    quotes, which open and close in turn."""
    drop = np.zeros(len(text), bool)
    for rx in (BEFORE, AFTER):
        for m in rx.finditer(text):
            drop[m.start():m.end()] = True
    for i, m in enumerate(re.finditer(r'"', text)):  # odd ones open, even ones close
        gap = re.match(r"[ \t]+", text[m.end():]) if i % 2 == 0 else re.search(r"[ \t]+$", text[:m.start()])
        if gap:
            a, b = (m.end(), m.end() + gap.end()) if i % 2 == 0 else (gap.start(), m.start())
            drop[a:b] = True
    keep = np.flatnonzero(~drop)
    out = "".join(text[i] for i in keep)
    assert re.sub(r"\s", "", out) == re.sub(r"\s", "", text)
    return out, np.append(keep, len(text))


def map_edits(text: str, keep: np.ndarray, edits) -> list:
    """(s, e, replacement, type) edits on the detokenized text, moved onto ``text``. A span
    keeps any tokenization space inside it; the spaces around it stay where they were, and an
    insertion goes right after the preceding character."""
    out = []
    for s, e, r, t in edits:
        a = keep[s - 1] + 1 if s > 0 and s == e else keep[s]
        out.append((int(a), int(keep[e - 1] + 1 if e > s else a), r, t))
    return out


_ann = None


def annotator():
    global _ann
    if _ann is None:
        import errant
        _ann = errant.load("en")
    return _ann


def char_edits(orig: str, cor: str) -> list[tuple[int, int, str, str]]:
    """ERRANT's edits from ``orig`` to ``cor`` as (s, e, replacement, type) on characters of ``orig``.

    Applied to ``orig`` without any whitespace repair, the edits give ``cor``
    exactly. As for the BEA-2019 M2 files, ERRANT sees the non-whitespace tokens
    only (spaCy's whitespace tokens, e.g. the second space of "take  exam", would
    otherwise be aligned with an inserted word). All whitespace lies in the gaps
    between tokens: with gap g the text between token g-1 and token g, an edit
    owns the gap before its first token and the gaps inside it on each side (an
    insertion owns no original gap). The remaining gaps pair up in order, and a
    pair that differs is a whitespace change. A change in the gap right after an
    edit joins that edit (FCE leaves a double space where a word is missing, and
    the annotator's insertion fills it: "take  exam" -> "take an exam" is one
    edit). Two changes one unchanged token apart are one edit (" ,I" -> ", I").
    Any other change (e.g. a removed paragraph break) is a whitespace-only edit,
    typed R:ORTH as in ``lib/errant.py``.
    """
    ann = annotator()

    def tokens(text):
        """The ERRANT parse of the non-whitespace tokens, and the gaps between them."""
        spans = [(t.idx, t.idx + len(t.text)) for t in ann.nlp.tokenizer(text) if not t.is_space]
        words = [text[a:b] for a, b in spans]
        assert not any(" " in w for w in words)
        doc = ann.parse(" ".join(words))
        assert len(doc) == len(spans)
        return doc, [0] + [b for _, b in spans], [a for a, _ in spans] + [len(text)]

    po, ogs, oge = tokens(orig)
    pc, cgs, cge = tokens(cor)
    out, o_own, c_own = [], set(), set()
    for e in ann.annotate(po, pc):
        if e.type == "noop":
            continue
        s = ogs[e.o_start]
        t = ogs[e.o_end] if e.o_end > e.o_start else s
        cs = cgs[e.c_start]
        ce = cgs[e.c_end] if e.c_end > e.c_start else cs
        out.append((s, t, cor[cs:ce], e.type))
        o_own.update(range(e.o_start, e.o_end))
        c_own.update(range(e.c_start, e.c_end))
    og = [g for g in range(len(po) + 1) if g not in o_own]
    cg = [g for g in range(len(pc) + 1) if g not in c_own]
    assert len(og) == len(cg)
    diff = [(a, b) for a, b in zip(og, cg) if orig[ogs[a]:oge[a]] != cor[cgs[b]:cge[b]]]
    ends = {t: i for i, (_, t, _, _) in enumerate(out)}
    i = 0
    while i < len(diff):
        a, b = diff[i]
        if ogs[a] in ends:  # the gap right after an edit
            j = ends.pop(ogs[a])
            s, _, r, typ = out[j]
            out[j] = (s, oge[a], r + cor[cgs[b]:cge[b]], typ)
            i += 1
        elif i + 1 < len(diff) and diff[i + 1][0] == a + 1 and ogs[a + 1] not in ends:
            # two changes one unchanged token apart, with no insertion between
            b2 = diff[i + 1][1]
            out.append((ogs[a], oge[a + 1], cor[cgs[b]:cge[b2]], "R:ORTH"))
            i += 2
        else:
            out.append((ogs[a], oge[a], cor[cgs[b]:cge[b]], "R:ORTH"))
            i += 1
    return sorted(out, key=lambda x: (x[0], x[1]))


def splice(text: str, edits) -> str:
    """Apply (s, e, replacement, ...) edits to ``text`` as they are."""
    out, p = [], 0
    for s, e, r, *_ in sorted(edits, key=lambda x: (x[0], x[1])):
        assert s >= p
        out += [text[p:s], r]
        p = e
    return "".join(out) + text[p:]


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
