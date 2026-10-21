"""Corrector, stage c3 (GPU) — generate corrections with plain transformers.

Two kinds of corrector:
    --adapter <run>     the base checkpoint + a LoRA adapter from c2_train.py
    --prompt zero|few   the instruct checkpoint with the minimal-edit prompt
                        (the baseline). "few" puts FEW_SHOT example pairs from the
                        tuning split (outside fold 0) before the text as earlier
                        turns of the conversation.

Texts (--texts):
    fold:<k>   the held-out fold k of pairs.parquet
    tune       fold 0's tuning split
    ellipse    every ELLIPSE essay

Greedy decoding, batched by length, at most 1.5 x the input length + 64 new
tokens. Thinking is off for Qwen3.5. Resumable: finished batches are kept in
<out>.part/ and skipped.

Writes data/study3/corrector/out/<corrector>__<texts>.parquet:
    text_id, text, output, n_in, n_new, hit_limit

    CUDA_VISIBLE_DEVICES=0 uv run --project src/3-surprisal-decomposition/corrector \
        python src/3-surprisal-decomposition/corrector/c3_correct.py --model qwen9b --prompt zero --texts fold:0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, load_lm, prompt_text  # noqa: E402

from util.paths import DATA_DIR  # noqa: E402

BATCH_TOKENS = 12_000  # prompt tokens per batch
FEW_SHOT = 3


def load_texts(spec: str) -> pd.DataFrame:
    p = pd.read_parquet(DATA / "pairs.parquet")
    if spec.startswith("fold:"):
        return p[p.fold == int(spec[5:])][["text_id", "text"]]
    if spec == "tune":
        return p[p.tune][["text_id", "text"]]
    if spec == "ellipse":
        return pd.read_parquet(DATA_DIR / "study3" / "appendix" / "ellipse" / "essays.parquet",
                               columns=["text_id", "text"])
    raise ValueError(spec)


def few_shot_examples() -> list[tuple[str, str]]:
    """One W&I example per CEFR level from the tuning split, 100-200 words, edited."""
    p = pd.read_parquet(DATA / "pairs.parquet")
    p = p[p.tune & (p.corpus == "wi") & (p.text != p.corrected)]
    n = p.text.str.split().str.len()
    p = p[(n >= 100) & (n <= 200)].sort_values("text_id")
    return [(g.text.iloc[0], g.corrected.iloc[0]) for _, g in p.groupby("group") if g.group.iloc[0] in "ABC"][:FEW_SHOT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter")
    g.add_argument("--prompt", choices=["zero", "few"])
    ap.add_argument("--texts", required=True)
    ap.add_argument("--limit", type=int, help="first n texts only (smoke test)")
    args = ap.parse_args()

    tag = args.adapter if args.adapter else f"{args.model}_prompt-{args.prompt}"
    out = DATA / "out" / f"{tag}__{args.texts.replace(':', '')}{'_n' + str(args.limit) if args.limit else ''}.parquet"
    if out.exists():
        print(f"{out.name} exists; skipping")
        return
    part = out.with_suffix(".part")
    part.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.texts).sort_values("text_id")
    if args.limit:
        texts = texts.head(args.limit)
    if args.adapter:
        from peft import PeftModel
        tok, model = load_lm(MODELS[args.model]["base"])
        model = PeftModel.from_pretrained(model, DATA / "adapters" / args.adapter).merge_and_unload()
        shots = []
    else:
        tok, model = load_lm(MODELS[args.model]["instruct"])
        shots = few_shot_examples() if args.prompt == "few" else []
    model.eval()
    tok.padding_side = "left"
    eos = [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<end_of_turn>"]
                                if t in tok.get_vocab()]

    prompts = [prompt_text(tok, args.model, t, shots) for t in texts.text]
    n_in = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    n_text = [len(tok(t, add_special_tokens=False).input_ids) for t in texts.text]
    order = sorted(range(len(prompts)), key=lambda i: -n_in[i])
    batches, cur = [], []
    for i in order:
        if cur and n_in[cur[0]] * (len(cur) + 1) > BATCH_TOKENS:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)

    t0 = time.time()
    for b, idx in enumerate(batches):
        f = part / f"{b:05d}.parquet"
        if f.exists():
            continue
        x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        max_new = int(1.5 * max(n_text[i] for i in idx)) + 64
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=max_new, do_sample=False, eos_token_id=eos,
                               pad_token_id=tok.pad_token_id)
        gen = y[:, x.input_ids.shape[1]:]
        rows = []
        for j, i in enumerate(idx):
            ids = gen[j].tolist()
            n_new = next((k for k, t in enumerate(ids) if t in eos or t == tok.pad_token_id), len(ids))
            rows.append({"text_id": texts.text_id.iloc[i], "text": texts.text.iloc[i],
                         "output": tok.decode(ids[:n_new], skip_special_tokens=True),
                         "n_in": n_text[i], "n_new": n_new, "hit_limit": n_new == len(ids)})
        pd.DataFrame(rows).to_parquet(f, index=False)
        print(f"batch {b + 1}/{len(batches)} ({len(idx)} texts) {time.time() - t0:.0f}s", flush=True)

    res = pd.concat([pd.read_parquet(p) for p in sorted(part.glob("*.parquet"))]).sort_values("text_id")
    res.to_parquet(out, index=False)
    for p in part.glob("*.parquet"):
        p.unlink()
    part.rmdir()
    meta = {"corrector": tag, "texts": args.texts, "n": len(res), "hours": (time.time() - t0) / 3600,
            "n_hit_limit": int(res.hit_limit.sum()), "few_shot_ids": len(shots)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
