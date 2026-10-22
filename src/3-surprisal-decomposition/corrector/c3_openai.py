"""Corrector, stage c3 via the OpenAI API — the prompting comparison (DECISIONS D43).

The same input as c3_correct.py's prompting baseline: the instruction and the
text as one user turn (``common.messages``), and for "few" the same three
example pairs from the tuning split as earlier turns (``few_shot_examples``).
Chat Completions, a few requests at a time; nothing is stored on OpenAI's side
(store=False). Temperature is left at the model's default (the GPT-6 models
accept only 1). Reasoning effort is sent only when given; "default" omits it.

Resumable: each finished text is appended to <out>.jsonl and skipped on a
rerun, so an interrupted run is not paid for twice.

Writes data/study3/corrector/out/<tag>__<texts>.parquet, tag =
<model>_prompt-<zero|few>_eff-<effort>, with c3_correct.py's columns (text_id,
text, output, n_in, n_new, hit_limit) plus the API's model, finish_reason and
token counts; <out>.json has the request settings, token totals and the cost
at list prices.

    uv run --project src/3-surprisal-decomposition/corrector python \
        src/3-surprisal-decomposition/corrector/c3_openai.py --model gpt-6-luna --prompt zero --texts tune --limit 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c3_correct import few_shot_examples, load_texts  # noqa: E402
from common import DATA, messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# USD per 1M tokens (input, cached input, output), developers.openai.com model pages, 2026-10-22
PRICES = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6.1-sol": (2.00, 0.10, 10.00), "gpt-6-astra": (10.00, None, 50.00)}
MAX_COMPLETION = 32_000  # reasoning + correction; the longest text is ~4,600 tokens


def api_key() -> str:
    if "OPENAI_API_KEY" not in os.environ:
        for line in (ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "OPENAI_API_KEY":
                return v.strip().strip("'\"")
    return os.environ["OPENAI_API_KEY"]


def conversation(text: str, shots) -> list[dict]:
    return [m for t, c in shots for m in messages(t, c)] + messages(text)


async def correct(client, sem, model, effort, tid, text, shots, sink):
    kw = {} if effort == "default" else {"reasoning_effort": effort}
    async with sem:
        r = await client.chat.completions.create(model=model, messages=conversation(text, shots),
                                                 max_completion_tokens=MAX_COMPLETION, store=False, **kw)
    u = r.usage
    reasoning = (u.completion_tokens_details.reasoning_tokens or 0) if u.completion_tokens_details else 0
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    c = r.choices[0]
    row = {"text_id": tid, "text": text, "output": c.message.content or "", "n_in": u.prompt_tokens,
           "n_new": u.completion_tokens - reasoning, "hit_limit": c.finish_reason == "length",
           "finish_reason": c.finish_reason, "refusal": c.message.refusal, "api_model": r.model,
           "prompt_tokens": u.prompt_tokens, "cached_tokens": cached,
           "completion_tokens": u.completion_tokens, "reasoning_tokens": reasoning}
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return row


async def run(args, todo, shots, cache):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key(), max_retries=8, timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    t0, done = time.time(), 0
    with cache.open("a") as sink:
        tasks = [asyncio.create_task(correct(client, sem, args.model, args.effort, t, x, shots, sink))
                 for t, x in zip(todo.text_id, todo.text)]
        for f in asyncio.as_completed(tasks):
            try:
                await f
            except Exception as e:  # noqa: BLE001 — a failed text is retried on the next run
                print(f"failed: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    await client.close()


def cost(model, df) -> float | None:
    p = PRICES.get(model)
    if p is None or (p[1] is None and df.cached_tokens.sum()):
        return None
    fresh = df.prompt_tokens.sum() - df.cached_tokens.sum()
    return (fresh * p[0] + df.cached_tokens.sum() * (p[1] or 0) + df.completion_tokens.sum() * p[2]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=["zero", "few"], required=True)
    ap.add_argument("--effort", default="default", help="reasoning effort; 'default' sends none")
    ap.add_argument("--texts", default="tune")
    ap.add_argument("--limit", type=int, help="first n texts only (smoke test)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    tag = f"{args.model}_prompt-{args.prompt}_eff-{args.effort}"
    out = DATA / "out" / f"{tag}__{args.texts.replace(':', '')}{'_n' + str(args.limit) if args.limit else ''}.parquet"
    if out.exists():
        print(f"{out.name} exists; skipping")
        return
    cache = out.with_suffix(".jsonl")
    texts = load_texts(args.texts).sort_values("text_id")
    if args.limit:
        texts = texts.head(args.limit)
    shots = few_shot_examples() if args.prompt == "few" else []
    have = set()
    if cache.exists():
        have = {json.loads(x)["text_id"] for x in cache.read_text().splitlines() if x.strip()}
    todo = texts[~texts.text_id.isin(have)]
    print(f"{tag}: {len(texts)} texts, {len(have)} cached, {len(todo)} to send", flush=True)
    t0 = time.time()
    asyncio.run(run(args, todo, shots, cache))

    res = pd.DataFrame([json.loads(x) for x in cache.read_text().splitlines() if x.strip()])
    res = res.drop_duplicates("text_id").sort_values("text_id")
    if set(res.text_id) != set(texts.text_id):
        print(f"{len(set(texts.text_id) - set(res.text_id))} texts missing; rerun to retry")
        return
    res.to_parquet(out, index=False)
    meta = {"corrector": tag, "texts": args.texts, "n": len(res), "model": args.model,
            "api_models": sorted(res.api_model.unique()), "endpoint": "chat.completions",
            "reasoning_effort": args.effort, "temperature": "default (not sent)",
            "max_completion_tokens": MAX_COMPLETION, "few_shot_ids": len(shots),
            "n_hit_limit": int(res.hit_limit.sum()), "n_refusal": int(res.refusal.notna().sum()),
            "finish_reasons": res.finish_reason.value_counts().to_dict(),
            "tokens": {k: int(res[k].sum()) for k in
                       ["prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"]},
            "cost_usd_list_price": cost(args.model, res), "minutes_last_run": (time.time() - t0) / 60}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    cache.unlink()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
