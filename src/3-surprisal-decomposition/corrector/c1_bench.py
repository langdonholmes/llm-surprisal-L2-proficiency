"""Corrector, stage c1 (GPU) — speed check of a candidate model before choosing it.

Loads the base checkpoint in bf16 on one GPU with the LoRA setup of
``common.py`` and times (a) LoRA training steps on real training pairs and (b)
batched greedy generation of corrections, with plain ``transformers``. Also
reports the token lengths of the pairs under the model's tokenizer.

Writes results/study3/corrector/bench_<model>.json.

    CUDA_VISIBLE_DEVICES=0 uv run --project src/3-surprisal-decomposition/corrector \
        python src/3-surprisal-decomposition/corrector/c1_bench.py --model qwen9b
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, RES, encode_pair, load_lm, lora_config, prompt_text  # noqa: E402

SEED = 42
N_TRAIN_STEPS = 12  # micro-batches timed after 2 warm-up steps
TRAIN_BS = 2
GEN_BS = [8, 32]
GEN_NEW = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(DATA / "pairs.parquet")
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    tok, model = load_lm(MODELS[args.model]["base"])
    rep = {"model": MODELS[args.model]["base"], "load_s": time.time() - t0,
           "class": type(model).__name__, "n_params_b": sum(p.numel() for p in model.parameters()) / 1e9}

    enc = [encode_pair(tok, args.model, t, c) for t, c in zip(pairs.text, pairs.corrected)]
    lens = np.array([len(f) for f, _ in enc])
    rep["pair_tokens"] = {q: float(np.percentile(lens, q)) for q in [50, 90, 99, 100]}
    rep["pair_tokens_total_m"] = float(lens.sum() / 1e6)

    # (a) LoRA training steps
    from peft import get_peft_model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, lora_config())
    rep["lora_trainable_m"] = sum(p.numel() for p in pm.parameters() if p.requires_grad) / 1e6
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4)
    pm.train()
    idx = rng.choice(len(enc), TRAIN_BS * (N_TRAIN_STEPS + 2), replace=False)
    torch.cuda.reset_peak_memory_stats()
    n_tok, t_train = 0, 0.0
    for step in range(N_TRAIN_STEPS + 2):
        b = [enc[i] for i in idx[step * TRAIN_BS:(step + 1) * TRAIN_BS]]
        L = max(len(f) for f, _ in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(b), L), -100, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (f, la) in enumerate(b):
            ids[j, :len(f)] = torch.tensor(f)
            lab[j, :len(f)] = torch.tensor(la)
            att[j, :len(f)] = 1
        torch.cuda.synchronize()
        t = time.time()
        loss = pm(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        if step >= 2:
            t_train += time.time() - t
            n_tok += int(att.sum())
    rep["train_tokens_per_s"] = n_tok / t_train
    rep["train_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    epoch_s = lens.sum() * 0.8 / rep["train_tokens_per_s"]  # one fold's training data (4/5)
    rep["est_hours_per_epoch_one_fold"] = epoch_s / 3600

    # (b) batched greedy generation, adapter merged (as in c3_correct.py), at a fixed
    # number of new tokens so that throughput does not depend on where the
    # untrained model stops: texts of similar length around the median, as c3 batches them
    gm = pm.merge_and_unload()
    gm.eval()
    tok.padding_side = "left"
    order = np.argsort(lens)
    rep["gen"] = {}
    for bs in GEN_BS:
        mid = len(order) // 2
        gi = order[mid - bs // 2: mid - bs // 2 + bs]
        prompts = [prompt_text(tok, args.model, pairs.text.iloc[i]) for i in gi]
        x = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            gm.generate(**x, max_new_tokens=GEN_NEW, min_new_tokens=GEN_NEW, do_sample=False)
        torch.cuda.synchronize()
        s = time.time() - t
        rep["gen"][bs] = {"s": s, "tokens_per_s": bs * GEN_NEW / s, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    best = max(v["tokens_per_s"] for v in rep["gen"].values())
    out_tokens = lens.sum() / 2  # roughly the output side of every pair
    rep["est_hours_generate_all_wi_fce"] = out_tokens / best / 3600
    print(json.dumps(rep, indent=2))
    (RES / f"bench_{args.model}.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
