"""Corrector, stage c2 (GPU) — LoRA fine-tuning of one model on one training set.

Trains on the (original, correction) pairs outside the held-out fold, with the
loss on the correction tokens only. The settings and their sources are in
README.md ("Training settings"); only the learning rate, the rank and the
epochs can be changed from the command line, for the fold-0 search.

    --fold k     train on folds != k (k = 0..4); with --tune, fold 0's tuning
                 units are also held out (the hyperparameter search)
    --fold all   train on every W&I/FCE pair (the final model for ELLIPSE)

Writes data/study3/corrector/adapters/<run>/ (the LoRA adapter, the tokenizer,
run.json with the settings, versions and timing) where
run = <model>_f<fold>[_tune]_lr<lr>_r<r>_e<epochs>.

    CUDA_VISIBLE_DEVICES=0 uv run --project src/3-surprisal-decomposition/corrector \
        python src/3-surprisal-decomposition/corrector/c2_train.py --model qwen9b --fold 0 --tune --lr 1e-4
"""

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, LORA_R, MODELS, encode_pair, load_lm, lora_config  # noqa: E402

SEED = 42
EPOCHS = 2
WARMUP = 0.10
PER_DEVICE_BS = 2
GRAD_ACCUM = 8  # effective batch 16


def run_name(model, fold, tune, lr, r, epochs):
    return f"{model}_f{fold}{'_tune' if tune else ''}_lr{lr:g}_r{r}_e{epochs:g}"


def training_pairs(fold: str, tune: bool) -> pd.DataFrame:
    p = pd.read_parquet(DATA / "pairs.parquet")
    if fold == "all":
        return p
    p = p[p.fold != int(fold)]
    return p[~p.tune] if tune else p


class Pairs(torch.utils.data.Dataset):
    """Chat-formatted pairs; labels are -100 on the prompt."""

    def __init__(self, tok, key, df):
        self.items = []
        for text, corrected in zip(df.text, df.corrected):
            ids, labels = encode_pair(tok, key, text, corrected)
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def f(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = torch.tensor(b["input_ids"])
            lab[j, :n] = torch.tensor(b["labels"])
            att[j, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--fold", required=True, help="0-4 or all")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--r", type=int, default=LORA_R)
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    args = ap.parse_args()
    assert not (args.tune and args.fold != "0"), "the tuning split belongs to fold 0"

    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(SEED)
    name = run_name(args.model, args.fold, args.tune, args.lr, args.r, args.epochs)
    out = DATA / "adapters" / name
    if (out / "run.json").exists():
        print(f"{name} exists; skipping")
        return

    df = training_pairs(args.fold, args.tune)
    tok, model = load_lm(MODELS[args.model]["base"])
    ds = Pairs(tok, args.model, df)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, lora_config(args.r))

    targs = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP,  # a float < 1 is the ratio of total steps (transformers 5)
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.0,
        train_sampling_strategy="group_by_length",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )
    t0 = time.time()
    trainer = Trainer(model=pm, args=targs, train_dataset=ds, data_collator=collate(tok.pad_token_id))
    res = trainer.train()
    pm.save_pretrained(out)
    tok.save_pretrained(out)
    run = {
        "run": name, "base": MODELS[args.model]["base"], "fold": args.fold, "tune": args.tune,
        "n_train_texts": len(ds), "lr": args.lr, "r": args.r, "epochs": args.epochs,
        "warmup_ratio": WARMUP, "schedule": "cosine", "batch": PER_DEVICE_BS * GRAD_ACCUM,
        "seed": SEED, "train_loss": res.training_loss, "steps": res.global_step,
        "hours": (time.time() - t0) / 3600, "log": trainer.state.log_history,
        "versions": {p: version(p) for p in ["torch", "transformers", "peft"]},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({k: v for k, v in run.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()
