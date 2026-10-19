"""Train a tuned lens for OLMo-2 1B on ELLIPSE dev essays, 8-subword windows.

Loss: sum over layers 0..15 of KL(p_final || p_lens), averaged over all window
positions. 5% of the dev essays are held out to report per-layer KL for the
tuned and the logit lens. No W&I text is read.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/3-surprisal-decomposition/pilot_lens/lens_train.py
Writes data/study3/pilot_lens/tuned_lens.pt, lens_train.json
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from lens_common import (HERE, N_LAYERS, OUT, WINDOW, TunedLens, hidden_states, lens_logits,
                         load_model, pad_windows)

SEED = 42
STEPS = 1500
BATCH = 256          # windows per step (x 8 positions)
LR = 1e-3


def windows_from(ids_list, n, rng):
    out = []
    lens = np.array([len(x) for x in ids_list])
    ess = rng.choice(len(ids_list), n, p=lens / lens.sum())
    for e in ess:
        ids = ids_list[e]
        end = rng.integers(1, len(ids)) + 1           # window ends at a real token; may start at essay start
        out.append(ids[max(0, end - WINDOW):end])
    return out


def kl_terms(model, lens, windows, pad_id, train=True):
    ids, att, pos = pad_windows(windows, pad_id)
    ids, att, pos = ids.cuda(), att.cuda(), pos.cuda()
    hs, logits = hidden_states(model, ids, att, pos)
    m = att.bool()
    logq = torch.log_softmax(logits.float()[m], -1)
    q = logq.exp()
    out = {}
    for layer in range(N_LAYERS):
        h = hs[layer][m]
        lp_t = torch.log_softmax(lens_logits(model, h, layer, lens), -1)
        kl_t = (q * (logq - lp_t)).sum(-1).mean()
        if train:
            kl_t.backward()
        out[layer] = kl_t.item()
        if not train:
            with torch.no_grad():
                lp_l = torch.log_softmax(lens_logits(model, h, layer, None), -1)
                out[f"logit_{layer}"] = (q * (logq - lp_l)).sum(-1).mean().item()
    return out


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok, model = load_model()
    split = pd.read_csv(HERE.parent / "split.csv", dtype={"text_id": str})
    e = pd.read_parquet(OUT.parent / "ellipse" / "essays.parquet", columns=["text_id", "text"])
    e = e[e.text_id.isin(set(split.text_id[split.split == "dev"]))].sample(frac=1, random_state=SEED)
    ids = [np.array(tok(t, add_special_tokens=False).input_ids) for t in e.text]
    ids = [x for x in ids if len(x) > 2]
    n_val = len(ids) // 20
    val, tr = ids[:n_val], ids[n_val:]
    print(f"{len(tr)} train essays, {len(val)} held out, {sum(map(len, tr))} train subwords", flush=True)

    lens = TunedLens(model.config.hidden_size).cuda()
    opt = torch.optim.Adam(lens.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    pad = tok.pad_token_id or 0
    val_w = windows_from(val, 2048, np.random.default_rng(SEED + 1))
    t0 = time.time()
    log = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        kl = kl_terms(model, lens, windows_from(tr, BATCH, rng), pad)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == STEPS - 1:
            row = {"step": step, "mean_kl": float(np.mean(list(kl.values()))), "sec": time.time() - t0}
            log.append(row)
            print(row, flush=True)
    torch.save(lens.state_dict(), OUT / "tuned_lens.pt")

    with torch.no_grad():
        vals = [kl_terms(model, lens, val_w[i:i + 256], pad, train=False) for i in range(0, len(val_w), 256)]
    held = {f"tuned_{k}": float(np.mean([v[k] for v in vals])) for k in range(N_LAYERS)}
    held.update({f"logit_{k}": float(np.mean([v[f"logit_{k}"] for v in vals])) for k in range(N_LAYERS)})
    (OUT / "lens_train.json").write_text(json.dumps(
        {"steps": STEPS, "batch_windows": BATCH, "lr": LR, "n_train_essays": len(tr),
         "train_sec": time.time() - t0, "log": log, "heldout_kl_nats": held}, indent=1))
    for k in range(N_LAYERS):
        print(f"layer {k:2d}  KL tuned {held[f'tuned_{k}']:.3f}  logit {held[f'logit_{k}']:.3f} nats")


if __name__ == "__main__":
    main()
