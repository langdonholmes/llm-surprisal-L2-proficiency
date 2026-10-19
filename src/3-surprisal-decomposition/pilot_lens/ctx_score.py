"""Contextuality views of each sampled W&I high token (observed first subword).

causal  OLMo-2 1B (Study 1 revision), left context only:
          w8    <= 8 subwords ending at the target (the scoring window)
          sent  from the start of the spaCy sentence (NaN if the target opens it)
          full  all of the essay before the target (one forward per essay)
        plus pool_ctx = mean over the top-32 candidates v under w8 of
        log2 p_w8(v) - log2 p_full(v) (Ondov et al. 2024's slot-level contrast,
        k = 32), KL(p_w8 || p_full) and entropy under full (bits).
mlm     ModernBERT-large, all subwords of the word masked, first-subword
        probability, given the sentence alone or the whole essay (Holmes et al.
        2026's adaptation of Ondov et al.).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/3-surprisal-decomposition/pilot_lens/ctx_score.py causal mlm
Writes data/study3/pilot_lens/ctx_{causal,mlm}.parquet (row-aligned with sample.parquet)
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

from lens_common import LOG2E, OUT, WINDOW, load_model, pad_windows

POOL_K = 32
MLM_ID = "answerdotai/ModernBERT-large"


def offsets(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([a + len(text[a:b]) - len(text[a:b].lstrip()) for a, b in enc.offset_mapping])
    return np.array(enc.input_ids), starts


@torch.no_grad()
def causal(s):
    tok, model = load_model()
    pad = tok.pad_token_id or 0
    cols = {k: np.full(len(s), np.nan) for k in
            ["s_w8", "s_sent", "s_full", "h_w8", "h_full", "pool_ctx", "kl_w8_full"]}
    for tid, g in s.groupby("text_id"):
        ids, starts = offsets(tok, g.text.iat[0])
        first = {}
        for k, c in enumerate(starts):
            first.setdefault(int(c), k)
        K = np.array([first.get(int(c), -1) for c in g.target_char])
        Ks = np.array([np.searchsorted(starts, c) for c in g.sent_char])  # first subword of the sentence
        keep = K >= 1
        if not keep.any():
            continue
        rows, K, Ks = g.index[keep], K[keep], Ks[keep]
        # full left context: one pass over the essay up to the last target
        full = torch.log_softmax(model(input_ids=torch.as_tensor(ids[:K.max() + 1])[None].cuda())
                                 .logits[0].float(), -1)
        lp_full = full[torch.as_tensor(K - 1)]                         # [n, V]
        w8 = [ids[max(0, k - WINDOW + 1):k + 1] for k in K]
        i8, a8, p8 = pad_windows(w8, pad)
        lp_w8 = torch.log_softmax(model(input_ids=i8.cuda(), attention_mask=a8.cuda(),
                                        position_ids=p8.cuda()).logits[:, -2].float(), -1)
        tgt = torch.as_tensor(ids[K]).cuda()[:, None]
        cols["s_full"][rows] = (-lp_full.gather(1, tgt)[:, 0] * LOG2E).cpu().numpy()
        cols["s_w8"][rows] = (-lp_w8.gather(1, tgt)[:, 0] * LOG2E).cpu().numpy()
        cols["h_full"][rows] = (-(lp_full.exp() * lp_full).sum(-1) * LOG2E).cpu().numpy()
        cols["h_w8"][rows] = (-(lp_w8.exp() * lp_w8).sum(-1) * LOG2E).cpu().numpy()
        cols["kl_w8_full"][rows] = ((lp_w8.exp() * (lp_w8 - lp_full)).sum(-1) * LOG2E).cpu().numpy()
        top = lp_w8.topk(POOL_K, -1).indices
        cols["pool_ctx"][rows] = ((lp_w8.gather(1, top) - lp_full.gather(1, top)).mean(-1) * LOG2E).cpu().numpy()
        # sentence-only left context
        sm = K > Ks
        if sm.any():
            ws = [ids[a:k + 1] for a, k in zip(Ks[sm], K[sm])]
            i_s, a_s, p_s = pad_windows(ws, pad)
            lp_s = torch.log_softmax(model(input_ids=i_s.cuda(), attention_mask=a_s.cuda(),
                                           position_ids=p_s.cuda()).logits[:, -2].float(), -1)
            cols["s_sent"][rows[sm]] = (-lp_s.gather(1, tgt[torch.as_tensor(sm).cuda()])[:, 0]
                                        * LOG2E).cpu().numpy()
    return pd.DataFrame(cols)


@torch.no_grad()
def mlm(s):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MLM_ID)
    model = AutoModelForMaskedLM.from_pretrained(MLM_ID, dtype=torch.bfloat16).cuda().eval()
    cols = {k: np.full(len(s), np.nan) for k in ["s_mlm_sent", "s_mlm_full", "h_mlm_sent", "h_mlm_full"]}

    def run(seqs, mpos, tgts, rows, name):
        for a in range(0, len(seqs), 16):
            sq = seqs[a:a + 16]
            L = max(map(len, sq))
            x = torch.full((len(sq), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(x)
            for j, q in enumerate(sq):
                x[j, :len(q)] = torch.as_tensor(q)
                att[j, :len(q)] = 1
            lg = model(input_ids=x.cuda(), attention_mask=att.cuda()).logits
            lg = lg[torch.arange(len(sq)), torch.as_tensor(mpos[a:a + 16])].float()
            lp = torch.log_softmax(lg, -1)
            t = torch.as_tensor(tgts[a:a + 16]).cuda()[:, None]
            r = rows[a:a + 16]
            cols[f"s_mlm_{name}"][r] = (-lp.gather(1, t)[:, 0] * LOG2E).cpu().numpy()
            cols[f"h_mlm_{name}"][r] = (-(lp.exp() * lp).sum(-1) * LOG2E).cpu().numpy()

    for tid, g in s.groupby("text_id"):
        text = g.text.iat[0]
        ids, starts = offsets(tok, text)
        for name in ["sent", "full"]:
            seqs, mpos, tgts, rows = [], [], [], []
            for r in g.itertuples():
                c0, c1 = int(r.target_char), int(r.target_char) + len(r.observed)
                k = np.flatnonzero(starts == c0)
                if len(k) == 0:
                    continue
                k = k[0]
                e = k + 1
                while e < len(ids) and starts[e] < c1:
                    e += 1
                if name == "sent":
                    lo = np.searchsorted(starts, r.sent_char)
                    hi = np.searchsorted(starts, r.sent_end_char)
                else:
                    lo, hi = 0, len(ids)
                q = [tok.cls_token_id] + list(ids[lo:k]) + [tok.mask_token_id] * (e - k) \
                    + list(ids[e:max(e, hi)]) + [tok.sep_token_id]
                seqs.append(q)
                mpos.append(1 + k - lo)
                tgts.append(int(ids[k]))
                rows.append(r.Index)
            if seqs:
                run(seqs, mpos, tgts, np.array(rows), name)
    return pd.DataFrame(cols)


def main():
    s = pd.read_parquet(OUT / "sample.parquet")
    for view in sys.argv[1:]:
        t0 = time.time()
        df = causal(s) if view == "causal" else mlm(s)
        df.to_parquet(OUT / f"ctx_{view}.parquet", index=False)
        print(view, f"{time.time() - t0:.0f}s", df.describe().T.round(2).to_string(), flush=True)
        if view == "causal":
            d = (df.s_w8 - s.surprisal_bits).abs()
            print(f"|s_w8 - stored| median {d.median():.4f} max {d.max():.4f}")


if __name__ == "__main__":
    main()
