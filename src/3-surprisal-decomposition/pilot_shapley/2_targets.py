"""Pilot step 2 (CPU) — targets, their windows and players, and the sample.

For every scored W&I+LOCNESS token the 8-subword window is rebuilt exactly as
``features.predictability.Predictor`` builds it (full-text tokenisation, first
subword found by ``np.intersect1d`` on whitespace-stripped starts). Only
targets past the first window (subword index k >= 8, the equal-length phase-2
windows) are used, so every window is 7 context subwords + the target.

Each context subword is assigned to the spaCy token in which its first
non-space character lies; consecutive subwords of one token form a *player*.
Player labels, from the first annotator's json edits (as in 2_select.gold_for_doc):
    edited   the token overlaps a sub/del edit (ERR_TOK) or a word is missing
             just before it (ERR_INS: the player right after an insertion point)
    flagged  the token overlaps a null-replacement (detection-only) span
Each edited player carries the ERRANT group(s) of its edit(s) (1_errant.py).

Target sets (``set``):
    cand     own aligned token, gold OK or POST (not ERR_INS: that is the
             omission part), >= 1 edited player in the window
    ctrl     own aligned token, gold OK, no edited player
    err      ERR_TOK token (a corrected token itself)
Excluded everywhere: windows with a flagged player; targets whose first
subword is whitespace-only (paragraph-initial newline alignment); windows
where a context subword maps to the target's own token.

Sample: essays drawn per level (seed 42), every cand and err target in them;
ctrl drawn 1:1 to cand, matched on level x surprisal decile x number of
players, from the same essays first and then from other essays of the level
(A has too few clean windows), falling back to level x decile when a cell
runs short. Draws are without replacement.

Outputs under data/study3/pilot_shapley/:
    targets_all.parquet   every eligible target (no window ids)
    sample.parquet        the sample, with window ids and player structure

    python src/3-surprisal-decomposition/pilot_shapley/2_targets.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LEVELS, OUT, PC, SEED, WIN, WI_RAW  # noqa: E402
from util.paths import WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

HF_ID, REV = "allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf"
N_ESSAYS = {"A": 80, "B": 80, "C": 80, "N": 50}


def json_edits():
    out = {}
    for f in sorted((WI_RAW / "json").glob("*.json")):
        for line in open(f):
            d = json.loads(line)
            out[d["id"]] = [x for c, x in d["edits"] if c == 0][0]
    return out


def essay_targets(text, toks, edits, types, tok):
    """Rows for one essay. toks: all spaCy tokens (incl. space) sorted by spacy_idx."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.array(enc.input_ids)
    adj = np.array([s + len(text[s:e]) - len(text[s:e].lstrip()) for s, e in enc.offset_mapping])
    ws_only = np.array([text[s:e].strip() == "" for s, e in enc.offset_mapping])
    sp_start = toks.idx.to_numpy()
    _, c2s, c2t = np.intersect1d(sp_start, adj, return_indices=True)
    first_sub = dict(zip(toks.spacy_idx.to_numpy()[c2s], c2t))

    # char -> spaCy token (non-space tokens only); subword -> token of first non-space char
    owner = np.full(len(text) + 1, -1)
    for si, a, t, sp in zip(toks.spacy_idx, toks.idx, toks.text, toks.is_space):
        if not sp:
            owner[a:a + len(t)] = si
    nxt = np.full(len(text) + 1, -1)
    cur = -1
    for c in range(len(text), -1, -1):
        if owner[c] >= 0:
            cur = owner[c]
        nxt[c] = cur
    sub_tok = nxt[np.minimum(adj, len(text))]

    # per non-space token: gold-like labels and edit indices
    ns = toks[~toks.is_space]
    rank = {si: r for r, si in enumerate(ns.spacy_idx)}
    spans = [(k, s, e, r) for k, (s, e, r) in enumerate(e_[:3] for e_ in edits)]
    lab = {}
    prev_end = 0
    for si, a, t in zip(ns.spacy_idx, ns.idx, ns.text):
        b = a + len(t)
        tok_e = [k for k, s, e, r in spans if e > s and s < b and e > a]
        ins_e = [k for k, s, e, r in spans if e == s and prev_end <= s <= a]
        flagged = any(spans[k][3] is None for k in tok_e + ins_e)
        err_tok = [k for k in tok_e if spans[k][3] is not None]
        err_ins = [k for k in ins_e if spans[k][3] is not None]
        lab[si] = (err_tok, err_ins, flagged)
        prev_end = b

    rows = []
    for si in ns.spacy_idx:
        k = first_sub.get(si)
        if k is None or k < WIN:
            continue
        ctx_tok = sub_tok[k - WIN + 1:k]
        r = {"spacy_idx": int(si), "k": int(k), "target_ws": bool(ws_only[k]),
             "self_overlap": bool((ctx_tok == si).any()) or bool((ctx_tok < 0).any())}
        # players
        pl_tok, pl_of_sub = [], []
        for t_ in ctx_tok:
            if not pl_tok or pl_tok[-1] != t_:
                pl_tok.append(int(t_))
            pl_of_sub.append(len(pl_tok) - 1)
        ed, fl, grp, kinds, gold_p = [], False, [], [], []
        for t_ in pl_tok:
            et, ei, f = lab.get(t_, ([], [], False))
            fl |= f
            ks = et + ei
            ed.append(bool(ks))
            grp.append("|".join(sorted({types.get(x, "unmapped") for x in ks})))
            kinds.append("tok" if et else ("ins" if ei else ""))
        r.update({
            "window": ids[k - WIN + 1:k + 1].tolist(), "player_of_sub": pl_of_sub,
            "player_tok": pl_tok, "n_players": len(pl_tok), "player_edited": ed,
            "player_group": grp, "player_kind": kinds,
            "player_dist": [rank[si] - rank[t_] if t_ in rank else -1 for t_ in pl_tok],
            "any_flagged": fl, "n_edited": int(sum(ed))})
        rows.append(r)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REV)
    essays = pd.read_parquet(PC / "essays.parquet", columns=["text_id", "level", "text"])
    al = pd.read_parquet(PC / "aligned.parquet",
                         columns=["text_id", "spacy_idx", "gold", "own", "aligned", "surprisal_bits",
                                  "s_corr", "delta", "same_window", "is_punct", "pos"])
    et = pd.read_parquet(OUT / "edit_types.parquet")
    types = {t: dict(zip(d.k, d.group)) for t, d in et.groupby("text_id")}
    ed = json_edits()
    orig_by = {}
    for doc in load_all_docbins(WI_DOCBINS_DIR):
        orig_by[doc.user_data["meta"]["text_id"]] = pd.DataFrame(
            {"spacy_idx": [t.i for t in doc], "idx": [t.idx for t in doc],
             "text": [t.text for t in doc], "is_space": [t.is_space for t in doc]})

    allrows = []
    for e in essays.itertuples():
        rows = essay_targets(e.text, orig_by[e.text_id], ed[e.text_id], types.get(e.text_id, {}), tok)
        for r in rows:
            r["text_id"] = e.text_id
            r["level"] = e.level
        allrows += rows
    t = pd.DataFrame(allrows).merge(al, on=["text_id", "spacy_idx"], how="inner")
    ok = ~t.target_ws & ~t.self_overlap & ~t.any_flagged
    t["set"] = None
    own = t.own & t.aligned
    t.loc[ok & own & t.gold.isin(["OK", "POST"]) & (t.n_edited > 0), "set"] = "cand"
    t.loc[ok & own & (t.gold == "OK") & (t.n_edited == 0), "set"] = "ctrl"
    t.loc[ok & (t.gold == "ERR_TOK"), "set"] = "err"
    # sanity: POST targets must have an edited last player; ERR_INS players must say so
    post = t[t.gold == "POST"]
    print("POST with edited last player:",
          np.mean([p[-1] for p in post.player_edited]).round(4), flush=True)
    print("exclusions:", {"target_ws": int(t.target_ws.sum()), "self_overlap": int(t.self_overlap.sum()),
                          "any_flagged": int(t.any_flagged.sum())})
    print(t.groupby(["level", "set"]).size().unstack().to_string(), flush=True)
    t.drop(columns=["window", "player_of_sub"]).to_parquet(OUT / "targets_all.parquet", index=False)

    # ---- sample ---------------------------------------------------------------
    t["decile"] = pd.qcut(t.surprisal_bits, 10, labels=False)
    picked = []
    for lvl in LEVELS:
        ids_ = essays.text_id[essays.level == lvl].to_numpy()
        picked += list(rng.choice(ids_, size=min(N_ESSAYS[lvl], len(ids_)), replace=False))
    s = t[t.text_id.isin(picked)]
    cand, err = s[s.set == "cand"], s[s.set == "err"]
    pool = t[(t.set == "ctrl") & t.level.isin(LEVELS)].copy()
    pool["same_essay"] = pool.text_id.isin(picked)
    pool["r"] = rng.random(len(pool))
    pool = pool.sort_values(["same_essay", "r"], ascending=[False, True])
    take = []
    for keys in (["level", "decile", "n_players"], ["level", "decile"]):
        need = cand.groupby(keys).size() - (t.loc[take].groupby(keys).size() if take else 0)
        need = need.fillna(cand.groupby(keys).size()).astype(int)
        have = pool[~pool.index.isin(take)]
        for kk, g in have.groupby(keys):
            n = int(need.get(kk, 0))
            if n > 0:
                take += g.index[:n].tolist()
        print(f"ctrl matched on {keys}: {len(take)} of {len(cand)}", flush=True)
    ctrl = t.loc[take]
    smp = pd.concat([cand, ctrl, err]).reset_index(drop=True)
    smp.to_parquet(OUT / "sample.parquet", index=False)
    print("sample essays:", len(picked), "ctrl from sampled essays:", ctrl.text_id.isin(picked).mean().round(3))
    print(smp.groupby(["level", "set"]).size().unstack().to_string())
    print("players per target:", smp.n_players.describe().round(2).to_dict())
    print("forward passes (2^p):", int((2 ** smp.n_players).sum()))


if __name__ == "__main__":
    main()
