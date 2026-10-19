"""Pilot step 5 (CPU) — KWIC for the qualitative pass (local only).

Among cand targets, unedited players that are the target's top positive player
with phi >= 2 bits. 15 per level drawn (seed 42). Each excerpt (< 100 words)
marks the player [[...]], the target <<...>>, and the gold edits in the shown
span {orig => repl}. Written to data/study3/pilot_shapley/kwic_strong_unedited.txt
(never to the repository). Labels go to kwic_labels.csv next to it, by hand.

    python src/3-surprisal-decomposition/pilot_shapley/5_kwic.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LEVELS, OUT, PC, SEED  # noqa: E402

spec = importlib.util.spec_from_file_location("an", HERE / "4_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def main():
    t = an.load_main()
    c = t[t.set == "cand"]
    rows = []
    for r in c.itertuples():
        a = np.asarray(r.phi)
        i = int(a.argmax())
        if a[i] >= 2 and not r.player_edited[i]:
            rows.append({"text_id": r.text_id, "level": r.level, "target": r.spacy_idx,
                         "player": r.player_tok[i], "phi": a[i], "d": r.player_dist[i],
                         "s": r.surprisal_bits, "delta": r.delta,
                         "phi_E": a[np.asarray(r.player_edited)].sum()})
    pool = pd.DataFrame(rows)
    print("strong unedited top players:", len(pool), "of", len(c), "cand;",
          pool.groupby("level").size().to_dict())
    pick = pd.concat([g.sample(min(15, len(g)), random_state=SEED) for _, g in pool.groupby("level")])
    pick = pick.reset_index(drop=True)
    essays = pd.read_parquet(PC / "essays.parquet", columns=["text_id", "text"]).set_index("text_id").text
    edits = pd.read_parquet(OUT / "edit_types.parquet")
    al = pd.read_parquet(PC / "aligned.parquet", columns=["text_id", "spacy_idx", "idx", "text"])
    pos = al.set_index(["text_id", "spacy_idx"])
    from util.paths import WI_DOCBINS_DIR
    from util.process_docs import load_all_docbins
    need = set(pick.text_id)
    docs = {d.user_data["meta"]["text_id"]: d for d in load_all_docbins(WI_DOCBINS_DIR)
            if d.user_data["meta"]["text_id"] in need}
    out = []
    for j, r in pick.iterrows():
        doc = docs[r.text_id]
        text = essays[r.text_id]
        pt = doc[int(r.player)]
        tg = doc[int(r.target)]
        # 25 tokens before the player, 6 after the target, capped well under 100 words
        a = doc[max(0, pt.i - 25)].idx
        b_tok = doc[min(len(doc) - 1, tg.i + 6)]
        b = b_tok.idx + len(b_tok.text)
        marks = [(pt.idx, pt.idx + len(pt.text), "[[", "]]"), (tg.idx, tg.idx + len(tg.text), "<<", ">>")]
        ed = edits[(edits.text_id == r.text_id) & (edits.s >= a) & (edits.e <= b) & (edits.kind != "flagged")]
        ev = []  # (position, order, text): closes, then insertions, then opens
        for s0, e0, o, c_ in marks:
            ev += [(s0, 2, o), (e0, 0, c_)]
        for e in ed.itertuples():
            if e.s == e.e:
                ev.append((e.s, 1, f"{{+{e.repl!r} ({e.group})}}"))
            else:
                ev += [(e.s, 2, "{"), (e.e, 0, f" => {e.repl!r} ({e.group})}}")]
        pieces = []
        p = a
        for at, _, sym in sorted(ev, key=lambda x: (x[0], x[1])):
            if at < a or at > b:
                continue
            pieces.append(text[p:at])
            pieces.append(sym)
            p = at
        pieces.append(text[p:b])
        snippet = "".join(pieces).replace("\n", " / ")
        assert len(snippet.split()) < 100
        out.append(f"#{j} level={r.level} phi={r.phi:.2f} d={r.d} s={r.s:.2f} delta={r.delta:.2f} "
                   f"phi_E={r.phi_E:.2f}\n{snippet}\n")
    (OUT / "kwic_strong_unedited.txt").write_text("\n".join(out))
    pick.to_csv(OUT / "kwic_pick.csv", index=False)
    print("wrote", len(out))


if __name__ == "__main__":
    main()
