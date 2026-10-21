"""Study 3, input to stage 9 — English Wikipedia articles, pre- and post-cutoff.

Two samples drawn the same way: random page ids in a creation window (namespace 0):

    wiki_old   created 2018-09-01 .. 2021-12-31, text as of the last revision
               before 2023-01-01 (the state OLMo-mix-1124's Wikipedia/DCLM
               snapshots would have seen)
    wiki_new   created 2026-01-01 .. 2026-09-30, current revision (after the
               OLMo-2 0425 training data, which ends in 2024)

Prose = the text of <p> elements of the parsed article (no lists, tables,
infoboxes, headings, captions); reference markers and edit links removed.
Articles with fewer than MIN_WORDS words of prose, redirects, disambiguation
pages, and articles in AI-generated / machine-translation cleanup categories
are skipped.

Output (data dir given as argument; one JSON line per kept article):
    wiki_old.jsonl, wiki_new.jsonl   title, pageid, revid, rev_timestamp, created, text

    python -I src/3-surprisal-decomposition/fetch_wiki.py data/study3/calibration/dl_wiki
"""

import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "predictive-processing-pilot/0.1 (research; langdonholmes@gmail.com)"
SEED = 42
N_TARGET = 300
MIN_WORDS = 200
N_CANDIDATES = 1500
BAD_CAT = re.compile(r"AI-generated|large language model|machine.translat|"
                     r"needing cleanup after translation|Disambiguation", re.I)
# Page-id windows, located with the creation log and first-revision timestamps:
# 58,353,942 = first article created 2018-09-01; ~69,630,000 = end of 2021;
# 82,001,497 = first article created 2026-01-01; 84,378,641 = last created 2026-09-30.
WINDOWS = {
    "wiki_old": (58_353_942, 69_630_000, "2023-01-01T00:00:00Z"),
    "wiki_new": (82_001_497, 84_378_642, None),
}

def api(**params) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(d["error"].get("code"))
            return d
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            err = e
    raise err


class ParaText(HTMLParser):
    """Collect the text of <p> elements, skipping reference sups, styles, math."""

    SKIP = {"style", "script", "math", "sup"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paras, self.cur, self.in_p, self.skip = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if self.skip or tag in self.SKIP or "mw-editsection" in cls or "noprint" in cls:
            if tag not in {"br", "img", "wbr", "meta", "link"}:
                self.skip += 1
            return
        if tag == "p":
            self.in_p += 1
            self.cur = []

    def handle_endtag(self, tag):
        if self.skip:
            if tag not in {"br", "img", "wbr", "meta", "link"}:
                self.skip -= 1
            return
        if tag == "p" and self.in_p:
            self.in_p -= 1
            t = re.sub(r"\s+", " ", "".join(self.cur)).strip()
            if t:
                self.paras.append(t)

    def handle_data(self, data):
        if self.in_p and not self.skip:
            self.cur.append(data)


def prose(html: str) -> str:
    p = ParaText()
    p.feed(html)
    text = "\n\n".join(p.paras)
    text = re.sub(r"\[(?:\d+|[a-z]|citation needed|note \d+)\]", "", text)
    return re.sub(r" +([.,;:])", r"\1", text)


def candidates(lo: int, hi: int, rng: random.Random) -> list[dict]:
    """Random page ids in [lo, hi) (page ids are assigned in creation order), kept
    if the page is a current mainspace non-redirect. The creation-log query was
    too slow for 2018-21 (DB timeouts), so the window is set by page id."""
    out, seen = [], set()
    while len(out) < N_CANDIDATES:
        ids = [rng.randrange(lo, hi) for _ in range(50)]
        d = api(action="query", prop="info", pageids="|".join(map(str, ids)))
        for pg in d["query"]["pages"]:
            if pg.get("ns") == 0 and not pg.get("missing") and not pg.get("redirect") \
                    and pg["pageid"] not in seen:
                seen.add(pg["pageid"])
                out.append({"title": pg["title"], "pageid": pg["pageid"], "created": None})
        time.sleep(0.2)
    return out


def fetch(c: dict, asof: str | None) -> dict | None:
    d = api(action="query", prop="revisions", pageids=c["pageid"], rvdir="newer", rvlimit=1,
            rvprop="timestamp")
    c = {**c, "created": d["query"]["pages"][0]["revisions"][0]["timestamp"]}
    if asof:
        d = api(action="query", prop="revisions", pageids=c["pageid"], rvstart=asof, rvdir="older",
                rvlimit=1, rvprop="ids|timestamp")
        revs = d["query"]["pages"][0].get("revisions")
        if not revs:
            return None
        revid, rts = revs[0]["revid"], revs[0]["timestamp"]
        d = api(action="parse", oldid=revid, prop="text|categories", disableeditsection=1,
                disablelimitreport=1)
    else:
        d = api(action="parse", pageid=c["pageid"], prop="text|categories|revid", redirects=0,
                disableeditsection=1, disablelimitreport=1)
        revid, rts = d["parse"].get("revid"), None
    p = d["parse"]
    html = p["text"]
    if "redirectMsg" in html or any(BAD_CAT.search(k["category"]) for k in p.get("categories", [])):
        return None
    text = prose(html)
    if len(text.split()) < MIN_WORDS:
        return None
    return {**c, "revid": revid, "rev_timestamp": rts, "text": text}


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (lo, hi, asof) in WINDOWS.items():
        path = out_dir / f"{name}.jsonl"
        if path.exists():
            continue
        rng = random.Random(f"{SEED}-{name}")
        cands = candidates(lo, hi, rng)
        kept, tried = [], 0
        for c in cands:
            if len(kept) >= N_TARGET:
                break
            tried += 1
            try:
                r = fetch(c, asof)
            except Exception as e:  # noqa: BLE001
                print("skip", c["title"], e, flush=True)
                r = None
            if r:
                kept.append(r)
            time.sleep(0.1)
            if tried % 100 == 0:
                print(name, "tried", tried, "kept", len(kept), flush=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
        tmp.rename(path)
        print(name, "candidates", len(cands), "tried", tried, "kept", len(kept), flush=True)


if __name__ == "__main__":
    main()
