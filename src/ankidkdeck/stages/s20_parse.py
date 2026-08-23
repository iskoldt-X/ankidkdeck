"""Stage 20: parse 2026 DDO pages into entries.json (primary key = entry_id).

Selector contract and field shapes follow the adjudicated final guide
(danish_pipelines research/2026-08-v3-review/v3_final_guide.md section 4.6).
Ordering constraint: forms are parsed FIRST because the classifier consumes
them. Never touch a Tailwind utility class; anchor on ids and modern-* only.
"""

import copy as copymod
import re

from bs4 import BeautifulSoup

from ..config import Config
from ..extract import cell_alternatives, is_tag, xt
from ..util import (FatalError, NFC, canonical_json, collapse_ws, nk,
                    read_json, sha256_str, write_json)
from .s12_download import Ledger, raw_path

# The character class excludes " and > only. An apostrophe-excluding form loses
# the witness for every apostrophe headword, because the fejlrapport subject
# embeds the headword: d'herrer (11008184) bridges under this form and not under
# [^"'>]. [verified against the 2025 corpus: 5,259 vs 5,258 files]
MAILTO_RE = re.compile(r"mailto:[^\">]*?\((\d{6,})\)")
AUDIO_ID_RE = re.compile(r'<audio[^>]*id="(\d{6,})_\d+"')
UDTRYK_ID_RE = re.compile(r"^udtryk-\d+$")

# Onym labels occur in singular AND plural; an unmapped label must be logged,
# never silently dropped.
ONYM_MAP = {
    "Synonym": "synonym", "Synonymer": "synonym",
    "Antonym": "antonym", "Antonymer": "antonym",
    "Se også": "see_also",
    "Forkortelse": "abbrev_of",
}

KNOWN_POS_KEYS = {
    "sb.", "vb.", "adj.", "adv.", "pron.", "præp.", "konj.", "artikel",
    "sb. pl.", "fork.", "symbol", "talord (mængdetal)", "talord (ordenstal)",
    "førsteled", "sidsteled", "suffiks", "præfiks", "formelt subjekt",
    "udråbsord", "lydord", "infinitivens", "num.",
}

DIGITS_TRAIL_RE = re.compile(r"\d+$")


def slice_articles(soup):
    """Yield (entry_id, scope, art). One <article> owns exactly one div.artikel
    plus its faste-udtryk sibling. Do NOT iterate .modern-article: the
    tabN-extra divs carry that class without being articles."""
    for art in soup.select("div.artikel"):
        scope = art.find_parent("article") or art
        if len(scope.select("div.artikel")) != 1:
            raise FatalError("article scope holds more than one div.artikel")
        yield entry_id_of(art, scope), scope, art


def entry_id_of(art, scope) -> str:
    # The id lives on the IMMEDIATE PARENT div (measured 216/216), never on
    # div.artikel itself. Two more witnesses (fejlrapport mailto, <audio id>)
    # must agree -- a free integrity assertion that the slice belongs to the id.
    par = art.parent
    eid = par.get("id") if par is not None else None
    if not (eid or "").isdigit():
        eid = next((a.get("id") for a in art.parents
                    if (a.get("id") or "").isdigit()), None)
    if not eid:
        raise FatalError("article slice without a numeric entry_id")
    blob = str(scope)
    for rx in (MAILTO_RE, AUDIO_ID_RE):
        m = rx.search(blob)
        if m and m.group(1) != eid:
            raise FatalError(f"entry_id witnesses disagree: {eid} vs {m.group(1)}")
    return eid


def _labelled_col(blk, label_text):
    for lc in blk.select("div.modern-label-cols"):
        h = lc.select_one("h3.modern-label") or lc.select_one("span.modern-label")
        if h and h.get_text(strip=True) == label_text:
            return lc
    return None


def _clean_links(container):
    out = []
    for a in container.select("a"):
        t = DIGITS_TRAIL_RE.sub("", xt(a, "onym_link"))
        if t:
            out.append(t)
    return out


def _preceding_diskret(kids):
    for prev in reversed(kids):
        if is_tag(prev):
            cls = prev.get("class") or []
            if "diskret" in cls:
                return xt(prev, "udtale_label")
            if "lydskrift" in cls:
                return None
    return None


def _parse_alt_spellings(art):
    """The bare div.modern-row > span.tekst prose row: 'ogsaa i formen'/
    'stavemaade' lines. span.match = official alternative; span.diskret =
    deprecated spelling."""
    out = []
    for row in art.select("div.modern-row"):
        tekst = row.select_one(":scope > span.tekst")
        if tekst is None:
            continue
        prose = tekst.get_text(" ", strip=True)
        if "stavemåde" not in prose and "også i formen" not in prose:
            continue
        for m in tekst.select("span.match"):
            f = collapse_ws(xt(m, "headword"))
            if f:
                out.append({"form": f, "official": True})
        for d in tekst.select("span.diskret"):
            f = collapse_ws(xt(d, "headword"))
            if f and len(f) > 1:
                out.append({"form": f, "official": False})
    return out


def _segment_loop(body):
    """Etymology segmenting, transferred verbatim from the 2025 parser."""
    segments, desc = [], ""
    for node in body.contents:
        if isinstance(node, str):
            desc += node
        elif is_tag(node) and (node.name == "a" or "ordform" in (node.get("class") or [])):
            segments.append({"form": xt(node, "etymology_form"),
                             "description": desc.strip(" ,")})
            desc = ""
        elif is_tag(node) and "dividerDot" in (node.get("class") or []):
            continue
        else:
            desc += xt(node, "etymology_desc") if is_tag(node) else str(node)
    if desc.strip():
        if segments:
            segments[-1]["description"] += " " + desc.strip()
        else:
            segments.append({"form": None, "description": desc.strip()})
    return segments


def parse_article(eid: str, scope, art, registry, report: dict) -> dict:
    e = {"entry_id": eid}
    h = art.select_one("div.modern-top-row h1.modern-match")
    if h is None:
        raise FatalError(f"no h1.modern-match in article slice {eid}")
    e["display_headword"] = xt(h, "headword")
    sup = h.select_one("span.super")
    e["super"] = xt(sup, "headword") if sup else None
    h2 = copymod.copy(h)
    for s in h2.select("span.super"):
        s.decompose()  # decompose the homograph index; never regex trailing digits
    e["lemma"] = NFC(xt(h2, "headword"))
    e["lemma_key"] = nk(e["lemma"])
    pt = art.select_one("div.modern-top-row span.text-large")
    e["pos_text"] = collapse_ws(xt(pt, "pos_text")) if pt else None
    pk = art.select_one("[data-pos-key]")
    e["pos_key"] = pk["data-pos-key"] if pk else None
    if e["pos_key"] and e["pos_key"] not in KNOWN_POS_KEYS:
        # The pos_key vocabulary is OPEN ('formelt subjekt' was in nobody's
        # list); log and continue, never reject on unknown.
        report.setdefault("new_pos_keys", {}).setdefault(e["pos_key"], []).append(eid)
    e["ddo_lastmod"] = art.get("lastmod")  # first-published date; provenance only

    # ---- forms FIRST: the classifier consumes them ----
    e["paradigm"] = {"short": None, "rows": []}
    boj = art.select_one("div.modern-row#id-boj")
    if boj is not None:
        btn = boj.select_one("button.kilde")
        if btn is not None:
            short = xt(btn, "headword")
            # Adjectives render 'Se boejning i skema' here -- not a suffix
            # notation; suppress anything that is not one.
            e["paradigm"]["short"] = short if short.lstrip().startswith("-") else None
        for t, tbl in enumerate(boj.select("table.flex-table")):
            for r_i, tr in enumerate(tbl.select("tbody tr")):
                cells = [c for td in tr.select("td") for c in cell_alternatives(td)]
                cells = list(dict.fromkeys(cells))
                if cells:
                    e["paradigm"]["rows"].append({"table": t, "row": r_i, "cells": cells})
    e["alt_spellings"] = _parse_alt_spellings(art)
    e["form_index"] = sorted(
        {nk(c) for row in e["paradigm"]["rows"] for c in row["cells"]}
        | {nk(e["lemma"])}
        | {nk(a["form"]) for a in e["alt_spellings"]}
    )
    shape = tuple(
        sum(1 for r in e["paradigm"]["rows"] if r["table"] == t)
        for t in sorted({r["table"] for r in e["paradigm"]["rows"]})
    )
    labels = registry.paradigm_labels(e["pos_key"], shape)
    for i, row in enumerate(e["paradigm"]["rows"]):
        row["slot_label"] = labels[i] if labels and i < len(labels) else None

    # ---- udtale + audio ----
    e["udtale"] = []
    cont = art.select_one("div.modern-row#id-udt")
    holder = None
    if cont is not None:
        holder = cont.select_one("span.modern-inline-text span.modern-text") or cont
    kids = list(holder.children) if holder is not None else []
    for i, node in enumerate(kids):
        if not (is_tag(node) and "lydskrift" in (node.get("class") or [])):
            continue
        btn = node.select_one("button.kilde")
        ipa = xt(btn or node, "ipa").strip("[]").strip()
        a_mp3 = node.select_one('a[href$=".mp3"]')
        aud = node.select_one("audio")
        slot = None
        if aud is not None and "_" in (aud.get("id") or ""):
            tail = aud["id"].rsplit("_", 1)[1]
            slot = int(tail) if tail.isdigit() else None
        if a_mp3 is not None and slot is not None:
            expected = f"/mp3/{eid[:5]}/{eid}_{slot}.mp3"
            if not a_mp3["href"].endswith(expected):
                raise FatalError(f"audio URL does not match entry {eid}: {a_mp3['href']}")
        e["udtale"].append({"ipa": ipa, "label": _preceding_diskret(kids[:i]),
                            "audio_url": a_mp3["href"] if a_mp3 else None,
                            "slot_n": slot})

    # ---- etymology ----
    body = art.select_one("div.modern-row#id-ety div.modern-text")
    e["etymology"] = ({"raw": xt(body, "etymology_raw"), "segments": _segment_loop(body)}
                      if body is not None else None)

    # ---- senses ----
    e["senses"] = []
    for box in art.select('div#content-betydninger div.modern-definition-box[id^="betydning-"]'):
        if not box.get("dannetid"):
            raise FatalError(f"sense box without dannetid in entry {eid}")
        blk = box.parent
        row = box.find_parent("div", class_="modern-row")
        no = row.select_one("div.modern-senseno") if row is not None else None
        dspan = box.select_one("span.modern-definition")
        d = xt(dspan, "definition") if dspan is not None else ""
        grammar = None
        for lab in blk.select("div.modern-label-cols > span.modern-label"):
            if lab.get_text(strip=True) == "grammatik":
                sib = lab.find_next_sibling("div", class_="modern-inline-text")
                if sib is not None:
                    grammar = xt(sib, "grammar")
                break
        onyms = {"synonym": [], "antonym": [], "see_also": [], "abbrev_of": []}
        for on in blk.select("div.modern-onym"):
            lab = on.select_one("h3.modern-label")
            ul = on.select_one("ul.modern-inline")
            if lab is None or ul is None:
                continue
            key = ONYM_MAP.get(lab.get_text(strip=True))
            if key is None:
                report.setdefault("unmapped_onym_labels", {}).setdefault(
                    lab.get_text(strip=True), 0)
                report["unmapped_onym_labels"][lab.get_text(strip=True)] += 1
                continue
            onyms[key].extend(_clean_links(ul))
        related_col = _labelled_col(blk, "Ord i nærheden")
        eks_col = _labelled_col(blk, "Eksempler")
        examples = []
        for cb in blk.select("div.modern-citation-box"):
            t = cb.select_one("span.modern-citat")
            if t is None:
                continue
            src = cb.select_one("div.modern-source button")
            full = cb.select_one('div.modern-source [role="dialog"] p')
            examples.append({
                "text": xt(t, "example"),
                "source_short": xt(src, "example_source") if src is not None else None,
                "source_full": xt(full, "example") if full is not None else None,
            })
        s = {
            "dannetid": box["dannetid"],
            "sense_path": box.get("id"),
            "number": xt(no, "sense_number") if no is not None else None,
            "definition": d,
            "grammar": grammar,
            "examples": examples,
            "eksempler": ([xt(li, "expression")
                           for li in eks_col.select("ul.modern-inline > li")]
                          if eks_col is not None else []),
            "onyms": onyms,
            "related": _clean_links(related_col) if related_col is not None else [],
            "src_sha": sha256_str(NFC(d)),
        }
        s["sense_sha"] = sha256_str(NFC(d) + "\x00" + "\x00".join(x["text"] for x in examples))
        e["senses"].append(s)

    # ---- fixed expressions: SIBLING of div.artikel inside the shared <article> ----
    e["expressions"] = []
    sec = scope.select_one("div#faste-udtryk div#content-udtryk")
    if sec is not None:
        for ed in sec.find_all("div", id=UDTRYK_ID_RE, recursive=False):
            names = [xt(x, "expression") for x in ed.select("div.modern-sublemma h3.modern-match")]
            names = [n for n in names if n]
            if not names:
                continue
            senses = []
            for db in ed.select('div.modern-definition-box[id*="betydning"]'):
                dspan = db.select_one("span.modern-definition")
                dd = xt(dspan, "expr_definition") if dspan is not None else ""
                exs = []
                for cb in db.parent.select("div.modern-citation-box"):
                    t = cb.select_one("span.modern-citat")
                    if t is not None:
                        exs.append({"text": xt(t, "expr_example")})
                senses.append({"dannetid": db.get("dannetid"), "definition": dd,
                               "examples": exs, "src_sha": sha256_str(NFC(dd))})
            e["expressions"].append({
                "dannetid": senses[0]["dannetid"] if senses else None,
                "expression": names[0], "variants": names[1:], "senses": senses,
            })

    # ---- orddannelser (all three categories parsed; export policy decides use) ----
    e["orddannelser"] = {}
    cont2 = art.select_one("div#content-orddannelser")
    if cont2 is not None:
        for row in cont2.select("div.modern-row"):
            lab = row.select_one("h3.modern-label")
            if lab is None:
                continue
            items = []
            for ul in row.select("ul.modern-inline"):
                for li in ul.select(":scope > li"):
                    a = li.select_one("a")
                    if a is None:
                        continue
                    pos = li.select_one("span.pos")
                    txt = xt(a, "orddannelse")
                    if pos is not None:
                        txt = (txt + " " + xt(pos, "orddannelse")).strip()
                    items.append(txt)
            e["orddannelser"][lab.get_text(strip=True)] = items

    e["empty"] = not e["senses"] and not e["expressions"]
    # 0-sense articles are real (godte2, vinge2); kept, marked, never rendered.
    e["article_sha"] = sha256_str(canonical_json({k: v for k, v in e.items()}))
    return e


def run(cfg: Config, registry) -> dict:
    ledger = Ledger(cfg)
    entries: dict[str, dict] = {}
    provenance: dict[str, list] = {}
    report: dict = {}
    n_pages = 0
    for word, led in sorted(ledger.data.items()):
        if led.get("status") != "ok":
            continue
        p = raw_path(cfg, word)
        if not p.exists():
            raise FatalError(f"ledger says ok but raw page missing for {word!r}")
        soup = BeautifulSoup(p.read_text(encoding="utf-8"), "html.parser")
        n = 0
        for eid, scope, art in slice_articles(soup):
            parsed = parse_article(eid, scope, art, registry, report)
            n += 1
            if eid in entries and entries[eid]["article_sha"] != parsed["article_sha"]:
                # The same article fetched via two words must parse identically;
                # record rather than die (tab-state noise is possible).
                report.setdefault("cross_page_mismatch", []).append(
                    {"entry_id": eid, "word": word})
            entries[eid] = parsed
            provenance.setdefault(eid, []).append(word)
        if n != led.get("article_count"):
            raise FatalError(
                f"page for {word!r}: parsed {n} articles, ledger says {led.get('article_count')}")
        if n == 0:
            raise FatalError(f"page for {word!r} parsed to zero articles but verdict was ok")
        n_pages += 1
    if not entries:
        raise FatalError("no entries parsed -- is the crawl done?")
    for eid, words in provenance.items():
        entries[eid]["source_words"] = sorted(set(words))
    write_json(cfg.json_dir / "entries.json", entries)
    report.update({"pages_parsed": n_pages, "entries": len(entries),
                   "empty_entries": sum(1 for e in entries.values() if e["empty"])})
    write_json(cfg.report_dir / "parse_report.json", report)
    return report
