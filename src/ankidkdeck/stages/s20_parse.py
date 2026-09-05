"""Stage 20: parse 2026 DDO pages into entries.json (primary key = entry_id).

Selector contract and field shapes follow the adjudicated final guide
(danish_pipelines research/2026-08-v3-review/v3_final_guide.md section 4.6).
Ordering constraint: forms are parsed FIRST because the classifier consumes
them. Never touch a Tailwind utility class; anchor on ids and modern-* only.
"""

import copy as copymod
import re

from bs4 import BeautifulSoup

from .. import extract
from ..config import Config
from ..extract import (ARTICLE_SHA_SCHEMA, ELISION, cell_alternatives,
                       expand_elision, is_tag)
from ..gates import G_LABEL, Gate, ledger_label_reconciliation, run_gates
from ..util import (FatalError, NFC, canonical_json, collapse_ws, nk,
                    read_json, sha256_str, write_json)
from .s12_download import Ledger, raw_path


def xt(node, field: str) -> str:
    """extract.xt, NFC-normalised at the point of extraction.

    src_sha hashes NFC(text) while stage 41 binds legacy rows by RAW byte
    equality, so a single NFD page would silently drop translations instead of
    failing loudly. Normalising here makes every stored field, key and hash
    agree. extract.SEP is still read at call time, which is what lets
    test_separators.py monkeypatch the table.
    """
    return NFC(extract.xt(node, field))


def _label(node) -> str:
    """A DDO display label used as a dict key (onym / orddannelse headings)."""
    return NFC(node.get_text(strip=True))

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
    # These three reached us through the "log and continue" path below: they
    # were the whole of report["new_pos_keys"] on the 2026 corpus (egennavn 8
    # entries, adj. pl. 3, infinitivpartikel 1). They are now hand-translated
    # in all four languages, so they are known, not new. infinitivpartikel is
    # the spelling this corpus serves for what `infinitivens` names; both stay,
    # and pos_translations.json gives them the same term per language.
    "adj. pl.", "egennavn", "infinitivpartikel",
}

DIGITS_TRAIL_RE = re.compile(r"\d+$")

# orddannelser is stored under FIXED schema keys, never under raw DDO display
# text: the section 1.11h "Oevrige into Derivatives" ruling must not become a
# downstream string match against whatever DDO renders. Unknown labels are
# logged to the parse report the same way unmapped onym labels are.
ORDDANNELSE_MAP = {
    "Afledninger": "Afledninger",
    "Afledning": "Afledninger",
    "Sammensætninger": "Sammensætninger",
    "Sammensætning": "Sammensætninger",
    "Sammensaetninger": "Sammensætninger",
    "Øvrige": "Øvrige",
    "Ovrige": "Øvrige",
}


# article_sha is the DDO-EDIT DETECTOR, so it must hash the article's CONTENT and
# nothing else. Schema 1 hashed canonical_json(entry) whole, which meant it also
# hashed:
#   headword_glued   provenance, never rendered
#   paradigm_index   derived from paradigm.rows
#   form_index       derived from paradigm.rows + lemma + alt_spellings
#   source_words     which of OUR queries happened to reach the article
#   slot_label       a registry lookup, i.e. OUR table and not DDO's page
# so renaming a derived field or editing registry/paradigm_slots.json reported
# every one of the 3,812 articles as "changed since last run" -- the exact line a
# human reads as "DDO moved". Excluded here, and the schema number is stamped in
# the ledger so a future change prints "parser schema changed" instead.
SHA_EXCLUDE_FIELDS = ("headword_glued", "paradigm_index", "form_index",
                      "source_words", "article_sha")
SHA_EXCLUDE_PARADIGM_ROW_FIELDS = ("slot_label",)


def content_sha(e: dict) -> str:
    """sha256 over the article's CONTENT fields only. See SHA_EXCLUDE_FIELDS."""
    payload = {k: v for k, v in e.items() if k not in SHA_EXCLUDE_FIELDS}
    par = e.get("paradigm") or {}
    payload["paradigm"] = {
        "short": par.get("short"),
        "rows": [{k: v for k, v in row.items()
                  if k not in SHA_EXCLUDE_PARADIGM_ROW_FIELDS}
                 for row in par.get("rows", [])],
    }
    return sha256_str(canonical_json(payload))


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
        if h and _label(h) == NFC(label_text):
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


def _is_form_shaped(f: str) -> bool:
    """A candidate alternative spelling must look like a word.

    Without this the prose row of `en` yields {"form": "1", "official": true},
    which then enters form_index and searchable_forms as a real alternative.
    """
    return len(f) > 1 and not f.isdigit()


def _parse_alt_spellings(art):
    """The bare div.modern-row > span.tekst prose row: 'ogsaa i formen'/
    'stavemaade' lines. span.match = official alternative; span.diskret =
    deprecated spelling.

    The official flag is load-bearing, not decoration: `khan` carries the
    deprecated spelling `kan`, and letting it count as classification evidence
    is what put the Mongol ruler on the rank-23 `kunne` card. Deprecated forms
    reach searchable_forms only -- never form_index, never variant matching.
    """
    out = []
    for row in art.select("div.modern-row"):
        tekst = row.select_one(":scope > span.tekst")
        if tekst is None:
            continue
        prose = NFC(tekst.get_text(" ", strip=True))
        if "stavemåde" not in prose and "også i formen" not in prose:
            continue
        for m in tekst.select("span.match"):
            f = collapse_ws(xt(m, "headword"))
            if _is_form_shaped(f):
                out.append({"form": f, "official": True})
        for d in tekst.select("span.diskret"):
            f = collapse_ws(xt(d, "headword"))
            if _is_form_shaped(f):
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
    sup = h.select_one("span.super")
    e["super"] = xt(sup, "headword") if sup else None
    h2 = copymod.copy(h)
    for s in h2.select("span.super"):
        s.decompose()  # decompose the homograph index; never regex trailing digits
    e["lemma"] = xt(h2, "headword")
    e["lemma_key"] = nk(e["lemma"])
    # display_headword is the LEMMA, and `super` stays in its own field.
    # xt(h, "headword") glues them ("al2", "udenfor1", "i5" -- 45 of 134 entries
    # on the fixture set), which is not what DDO shows (a superscript) and which
    # leaked into searchable_forms as a junk Anki search token. Stage 70 renders
    # the superscript from `super`.
    e["display_headword"] = e["lemma"]
    e["headword_glued"] = xt(h, "headword")  # provenance only; never rendered
    pt = art.select_one("div.modern-top-row span.text-large")
    e["pos_text"] = collapse_ws(xt(pt, "pos_text")) if pt else None
    pk = art.select_one("[data-pos-key]")
    # NFC: pos_key is a dict key in the demotion registry, the POS translation
    # table and the export's front-side grouping. 'foersteled' carries an
    # o-slash and an NFD attribute would silently miss every lookup.
    e["pos_key"] = NFC(pk["data-pos-key"]) if pk else None
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
                cells = [NFC(c) for td in tr.select("td")
                         for c in cell_alternatives(td)]
                # DDO's own ".." prefix elision, expanded here so that the ONE
                # stored form is the real one: paradigm.rows[].cells is both what
                # stage 70 prints on the card and what paradigm_index (bucket 2's
                # only evidence) is derived from. Leaving it raw printed
                # "..maend" on 14 cards and made julemaend/oversat/efterlod
                # unmatchable. article_sha hashes these cells, hence schema 3.
                cells = [expand_elision(c, e["lemma"]) for c in cells]
                cells = list(dict.fromkeys(cells))
                if cells:
                    if any(ELISION in c for c in cells):
                        bad = {"entry_id": eid, "lemma": e["lemma"], "cells": cells}
                        rows = report.setdefault("unexpanded_elided_cells", [])
                        if bad not in rows:      # parsed once per source page
                            rows.append(bad)
                    e["paradigm"]["rows"].append({"table": t, "row": r_i, "cells": cells})
    e["alt_spellings"] = _parse_alt_spellings(art)
    # TWO indexes, and the difference is the whole classifier:
    #
    #   paradigm_index  real inflection cells ONLY. This is bucket 2's evidence
    #                   ("the word is in this article's own flex table"). It
    #                   deliberately excludes nk(lemma): a case-only pair such
    #                   as er/Er matched form_index through the lemma key, which
    #                   made exact_ci and case_only_demoted_pos unreachable for
    #                   every possible input -- erbium became an inflection of
    #                   `er`.
    #   form_index      cells + lemma + OFFICIAL alternative spellings. This is
    #                   stage 21's reverse index, which WANTS the lemma key --
    #                   that is how an override or a lemma_key lookup finds an
    #                   article. Deprecated spellings are excluded (khan/kan).
    e["paradigm_index"] = sorted(
        {nk(c) for row in e["paradigm"]["rows"] for c in row["cells"]})
    e["form_index"] = sorted(
        set(e["paradigm_index"])
        | {nk(e["lemma"])}
        | {nk(a["form"]) for a in e["alt_spellings"] if a.get("official")}
    )
    # UPSTREAM-DIRTY CELLS, recorded and never patched. paradigm.short is DDO's
    # own short notation for the same slots ("-r, ..lagde, ..lagt"), so an
    # elided token in it whose expansion is absent from the flex table means the
    # page's two renderings of one inflection disagree. Measured: 15 articles
    # carry ".." in `short`, 14 spell the same form out in the cell, and exactly
    # one -- planlaegge (11039990), whose past-tense cell is the truncated
    # "<td>p<span>lagde</span></td>" -> plagde -- does not. That is DDO's data,
    # not ours: a registry hand-patch would put invented text in a content field
    # article_sha treats as DDO's, so it is REPORTED for the owner instead.
    for tok in (e["paradigm"]["short"] or "").split(","):
        tok = tok.strip()
        if not tok.startswith(ELISION):
            continue
        want = expand_elision(tok, e["lemma"])
        if want == tok or nk(want) in e["paradigm_index"]:
            continue
        row = {"entry_id": eid, "lemma": e["lemma"],
               "short": e["paradigm"]["short"], "expected_form": want}
        # An article reached through two query words is parsed twice (11039990
        # comes from both /planlagt and /planlaegger), so a bare append reported
        # one defect as four. This file is a count a human acts on.
        rows = report.setdefault("short_form_missing_from_cells", [])
        if row not in rows:
            rows.append(row)
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
            label = _label(lab)
            key = ONYM_MAP.get(label)
            if key is None:
                report.setdefault("unmapped_onym_labels", {}).setdefault(label, 0)
                report["unmapped_onym_labels"][label] += 1
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
            label = _label(lab)
            key = ORDDANNELSE_MAP.get(label)
            if key is None:
                report.setdefault("unmapped_orddannelse_labels", {}).setdefault(
                    label, 0)
                report["unmapped_orddannelse_labels"][label] += 1
                continue
            # MERGED, not overwritten: a second row carrying the same label used
            # to silently replace the first one's items.
            bucket = e["orddannelser"].setdefault(key, [])
            for txt in items:
                if txt not in bucket:
                    bucket.append(txt)

    e["empty"] = not e["senses"] and not e["expressions"]
    # 0-sense articles are real (godte2, vinge2); kept, marked, never rendered.
    e["article_sha"] = content_sha(e)
    return e


def run(cfg: Config, registry) -> dict:
    ledger = Ledger(cfg)
    entries: dict[str, dict] = {}
    provenance: dict[str, list] = {}
    report: dict = {}
    parsed_counts: dict[str, int] = {}
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
        parsed_counts[word] = n
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

    # G-LABEL as a RECORDED verdict. The reconciliation rule itself has always
    # run (s12.verdict_of, and the per-page FatalError above), but its result
    # never reached gates_report.json, so a release could not show that the
    # crawl's own checksum held. Runs before entries.json is written, like every
    # other gate in this pipeline.
    gates_cfg = getattr(registry, "gates", {}) or {}
    run_gates([
        Gate(G_LABEL, "every ok page's #results-label reconciles with its "
                      "article count, the parse agrees, and the error-page "
                      "population is inside its baseline",
             lambda: ledger_label_reconciliation(
                 ledger.data, parsed_counts,
                 float(gates_cfg.get("label_error_max_rate", 0.01))),
             stage="20"),
    ], cfg, stage="20")

    write_json(cfg.json_dir / "entries.json", entries)
    report.update({"pages_parsed": n_pages, "entries": len(entries),
                   "article_sha_schema": ARTICLE_SHA_SCHEMA,
                   "empty_entries": sum(1 for e in entries.values() if e["empty"])})
    write_json(cfg.report_dir / "parse_report.json", report)
    return report
