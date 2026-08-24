#!/usr/bin/env python3
"""Assemble tests/ fixtures from the saved DDO pages and the 2025 corpus.

RUN THIS ON THE HOST THAT HOLDS THE DATA (home-vm). The output contains DDO
text, so it is gitignored by policy and must never be committed or uploaded.
Nothing here talks to the network.

    python3 tools/build_fixtures.py \
        --pages ~/scratch/ddo-probe-b/html_2026 ~/scratch/ddo-probe-a \
        --legacy-workspace ~/GitHub/danish_pipelines/data/recovered-v2.1-workspace \
        --out work/fixtures

    python3 tools/build_fixtures.py --work work \
        --legacy-workspace .../recovered-v2.1-workspace --out work/fixtures

TWO INPUT MODES, and --work is the one that matters at scale. --pages reads
directories of human-named .html files, which is all the probe dirs are; but the
pipeline stores its own crawl as raw/<sha1(NFC(word))>.html plus
json/fetch_ledger.json, so without --work the 39-page (and later ~5,000-page)
crawl corpus was simply unusable as a fixture source. G-SEP is "THE most
important test here" and it was stuck at 27 joinable entry_ids against a target of
">= 500 after the first full crawl". --work recovers the word from the ledger, so
the page name is the word and every crawled page is a candidate. Both modes may
be given at once; --pages wins a name collision (pass the cleanest dir first).
tests/test_parse_pages.py looks for pages named exactly "hus" and "god".

What it writes:

    work/fixtures/manifest.json                      pages, counts, joinable ids
    work/fixtures/pages_2026/<name>.html             the saved pages, verbatim
    work/fixtures/expected/definitions_by_entry.json {entry_id: [danish text]}
    work/fixtures/expected/expressions_by_entry.json {entry_id: [danish text]}
    work/fixtures/expected/legacy_meta_by_entry.json {entry_id: {...}}

The expected values come from the 2025 corpus -- the strings the existing
22,734 translation cells are keyed by -- so test_separators.py is a real
two-sided golden test: the right separator reproduces them, the wrong one does
not. Only entry_ids that exist on BOTH sides are listed as joinable.

Dependencies: the standard library plus beautifulsoup4. It deliberately does not
import ankidkdeck: a fixture builder that shares code with the parser under test
can hide the bug it is supposed to expose.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

MAILTO_RE = re.compile(r'mailto:[^">]*?\((\d{6,})\)')
AUDIO_ID_RE = re.compile(r'<audio[^>]*id="(\d{6,})_\d+"')
NOHIT_MARKER = "matcher ingen opslag i ordbogen"
NAME_PREFIX_RE = re.compile(r"^\d+[_-]")
NAME_KIND_RE = re.compile(r"^(query|path|select|deeplink)[_-]")


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def nk(s):
    return nfc(s).casefold()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text_nfc_tolerant(path: Path) -> str:
    """204 of the 5,267 legacy filenames are NFD on disk while 0 map keys are:
    macOS normalised on write, so a Linux open() by key misses them."""
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    alt = path.parent / unicodedata.normalize("NFD", path.name)
    if alt.exists():
        return alt.read_text(encoding="utf-8", errors="replace")
    return ""


def page_name(path: Path) -> str:
    stem = path.stem.lstrip("_")
    stem = NAME_PREFIX_RE.sub("", stem)
    stem = NAME_KIND_RE.sub("", stem)
    return stem or path.stem


def describe_page(path: Path) -> dict:
    """Article count, results label and entry_ids, using only the primitives the
    pipeline itself anchors on (ids and modern-* classes, never Tailwind)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    label_el = soup.select_one("#results-label")
    arts = soup.select("div.artikel")
    entry_ids = []
    for art in arts:
        eid = None
        node = art.parent
        while node is not None and getattr(node, "get", None) is not None:
            candidate = node.get("id")
            if (candidate or "").isdigit():
                eid = candidate
                break
            node = node.parent
        if eid:
            entry_ids.append(eid)
    kind = "word"
    if not arts and NOHIT_MARKER in text:
        kind = "nonword"
    elif path.stem.lower().find("deeplink") >= 0:
        kind = "deeplink"
    lemmas = []
    for h in soup.select("div.modern-top-row h1.modern-match"):
        copy = BeautifulSoup(str(h), "html.parser")
        for sup in copy.select("span.super"):
            sup.decompose()
        lemmas.append(nfc(copy.get_text("", strip=True)))
    return {
        "name": page_name(path),
        "source": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_articles": len(arts),
        "results_label": (label_el.get_text(strip=True)
                          if label_el is not None else None),
        "entry_ids": entry_ids,
        "lemmas": lemmas,
        "kind": kind,
    }


def is_ddo_page(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:200000]
    return ("modern-match" in head or NOHIT_MARKER in head
            or "results-label" in head)


def _adopt(path: Path, out_dir: Path, pages: list, used_names: set,
           name: str | None = None, extra: dict | None = None) -> None:
    info = describe_page(path)
    base = name or info["name"]
    candidate, n = base, 2
    while candidate in used_names:
        candidate = "%s__%d" % (base, n)
        n += 1
    used_names.add(candidate)
    info["name"] = candidate
    info.update(extra or {})
    rel = "pages_2026/%s.html" % candidate
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    info["file"] = rel
    pages.append(info)


def collect_pages(dirs, out_dir: Path, pages=None, used_names=None) -> list:
    pages = [] if pages is None else pages
    used_names = set() if used_names is None else used_names
    for d in dirs or ():
        d = Path(d).expanduser()
        if not d.is_dir():
            print("skip (not a directory): %s" % d, file=sys.stderr)
            continue
        for path in sorted(d.glob("*.html")):
            if not is_ddo_page(path):
                continue
            _adopt(path, out_dir, pages, used_names)
    return pages


def safe_name(word: str) -> str:
    """A filesystem-safe fixture name for a DDO query word.

    The pipeline stores pages under sha1(NFC(word)) precisely because a word is
    not a filename (case-insensitive filesystems, spaces, dots). The fixture set
    is read by name, so the word comes back here -- with the same hazards handled
    explicitly and the sha1 prefix appended whenever anything had to be replaced,
    so `min.` and `min` cannot collide into one page.
    """
    w = nfc(word)
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in w)
    cleaned = cleaned.strip("_") or "page"
    if cleaned != w:
        cleaned = "%s-%s" % (cleaned, hashlib.sha1(w.encode("utf-8")).hexdigest()[:8])
    return cleaned


def collect_from_work(work: Path, out_dir: Path, pages=None,
                      used_names=None) -> tuple:
    """The pipeline's OWN crawl corpus: raw/<sha1(NFC(word))>.html + the ledger.

    Only `ok` pages and `nohit` pages are adopted: an `error` page is a fetch or
    parse failure, not a specimen of DDO markup, and `attempted` means the run
    never got an answer. The ledger's own results_label / article_count are
    carried into the manifest so tests/test_results_label.py can reconcile
    against what the CRAWL recorded, not only against what the page says now.
    """
    pages = [] if pages is None else pages
    used_names = set() if used_names is None else used_names
    work = Path(work).expanduser()
    ledger_path = work / "json" / "fetch_ledger.json"
    raw_dir = work / "raw"
    if not ledger_path.exists():
        print("skip --work %s: no json/fetch_ledger.json" % work, file=sys.stderr)
        return pages, {"work": str(work), "adopted": 0, "reason": "no ledger"}
    if not raw_dir.is_dir():
        print("skip --work %s: no raw/" % work, file=sys.stderr)
        return pages, {"work": str(work), "adopted": 0, "reason": "no raw dir"}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    stats = {"work": str(work), "ledger_words": len(ledger), "adopted": 0,
             "skipped_status": {}, "missing_raw": 0}
    for word in sorted(ledger):
        row = ledger[word] or {}
        status = row.get("status")
        if status not in ("ok", "nohit"):
            stats["skipped_status"][status] = \
                stats["skipped_status"].get(status, 0) + 1
            continue
        path = raw_dir / ("%s.html" % hashlib.sha1(
            nfc(word).encode("utf-8")).hexdigest())
        if not path.exists():
            stats["missing_raw"] += 1
            continue
        if not is_ddo_page(path):
            stats["skipped_status"]["not_a_ddo_page"] = \
                stats["skipped_status"].get("not_a_ddo_page", 0) + 1
            continue
        _adopt(path, out_dir, pages, used_names, name=safe_name(word),
               extra={"query_word": nfc(word), "ledger_status": status,
                      "ledger_results_label": row.get("results_label"),
                      "ledger_article_count": row.get("article_count"),
                      "from_work": str(work)})
        stats["adopted"] += 1
    return pages, stats


def bridge_legacy(workspace: Path, wanted_keys: set) -> tuple:
    """filename -> entry_id, for the legacy files that plausibly cover the saved
    2026 pages. Bridging all 5,267 files takes minutes and buys nothing here.

    The witness is the fejlrapport mailto (5,259 of 5,267 files carry it); the
    <audio> id is the fallback. NOTE the character class: only " and > are
    excluded, NOT the apostrophe -- d'herrer embeds it in the mailto subject.
    """
    entries = json.loads((workspace / "ddo_entries.json").read_text(
        encoding="utf-8"))
    html_dir = workspace / "ddo_html_all_versions"
    by_file = {}
    for e in entries:
        by_file[e["filename"]] = e
    candidates = [fn for fn, e in by_file.items()
                  if nk(e.get("headword")) in wanted_keys
                  or nk(e.get("query_word")) in wanted_keys]
    bridge = {}
    for fn in sorted(candidates):
        html = read_text_nfc_tolerant(html_dir / fn)
        if not html:
            continue
        m = MAILTO_RE.search(html) or AUDIO_ID_RE.search(html)
        if m:
            bridge[fn] = m.group(1)
    return bridge, by_file


def tie_break_key(fn: str, entry: dict):
    """The written tie-break (guide 1.3), reproduced so the fixture picks the
    same 2025 file whose translations win in stage 40: the lemma's own file
    first, then frequency, then byte order."""
    qw, hw = nk(entry.get("query_word")), nk(entry.get("headword"))
    try:
        rank = int(entry.get("query_rank") or 10 ** 9)
    except (TypeError, ValueError):
        rank = 10 ** 9
    return (0 if hw and qw == hw else 1, rank, fn)


def legacy_texts(bridge: dict, by_file: dict) -> tuple:
    """{entry_id: [definitions]}, {entry_id: [expressions]}, {entry_id: meta}."""
    files_by_eid = {}
    for fn, eid in bridge.items():
        files_by_eid.setdefault(eid, []).append(fn)
    defs, exprs, meta = {}, {}, {}
    for eid, files in files_by_eid.items():
        files.sort(key=lambda f: tie_break_key(f, by_file[f]))
        winner = files[0]
        e = by_file[winner]
        d = [nfc(x.get("definition")) for x in e.get("definitions", [])
             if x.get("definition")]
        x_texts, x_defs = [], []
        for fx in e.get("fixed_expressions", []):
            if fx.get("expression"):
                x_texts.append(nfc(fx["expression"]))
            for detail in fx.get("details", []):
                if detail.get("type") == "definition" and detail.get("text"):
                    x_defs.append(nfc(detail["text"]))
        defs[eid] = sorted(set(d + x_defs))
        exprs[eid] = sorted(set(x_texts))
        meta[eid] = {"winner_file": winner, "files": files,
                     "headword": nfc(e.get("headword")),
                     "pos": e.get("pos"), "query_word": nfc(e.get("query_word")),
                     "query_rank": e.get("query_rank"),
                     "n_definitions": len(d),
                     "n_expressions": len(x_texts)}
    return defs, exprs, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", nargs="+",
                    help="directories of saved 2026 pages (cleanest one first)")
    ap.add_argument("--work", action="append", default=[],
                    help="an ankidkdeck workspace: reads raw/<sha1(word)>.html "
                         "plus json/fetch_ledger.json, so the pipeline's own "
                         "crawl corpus feeds the fixtures. Repeatable.")
    ap.add_argument("--legacy-workspace",
                    help="recovered 2025 workspace (ddo_entries.json + "
                         "ddo_html_all_versions/); omit to build pages only")
    ap.add_argument("--out", default="work/fixtures", help="output directory")
    args = ap.parse_args(argv)
    if not args.pages and not args.work:
        ap.error("give at least one of --pages <dirs> or --work <workspace>")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pages: list = []
    used_names: set = set()
    # --pages first: those directories are human-curated and their names are the
    # ones tests/test_parse_pages.py looks for.
    collect_pages(args.pages, out, pages, used_names)
    work_stats = []
    for w in args.work:
        _, st = collect_from_work(w, out, pages, used_names)
        work_stats.append(st)
    if not pages:
        print("no DDO pages found; nothing written", file=sys.stderr)
        return 1

    page_entry_ids = {eid for p in pages for eid in p["entry_ids"]}
    wanted_keys = {nk(x) for p in pages for x in p["lemmas"]}
    wanted_keys |= {nk(p["name"]) for p in pages}
    # The QUERY WORD, when we know it: with --work the fixture name is a
    # filesystem-safe mangling and the word itself is the key that joins the 2025
    # corpus (2025 buckets were keyed by query_word / headword).
    wanted_keys |= {nk(p["query_word"]) for p in pages if p.get("query_word")}
    wanted_keys.discard("")

    manifest = {
        "built_by": "tools/build_fixtures.py",
        "note": "contains DDO text: gitignored, never committed or uploaded",
        "pages": pages,
        "n_pages": len(pages),
        "sources": {"pages_dirs": list(args.pages or []),
                    "workspaces": work_stats},
        "entry_ids_2026": sorted(page_entry_ids),
        "expected": {"definitions": "expected/definitions_by_entry.json",
                     "expressions": "expected/expressions_by_entry.json",
                     "legacy_meta": "expected/legacy_meta_by_entry.json"},
        "joinable_entry_ids": [],
    }

    defs = exprs = meta = {}
    if args.legacy_workspace:
        ws = Path(args.legacy_workspace).expanduser()
        if not (ws / "ddo_entries.json").exists():
            print("legacy workspace has no ddo_entries.json: %s" % ws,
                  file=sys.stderr)
            return 1
        bridge, by_file = bridge_legacy(ws, wanted_keys)
        defs, exprs, meta = legacy_texts(bridge, by_file)
        joinable = sorted(set(defs) & page_entry_ids)
        manifest["joinable_entry_ids"] = joinable
        manifest["legacy"] = {
            "workspace": str(ws),
            "files_bridged": len(bridge),
            "entry_ids_bridged": len(defs),
            "joinable": len(joinable),
            "definition_texts": sum(len(defs[e]) for e in joinable),
            "expression_texts": sum(len(exprs[e]) for e in joinable),
        }

    exp_dir = out / "expected"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("definitions_by_entry.json", defs),
                          ("expressions_by_entry.json", exprs),
                          ("legacy_meta_by_entry.json", meta)):
        (exp_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print("fixtures written to %s" % out)
    print("  pages          %d (%s)" % (len(pages),
                                        ", ".join(p["name"] for p in pages[:8])))
    for st in work_stats:
        print("  from --work    %s: %d adopted of %s ledger words (%s)"
              % (st.get("work"), st.get("adopted", 0),
                 st.get("ledger_words", "?"), st.get("skipped_status")
                 or st.get("reason") or "nothing skipped"))
    print("  2026 entry_ids %d" % len(page_entry_ids))
    if args.legacy_workspace:
        print("  joinable ids   %d  (definition texts %d, expression texts %d)"
              % (manifest["legacy"]["joinable"],
                 manifest["legacy"]["definition_texts"],
                 manifest["legacy"]["expression_texts"]))
        if manifest["legacy"]["joinable"] == 0:
            print("  WARNING: nothing joins; test_separators.py will skip",
                  file=sys.stderr)
    print("run the suite with: ANKIDKDECK_FIXTURES=%s python -m pytest tests"
          % out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
