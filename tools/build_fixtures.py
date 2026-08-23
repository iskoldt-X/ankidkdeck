#!/usr/bin/env python3
"""Assemble tests/ fixtures from the saved DDO pages and the 2025 corpus.

RUN THIS ON THE HOST THAT HOLDS THE DATA (home-vm). The output contains DDO
text, so it is gitignored by policy and must never be committed or uploaded.
Nothing here talks to the network.

    python3 tools/build_fixtures.py \
        --pages ~/scratch/ddo-probe-b/html_2026 ~/scratch/ddo-probe-a \
        --legacy-workspace ~/GitHub/danish_pipelines/data/recovered-v2.1-workspace \
        --out work/fixtures

Pass the cleanest page directory FIRST: page names come from the filenames, and
the first directory wins a name collision. tests/test_parse_pages.py looks for
pages named exactly "hus" and "god", which html_2026/ provides.

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


def collect_pages(dirs, out_dir: Path) -> list:
    pages, used_names = [], set()
    for d in dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            print("skip (not a directory): %s" % d, file=sys.stderr)
            continue
        for path in sorted(d.glob("*.html")):
            if not is_ddo_page(path):
                continue
            info = describe_page(path)
            name = info["name"]
            n = 2
            while name in used_names:
                name = "%s__%d" % (info["name"], n)
                n += 1
            used_names.add(name)
            info["name"] = name
            rel = "pages_2026/%s.html" % name
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            info["file"] = rel
            pages.append(info)
    return pages


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
    ap.add_argument("--pages", nargs="+", required=True,
                    help="directories of saved 2026 pages (cleanest one first)")
    ap.add_argument("--legacy-workspace",
                    help="recovered 2025 workspace (ddo_entries.json + "
                         "ddo_html_all_versions/); omit to build pages only")
    ap.add_argument("--out", default="work/fixtures", help="output directory")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pages = collect_pages(args.pages, out)
    if not pages:
        print("no DDO pages found; nothing written", file=sys.stderr)
        return 1

    page_entry_ids = {eid for p in pages for eid in p["entry_ids"]}
    wanted_keys = {nk(x) for p in pages for x in p["lemmas"]}
    wanted_keys |= {nk(p["name"]) for p in pages}

    manifest = {
        "built_by": "tools/build_fixtures.py",
        "note": "contains DDO text: gitignored, never committed or uploaded",
        "pages": pages,
        "n_pages": len(pages),
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
