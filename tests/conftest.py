"""Shared test scaffolding. Everything here is offline.

Two classes of test live in this suite:

  * PURE LOGIC (urls, extract, registry, classifier, merge, gates) -- runs
    anywhere, on any checkout, with no DDO content on disk. These are the tests
    CI must keep green.
  * FIXTURE-DEPENDENT (separators, parse, results label) -- needs saved DDO
    pages, which are gitignored and never committed. Those modules skip
    themselves when the fixtures are absent.

Fixtures live in $ANKIDKDECK_FIXTURES, or work/fixtures/ under the repo root;
build them with tools/build_fixtures.py on the run host.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # src layout: the tests must run from a plain checkout, not only from an
    # installed wheel.
    sys.path.insert(0, str(SRC_DIR))

FIXTURES_DIR = Path(os.environ.get("ANKIDKDECK_FIXTURES")
                    or (REPO_ROOT / "work" / "fixtures"))
FIXTURE_MANIFEST = FIXTURES_DIR / "manifest.json"

NO_FIXTURES_REASON = (
    "DDO page fixtures are not present (%s). They are gitignored by policy; "
    "build them with tools/build_fixtures.py on the host that holds the "
    "corpus." % FIXTURE_MANIFEST)


def fixtures_available() -> bool:
    return FIXTURE_MANIFEST.exists()


def require_fixtures() -> dict:
    """Module-level guard for the fixture-dependent test files."""
    if not fixtures_available():
        pytest.skip(NO_FIXTURES_REASON, allow_module_level=True)
    import json
    with open(FIXTURE_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


requires_fixtures = pytest.mark.skipif(not fixtures_available(),
                                       reason=NO_FIXTURES_REASON)


# --------------------------------------------------------------------------
# config / registry
# --------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    from ankidkdeck.config import Config
    c = Config(work_dir=tmp_path / "work")
    for d in (c.json_dir, c.report_dir, c.review_dir, c.audio_dir,
              c.registry_local):
        d.mkdir(parents=True, exist_ok=True)
    return c


@pytest.fixture
def registry(cfg):
    from ankidkdeck.registry import Registry
    return Registry(cfg)


# --------------------------------------------------------------------------
# synthetic entries: the same shape stage 20 writes, minus the HTML
# --------------------------------------------------------------------------

def make_sense(dannetid: str, definition: str, examples=(), number=None) -> dict:
    from ankidkdeck.util import NFC, sha256_str
    return {
        "dannetid": dannetid,
        "sense_path": "betydning-%s" % dannetid,
        "number": number,
        "definition": definition,
        "grammar": None,
        "examples": [{"text": t, "source_short": None, "source_full": None}
                     for t in examples],
        "eksempler": [],
        "onyms": {"synonym": [], "antonym": [], "see_also": [], "abbrev_of": []},
        "related": [],
        "src_sha": sha256_str(NFC(definition)),
        "sense_sha": sha256_str(NFC(definition)),
    }


def make_expression(dannetid: str, expression: str, definition: str = "",
                    variants=()) -> dict:
    from ankidkdeck.util import NFC, sha256_str
    senses = []
    if definition:
        senses.append({"dannetid": dannetid, "definition": definition,
                       "examples": [], "src_sha": sha256_str(NFC(definition))})
    return {"dannetid": dannetid, "expression": expression,
            "variants": list(variants), "senses": senses}


def make_entry(entry_id: str, lemma: str, *, pos_key="sb.", pos_text="substantiv",
               senses=(), expressions=(), forms=(), alt_spellings=(),
               display_headword=None, super_=None, paradigm_rows=None,
               paradigm_short=None, udtale=(), source_words=(),
               orddannelser=None, etymology=None) -> dict:
    """One entries.json row. `forms` are extra flex-table cells; the lemma is
    always in form_index, exactly as stage 20 writes it."""
    from ankidkdeck.util import NFC, canonical_json, nk, sha256_str
    rows = paradigm_rows
    if rows is None:
        rows = ([{"table": 0, "row": i, "cells": [f], "slot_label": None}
                 for i, f in enumerate(forms)] if forms else [])
    alt = [a if isinstance(a, dict) else {"form": a, "official": True}
           for a in alt_spellings]
    e = {
        "entry_id": entry_id,
        "display_headword": display_headword or lemma,
        "lemma": NFC(lemma),
        "lemma_key": nk(lemma),
        "super": super_,
        "pos_text": pos_text,
        "pos_key": pos_key,
        "ddo_lastmod": None,
        "paradigm": {"short": paradigm_short, "rows": rows},
        "alt_spellings": alt,
        "udtale": [dict(u) for u in udtale],
        "etymology": etymology,
        "senses": list(senses),
        "expressions": list(expressions),
        "orddannelser": orddannelser or {},
        "source_words": list(source_words),
    }
    e["form_index"] = sorted({nk(c) for r in rows for c in r["cells"]}
                             | {nk(lemma)}
                             | {nk(a["form"]) for a in alt})
    e["empty"] = not e["senses"] and not e["expressions"]
    e["article_sha"] = sha256_str(canonical_json(e))
    return e


def parse_fixture_page(path, registry) -> dict:
    """{entry_id: entry} for one saved page, through the real stage-20 parser.

    bs4 is imported here so the pure-logic tests never need it.
    """
    from bs4 import BeautifulSoup

    from ankidkdeck.stages.s20_parse import parse_article, slice_articles

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    report: dict = {}
    out = {}
    for eid, scope, art in slice_articles(soup):
        out[eid] = parse_article(eid, scope, art, registry, report)
    return out


def write_workspace(cfg, entries: dict, wordlist: list, classification=None,
                    v2_querywords=None) -> None:
    """Lay down the json inputs a stage expects, so a stage can be run for
    real in a tmp dir instead of being reimplemented by the test."""
    from ankidkdeck.util import write_json
    write_json(cfg.json_dir / "entries.json", entries)
    write_json(cfg.json_dir / "wordlist.json",
               {"source": "test", "sha256": "test",
                "words": [{"rank": r, "raw": w, "word": w}
                          for r, w in wordlist]})
    if classification is not None:
        write_json(cfg.json_dir / "classification.json", classification)
    if v2_querywords is not None:
        write_json(cfg.json_dir / "legacy" / "v2_querywords.json", v2_querywords)
