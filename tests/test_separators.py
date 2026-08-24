"""G-SEP: the golden, two-sided separator test. THE most important test here.

The 2025 translation asset is keyed by the exact Danish strings the 2025 parser
produced. Stage 41 binds a legacy row by BYTE-EQUAL text first, so if a single
field's get_text() separator drifts, 22,734 cells x 4 languages stop matching
and the deck silently ships bare definitions.

Two-sided means: the right separator reproduces the 2025 strings, AND the wrong
one provably does not. A one-sided test would pass on a parser that space-joined
everything.

Fixtures: the entry_ids that join 2025 <-> 2026 (built by tools/build_fixtures.py
from the saved 2026 pages and the legacy corpus). Extend to >= 500 after the
first full crawl.
"""

import json

import pytest
from conftest import FIXTURES_DIR, parse_fixture_page, require_fixtures

from ankidkdeck import extract

MANIFEST = require_fixtures()

# Guide 1.1, measured: definitions reproduce 388/388 under " " and 330/388
# under ""; expressions 689/689 under "" and 499/689 under " ".
MIN_CORRECT_RATE = 0.98
MAX_WRONG_RATE = 0.90


def _load(name):
    path = FIXTURES_DIR / MANIFEST["expected"][name]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_all(registry) -> dict:
    entries = {}
    for page in MANIFEST["pages"]:
        entries.update(parse_fixture_page(FIXTURES_DIR / page["file"], registry))
    return entries


def _rate(entries: dict, expected: dict, field: str) -> tuple:
    hit = total = 0
    misses = []
    for eid, wanted in expected.items():
        e = entries.get(eid)
        if e is None:
            continue
        if field == "definitions":
            have = {s["definition"] for s in e["senses"]}
            have |= {s["definition"] for x in e["expressions"]
                     for s in x.get("senses", [])}
        else:
            have = {x["expression"] for x in e["expressions"]}
            have |= {v for x in e["expressions"] for v in x.get("variants", [])}
        for text in wanted:
            total += 1
            if text in have:
                hit += 1
            elif len(misses) < 10:
                misses.append({"entry_id": eid, "text": text})
    return (hit / total if total else None), total, misses


@pytest.fixture(scope="module")
def joinable():
    ids = MANIFEST.get("joinable_entry_ids") or []
    if not ids:
        pytest.skip("no entry_id joins 2025 and 2026 in this fixture set")
    return set(ids)


def test_definitions_reproduce_the_2025_strings(registry, joinable):
    expected = {k: v for k, v in _load("definitions").items() if k in joinable}
    if not expected:
        pytest.skip("no joinable definition texts in this fixture set")
    rate, total, misses = _rate(_parse_all(registry), expected, "definitions")
    assert rate >= MIN_CORRECT_RATE, (
        "definition reproduction %.3f over %d texts; first misses: %s"
        % (rate, total, misses))


def test_expressions_reproduce_the_2025_strings(registry, joinable):
    expected = {k: v for k, v in _load("expressions").items() if k in joinable}
    if not expected:
        pytest.skip("no joinable expression texts in this fixture set")
    rate, total, misses = _rate(_parse_all(registry), expected, "expressions")
    assert rate >= MIN_CORRECT_RATE, (
        "expression reproduction %.3f over %d texts; first misses: %s"
        % (rate, total, misses))


def test_the_wrong_definition_separator_provably_fails(registry, joinable,
                                                       monkeypatch):
    expected = {k: v for k, v in _load("definitions").items() if k in joinable}
    if not expected:
        pytest.skip("no joinable definition texts in this fixture set")
    right, _, _ = _rate(_parse_all(registry), expected, "definitions")
    monkeypatch.setitem(extract.SEP, "definition", "")
    monkeypatch.setitem(extract.SEP, "expr_definition", "")
    wrong, _, _ = _rate(_parse_all(registry), expected, "definitions")
    assert wrong < right
    assert wrong <= MAX_WRONG_RATE, (
        "removing the definition separator still reproduced %.3f of the 2025 "
        "strings; this fixture set cannot detect the drift it exists to detect"
        % wrong)


def test_the_wrong_expression_separator_provably_fails(registry, joinable,
                                                       monkeypatch):
    expected = {k: v for k, v in _load("expressions").items() if k in joinable}
    if not expected:
        pytest.skip("no joinable expression texts in this fixture set")
    right, _, _ = _rate(_parse_all(registry), expected, "expressions")
    monkeypatch.setitem(extract.SEP, "expression", " ")
    wrong, _, _ = _rate(_parse_all(registry), expected, "expressions")
    assert wrong < right
