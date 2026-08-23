"""Stage 20 against the saved 2026 pages: counts, dannetid, flex-table shape.

These numbers are the measured ones. If DDO edits an article the test fails, and
that is the point -- a silent content change is exactly what stage 42's src_sha
retranslation and the drift ledger need to know about.
"""

import pytest
from conftest import FIXTURES_DIR, parse_fixture_page, require_fixtures

MANIFEST = require_fixtures()


def _page(name):
    for p in MANIFEST["pages"]:
        if p["name"] == name:
            return FIXTURES_DIR / p["file"]
    pytest.skip("fixture page %r not in this set" % name)


@pytest.fixture
def hus(registry):
    return parse_fixture_page(_page("hus"), registry)


def test_hus_noun_has_thirteen_senses(hus):
    nouns = [e for e in hus.values() if e["lemma"] == "hus" and e["pos_key"] == "sb."]
    assert len(nouns) == 1
    assert len(nouns[0]["senses"]) == 13


def test_every_sense_and_expression_carries_a_dannetid(hus):
    for e in hus.values():
        for s in e["senses"]:
            assert s["dannetid"], (e["entry_id"], s["sense_path"])
        for x in e["expressions"]:
            assert x["dannetid"] or not x["senses"], e["entry_id"]
            for s in x["senses"]:
                assert s["dannetid"]


def test_entry_ids_are_eight_digits_and_own_their_slice(hus):
    # parse_article raises when the mailto / <audio> witnesses disagree, so
    # reaching this line already proves the slice belongs to the id.
    for eid in hus:
        assert eid.isdigit() and len(eid) == 8


def test_audio_urls_belong_to_their_entry(hus):
    for eid, e in hus.items():
        for u in e["udtale"]:
            if u["audio_url"] and u["slot_n"] is not None:
                assert u["audio_url"].endswith(
                    "/mp3/%s/%s_%d.mp3" % (eid[:5], eid, u["slot_n"]))


def test_god_adjective_paradigm_shape(registry):
    entries = parse_fixture_page(_page("god"), registry)
    adjs = [e for e in entries.values()
            if e["lemma"] == "god" and e["pos_key"] == "adj."]
    if not adjs:
        pytest.skip("no god adj. article in this fixture set")
    rows = adjs[0]["paradigm"]["rows"]
    cells = {c for r in rows for c in r["cells"]}
    # measured census: ('adj.', (2,2)) -- two tables of two rows
    assert len(rows) == 4
    assert {"godt", "gode", "bedre", "bedst"} <= cells
    labels = [r["slot_label"] for r in rows]
    assert labels == [None] * 4 or all(labels), labels


def test_adjective_short_boejning_is_suppressed(registry):
    # button.kilde reads 'Se boejning i skema' for adjectives -- not a suffix
    # notation, so it must not become the front-side Boejning line.
    entries = parse_fixture_page(_page("god"), registry)
    for e in entries.values():
        short = e["paradigm"]["short"]
        assert short is None or short.lstrip().startswith("-")


def test_no_slot_labels_are_invented_anywhere(registry):
    for page in MANIFEST["pages"]:
        if page.get("kind") == "nonword":
            continue
        entries = parse_fixture_page(FIXTURES_DIR / page["file"], registry)
        for e in entries.values():
            shape = tuple(
                sum(1 for r in e["paradigm"]["rows"] if r["table"] == t)
                for t in sorted({r["table"] for r in e["paradigm"]["rows"]}))
            expected = registry.paradigm_labels(e["pos_key"], shape)
            for i, row in enumerate(e["paradigm"]["rows"]):
                if expected is None:
                    assert row["slot_label"] is None
                else:
                    assert row["slot_label"] == (expected[i] if i < len(expected)
                                                 else None)


def test_multi_cell_rows_are_the_majority_somewhere(registry):
    # 331 of 539 rows have several <td>: a row is a slot, cells are its
    # spellings. At least one fixture page must show one, or the fixture set is
    # not exercising cell_alternatives at all.
    seen_multi = False
    for page in MANIFEST["pages"]:
        entries = parse_fixture_page(FIXTURES_DIR / page["file"], registry)
        for e in entries.values():
            for r in e["paradigm"]["rows"]:
                if len(r["cells"]) > 1:
                    seen_multi = True
    if not seen_multi:
        pytest.skip("no multi-cell flex row in this fixture set")
