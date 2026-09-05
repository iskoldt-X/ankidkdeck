"""Stage 70: the byte-frozen identity, the rendered fields, and the package.

THE IDENTITY CONSTANTS ARE THE POINT OF THIS FILE. deck_id, model_id, the
Chinese 1745409457359 override, the 8 field names, sort_field_index = 7 and
guid_for(seed, lang) are the bytes that decide whether a re-import UPGRADES a
user's 4,442 existing cards or duplicates them. Round 2 found them verified only
by hand, against the released decks, in a scratch directory. They belong in the
suite.

The rendering half pins the defects round 2 measured on the real build: a
`Variants` field that could never be empty (so AFMT printed a bare "Variants:"
label and G-RATE's Variants row became vacuous), the glued homograph form on the
card face, and per-family double counting of shared idioms.
"""

import tempfile
from pathlib import Path

import pytest
from conftest import (make_entry, make_expression, make_sense, write_workspace)

from ankidkdeck.stages import s70_export as S70
from ankidkdeck.util import FatalError, read_json, write_json

genanki = pytest.importorskip("genanki",
                              reason="genanki is a runtime dependency")


# --------------------------------------------------------------- identity

def test_lang_hash_and_deck_and_model_ids_are_byte_frozen():
    # Measured against the four released v2.1 decks. Changing any of these
    # retires every existing card and creates a duplicate in its place.
    assert S70.lang_hash("German") == 132645467
    assert S70.deck_id("German") == 401080923
    assert S70.model_id("German") == 669516379
    assert S70.deck_id("English") == 0x10000000 + S70.lang_hash("English")
    assert S70.model_id("English") == 0x20000000 + S70.lang_hash("English")


def test_the_chinese_deck_id_is_the_legacy_override_not_the_formula():
    """The Chinese deck shipped before the adler32 scheme existed."""
    assert S70.deck_id("Chinese") == 1745409457359
    assert S70.deck_id("Chinese") != 0x10000000 + S70.lang_hash("Chinese")
    # ...but its MODEL id follows the formula, as it does on every deck.
    assert S70.model_id("Chinese") == 0x20000000 + S70.lang_hash("Chinese")


def test_field_names_and_sort_field_index_are_frozen():
    assert S70.FIELD_NAMES == ["QueryWord", "FrontSideSummary", "Content",
                               "Collocations", "Variants", "Derivatives",
                               "Etymology", "FrequencyRank"]
    assert len(S70.FIELD_NAMES) == 8
    # sfld is FrequencyRank, and SQLite integer affinity is what orders the deck.
    assert S70.SORT_FIELD_INDEX == 7
    assert S70.FIELD_NAMES[S70.SORT_FIELD_INDEX] == "FrequencyRank"


def test_model_name_is_the_one_already_in_every_users_collection():
    assert S70.model_name("German") == "Danish Frequency Deck V2.1 (German)"


def test_guid_is_genanki_guid_for_seed_and_lang():
    for seed in ("hus", "Er", "er", "uden for", "lån"):
        assert S70.guid_for(seed, "German") == genanki.guid_for(seed, "German")
        # the language is part of the hash: the same word is a different note
        # in a different deck
        assert S70.guid_for(seed, "German") != S70.guid_for(seed, "Chinese")


def test_the_copyright_year_comes_from_config_never_from_the_clock():
    meta = S70.deck_meta("German", 2026)
    assert "© 2026" in meta["copyright_html"]
    assert "© 2026" in meta["deck_description"]
    assert S70.deck_meta("German", 2025)["copyright_html"] != meta["copyright_html"]


def test_deck_names_are_the_released_ones():
    assert S70.deck_meta("German", 2026)["deck_name"] == \
        "Avadanskedavra: 3000 Words to Slay German"
    assert S70.deck_meta("Chinese", 2026)["deck_name"] == "丹麦语要你命3000词"


# ------------------------------------------------------------- build_note

def _fam(entries, **over):
    eids = list(entries)
    fam = {"family_id": eids[0], "anchor_entry_id": eids[0], "entry_ids": eids,
           "lemma": entries[eids[0]]["lemma"],
           "display_headword": entries[eids[0]]["lemma"],
           "super": entries[eids[0]].get("super"),
           "guid_seed": entries[eids[0]]["lemma"], "freq_rank": 1,
           "members": [{"word": entries[eids[0]]["lemma"],
                        "wiktionary_rank": 1, "relation": "anchor"}],
           "searchable_forms": [entries[eids[0]]["lemma"]]}
    fam.update(over)
    return fam


def _tr(defs=None, exprs=None, pos=None):
    return {"definitions": defs or {}, "expressions": exprs or {},
            "pos": {"sb.": "Substantiv"} if pos is None else pos}


def _note(cfg, entries, fam, tr, lang="German"):
    stats, report, misses = {}, {}, []
    note = S70.build_note(fam, entries, tr, S70.Media(cfg), misses, lang,
                          stats, report)
    return note, misses, stats


def test_build_note_numbers_senses_and_folds_from_the_third(cfg):
    e = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                   senses=[make_sense("2100000%d" % i, "hus betydning %d" % i)
                           for i in range(4)])
    entries = {e["entry_id"]: e}
    defs = {"11021722:2100000%d" % i: {"lemma": "Haus %d" % i,
                                       "gloss": "ein Haus %d" % i}
            for i in range(4)}
    note, misses, stats = _note(cfg, entries, _fam(entries), _tr(defs))
    content = note["fields"][S70.CONTENT_INDEX]
    assert misses == []
    # renumbered 1..N, DDO's own 1.a numbering flattened
    for n in (1, 2, 3, 4):
        assert "%d. hus betydning %d" % (n, n - 1) in content
    # the first two are visible, the rest folded behind v2.1's summary text
    head, _, folded = content.partition("<details>")
    assert "1. hus betydning 0" in head and "2. hus betydning 1" in head
    assert "3. hus betydning 2" in folded and "4. hus betydning 3" in folded
    assert "Show more definitions..." in folded
    # the translation is its own div, lemma bolded
    assert '<div class="translation"><b>Haus 0</b>: ein Haus 0</div>' in content
    assert stats["senses_rendered"] == 4


def test_build_note_renders_the_homograph_index_as_a_superscript(cfg):
    e = make_entry("11001153", "al", super_="2", pos_key="pron.",
                   pos_text="pronomen",
                   senses=[make_sense("21000010", "hele mængden")])
    entries = {e["entry_id"]: e}
    note, _, _ = _note(cfg, entries, _fam(entries),
                       _tr({"11001153:21000010": {"lemma": "all", "gloss": "alles"}},
                           pos={"pron.": "Pronomen"}))
    content = note["fields"][S70.CONTENT_INDEX]
    assert 'al<span class="super">2</span>' in content
    # the glued form the parser keeps for provenance never reaches the card
    assert "al2<" not in content


def test_variants_is_empty_when_there_is_no_paradigm_and_no_alias(cfg):
    """R3 M1 / R4 M5. AFMT wraps Variants in {{#Variants}}, and Mustache's
    section test is "the field is non-empty" -- so a field that always carried
    the hidden searchable-forms span printed `Variants:` followed by nothing on
    every such card (5 of 34 measured), and pinned the measured empty rate at
    0.00% against a 10.96% baseline so G-RATE could never fire for it again."""
    e = make_entry("11099001", "engroshandel", pos_key="sb.",
                   pos_text="substantiv",
                   senses=[make_sense("21990002", "handel i store partier")])
    entries = {e["entry_id"]: e}
    fam = _fam(entries)
    assert S70.variants_html(fam, [e]) == ""
    note, _, _ = _note(cfg, entries, fam,
                       _tr({"11099001:21990002": {"lemma": "Grosshandel",
                                                  "gloss": "Handel in Mengen"}}))
    assert note["fields"][S70.FIELD_NAMES.index("Variants")] == ""
    # ...and the span is at the END of Content, which is rendered unconditionally
    content = note["fields"][S70.CONTENT_INDEX]
    assert content.endswith('<span class="searchable-forms">engroshandel</span>')
    assert "definition-item" in S70.visible(content)


def test_variants_carries_the_paradigm_block_and_the_alt_spellings(cfg):
    e = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                   paradigm_rows=[{"table": 0, "row": 0,
                                   "cells": ["huset"],
                                   "slot_label": "definite singular"},
                                  {"table": 0, "row": 1,
                                   "cells": ["huse", "husene"],
                                   "slot_label": "indefinite plural"}],
                   alt_spellings=[{"form": "huus", "official": True},
                                  {"form": "hvs", "official": False}],
                   senses=[make_sense("21000020", "bygning")])
    entries = {e["entry_id"]: e}
    v = S70.variants_html(_fam(entries), [e])
    assert '<span class="paradigm-label">definite singular:</span> huset' in v
    # cells of ONE slot are joined with " / ", never presented as slots
    assert "huse / husene" in v
    # official alternatives are shown; a DEPRECATED spelling never is
    assert "huus" in v and "hvs" not in v
    assert "searchable-forms" not in v


def test_a_shared_idiom_is_counted_and_translated_once(cfg):
    """10 dannetids sit on two entry_ids; collocations_html() collapses them to
    one translation on purpose, so summing per-family distinct dannetids
    over-counted (measured 481 reported vs 477 real)."""
    x = make_expression("21990100", "kold krig", "spændt tilstand")
    a = make_entry("11000100", "krig", pos_key="sb.", pos_text="substantiv",
                   expressions=[x], senses=[make_sense("21990101", "kamp")])
    b = make_entry("11000101", "kold", pos_key="adj.", pos_text="adjektiv",
                   expressions=[x], senses=[make_sense("21990102", "lav temp")])
    entries = {e["entry_id"]: e for e in (a, b)}
    fam = _fam(entries)
    tr = _tr({"11000100:21990101": {"lemma": "Krieg", "gloss": "Kampf"},
              "11000101:21990102": {"lemma": "kalt", "gloss": "niedrige Temp"}},
             {"21990100": {"lemma": "Kalter Krieg", "gloss": "die Spannung"}},
             pos={"sb.": "Substantiv", "adj.": "Adjektiv"})
    note, misses, stats = _note(cfg, entries, fam, tr)
    colloc = note["fields"][S70.FIELD_NAMES.index("Collocations")]
    assert colloc.count("kold krig") == 1
    assert misses == []
    # build_all() turns the accumulated dannetid set into the reported number
    assert stats["_expression_dannetids"] == {"21990100"}


def test_a_missing_translation_is_a_coverage_miss_never_a_bare_definition(cfg):
    e = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                   senses=[make_sense("21000030", "bygning")])
    entries = {e["entry_id"]: e}
    note, misses, _ = _note(cfg, entries, _fam(entries),
                            _tr(pos={}))       # no pos table either
    # The SET is the contract; the order of the miss list is not.
    assert {m["kind"] for m in misses} == {"definition", "pos"}
    assert "bygning" in note["fields"][S70.CONTENT_INDEX]
    assert "translation" not in note["fields"][S70.CONTENT_INDEX]


def test_a_family_with_no_guid_seed_is_fatal(cfg):
    e = make_entry("11021722", "hus", pos_key="sb.",
                   senses=[make_sense("21000040", "bygning")])
    entries = {e["entry_id"]: e}
    fam = _fam(entries, guid_seed=None)
    with pytest.raises(FatalError):
        _note(cfg, entries, fam, _tr())


def test_a_non_nfc_guid_seed_is_fatal(cfg):
    import unicodedata
    e = make_entry("11021722", "lån", pos_key="sb.",
                   senses=[make_sense("21000041", "penge man skylder")])
    entries = {e["entry_id"]: e}
    fam = _fam(entries, guid_seed=unicodedata.normalize("NFD", "lån"))
    with pytest.raises(FatalError):
        _note(cfg, entries, fam, _tr())


# ------------------------------------------------------------ the package

def _relaxed_gates(cfg):
    write_json(cfg.registry_local / "gates.json",
               {"media_floor": 0, "note_count_range": [1, 99],
                "empty_rate_tolerance_pp": 100.0,
                "sitemap_shortfall_max_rate": 1.0})


def _tiny_workspace(cfg, lang="German"):
    from ankidkdeck.registry import Registry
    from ankidkdeck.stages import s30_merge
    _relaxed_gates(cfg)
    e = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                   forms=["huset"],
                   senses=[make_sense("21000050", "bygning man bor i")],
                   source_words=["hus"])
    write_workspace(cfg, {e["entry_id"]: e}, [(1, "hus")],
                    classification={"hus": {"members": [
                        {"entry_id": "11021722", "bucket": "exact_cs",
                         "demoted": False}], "xrefs": [], "rejected": [],
                        "resolved_by": "forward"}},
                    v2_querywords={"hus": 1})
    s30_merge.run(cfg, Registry(cfg))
    tdir = cfg.json_dir / "translations" / lang
    write_json(tdir / "definitions.json",
               {"11021722:21000050": {"lemma": "Haus", "gloss": "ein Gebäude",
                                      "src_sha": "x", "provenance": "t"}})
    write_json(tdir / "expressions.json", {})
    return Registry(cfg)


def test_write_package_and_read_package_round_trip(cfg, tmp_path):
    reg = _tiny_workspace(cfg)
    built = S70.build_all(cfg, reg, "German")
    out = tmp_path / "deck.apkg"
    info = S70.write_package(cfg, "German", built["notes"],
                             built["media"].sorted_paths(), out)
    assert info["notes"] == len(built["notes"]) == 1
    back = S70.read_package(out)
    guid, flds, tags = back["notes"][0]
    assert guid == built["notes"][0]["guid"]
    assert flds.split("\x1f") == built["notes"][0]["fields"]
    assert len(flds.split("\x1f")) == 8
    assert tags.strip() == ""       # the main package tags nothing


def test_the_written_package_carries_the_frozen_ids(cfg, tmp_path):
    import json
    import sqlite3
    import tempfile
    import zipfile
    reg = _tiny_workspace(cfg)
    built = S70.build_all(cfg, reg, "German")
    out = tmp_path / "deck.apkg"
    S70.write_package(cfg, "German", built["notes"], [], out)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            db = ("collection.anki21" if "collection.anki21" in names
                  else "collection.anki2")
            z.extract(db, tmp)
        con = sqlite3.connect(str(__import__("pathlib").Path(tmp) / db))
        models, decks = con.execute("SELECT models, decks FROM col").fetchone()
        con.close()
    models, decks = json.loads(models), json.loads(decks)
    model = list(models.values())[0]
    assert int(model["id"]) == S70.model_id("German")
    assert model["name"] == S70.model_name("German")
    assert [f["name"] for f in model["flds"]] == S70.FIELD_NAMES
    assert model["sortf"] == S70.SORT_FIELD_INDEX
    deck = [d for k, d in decks.items() if k != "1"][0]
    assert int(deck["id"]) == S70.deck_id("German")
    assert deck["name"] == S70.deck_meta("German", cfg.copyright_year)["deck_name"]


def test_media_paths_are_a_sorted_list_never_a_set(cfg):
    media = S70.Media(cfg)
    for name in ("z.mp3", "a.mp3", "m.mp3"):
        (cfg.audio_dir / name).write_bytes(b"id3")
        media.used[name] = cfg.audio_dir / name
    paths = media.sorted_paths()
    assert paths == sorted(paths)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["a.mp3", "m.mp3", "z.mp3"]


def test_the_registry_pos_table_needs_no_llm_call_and_no_pos_json(cfg):
    """R4 M4: nothing offline wrote translations/<lang>/pos.json, so G-COV
    blocked every export until a paid Gemini POS call had run -- there was no
    offline path to an .apkg at all."""
    reg = _tiny_workspace(cfg)
    assert not (cfg.json_dir / "translations" / "German" / "pos.json").exists()
    pos, note = S70.pos_table(cfg, reg, "German")
    assert pos["sb."] == "Substantiv"
    assert note["from_registry"] >= 20 and note["override_rows"] == 0
    for lang, expected in (("Chinese", "名词"), ("English", "noun"),
                           ("Spanish", "sustantivo")):
        assert S70.pos_table(cfg, reg, lang)[0]["sb."] == expected


def test_a_work_dir_pos_json_overrides_the_registry(cfg):
    reg = _tiny_workspace(cfg)
    write_json(cfg.json_dir / "translations" / "German" / "pos.json",
               {"sb.": "Hauptwort"})
    pos, note = S70.pos_table(cfg, reg, "German")
    assert pos["sb."] == "Hauptwort"
    assert note["override_rows"] == 1


def test_export_refuses_a_language_that_is_not_configured(cfg):
    reg = _tiny_workspace(cfg)
    with pytest.raises(FatalError) as exc:
        S70.run(cfg, reg, "Gemran")
    assert "Gemran" in str(exc.value)
    for lang in cfg.langs:
        assert lang in str(exc.value)


def test_a_full_offline_export_passes_every_gate(cfg, fixtures_env):
    """The whole point of the registry POS table: an .apkg with no LLM call at
    all. G-SEP needs the fixture set, which is why this one is fixture-gated."""
    reg = _tiny_workspace(cfg)
    report = S70.run(cfg, reg, "German")
    assert report["coverage_misses"] == 0
    assert (cfg.dist_dir / "DDO_Danish_Frequency_Deck_German.apkg").exists()
    gates = read_json(cfg.report_dir / "gates_report.json")
    assert gates["failed"] == []
    assert "G-SEP" in {r["id"] for r in gates["results"]}


# G-SEED reads the append-only registry and G-RANK reads words.json, so both
# verdicts are identical for every language: recorded ONCE, unscoped.
LANGUAGE_INDEPENDENT_GATES = {"G-SEED", "G-RANK"}


def _stage_70_rows(cfg, reg, lang):
    """Drive the export for its GATE REPORT only.

    run_gates() writes the report BEFORE it raises, so this works with or
    without the fixture set: G-SEP fails when the fixtures are absent, and that
    failure is orthogonal to how rows are scoped.
    """
    try:
        S70.run(cfg, reg, lang)
    except FatalError:
        pass
    return [r for r in read_json(cfg.report_dir / "gates_report.json")["results"]
            if r["stage"] == "70"]


def test_a_stage_70_gate_row_is_scoped_exactly_when_its_verdict_is(cfg):
    reg = _tiny_workspace(cfg)
    stage70 = _stage_70_rows(cfg, reg, "German")
    assert stage70
    scoped = {r["id"] for r in stage70 if r["extra"] == {"lang": "German"}}
    unscoped = {r["id"] for r in stage70 if not r["extra"]}
    assert unscoped == LANGUAGE_INDEPENDENT_GATES
    assert scoped and not (scoped & LANGUAGE_INDEPENDENT_GATES)


def test_a_second_language_adds_rows_only_for_the_per_language_gates(cfg):
    """Two languages give two G-COV rows -- and exactly ONE G-SEED row."""
    reg = _tiny_workspace(cfg)
    zh = cfg.json_dir / "translations" / "Chinese"
    write_json(zh / "definitions.json",
               {"11021722:21000050": {"lemma": "fangzi", "gloss": "bygning",
                                      "src_sha": "x", "provenance": "t"}})
    write_json(zh / "expressions.json", {})
    _stage_70_rows(cfg, reg, "German")
    rows = _stage_70_rows(cfg, reg, "Chinese")
    per_id = {}
    for r in rows:
        per_id.setdefault(r["id"], []).append(r["extra"])
    assert sorted(per_id["G-COV"], key=str) == [{"lang": "Chinese"},
                                                {"lang": "German"}]
    for gid in LANGUAGE_INDEPENDENT_GATES:
        assert per_id[gid] == [{}], (gid, per_id[gid])


def _g_rel_row(cfg, reg, lang):
    return next(r for r in _stage_70_rows(cfg, reg, lang) if r["id"] == "G-REL")


def test_the_guid_diff_tool_writes_one_report_per_language(cfg, tmp_path):
    """The tool wrote one reports/guid_diff.json whatever --lang said,
    so a release month that ships two languages ends with a file about whichever
    was diffed LAST -- and the exporter reads it as the other one's."""
    import importlib.util

    reg = _tiny_workspace(cfg)
    built = S70.build_all(cfg, reg, "German")
    released = tmp_path / "old.apkg"
    S70.write_package(cfg, "German", built["notes"], [], released)

    spec = importlib.util.spec_from_file_location(
        "guid_diff", Path(__file__).resolve().parents[1] / "tools"
        / "guid_diff.py")
    guid_diff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guid_diff)
    assert guid_diff.main(["--apkg", str(released), "--lang", "German",
                           "--work", str(cfg.work_dir)]) == 0
    assert not (cfg.report_dir / "guid_diff.json").exists()
    written = read_json(cfg.report_dir / "guid_diff.German.json")
    assert written["summary"]["language"] == "German"
    # --out still overrides the name; it is the only way to write a second
    # report for one language without clobbering the first.
    assert guid_diff.main(["--apkg", str(released), "--lang", "German",
                           "--work", str(cfg.work_dir),
                           "--out", str(tmp_path / "elsewhere.json")]) == 0
    assert (tmp_path / "elsewhere.json").exists()


def test_g_rel_reads_the_report_named_for_this_language(cfg):
    """The Russian month: reports/guid_diff.json still described the
    Chinese deck, so the Russian export failed with language_mismatch on a file
    that was never about Russian."""
    reg = _tiny_workspace(cfg)
    n_notes = len(S70.build_all(cfg, reg, "German")["notes"])
    write_json(cfg.report_dir / "guid_diff.German.json",
               {"summary": {"language": "German", "card_count": n_notes}})
    # a legacy report about ANOTHER language is not this language's report: it
    # is ignored, not failed. Failing on it is the defect.
    write_json(cfg.report_dir / "guid_diff.json",
               {"summary": {"language": "Chinese", "card_count": 4242}})
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is True
    assert row["detail"]["read_from"] == "guid_diff.German.json"
    assert row["detail"]["violations"] == {}


def test_g_rel_still_honours_a_legacy_report_about_this_language(cfg):
    """Backward compatibility, and no further: the unsuffixed name is read when
    the file itself says it describes THIS language."""
    reg = _tiny_workspace(cfg)
    n_notes = len(S70.build_all(cfg, reg, "German")["notes"])
    write_json(cfg.report_dir / "guid_diff.json",
               {"summary": {"language": "German", "card_count": n_notes}})
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is True and row["detail"]["read_from"] == "guid_diff.json"
    # ...and it FAILS on that name too when the count disagrees. Per-language
    # naming must not turn the legacy path into a soft one.
    write_json(cfg.report_dir / "guid_diff.json",
               {"summary": {"language": "German", "card_count": 4242}})
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is False
    assert row["detail"]["read_from"] == "guid_diff.json"
    assert "card_count_mismatch" in row["detail"]["violations"]


def test_one_reading_of_which_language_a_guid_diff_report_describes(cfg):
    """The exporter read `summary.language` and tools/retired_notes.py read the
    top-level `language` key, so a report written BEFORE the summary row existed
    was "not about German" to the exporter and "about German" to the companion
    builder -- one file, two answers. Now there is one function.
    """
    reg = _tiny_workspace(cfg)
    assert S70.guid_diff_language({"summary": {"language": "German"}}) \
        == "German"
    assert S70.guid_diff_language({"language": "German"}) == "German"
    # the summary wins, because that is the row G-REL asserts against
    assert S70.guid_diff_language(
        {"language": "Chinese", "summary": {"language": "German"}}) == "German"
    for junk in ({}, None, [], {"language": ""}, {"language": 7}):
        assert S70.guid_diff_language(junk) is None
    # a pre-summary-row legacy file is therefore READ, and fails honestly on
    # the missing row rather than being silently declared irrelevant
    write_json(cfg.report_dir / "guid_diff.json",
               {"language": "German", "retired": [], "kept": []})
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is False
    assert row["detail"]["read_from"] == "guid_diff.json"
    assert "no_summary_row" in row["detail"]["violations"]


def test_the_not_applicable_row_names_what_it_looked_at_and_skipped(cfg):
    """A gate row whose prose does not describe what happened is the species of
    defect this whole round exists to remove.

    `guid_diff_report` reaches "no report" from more than one state, and the
    obvious wording -- "there is no previous release to reconcile against" --
    is a claim about the disk that is FALSE when a file was found and rejected.
    """
    reg = _tiny_workspace(cfg)
    # a legacy report about another language: found, rejected, and SAID SO
    write_json(cfg.report_dir / "guid_diff.json",
               {"summary": {"language": "Chinese", "card_count": 4242}})
    detail = _g_rel_row(cfg, reg, "German")["detail"]
    assert detail["ignored"] == [{"file": "guid_diff.json",
                                  "why": "it describes 'Chinese', not "
                                         "'German'"}]
    assert "guid_diff.json" in detail["why"] and "Chinese" in detail["why"]
    assert "is not on disk" not in detail["why"]
    # a per-language file that EXISTS but is empty: the old message said the
    # file did not exist, to an operator staring at it
    write_json(cfg.report_dir / "guid_diff.German.json", {})
    detail = _g_rel_row(cfg, reg, "German")["detail"]
    assert [row["file"] for row in detail["ignored"]] == [
        "guid_diff.German.json", "guid_diff.json"]
    assert "empty" in detail["ignored"][0]["why"]
    # and with nothing on disk at all, the release-shaped explanation is the
    # honest one
    (cfg.report_dir / "guid_diff.German.json").unlink()
    (cfg.report_dir / "guid_diff.json").unlink()
    detail = _g_rel_row(cfg, reg, "German")["detail"]
    assert detail["ignored"] == []
    assert "is not on disk" in detail["why"]
    assert "a first release has none" in detail["why"]


def test_g_rel_still_fails_a_same_language_card_count_mismatch(cfg):
    """What the gate is FOR. Per-language naming must not soften it: the churn
    numbers in the release note have to be about the deck being written."""
    reg = _tiny_workspace(cfg)
    write_json(cfg.report_dir / "guid_diff.German.json",
               {"summary": {"language": "German", "card_count": 4242}})
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is False
    assert "card_count_mismatch" in row["detail"]["violations"]
    # ...and a report that is named for this language but describes another one
    # is a mislabelled file, not an absent one: trusted by NAME, verified by
    # CONTENT.
    write_json(cfg.report_dir / "guid_diff.German.json",
               {"summary": {"language": "Chinese", "card_count": 4242}})
    assert "language_mismatch" in _g_rel_row(cfg, reg, "German")[
        "detail"]["violations"]


def test_g_rel_records_a_not_applicable_row_when_no_report_exists(cfg):
    """Why a not-applicable ROW exists at all, rather than no row.

    Rows merge on (id, stage, extra) and are never pruned, so the ONLY thing
    that can clear a stale FAIL is a later row for the SAME scope. G-REL used to
    be appended only when a report was on disk, so a first release in a language
    -- which can never have one -- wrote nothing, and the Russian month's
    language_mismatch FAIL stayed in gates_report.json for good.
    """
    from ankidkdeck import gates as G

    reg = _tiny_workspace(cfg)
    row = _g_rel_row(cfg, reg, "German")
    assert row["ok"] is True and row["extra"] == {"lang": "German"}
    assert G.row_is_not_applicable(row)
    assert "is not on disk" in row["detail"]["why"]
    assert "NOT a verified pass" in row["detail"]["why"]

    # G-SEP fails here (no fixture set), so the assertions are about G-REL's own
    # row, not about an all-green report.
    def report():
        return read_json(cfg.report_dir / "gates_report.json")

    assert "G-REL[lang=German]" not in report()["failed_rows"]
    # the third verdict gets a derived top-level list of its own, alongside the
    # other four, so nothing has to walk `results` to find it
    assert report()["gate_rows_not_applicable"] == ["G-REL[lang=German]"]
    # the whole mechanism: a stale FAIL for this scope is REPLACED by the
    # not-applicable row on the next run, with no prune step anywhere
    write_json(cfg.report_dir / "guid_diff.German.json",
               {"summary": {"language": "German", "card_count": 4242}})
    assert _g_rel_row(cfg, reg, "German")["ok"] is False
    assert "G-REL[lang=German]" in report()["failed_rows"]
    assert report()["gate_rows_not_applicable"] == []
    (cfg.report_dir / "guid_diff.German.json").unlink()
    assert G.row_is_not_applicable(_g_rel_row(cfg, reg, "German"))
    assert "G-REL[lang=German]" not in report()["failed_rows"]
    assert report()["gate_rows_not_applicable"] == ["G-REL[lang=German]"]


def test_the_printed_gate_report_never_calls_a_non_check_a_pass(cfg, capsys):
    """A human reading N/A as PASS is the reader this project cannot afford: it
    is the difference between "the churn numbers were reconciled" and "there was
    nothing to reconcile them against"."""
    from ankidkdeck.cli import gates_report

    reg = _tiny_workspace(cfg)
    _stage_70_rows(cfg, reg, "German")
    gates_report(cfg)
    lines = capsys.readouterr().out.splitlines()
    rel = next(l for l in lines if l.split()[1:2] == ["G-REL[lang=German]"])
    assert rel.startswith("N/A ")
    assert any(l.startswith("PASS") for l in lines)
    assert any(l.startswith("FAIL") for l in lines)     # G-SEP, no fixtures
    summary = next(l for l in lines if "gate row(s) recorded" in l)
    assert "1 not applicable: G-REL[lang=German]" in summary


def _retired_notes_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retired_notes", Path(__file__).resolve().parents[1] / "tools"
        / "retired_notes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_retired_companion_prefers_the_per_language_report(cfg, tmp_path):
    """The exact disk state Fix 1 exists for -- a stale unsuffixed report plus a
    fresh per-language one -- read by the OTHER consumer of that file.

    The companion package is built from the `retired` list, so reading the wrong
    file tags the wrong notes: the operator's release note then says "search the
    tag, select all, delete" over another language's GUIDs.
    """
    retired_notes = _retired_notes_module()
    fresh = S70.guid_for("huse", "German")
    stale = S70.guid_for("hus", "German")
    write_json(cfg.report_dir / "guid_diff.German.json",
               {"language": "German",
                "retired": [{"guid": fresh, "query_word": "huse",
                             "merged_into": "hus"}]})
    write_json(cfg.report_dir / "guid_diff.json",
               {"language": "German",
                "retired": [{"guid": stale, "query_word": "STALE",
                             "merged_into": "hus"}]})
    out = tmp_path / "retired.apkg"
    assert retired_notes.main(["--lang", "German", "--work", str(cfg.work_dir),
                              "--out", str(out)]) == 0
    assert [g for g, _, _ in S70.read_package(out)["notes"]] == [fresh]

    # the legacy name is still the FALLBACK when the per-language file is absent
    (cfg.report_dir / "guid_diff.German.json").unlink()
    out2 = tmp_path / "retired_legacy.apkg"
    assert retired_notes.main(["--lang", "German", "--work", str(cfg.work_dir),
                              "--out", str(out2)]) == 0
    assert [g for g, _, _ in S70.read_package(out2)["notes"]] == [stale]

    # ...and a report for ANOTHER language is refused, naming the remedy rather
    # than the file to go and edit
    write_json(cfg.report_dir / "guid_diff.json",
               {"language": "Chinese", "retired": [{"guid": stale,
                                                    "query_word": "x",
                                                    "merged_into": "y"}]})
    with pytest.raises(SystemExit) as exc:
        retired_notes.main(["--lang", "German", "--work", str(cfg.work_dir),
                            "--out", str(tmp_path / "never.apkg")])
    assert "guid_diff.json describes 'Chinese', not 'German'" in str(exc.value)
    assert "tools/guid_diff.py" in str(exc.value)
    assert "guid_diff.German.json" in str(exc.value)


def test_the_retired_companion_fills_the_sort_field(cfg, tmp_path):
    """R4 m9: every retired note left sfld empty, so the 4,410 of them scattered
    through the browser's frequency ordering instead of collecting in one block
    -- in a package whose entire purpose is "search the tag, select all,
    delete"."""
    import importlib.util
    import sqlite3
    import zipfile

    from ankidkdeck.util import write_json
    spec = importlib.util.spec_from_file_location(
        "retired_notes", Path(__file__).resolve().parents[1] / "tools"
        / "retired_notes.py")
    retired_notes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retired_notes)

    guids = [S70.guid_for(w, "German") for w in ("huse", "husene")]
    write_json(cfg.report_dir / "guid_diff.json",
               {"language": "German",
                "retired": [{"guid": guids[0], "query_word": "huse",
                             "merged_into": "hus"},
                            {"guid": guids[1], "query_word": "husene",
                             "merged_into": None}]})
    out = tmp_path / "retired.apkg"
    rc = retired_notes.main(["--lang", "German", "--work", str(cfg.work_dir),
                             "--out", str(out)])
    assert rc == 0 and out.exists()

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            db = ("collection.anki21" if "collection.anki21" in names
                  else "collection.anki2")
            z.extract(db, tmp)
        con = sqlite3.connect(str(Path(tmp) / db))
        rows = con.execute("SELECT guid, flds, sfld FROM notes").fetchall()
        con.close()
    assert len(rows) == 2
    idx = S70.FIELD_NAMES.index("FrequencyRank")
    for guid, flds, sfld in rows:
        assert guid in guids
        fields = flds.split("\x1f")
        assert fields[idx] == retired_notes.RETIRED_RANK == "0"
        # sfld is the sort column Anki derives from that field; it must not be
        # blank, and it must be the SAME on every retired note so they sort
        # together, ahead of every live rank (1..N).
        assert str(sfld).strip() != ""
        assert str(sfld) == "0"
    assert len({str(r[2]) for r in rows}) == 1


def test_g_det_compares_two_independently_derived_media_lists(cfg, fixtures_env):
    """R3 m1 / R4 m2: both packages used to be written from build A's media
    list, so package_media_equal compared a list against itself."""
    reg = _tiny_workspace(cfg)
    report = S70.run(cfg, reg, "German", check_determinism=True)
    assert report["determinism_checked"] is True
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    det = [r for r in rows if r["id"] == "G-DET"][0]
    assert det["ok"] is True
    assert det["detail"]["package_media_equal"] is True
    assert det["detail"]["notes_equal"] is True
