"""The registry is the users' study progress. It is append-only, or it is nothing.

card_keys.json maps family_id -> guid_seed, and the GUID is
genanki.guid_for(guid_seed, lang) -- a hash of the bytes. Changing one seed
silently retires one card and creates another, and 55 (headword, wordlist word)
pairs differ only by case with ('Er','er') at rank 1, so this is not a
hypothetical failure mode.
"""

import re

import pytest

from ankidkdeck.registry import FILES, Registry
from ankidkdeck.util import FatalError, read_json, write_json

# PEP 440 pre-release / dev suffixes: a1, b2, rc1, .dev0
PRE_RELEASE = re.compile(r"(a|b|rc)\d+$|\.dev\d+$")


def test_all_registry_files_ship_as_package_data(registry):
    for name in FILES:
        assert name.removesuffix(".json") in registry.data


def test_defaults_are_the_measured_ones(registry):
    assert "symbol" in registry.demoted_pos_keys
    assert "førsteled" in registry.demoted_pos_keys
    gates = registry.gates
    assert gates["note_count_range"] == [2800, 3100]
    assert gates["empty_rate_baseline_pct"]["Collocations"] == pytest.approx(33.86)


def test_freeze_adds_new_families_and_writes_the_overlay(cfg, registry):
    path = cfg.registry_local / "card_keys.json"
    counts = registry.freeze_card_keys(
        {"11021722": {"guid_seed": "hus", "lemma_at_freeze": "hus",
                      "since": "3.0.0a0", "carried_from_v2": True}}, path)
    assert counts["added"] == 1
    assert read_json(path)["11021722"]["guid_seed"] == "hus"
    assert registry.card_keys["11021722"]["guid_seed"] == "hus"


def test_freeze_is_idempotent(cfg, registry):
    path = cfg.registry_local / "card_keys.json"
    row = {"11021722": {"guid_seed": "hus", "lemma_at_freeze": "hus",
                        "since": "3.0.0a0", "carried_from_v2": True}}
    registry.freeze_card_keys(row, path)
    counts = registry.freeze_card_keys(row, path)
    assert counts["added"] == 0
    assert counts["total"] == 1


def test_freeze_refuses_to_change_an_existing_seed(cfg, registry):
    path = cfg.registry_local / "card_keys.json"
    registry.freeze_card_keys(
        {"11021722": {"guid_seed": "hus", "lemma_at_freeze": "hus",
                      "since": "3.0.0a0", "carried_from_v2": True}}, path)
    with pytest.raises(FatalError) as exc:
        registry.freeze_card_keys(
            {"11021722": {"guid_seed": "Hus", "lemma_at_freeze": "hus",
                          "since": "3.0.0a0", "carried_from_v2": True}}, path)
    assert "guid_seed" in str(exc.value)


def test_local_overlay_merges_over_the_packaged_dict(cfg):
    # A run may lower a gate baseline for a partial build without editing the
    # checked-in registry.
    write_json(cfg.registry_local / "gates.json",
               {"media_floor": 0, "note_count_range": [1, 10]})
    reg = Registry(cfg)
    assert reg.gates["media_floor"] == 0
    assert reg.gates["note_count_range"] == [1, 10]
    # untouched keys still come from the package default
    assert "empty_rate_baseline_pct" in reg.gates


def test_local_overlay_appends_to_the_packaged_list(cfg):
    write_json(cfg.registry_local / "demoted_pos_keys.json", ["hilarious"])
    reg = Registry(cfg)
    assert "hilarious" in reg.demoted_pos_keys
    assert "symbol" in reg.demoted_pos_keys      # default preserved


def test_no_frozen_row_claims_a_pre_release_version(cfg, registry):
    """`since` is written once and never rewritten -- correctly, the file is
    append-only. Stamping it from __version__ ("3.0.0a0") would therefore brand
    every v3.0 family with a dev pre-release forever, in the file whose
    human-reviewed diff IS the release artifact. It comes from
    cfg.registry_version instead.
    """
    assert cfg.registry_version == "3.0"
    assert not PRE_RELEASE.search(cfg.registry_version)

    from conftest import make_entry, make_sense, write_workspace
    from ankidkdeck.stages.s30_merge import run as merge_run
    e = make_entry("11000400", "hus", pos_key="sb.",
                   senses=[make_sense("21000050", "bygning")],
                   source_words=["hus"])
    write_workspace(cfg, {"11000400": e}, [(1, "hus")],
                    classification={"hus": {"members": [
                        {"entry_id": "11000400", "bucket": "exact_cs",
                         "demoted": False}],
                        "xrefs": [], "rejected": [],
                        "resolved_by": "forward"}},
                    v2_querywords={"hus": 1})
    merge_run(cfg, registry)
    rows = read_json(cfg.registry_local / "card_keys.json")
    assert rows["11000400"]["since"] == "3.0"
    for fid, row in rows.items():
        assert not PRE_RELEASE.search(str(row.get("since", ""))), (fid, row)


def test_the_shipped_registry_carries_no_pre_release_since(registry):
    for fid, row in registry.card_keys.items():
        assert not PRE_RELEASE.search(str(row.get("since", ""))), (fid, row)


def test_paradigm_labels_only_for_recognised_shapes(registry):
    assert registry.paradigm_labels("sb.", (3,)) == [
        "definite singular", "indefinite plural", "definite plural"]
    # An unrecognised shape returns None: a label is never invented.
    assert registry.paradigm_labels("sb.", (7,)) is None
    assert registry.paradigm_labels("adv.", (3,)) is None


def test_paradigm_slot_labels_stay_english_on_every_deck(registry):
    """v2.1 precedent, re-affirmed in round 2: EVERY UI string the exporter
    emits is English on every deck ("Fixed Expressions", "Variants:",
    "Derivatives:", "Rank:", "Show more definitions..."), so translating this
    one table would make the paradigm block the single inconsistent element
    rather than fixing an inconsistency. Per-language slot labels are out of
    scope for v3.0."""
    table = registry.data["paradigm_slots"]
    note = " ".join(table["_note"])
    assert "STAY ENGLISH" in note
    for key, labels in table.items():
        if key.startswith("_"):
            continue
        for label in labels:
            assert label.isascii(), (key, label)


def test_a_nested_overlay_dict_is_merged_not_replaced(cfg):
    """R4 n6: `{**base, **extra}` meant a work/registry/gates.json touching
    empty_rate_baseline_pct AT ALL replaced the whole 5-field baseline."""
    write_json(cfg.registry_local / "gates.json",
               {"empty_rate_baseline_pct": {"Content": 5.0}})
    reg = Registry(cfg)
    base = reg.gates["empty_rate_baseline_pct"]
    assert base["Content"] == 5.0
    assert base["Collocations"] == pytest.approx(33.86)   # NOT dropped
    assert len(base) == 5


def test_a_nested_overlay_list_replaces_rather_than_appending(cfg):
    write_json(cfg.registry_local / "gates.json", {"note_count_range": [1, 10]})
    assert Registry(cfg).gates["note_count_range"] == [1, 10]


# ------------------------------------------------------- the POS label table

def test_pos_translations_cover_every_known_pos_key(registry):
    """The pos_key vocabulary is OPEN -- `formelt subjekt` was in nobody's list
    -- so a key nobody has translated must fail G-COV loudly. That only works if
    the table covers everything the parser already knows about."""
    from ankidkdeck.stages.s20_parse import KNOWN_POS_KEYS
    table = registry.pos_translations
    assert set(table) == {"Chinese", "English", "German", "Spanish"}
    for lang, rows in table.items():
        missing = sorted(KNOWN_POS_KEYS - set(rows))
        assert missing == [], (lang, missing)
        extra = sorted(set(rows) - KNOWN_POS_KEYS)
        assert extra == [], (lang, extra)
        assert all(v and v.strip() for v in rows.values()), lang


def test_the_pos_terms_are_the_right_language(registry):
    t = registry.pos_translations
    assert t["English"]["sb."] == "noun"
    assert t["Chinese"]["sb."] == "名词"
    assert t["German"]["sb."] == "Substantiv"          # German capitalises nouns
    assert t["Spanish"]["sb."] == "sustantivo"
    # no pinyin on the Chinese deck (the 04 prompt says so too)
    assert all(v.isascii() is False or v.isascii()
               for v in t["Chinese"].values())
    assert not any(c.isascii() and c.isalpha()
                   for v in t["Chinese"].values() for c in v)
    # German nouns are capitalised; the two adjective-phrase rows are not nouns
    assert t["German"]["udråbsord"] == "Interjektion"
    assert t["German"]["formelt subjekt"] == "formales Subjekt"


def test_pos_for_returns_a_copy_per_language(registry):
    a = registry.pos_for("German")
    a["sb."] = "mutated"
    assert registry.pos_for("German")["sb."] == "Substantiv"
    assert registry.pos_for("Klingon") == {}


def test_a_work_dir_pos_overlay_merges_per_language(cfg):
    write_json(cfg.registry_local / "pos_translations.json",
               {"German": {"sb.": "Hauptwort"}})
    reg = Registry(cfg)
    assert reg.pos_for("German")["sb."] == "Hauptwort"
    assert reg.pos_for("German")["vb."] == "Verb"        # the other 21 survive
    assert reg.pos_for("Chinese")["sb."] == "名词"


# ------------------------------------------------------------------- config

def test_a_toml_key_naming_a_read_only_property_is_a_fatal_error(tmp_path):
    """R3 m6: `hasattr` is True for every derived property, so `json_dir = "x"`
    raised a bare AttributeError out of main(), which only catches FatalError."""
    from ankidkdeck.config import load_config
    p = tmp_path / "ankidkdeck.toml"
    p.write_text('json_dir = "/tmp/nope"\n', encoding="utf-8")
    with pytest.raises(FatalError) as exc:
        load_config(p)
    assert "json_dir" in str(exc.value)
    assert "read-only property" in str(exc.value)


def test_the_likely_typo_names_the_settable_field(tmp_path):
    from ankidkdeck.config import load_config
    p = tmp_path / "ankidkdeck.toml"
    p.write_text('expressions_model = "gemini-2.5-flash"\n', encoding="utf-8")
    with pytest.raises(FatalError) as exc:
        load_config(p)
    assert "gemini_model_expressions" in str(exc.value)


def test_an_unknown_toml_key_lists_the_accepted_ones(tmp_path):
    from ankidkdeck.config import load_config
    p = tmp_path / "ankidkdeck.toml"
    p.write_text('wordlist_sha = "abc"\n', encoding="utf-8")
    with pytest.raises(FatalError) as exc:
        load_config(p)
    assert "wordlist_sha256" in str(exc.value)     # in the accepted-keys list


def test_a_settable_field_still_loads(tmp_path):
    from ankidkdeck.config import load_config, settable_fields
    p = tmp_path / "ankidkdeck.toml"
    p.write_text('langs = ["German"]\ncopyright_year = 2027\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.langs == ["German"] and cfg.copyright_year == 2027
    assert "gemini_model_expressions" in settable_fields()
    assert "json_dir" not in settable_fields()


def test_freeze_reports_a_stale_seed_without_touching_the_row(cfg, registry):
    """`proposed_seeds` is what today's data WOULD choose for every family,
    already-frozen ones included. Without it this method never saw them: stage
    30 filtered existing family_ids out before calling, so the guard above was
    dead code and the append-only rule had no voice -- which is how 22 families
    locked in the less frequent of two spellings silently."""
    path = cfg.registry_local / "card_keys.json"
    registry.freeze_card_keys(
        {"11021722": {"guid_seed": "huse", "lemma_at_freeze": "hus",
                      "since": "3.0", "carried_from_v2": True}}, path)
    counts = registry.freeze_card_keys({}, path,
                                       proposed_seeds={"11021722": "hus"})
    assert counts["added"] == 0
    assert counts["stale_seeds"] == [{"family_id": "11021722",
                                      "frozen_seed": "huse",
                                      "seed_today": "hus",
                                      "frozen_since": "3.0",
                                      "lemma_at_freeze": "hus"}]
    # REPORTED, not rewritten
    assert read_json(path)["11021722"]["guid_seed"] == "huse"


def test_freeze_reports_no_stale_seed_when_the_choice_still_agrees(cfg, registry):
    path = cfg.registry_local / "card_keys.json"
    registry.freeze_card_keys(
        {"11021722": {"guid_seed": "hus", "lemma_at_freeze": "hus",
                      "since": "3.0", "carried_from_v2": True}}, path)
    counts = registry.freeze_card_keys({}, path,
                                       proposed_seeds={"11021722": "hus"})
    assert counts["stale_seeds"] == []
    assert counts["unchanged"] == 1


def test_the_alias_merge_quarantine_ships_and_holds_the_three_frozen_pairs(
        registry):
    """These three are the pairs where BOTH sides own a DDO article AND an
    already-frozen card_keys row, so merging them retires a frozen guid_seed.
    The classifier keeps admitting them; the merge keeps two heads until the
    owner's single re-freeze."""
    pairs = {tuple(p) for p in registry.alias_merge_pending}
    assert pairs == {("ok", "o.k."), ("næ", "næh"), ("check", "tjek")}
    # every quarantined pair must actually BE an alias pair, or it is a no-op
    assert pairs <= registry.alias_pairs
