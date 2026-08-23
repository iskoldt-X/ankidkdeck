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
