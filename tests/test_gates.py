"""A gate that cannot fail is not a gate.

Every gate body gets one passing input and one deliberately broken input. The
broken inputs are the historical defects, not invented ones: a duplicate
FrequencyRank, a carried GUID seed that is not a v2 QueryWord, a form claimed by
two families, an affix-anchored family, a translation drop with no reason code,
a media list that is a set, a bare definition with no translation.
"""

import pytest
from conftest import make_entry, make_sense

from ankidkdeck import gates as G
from ankidkdeck.stages import s42_translate as S42
from ankidkdeck.stages import s50_priority as S50
from ankidkdeck.stages import s70_export as S70


# ---------------------------------------------------------------- stage gates

def test_dense_unique_ranks():
    assert G.dense_unique_ranks([1, 2, 3], 3)[0] is True
    assert G.dense_unique_ranks([1, 2, 2], 3)[0] is False      # duplicate
    assert G.dense_unique_ranks([1, 2, 4], 3)[0] is False      # hole
    ok, detail = G.dense_unique_ranks([1, 2, 2], 3)
    assert detail["duplicates"] == [2]


def test_registry_seed_bytes():
    good = {"1": {"guid_seed": "hus", "carried_from_v2": True}}
    assert G.registry_seed_bytes(good, {"hus": 1})[0] is True
    # the 55 case-only pairs: a carried seed must be BYTE-equal, not
    # casefold-equal, to the word that shipped
    bad_case = {"1": {"guid_seed": "Hus", "carried_from_v2": True}}
    assert G.registry_seed_bytes(bad_case, {"hus": 1})[0] is False
    import unicodedata
    bad_nfc = {"1": {"guid_seed": unicodedata.normalize("NFD", "lån"),
                     "carried_from_v2": False}}
    assert G.registry_seed_bytes(bad_nfc, {})[0] is False
    dupes = {"1": {"guid_seed": "hus", "carried_from_v2": False},
             "2": {"guid_seed": "hus", "carried_from_v2": False}}
    assert G.registry_seed_bytes(dupes, {})[0] is False
    missing_row = {}
    assert G.registry_seed_bytes(missing_row, {}, family_ids=["9"])[0] is False


def test_unique_assignment():
    fams = {"a": {"members": [{"word": "mit"}]},
            "b": {"members": [{"word": "dit"}]}}
    assert G.unique_assignment({"mit": {"family_id": "a"},
                                "dit": {"family_id": "b"}}, fams)[0] is True
    # /mit returns jeg AND min: counted twice, FrequencyRank stops being unique
    fams_bad = {"a": {"members": [{"word": "mit"}]},
                "b": {"members": [{"word": "mit"}]}}
    ok, detail = G.unique_assignment({"mit": {"family_id": "a"}}, fams_bad)
    assert ok is False
    assert "mit" in detail["words_in_more_than_one_family"]


def test_anchors_not_demoted_or_affix():
    entries = {
        "1": make_entry("1", "kvinde", pos_key="sb."),
        "2": make_entry("2", "-kvinde", pos_key="sidsteled"),
        "3": make_entry("3", "cm", pos_key="fork."),
        "4": make_entry("4", "meter", pos_key="sb.",
                        senses=[make_sense("21", "laengdeenhed")]),
    }
    demoted = {"fork.", "symbol", "sidsteled"}
    good = {"f1": {"anchor_entry_id": "1", "entry_ids": ["1"]}}
    assert G.anchors_not_demoted_or_affix(good, entries, demoted)[0] is True
    affix = {"f2": {"anchor_entry_id": "2", "entry_ids": ["2"]}}
    assert G.anchors_not_demoted_or_affix(affix, entries, demoted)[0] is False
    # a demoted anchor WITH a real alternative in the family is a failure
    demoted_anchor = {"f3": {"anchor_entry_id": "3", "entry_ids": ["3", "4"]}}
    assert G.anchors_not_demoted_or_affix(demoted_anchor, entries,
                                          demoted)[0] is False
    # a family where everything is demoted is reported, not fatal (owner ruling)
    all_demoted = {"f4": {"anchor_entry_id": "3", "entry_ids": ["3"]}}
    ok, detail = G.anchors_not_demoted_or_affix(all_demoted, entries, demoted)
    assert ok is True and detail["all_demoted_families"] == 1


def test_bind_accounting():
    good = {"German": {"n_legacy": 10, "n_bound": 8, "n_dropped": 2,
                       "n_unexplained": 0,
                       "reasons": {"sense_text_changed": 2}}}
    assert G.bind_accounting(good)[0] is True
    lost = {"German": {"n_legacy": 10, "n_bound": 8, "n_dropped": 1,
                       "n_unexplained": 0, "reasons": {"sense_text_changed": 1}}}
    assert G.bind_accounting(lost)[0] is False
    invented_reason = {"German": {"n_legacy": 1, "n_bound": 0, "n_dropped": 1,
                                  "n_unexplained": 0,
                                  "reasons": {"because_i_said_so": 1}}}
    assert G.bind_accounting(invented_reason)[0] is False


def test_tie_break_resolved():
    assert G.tie_break_resolved({"En": {"unresolved_conflicts": 0,
                                        "discard_file_written": True}})[0] is True
    assert G.tie_break_resolved({"En": {"unresolved_conflicts": 3,
                                        "discard_file_written": True}})[0] is False
    assert G.tie_break_resolved({"En": {"unresolved_conflicts": 0,
                                        "discard_file_written": False}})[0] is False


def test_sitemap_shortfall():
    assert G.sitemap_shortfall([], 100, 0.01)[0] is True
    assert G.sitemap_shortfall([{"lemma": "vinge"}], 100, 0.01)[0] is True
    assert G.sitemap_shortfall([{"lemma": "vinge"}, {"lemma": "vove"}],
                               100, 0.01)[0] is False


def test_run_gates_records_everything_before_it_raises(cfg):
    from ankidkdeck.util import FatalError, read_json
    gates = [G.Gate("G-OK", "passes", lambda: (True, {"n": 1}), stage="t"),
             G.Gate("G-BAD", "fails", lambda: (False, {"why": "on purpose"}),
                    stage="t"),
             G.Gate("G-ALSO-BAD", "fails too", lambda: (False, {}), stage="t")]
    with pytest.raises(FatalError):
        G.run_gates(gates, cfg, stage="t")
    report = read_json(cfg.report_dir / "gates_report.json")
    assert {r["id"] for r in report["results"]} == {"G-OK", "G-BAD", "G-ALSO-BAD"}
    assert report["failed"] == ["G-ALSO-BAD", "G-BAD"]


# --------------------------------------------------------------- export gates

def _note(guid="g1", content="1. hus", fields=None, rank="1"):
    f = fields or ["hus", "front", content, "", "", "", "", rank]
    return {"family_id": "11021722", "guid_seed": "hus", "guid": guid,
            "fields": f, "freq_rank": int(rank)}


def test_empty_content_gate():
    assert S70.empty_content_gate([_note()])[0] is True
    assert S70.empty_content_gate([_note(content="")])[0] is False


def test_empty_rate_gate_uses_rates_not_counts():
    baseline = {"Collocations": 33.86}
    full = ["hus", "front", "content", "kold krig", "v", "d", "e", "1"]
    no_colloc = ["hav", "front", "content", "", "v", "d", "e", "2"]
    # 1 of 2 notes has an empty Collocations field -> 50% > 33.86 + 1
    ok, detail = S70.empty_rate_gate([_note(fields=full), _note(fields=no_colloc)],
                                     baseline, 1.0)
    assert ok is False
    assert detail["fields"]["Collocations"]["rate_pct"] == 50.0
    # an IMPROVEMENT always passes; and a field with no baseline must be full
    assert S70.empty_rate_gate([_note(fields=full)], baseline, 1.0)[0] is True


def test_coverage_gate():
    assert S70.coverage_gate([], {"notes": 1})[0] is True
    misses = [{"kind": "definition", "key": "11021722:21000001"}]
    ok, detail = S70.coverage_gate(misses, {"notes": 1})
    assert ok is False and detail["missing_by_kind"]["definition"] == 1


def test_guid_gate():
    assert S70.guid_gate([_note(guid="a"), _note(guid="b")])[0] is True
    assert S70.guid_gate([_note(guid="a"), _note(guid="a")])[0] is False


def test_note_count_gate():
    assert S70.note_count_gate([_note()] * 2900, [2800, 3100])[0] is True
    assert S70.note_count_gate([_note()] * 12, [2800, 3100])[0] is False


def test_media_gate(cfg):
    from ankidkdeck.util import write_json
    (cfg.audio_dir / "11021722_1.mp3").write_bytes(b"id3")
    write_json(cfg.audio_dir / "manifest.json",
               {"https://static.ordnet.dk/mp3/11021/11021722_1.mp3":
                {"file": "11021722_1.mp3", "sha256": "x", "bytes": 3,
                 "entry_id": "11021722", "slot_n": 1}})
    media = S70.Media(cfg)
    tag = media.sound_tag("https://static.ordnet.dk/mp3/11021/11021722_1.mp3",
                          "11021722", 1)
    assert tag == "[sound:11021722_1.mp3]"
    assert S70.media_gate(media, 1, ["11021722_1.mp3"])[0] is True
    # a declared file that is not on disk fails, and no dead tag is emitted
    assert media.sound_tag("https://static.ordnet.dk/mp3/11021/11021722_9.mp3",
                           "11021722", 9) == ""
    assert S70.media_gate(media, 1, ["11021722_1.mp3"])[0] is False
    # the floor catches a build that silently lost the audio cache
    media2 = S70.Media(cfg)
    assert S70.media_gate(media2, 4629, [])[0] is False


def test_media_list_must_be_sorted(cfg):
    media = S70.Media(cfg)
    for name in ("b.mp3", "a.mp3"):
        (cfg.audio_dir / name).write_bytes(b"x")
        media.used[name] = cfg.audio_dir / name
    paths = media.sorted_paths()
    assert paths == sorted(paths)      # a set here is what broke reproducibility


def test_determinism_gate():
    a, b = [_note()], [_note()]
    assert S70.determinism_gate(a, b)[0] is True
    drifted = [_note(content="1. hus (rebuilt 2027)")]
    ok, detail = S70.determinism_gate(a, drifted)
    assert ok is False and detail["first_difference"]["index"] == 0
    assert S70.determinism_gate(a, b, {"notes": [1], "media": ["a"]},
                                {"notes": [1], "media": ["b"]})[0] is False


def test_sound_names_are_extracted_from_every_field():
    n = _note(fields=["hus", "[sound:a.mp3] and [sound:b.mp3]", "c", "", "", "",
                      "", "1"])
    assert S70.sound_names_of([n]) == ["a.mp3", "b.mp3"]


# ------------------------------------------------- stage 42 and 50 local gates

def test_orphans_gate():
    clean = {"German": {"archived": {"definitions": 1, "expressions": 0},
                        "rows_live": {"definitions": 5, "expressions": 2},
                        "orphans_remaining": []}}
    assert S42.orphans_gate(clean)[0] is True
    dirty = dict(clean)
    dirty["German"] = {**clean["German"], "orphans_remaining": ["1:2"]}
    assert S42.orphans_gate(dirty)[0] is False


def test_order_gate():
    assert S50._order_gate([{"family_id": "1", "ok": True}])[0] is True
    assert S50._order_gate([{"family_id": "1", "ok": False}])[0] is False


def test_stable_merge_keeps_the_tone_setter():
    # the lowest-rank source sets the tone; a newcomer is placed
    # deterministically, never by dict order
    assert S50.stable_merge(["a", "b", "c"], ["b", "x", "c"]) == ["a", "b", "x", "c"]
    assert S50.stable_merge([], ["a", "b"]) == ["a", "b"]
    assert S50.stable_merge(["a"], ["z"]) == ["a", "z"]
    assert S50.stable_merge(["a", "b"], ["b", "a"]) == ["a", "b"]


def test_dedupe_keep_first_collapses_the_re_downloads():
    # rom__e/rom__2 and friends: 4 of the 615 priority keys collapse
    assert S50.dedupe_keep_first(["1", "2", "1", "3"]) == ["1", "2", "3"]


def test_anchor_first_survives_ranking():
    assert S50.anchor_first("11000011", ["11000010", "11000011"]) == [
        "11000011", "11000010"]
