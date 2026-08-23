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
    # ...but the POPULATION is baselined, so the softening cannot grow silently
    ok, detail = G.anchors_not_demoted_or_affix(all_demoted, entries, demoted,
                                                all_demoted_max=0)
    assert ok is False and detail["all_demoted_over_baseline"] is True
    assert G.anchors_not_demoted_or_affix(all_demoted, entries, demoted,
                                          all_demoted_max=1)[0] is True


def test_no_affix_members():
    """The affix check used to be a bare AssertionError AFTER the outputs were
    written, so G-AFFIX never appeared in gates_report.json."""
    entries = {"1": make_entry("1", "kvinde", pos_key="sb."),
               "2": make_entry("2", "-kvinde", pos_key="sidsteled"),
               "3": make_entry("3", "for-", pos_key="sb.")}
    clean = {"kvinder": {"members": [{"entry_id": "1", "bucket": "form"}]}}
    assert G.no_affix_members(clean, entries)[0] is True
    # rejected is fine; a MEMBER is not
    with_reject = {"kvinder": {"members": [{"entry_id": "1", "bucket": "form"}],
                               "rejected": [{"entry_id": "2"}]}}
    assert G.no_affix_members(with_reject, entries)[0] is True
    by_pos = {"kvinder": {"members": [{"entry_id": "2", "bucket": "form"}]}}
    ok, detail = G.no_affix_members(by_pos, entries)
    assert ok is False and detail["sample"][0]["lemma"] == "-kvinde"
    # detected by SHAPE too, not only by data-pos-key
    by_shape = {"foran": {"members": [{"entry_id": "3", "bucket": "form"}]}}
    assert G.no_affix_members(by_shape, entries)[0] is False


def test_registry_family_ids():
    """`11028611#kunne` put a wordlist form into the registry whose whole point
    is to make wordlist changes GUID-neutral."""
    assert G.registry_family_ids({"11028611": {}, "11021722": {}})[0] is True
    ok, detail = G.registry_family_ids({"11028611#kunne": {}})
    assert ok is False and detail["sample"] == ["11028611#kunne"]
    assert G.registry_family_ids({"11028611/deadbeef": {}})[0] is False
    assert G.registry_family_ids({"12345": {}})[0] is False        # too short
    assert G.registry_family_ids({})[0] is True


def test_tie_break_gate_can_actually_fail_and_has_a_baseline():
    """With the filename in the comparison key the count was structurally 0 --
    filenames are unique -- so G-TIE proved a tautology. Without it the number
    means "the written rule fell through to byte order", which is reproducible
    but worth watching, so it is baselined rather than forbidden."""
    two = {"En": {"unresolved_conflicts": 2, "discard_file_written": True}}
    assert G.tie_break_resolved(two)[0] is False
    assert G.tie_break_resolved(two, byte_order_max=2)[0] is True
    assert G.tie_break_resolved(two, byte_order_max=1)[0] is False


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


# --------------------------------------------- stage 70 rendering regressions

def test_the_homograph_index_is_rendered_as_a_superscript():
    """`al2` / `udenfor1` / `i5` is not what DDO shows (a superscript) and it
    also became an Anki search token. lemma and super are separate fields; the
    markup is assembled at render time."""
    e = make_entry("11001153", "al", super_="2", pos_key="pron.")
    assert S70.headword_html(e) == 'al<span class="super">2</span>'
    plain = make_entry("11021722", "hus", pos_key="sb.")
    assert S70.headword_html(plain) == "hus"
    # and the glued form never reaches the card face
    assert "al2" not in S70.headword_html(e)


def test_ipa_dedup_keeps_rows_that_have_audio_but_no_ipa(cfg):
    """21 udtale rows (all multiword compounds -- `alle sammen`,
    `dag til dag-service`) carry ipa: "" with a valid audio_url. Deduping on the
    empty IPA string collapsed them into one row and lost every [sound:] tag but
    the first."""
    from ankidkdeck.util import write_json
    urls, manifest = [], {}
    for slot in (1, 2):
        name = "11000900_%d.mp3" % slot
        (cfg.audio_dir / name).write_bytes(b"id3")
        url = "https://static.ordnet.dk/mp3/11000/%s" % name
        urls.append((url, slot))
        manifest[url] = {"file": name}
    write_json(cfg.audio_dir / "manifest.json", manifest)
    media = S70.Media(cfg)
    e = make_entry("11000900", "alle sammen", pos_key="sb.",
                   udtale=[{"ipa": "", "label": None, "audio_url": u,
                            "slot_n": s} for u, s in urls])
    html = S70.pron_html_for_group([e], media)
    assert html.count("[sound:") == 2
    assert "[]" not in html                    # no empty IPA brackets rendered
    # a row with neither IPA nor audio is still nothing to render
    empty = make_entry("11000901", "x", pos_key="sb.",
                       udtale=[{"ipa": "", "label": None, "audio_url": None,
                                "slot_n": None}])
    assert S70.pron_html_for_group([empty], S70.Media(cfg)) == ""
    # and a genuine IPA duplicate is still collapsed
    dup = make_entry("11000902", "hus", pos_key="sb.",
                     udtale=[{"ipa": "huˀs", "label": None, "audio_url": None,
                              "slot_n": None}] * 2)
    assert S70.pron_html_for_group([dup], S70.Media(cfg)).count("ipa-row") == 1


# ------------------------------------------------- stage 10 governance / gzip

def test_robots_blanket_disallow_is_a_governance_stop():
    from ankidkdeck.stages.s10_sitemap import robots_forbids_ddo
    assert robots_forbids_ddo("User-agent: *\nAllow: /\n") is None
    assert robots_forbids_ddo("User-agent: *\nDisallow: /ddo\n") is not None
    # the strictest possible robots.txt used to PASS the gate
    assert robots_forbids_ddo("User-agent: *\nDisallow: /\n") is not None
    assert robots_forbids_ddo("User-agent: *\nDisallow: /   # comment\n") is not None
    # another agent's blanket ban is not ours
    assert robots_forbids_ddo("User-agent: BadBot\nDisallow: /\n") is None


def test_a_gz_url_serving_plain_xml_is_not_gunzipped():
    """CloudFront serves these with content-encoding: br, so what requests hands
    back depends on whether brotli is installed. Trusting the .gz suffix raised
    a bare BadGzipFile traceback."""
    import gzip as gziplib

    from ankidkdeck.stages.s10_sitemap import _shard_xml
    from ankidkdeck.util import FatalError

    class R:
        def __init__(self, content, text=None):
            self.content = content
            self.text = text if text is not None else content.decode("utf-8")

    plain = b"<urlset><url><loc>https://ordnet.dk/ddo/ordbog?query=hus</loc>"
    assert _shard_xml("shard_a_d.xml.gz", R(plain)) == plain.decode("utf-8")
    packed = gziplib.compress(plain)
    assert _shard_xml("shard_a_d.xml.gz", R(packed, text="junk")) == \
        plain.decode("utf-8")
    # a truncated gzip is a FatalError that names the shard, not a traceback
    with pytest.raises(FatalError) as exc:
        _shard_xml("shard_a_d.xml.gz", R(packed[:12], text="junk"))
    assert "shard_a_d.xml.gz" in str(exc.value)
