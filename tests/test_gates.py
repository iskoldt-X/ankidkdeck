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


# ------------------------------------------------- the report's merge key

def test_a_passing_language_cannot_erase_a_failing_one(cfg):
    """R4 M1. G-COV / G-RATE / G-MEDIA / G-DET are verdicts about ONE language's
    deck, and the report merged on the bare gate id with "later stage wins" --
    so `export --lang Chinese` (FATAL G-COV) followed by `export --lang German`
    (passing) left `ankidkdeck gates` certifying the release all-green."""
    from ankidkdeck.util import FatalError, read_json
    zh = [G.Gate(G.G_COV, "coverage", lambda: (False, {"missing": 8}),
                 stage="70", extra={"lang": "Chinese"})]
    de = [G.Gate(G.G_COV, "coverage", lambda: (True, {"missing": 0}),
                 stage="70", extra={"lang": "German"})]
    with pytest.raises(FatalError):
        G.run_gates(zh, cfg, stage="70")
    G.run_gates(de, cfg, stage="70")
    report = read_json(cfg.report_dir / "gates_report.json")
    rows = {G.row_label(r): r for r in report["results"]}
    assert set(rows) == {"G-COV[lang=Chinese]", "G-COV[lang=German]"}
    assert rows["G-COV[lang=Chinese]"]["ok"] is False
    assert rows["G-COV[lang=German]"]["ok"] is True
    # the aggregate still names the failure
    assert report["failed"] == ["G-COV"]
    assert report["failed_rows"] == ["G-COV[lang=Chinese]"]


def test_re_running_the_same_scope_replaces_its_own_row(cfg):
    from ankidkdeck.util import read_json
    scope = {"lang": "German"}
    G.run_gates([G.Gate(G.G_COV, "coverage", lambda: (True, {"n": 1}),
                        stage="70", extra=scope)], cfg)
    G.run_gates([G.Gate(G.G_COV, "coverage", lambda: (True, {"n": 2}),
                        stage="70", extra=scope)], cfg)
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    assert len(rows) == 1 and rows[0]["detail"]["n"] == 2


def test_the_same_gate_id_at_two_stages_keeps_two_rows(cfg):
    from ankidkdeck.util import read_json
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {}), stage="30")], cfg)
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {}), stage="70")], cfg)
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    assert sorted(r["stage"] for r in rows) == ["30", "70"]


def test_the_report_separates_rows_this_run_ran_from_rows_it_inherited(cfg):
    """RAN / CARRIED / NEVER-RUN are three states, and the report could only
    tell two of them apart.

    The report is a merged ledger on purpose -- stages 20..41 all append to it
    and rows survive across runs -- so "12 gate row(s) recorded, 0 failing" was
    read as a 12-gate verdict when 10 rows had just been executed and two
    (G-TIE at stage 40, G-SITEMAP-INV at stage 10) were inherited from an
    earlier run at stages `build` never touches. gate_ids_with_a_verdict counts
    an inherited row as a verdict, and nothing else in the file said otherwise.
    """
    from ankidkdeck.util import read_json, write_json
    write_json(cfg.report_dir / "gates_report.json",
               {"results": [{"id": G.G_TIE, "description": "tie-breaks",
                             "stage": "40", "extra": {}, "ok": True,
                             "detail": {}, "executed_this_run": True}]})
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {}), stage="30")], cfg)
    rep = read_json(cfg.report_dir / "gates_report.json")
    rows = {r["id"]: r for r in rep["results"]}
    assert rows[G.G_SEED]["executed_this_run"] is True
    # the inherited row asserted `true` about itself; provenance is recomputed
    # from this process, never read back out of the file
    assert rows[G.G_TIE]["executed_this_run"] is False
    assert rep["gate_rows_executed_this_run"] == [G.G_SEED]
    assert rep["gate_rows_carried_from_an_earlier_run"] == [G.G_TIE]
    assert rep["stages_executed_this_run"] == ["30"]
    assert rep["stages_reported"] == ["30", "40"]
    # both rows still count as "has a verdict here"; the third state is the
    # declared gates with no row at all
    assert rep["gate_ids_with_a_verdict"] == sorted([G.G_SEED, G.G_TIE])
    assert G.G_GUID in rep["gate_ids_never_run"]
    assert G.G_SEED not in rep["gate_ids_never_run"]


def test_the_provenance_field_keeps_a_repeated_run_byte_identical(cfg,
                                                                 monkeypatch):
    """Why this is a per-row flag and not a timestamp or a run counter.

    gates_report.json has to stay byte-stable when the same chain runs twice, or
    the workspace determinism check means nothing. Provenance is a function of
    WHICH stages the invocation executed, so a repeat of the same chain writes
    the same bytes -- it moves only when the executed set moves, which is
    exactly when a reader needs it to.
    """
    from ankidkdeck.util import read_json
    path = cfg.report_dir / "gates_report.json"
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {"n": 1}),
                        stage="30")], cfg)
    first = path.read_bytes()
    # a second process, same chain: no memory of the first run's rows
    monkeypatch.setattr(G, "_ROWS_EXECUTED_THIS_RUN", set())
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {"n": 1}),
                        stage="30")], cfg)
    assert path.read_bytes() == first
    # ...and a run that executes a DIFFERENT subset does move the field
    monkeypatch.setattr(G, "_ROWS_EXECUTED_THIS_RUN", set())
    G.run_gates([G.Gate(G.G_RANK, "ranks", lambda: (True, {}), stage="30")], cfg)
    rep = read_json(path)
    assert rep["gate_rows_executed_this_run"] == [G.G_RANK]
    assert rep["gate_rows_carried_from_an_earlier_run"] == [G.G_SEED]


def test_the_gates_command_marks_a_carried_row(cfg, capsys):
    """`ankidkdeck gates` reads the report, it never runs a gate -- so the
    distinction has to come out of the file, or the release checklist reads an
    inherited PASS as a fresh one."""
    from ankidkdeck.cli import gates_report
    from ankidkdeck.util import write_json
    write_json(cfg.report_dir / "gates_report.json",
               {"results": [{"id": G.G_TIE, "description": "tie-breaks",
                             "stage": "40", "extra": {}, "ok": True,
                             "detail": {}}]})
    G.run_gates([G.Gate(G.G_SEED, "seeds", lambda: (True, {}), stage="30")], cfg)
    assert gates_report(cfg) == 0
    out = capsys.readouterr().out
    assert "[CARRIED]" in out
    assert "1 row(s) were executed by the run that wrote this report" in out
    assert "1 carried from an earlier run: G-TIE" in out
    seed_line = [ln for ln in out.splitlines() if ln.startswith("PASS G-SEED")]
    assert seed_line and "[CARRIED]" not in seed_line[0]


def test_the_gates_command_says_so_when_the_report_predates_the_accounting(
        cfg, capsys):
    """A gates_report.json written before this accounting existed has no
    provenance to show; saying so beats printing "0 carried"."""
    from ankidkdeck.cli import gates_report
    from ankidkdeck.util import write_json
    write_json(cfg.report_dir / "gates_report.json",
               {"results": [{"id": G.G_TIE, "description": "tie-breaks",
                             "stage": "40", "extra": {}, "ok": True,
                             "detail": {}}]})
    assert gates_report(cfg) == 0
    out = capsys.readouterr().out
    assert "predates the executed-here accounting" in out
    # "unknown" must not print as "carried"
    assert "[CARRIED]" not in out


def test_the_override_gate_counts_edges_and_mappings_apart():
    """One curated mapping can admit several (word, entry_id) edges -- `vent`
    binds both homograph articles of `vente` -- so 140 mappings admitted 177
    edges. The detail field called that number `mappings_that_bound`, in a file
    the release checklist reads."""
    ok, detail = G.curated_overrides_bind([], 177)
    assert ok is True
    assert detail["edges_admitted_by_the_curated_path"] == 177
    assert detail["mappings_that_bound_nothing"] == 0
    assert "mappings_that_bound" not in detail
    bad, detail = G.curated_overrides_bind(
        [{"word": "dan", "reason": "no_survivor"}], 0)
    assert bad is False
    assert detail["mappings_that_bound_nothing"] == 1
    assert detail["by_reason"] == {"no_survivor": 1}


def test_the_cli_gates_command_exits_non_zero_when_any_row_fails(cfg, capsys):
    from ankidkdeck.cli import gates_report
    from ankidkdeck.util import FatalError
    with pytest.raises(FatalError):
        G.run_gates([G.Gate(G.G_COV, "coverage", lambda: (False, {}),
                            stage="70", extra={"lang": "Chinese"})], cfg)
    G.run_gates([G.Gate(G.G_COV, "coverage", lambda: (True, {}),
                        stage="70", extra={"lang": "German"})], cfg)
    assert gates_report(cfg) == 1
    out = capsys.readouterr().out
    assert "G-COV[lang=Chinese]" in out and "1 failing" in out


# ------------------------------------------------------- the new gate bodies

def test_case_only_members_is_baselined_not_forbidden():
    rows = [{"family_id": "1", "word": "var"}] * 5
    assert G.case_only_members(rows)[0] is True          # unbaselined: report
    assert G.case_only_members(rows, 5)[0] is True
    ok, detail = G.case_only_members(rows, 4)
    assert ok is False and detail["over_baseline"] is True
    assert detail["rows"] == 5


def test_sitemap_inventory_is_report_only_until_it_is_baselined():
    # null range = report-only, which is what ships: an absolute floor
    # extrapolated from a partial measurement is a vacuous gate.
    ok, detail = G.sitemap_inventory(91234, None, 285, [150, 600])
    assert ok is True and "report-only" in detail["total_note"]
    # baselined: a collapse AND a 3x jump both fail (the shard set changed)
    assert G.sitemap_inventory(91234, [70000, 110000], 285, [150, 600])[0] is True
    assert G.sitemap_inventory(4000, [70000, 110000], 285, [150, 600])[0] is False
    assert G.sitemap_inventory(300000, [70000, 110000], 285, [150, 600])[0] is False
    # the affix range is already measured, so it stays enforced
    ok, detail = G.sitemap_inventory(91234, None, 9, [150, 600])
    assert ok is False
    assert "affix_slugs_outside_range" in detail["violations"]


def test_ledger_label_reconciliation():
    ok, detail = G.ledger_label_reconciliation(
        {"hus": {"status": "ok", "results_label": "3 resultater",
                 "article_count": 3},
         "har": {"status": "ok", "results_label": "", "article_count": 1},
         "zzz": {"status": "nohit"}},
        {"hus": 3, "har": 1})
    assert ok is True and detail["ok_pages"] == 2 and detail["error_pages"] == 0
    # a label that does not reconcile with its own stored count
    bad, detail = G.ledger_label_reconciliation(
        {"hus": {"status": "ok", "results_label": "5 resultater",
                 "article_count": 3}}, {"hus": 3})
    assert bad is False and detail["label_does_not_reconcile"]
    # the parse disagreeing with the ledger
    bad, detail = G.ledger_label_reconciliation(
        {"hus": {"status": "ok", "results_label": "3 resultater",
                 "article_count": 3}}, {"hus": 2})
    assert bad is False and detail["parsed_count_disagrees_with_ledger"]
    # error pages are baselined, not forbidden: they are skipped by design
    rows = {"w%d" % i: {"status": "ok", "results_label": "", "article_count": 1}
            for i in range(99)}
    rows["broken"] = {"status": "error", "results_label": "5 resultater",
                      "article_count": 1}
    counts = {w: 1 for w in rows if rows[w]["status"] == "ok"}
    assert G.ledger_label_reconciliation(rows, counts, 0.01)[0] is True
    assert G.ledger_label_reconciliation(rows, counts, 0.0)[0] is False


def test_guid_diff_reconciles_against_the_deck_being_written():
    report = {"summary": {"language": "German", "card_count": 34}}
    assert G.guid_diff_reconciles(report, 34, "German")[0] is True
    ok, detail = G.guid_diff_reconciles(report, 35, "German")
    assert ok is False and "card_count_mismatch" in detail["violations"]
    ok, detail = G.guid_diff_reconciles(report, 34, "Chinese")
    assert ok is False and "language_mismatch" in detail["violations"]
    # a report written before the summary row existed
    ok, detail = G.guid_diff_reconciles({"counts": {}}, 34, "German")
    assert ok is False and "no_summary_row" in detail["violations"]


def test_g_rel_with_no_report_is_not_applicable_rather_than_skipped():
    """The Russian month's defect. G-REL was appended to the export's gate list
    only when a report existed, so the run that should have replaced a stale
    G-REL[lang=Russian] FAIL wrote no row at all -- and report rows merge on
    (id, stage, extra) with no prune, so the failure was permanent on a language
    that can never have a guid_diff report (no previous release to diff).

    ok stays a BOOL: `failed` is a list of ids the release checklist reads. The
    third state lives in the detail, where the printed view can find it.
    """
    ok, detail = G.guid_diff_reconciles({}, 34, "Russian")
    assert ok is True
    assert detail[G.NOT_APPLICABLE] is True
    assert "guid_diff.Russian.json" in detail["why"]
    assert "NOT a verified pass" in detail["why"]
    assert G.row_is_not_applicable({"ok": ok, "detail": detail})
    # a real verdict is never mistaken for one, in either direction
    assert not G.row_is_not_applicable(
        {"ok": True, "detail": G.guid_diff_reconciles(
            {"summary": {"language": "German", "card_count": 34}},
            34, "German")[1]})
    assert not G.row_is_not_applicable({"ok": False, "detail":
                                        {G.NOT_APPLICABLE: True}})
    assert not G.row_is_not_applicable({"ok": True, "detail": "a string"})


def test_g_sep_fails_when_the_fixtures_are_absent(cfg, registry, monkeypatch):
    """"Fixtures unavailable" is a FAILURE, never a silent pass: a release host
    that cannot check the separator table has not checked it."""
    monkeypatch.delenv("ANKIDKDECK_FIXTURES", raising=False)
    ok, detail = G.separator_golden(registry, cfg.work_dir / "fixtures")
    assert ok is False
    assert detail["checked"] is False
    assert detail["reason"] == "fixtures unavailable"
    assert "build_fixtures.py" in detail["hint"]


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


def test_the_export_media_gate_is_baselined_against_the_upstream_dead_registry(cfg):
    """The EXPORT half of G-MEDIA, over the notes actually written. Stage 60
    passing the cache is not enough: this gate re-runs the same question and
    would have blocked the release on the same four slots."""
    dead = "https://static.ordnet.dk/mp3/11034/11034312_2.mp3"   # shipped row
    live = "https://static.ordnet.dk/mp3/11034/11034312_1.mp3"
    (cfg.audio_dir / "11034312_1.mp3").write_bytes(b"ID3-real-audio")
    media = S70.Media(cfg)                 # reads the registry by default
    assert len(media.known_missing) == 4
    assert media.sound_tag(live, "11034312", 1) == "[sound:11034312_1.mp3]"
    # no dead [sound:] tag is emitted either way -- that part never changed
    assert media.sound_tag(dead, "11034312", 2) == ""
    ok, detail = S70.media_gate(media, 1, ["11034312_1.mp3"], 4)
    assert ok is True
    assert detail["declared_but_absent"] == 0        # excused
    assert detail["declared_but_absent_total"] == 1   # but still counted
    known = detail["known_missing_upstream"]
    assert known["n_still_missing"] == 1 and known["still_missing"] == [dead]
    assert len(known["no_longer_declared"]) == 3

    # a miss the registry does NOT name still fails
    assert media.sound_tag("https://static.ordnet.dk/mp3/11034/11034312_9.mp3",
                           "11034312", 9) == ""
    ok, detail = S70.media_gate(media, 1, ["11034312_1.mp3"], 4)
    assert ok is False and detail["declared_but_absent"] == 1

    # fail-closed: deleting the baseline must not grant the excuse
    assert S70.media_gate(S70.Media(cfg), 0, [], 0)[0] is False
    # a repaired slot is reported, and its tag IS emitted
    (cfg.audio_dir / "11034312_2.mp3").write_bytes(b"ID3-repaired")
    media2 = S70.Media(cfg)
    assert media2.sound_tag(dead, "11034312", 2) == "[sound:11034312_2.mp3]"
    known = S70.media_gate(media2, 1, ["11034312_2.mp3"], 4)[1][
        "known_missing_upstream"]
    assert known["recovered"] == [dead] and known["n_still_missing"] == 0


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


def test_a_dead_gate_id_is_not_declared_and_gate_extra_is_used():
    """R4 m14 / item 20. Both halves of the dead-code finding: cli's
    NETWORK_COMMANDS is gone (the invariant is enforced by which body calls
    _net), and Gate.extra is now the report's scope key."""
    import ankidkdeck.cli as cli
    assert not hasattr(cli, "NETWORK_COMMANDS")
    g = G.Gate("G-X", "d", lambda: (True, {}), stage="70",
               extra={"lang": "German"})
    assert G.row_label({"id": g.id, "extra": g.extra}) == "G-X[lang=German]"
    assert G.row_label({"id": g.id, "extra": {}}) == "G-X"


# --------------------------------------------- the net layer's own accounting

class _Resp:
    def __init__(self, status=200, content=b"id3", headers=None):
        self.status_code = status
        self.content = content
        # Real headers matter for the audio ladder: the four upstream-dead slots
        # are a 200 whose content-type is text/html, and a body's declared type is
        # the only signal static.ordnet.dk gives (it has no WAF to check for).
        self.headers = dict(headers or {})
        self.text = ""
        self.history = []


def _fake_net(cfg, monkeypatch, responses):
    from ankidkdeck import net as N
    monkeypatch.setattr(N.time, "sleep", lambda *a, **k: None)
    n = N.Net(cfg)
    seq = list(responses)
    monkeypatch.setattr(n.session, "get",
                        lambda *a, **k: seq.pop(0) if seq else _Resp())
    return n


def test_an_audio_fetch_is_counted_like_every_other_request(cfg, monkeypatch):
    """R4 n5: get_audio bypassed request_count, so a stage-60 run that fetched
    ~4,600 files reported 0 requests -- the one number a polite crawler is
    judged on."""
    n = _fake_net(cfg, monkeypatch, [_Resp(), _Resp(), _Resp()])
    for i in range(3):
        n.get_audio("https://static.ordnet.dk/mp3/11021/1102172%d_1.mp3" % i)
    assert n.request_count == 3
    assert n.circuit.results == [True, True, True]


def test_a_degraded_audio_host_trips_the_circuit_breaker(cfg, monkeypatch):
    """It also bypassed the breaker, so a dead audio host produced ~4,600
    individual FatalErrors on every resume instead of one stop."""
    from ankidkdeck.util import FatalError
    n = _fake_net(cfg, monkeypatch, [_Resp(503), _Resp(503), _Resp(503)])
    url = "https://static.ordnet.dk/mp3/11021/11021722_1.mp3"
    for _ in range(2):
        with pytest.raises(FatalError) as exc:
            n.get_audio(url)
        assert "audio fetch failed" in str(exc.value)
    with pytest.raises(FatalError) as exc:
        n.get_audio(url)
    assert "circuit breaker" in str(exc.value)
    assert n.request_count == 3


def test_one_audio_failure_does_not_trip_the_breaker(cfg, monkeypatch):
    from ankidkdeck.util import FatalError
    n = _fake_net(cfg, monkeypatch, [_Resp(404), _Resp(), _Resp()])
    with pytest.raises(FatalError):
        n.get_audio("https://static.ordnet.dk/mp3/11021/11021722_1.mp3")
    n.get_audio("https://static.ordnet.dk/mp3/11021/11021722_2.mp3")
    assert n.circuit.consecutive_failures == 0


# The zero-byte placeholder DDO serves on the four dead slots, verbatim:
# HTTP 200, content-length 0, content-type text/html, etag "5b06c6d8-0".
_DEAD_HEADERS = {"content-type": "text/html", "content-length": "0",
                 "etag": '"5b06c6d8-0"'}
_DEAD_SLOT = "https://static.ordnet.dk/mp3/11030/11030243_2.mp3"   # shipped row


def test_a_200_with_no_audio_is_retried_once_and_never_returned(cfg, monkeypatch):
    """get_audio had NO rung for a 200 that carries no audio -- a 5xx was
    retried, a 404 was fatal, and a 200-with-nothing was simply RETURNED. That
    is the exact failure DDO produced on four declared slots, and it meant only
    stage 60's own `if not r.content` kept the body off the disk; a challenge
    page served as 200 (this host has no WAF header to check) would have been
    written as an mp3 and G-MEDIA, which tests for zero bytes, would have passed
    it."""
    from ankidkdeck.util import AudioUnavailable
    n = _fake_net(cfg, monkeypatch, [_Resp(200, b"", headers=_DEAD_HEADERS)] * 2)
    with pytest.raises(AudioUnavailable) as exc:
        n.get_audio(_DEAD_SLOT)
    assert exc.value.why == "empty body" and exc.value.retried is True
    assert exc.value.status == 200 and exc.value.n_bytes == 0
    assert n.request_count == 2            # the first attempt plus ONE retry
    # ONE host failure for one unfetchable URL, recorded on the first attempt
    # and not again on the raise -- the 5xx rung's accounting exactly. Recording
    # both attempts would make two dead slots look like four to a breaker that
    # trips on 3 failures in a rolling 50 requests.
    assert n.circuit.results == [False]

    # A non-audio body with real bytes in it is the same failure: the point is
    # that the response never comes back, so it can never be written to a file.
    page = _Resp(200, b"<html>are you a robot</html>",
                 headers={"content-type": "text/html; charset=utf-8"})
    n2 = _fake_net(cfg, monkeypatch, [page, page])
    with pytest.raises(AudioUnavailable) as exc:
        n2.get_audio(_DEAD_SLOT)
    assert exc.value.why == "content-type is not audio"

    # A slot already recorded upstream-dead is probed ONCE and its failure is
    # kept off the breaker: four known-dead probes in a row are four consecutive
    # failures, and tripping on them would abort an otherwise fully cached rerun.
    n3 = _fake_net(cfg, monkeypatch, [_Resp(200, b"", headers=_DEAD_HEADERS)] * 2)
    with pytest.raises(AudioUnavailable):
        n3.get_audio(_DEAD_SLOT, expected_missing=True)
    assert n3.request_count == 1 and n3.circuit.results == []

    # A real mp3 is unaffected, with or without a content-type header.
    n4 = _fake_net(cfg, monkeypatch,
                   [_Resp(200, b"ID3real", headers={"content-type": "audio/mpeg"}),
                    _Resp(200, b"\xff\xfbPreal")])
    assert n4.get_audio(_DEAD_SLOT).content == b"ID3real"
    assert n4.get_audio(_DEAD_SLOT).content == b"\xff\xfbPreal"
    assert n4.circuit.results == [True, True]

    # But three UNKNOWN no-audio URLs in a row still stop the run: a host that
    # has started serving empty bodies wholesale wants a human, not a report
    # listing 5,893 gaps.
    from ankidkdeck.util import FatalError
    n5 = _fake_net(cfg, monkeypatch, [_Resp(200, b"", headers=_DEAD_HEADERS)] * 6)
    for _ in range(2):
        with pytest.raises(AudioUnavailable):
            n5.get_audio(_DEAD_SLOT)
    with pytest.raises(FatalError) as exc:
        n5.get_audio(_DEAD_SLOT)
    assert "circuit breaker" in str(exc.value)


# ------------------------------------------------------- stage 60 audio cache

def test_the_audio_url_assertion_demands_eight_digits():
    """Guide 4.11 asserts an 8-digit entry_id and the corpus agrees
    (4,629/4,629). The looser 6-or-more form would have let a malformed slice
    past the only free integrity check this stage has -- and the wrong sound on
    a card is silent by construction."""
    from ankidkdeck.stages.s60_audio import assert_url_belongs
    from ankidkdeck.util import FatalError
    ok = "https://static.ordnet.dk/mp3/11021/11021722_1.mp3"
    assert assert_url_belongs(ok, "11021722", 1) == 1
    for bad in ("https://static.ordnet.dk/mp3/11021/110217_1.mp3",
                "https://static.ordnet.dk/mp3/11021/1102172_1.mp3",
                "https://static.ordnet.dk/mp3/11021/110217223_1.mp3"):
        eid = bad.rsplit("/", 1)[-1].split("_")[0]
        with pytest.raises(FatalError) as exc:
            assert_url_belongs(bad, eid, 1)
        assert "does not match the DDO pattern" in str(exc.value), bad
    # the other two halves of the assertion still fire
    with pytest.raises(FatalError) as exc:
        assert_url_belongs(ok, "11021722", 2)
    assert "slot mismatch" in str(exc.value)
    with pytest.raises(FatalError) as exc:
        assert_url_belongs("https://static.ordnet.dk/mp3/11021/11021799_1.mp3",
                           "11021722", 1)
    assert "does not belong to entry" in str(exc.value)


def test_the_audio_cache_hashes_only_what_the_cheap_check_flags(cfg, monkeypatch):
    """R3 m12: _sha_of() ran for every already-cached file on every invocation,
    i.e. a full sha256 sweep of 4,629+ files per resume."""
    from ankidkdeck.stages import s60_audio as S60
    from ankidkdeck.util import sha256_bytes, write_json
    url = "https://static.ordnet.dk/mp3/11021/11021722_1.mp3"
    data = b"id3-payload"
    (cfg.audio_dir / "11021722_1.mp3").write_bytes(data)
    write_json(cfg.audio_dir / "manifest.json",
               {url: {"url": url, "file": "11021722_1.mp3",
                      "sha256": sha256_bytes(data), "bytes": len(data),
                      "entry_id": "11021722", "slot_n": 1, "source": "legacy"}})
    e = make_entry("11021722", "hus", pos_key="sb.",
                   senses=[make_sense("21000001", "bygning")],
                   udtale=[{"ipa": "huˀs", "label": None, "audio_url": url,
                            "slot_n": 1}])
    write_json(cfg.json_dir / "entries.json", {"11021722": e})
    hashed = []
    real = S60._sha_of
    monkeypatch.setattr(S60, "_sha_of",
                        lambda p: (hashed.append(str(p)), real(p))[1])
    report = S60.run(cfg, None)
    assert report["already_cached"] == 1
    assert hashed == [], "the size check was skipped"
    # a size that disagrees DOES get hashed
    write_json(cfg.audio_dir / "manifest.json",
               {url: {"url": url, "file": "11021722_1.mp3",
                      "sha256": sha256_bytes(data), "bytes": 999,
                      "entry_id": "11021722", "slot_n": 1, "source": "legacy"}})
    S60.run(cfg, None)
    assert hashed


def test_orphans_are_counted_always_and_swept_only_on_request(cfg):
    """R4 m10: the sweep was unconditional, so running `audio` against a partial
    entries.json quarantined the whole 2025 cache -- which the next full run then
    re-downloads."""
    from ankidkdeck.stages import s60_audio as S60
    from ankidkdeck.util import write_json
    (cfg.audio_dir / "19999999_1.mp3").write_bytes(b"stale")
    e = make_entry("11021722", "hus", pos_key="sb.",
                   senses=[make_sense("21000001", "bygning")])
    write_json(cfg.json_dir / "entries.json", {"11021722": e})
    report = S60.run(cfg, None)
    assert report["unreferenced_files"] == 1
    assert report["quarantined_orphans"] == 0
    assert (cfg.audio_dir / "19999999_1.mp3").exists()
    assert "--sweep-orphans" in report["sweep_hint"]
    report = S60.run(cfg, None, sweep_orphans=True)
    assert report["quarantined_orphans"] == 1
    assert (cfg.audio_dir / "_orphans" / "19999999_1.mp3").exists()


# ------------------------------- the upstream-dead audio registry (G-MEDIA)
#
# DDO's own article declares 5,893 audio slots and DDO's own host cannot serve
# four of them: HTTP 200, content-length 0, content-type text/html, and the same
# etag "5b06c6d8-0" -- nginx's etag for a zero-byte file -- on all four, while
# sibling slots of the same entries serve real mp3 bodies. The four are recorded
# in registry/known_missing_audio.json and BASELINED by
# gates.json:known_missing_audio_max, so G-MEDIA passes on a defect that is not
# ours without ceasing to be a gate.

_LIVE_SLOT = "https://static.ordnet.dk/mp3/11030/11030243_1.mp3"


def _lev_workspace(cfg, slots=(1, 2)):
    """entries.json with `lev` (11030243), whose slot 2 is a shipped registry
    row and whose slot 1 is a normal, working slot."""
    from ankidkdeck.util import write_json
    udtale = [{"ipa": "leːˀv" if n == 1 else "", "label": None, "slot_n": n,
               "audio_url": "https://static.ordnet.dk/mp3/11030/11030243_%d.mp3" % n}
              for n in slots]
    e = make_entry("11030243", "lev", pos_key="sb.",
                   senses=[make_sense("21000001", "livet")], udtale=udtale)
    write_json(cfg.json_dir / "entries.json", {"11030243": e})
    return e


def _media_row(cfg):
    from ankidkdeck.util import read_json
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    return next(r for r in rows if r["id"] == "G-MEDIA" and r["stage"] == "60")


def test_an_upstream_dead_slot_is_reported_and_never_lands_on_disk(cfg, monkeypatch):
    """The registry row is honoured: G-MEDIA passes, the slot is reported as
    known_missing_upstream rather than counted absent, and the zero-byte
    text/html body is not written -- a zero-byte mp3 imports into Anki as a
    silent card, which no later gate would notice."""
    from ankidkdeck.stages import s60_audio as S60
    _lev_workspace(cfg)
    (cfg.audio_dir / "11030243_1.mp3").write_bytes(b"ID3-real-audio")
    n = _fake_net(cfg, monkeypatch, [_Resp(200, b"", headers=_DEAD_HEADERS)])

    report = S60.run(cfg, n)               # no FatalError == G-MEDIA passed

    assert report["known_missing_upstream"] == 1
    assert report["recovered_upstream"] == 0
    assert report["n_no_audio_from_host"] == 1
    assert report["no_audio_from_host"][0]["url"] == _DEAD_SLOT
    assert report["no_audio_from_host"][0]["why"] == "empty body"
    assert report["known_missing_audio"]["still_missing"] == [_DEAD_SLOT]
    # the other three shipped rows are not declared by this one-entry corpus,
    # which is reported so a row the corpus dropped stops being carried silently
    assert len(report["known_missing_audio"]["no_longer_declared"]) == 3
    assert not (cfg.audio_dir / "11030243_2.mp3").exists()
    from ankidkdeck.util import read_json
    assert _DEAD_SLOT not in read_json(cfg.audio_dir / "manifest.json")
    row = _media_row(cfg)
    assert row["ok"] is True
    assert row["detail"]["n_missing"] == 0 and row["detail"]["n_zero_byte"] == 0
    known = row["detail"]["known_missing_upstream"]
    assert known["registry_rows"] == 4 and known["max"] == 4
    assert known["n_still_missing"] == 1 and known["over_baseline"] is False


def test_a_missing_slot_the_registry_does_not_name_still_fails_g_media(cfg,
                                                                       monkeypatch):
    """The population the gate exists for -- a lost cache, a failed download, a
    skipped seed -- is untouched by the registry."""
    from ankidkdeck.stages import s60_audio as S60
    from ankidkdeck.util import FatalError
    _lev_workspace(cfg)
    # slot 1 is NOT a registry row, and the host serves it the same empty 200
    n = _fake_net(cfg, monkeypatch, [_Resp(200, b"", headers=_DEAD_HEADERS)] * 4)
    with pytest.raises(FatalError) as exc:
        S60.run(cfg, n)
    assert "G-MEDIA" in str(exc.value)
    detail = _media_row(cfg)["detail"]
    assert detail["missing"] == [_LIVE_SLOT] and detail["n_missing"] == 1
    assert detail["known_missing_upstream"]["n_still_missing"] == 1
    assert not (cfg.audio_dir / "11030243_1.mp3").exists()
    # and with no net at all, an absent non-registry file is still a failure
    with pytest.raises(FatalError):
        S60.run(cfg, None)


def test_a_recovered_upstream_slot_is_reported_not_hidden(cfg, monkeypatch):
    """The day DDO repairs the file is the mechanism working, not a regression:
    the row is reported as recovered -- through the DISK, so a legacy seed or an
    earlier run counts too -- and the report says to delete it."""
    from ankidkdeck.stages import s60_audio as S60
    _lev_workspace(cfg)
    for name in ("11030243_1.mp3", "11030243_2.mp3"):
        (cfg.audio_dir / name).write_bytes(b"ID3-real-audio")

    report = S60.run(cfg, None)            # nothing to fetch, nothing to fail

    assert report["known_missing_audio"]["recovered"] == [_DEAD_SLOT]
    assert report["known_missing_upstream"] == 0 and report["recovered_upstream"] == 1
    assert "known_missing_audio_max" in report["recovered_hint"]
    known = _media_row(cfg)["detail"]["known_missing_upstream"]
    assert known["recovered"] == [_DEAD_SLOT] and known["n_still_missing"] == 0


def test_the_upstream_dead_registry_cannot_grow_past_its_baseline(cfg):
    """Bump-in-the-same-commit, and fail-closed on a missing key -- the same
    discipline G-SUPPRESS and G-ADMIT have. A registry that excuses rows from the
    only gate that adjudicates them must not be able to grow silently, and the
    excuse must not be grantable by DELETING the baseline line."""
    from ankidkdeck.stages import s60_audio as S60
    status = {"registry_rows": 5, "still_missing": [_DEAD_SLOT],
              "recovered": [], "no_longer_declared": []}
    assert S60._media_gate([_DEAD_SLOT], [], 1, status, 5)[0] is True
    ok, detail = S60._media_gate([_DEAD_SLOT], [], 1, status, 4)
    assert ok is False and detail["known_missing_upstream"]["over_baseline"] is True
    # no baseline at all: every row is over it
    assert S60._media_gate([], [], 1, status, 0)[0] is False
    # and with no registry in play the gate is exactly what it always was
    assert S60._media_gate([], [], 1)[0] is True
    assert S60._media_gate([_LIVE_SLOT], [], 1)[0] is False


# ------------------------------------------------- stage 10 governance / gzip

class _FakeNet:
    """Offline stand-in for net.Net: robots.txt, the index, and 1 shard."""

    ROBOTS = ("User-agent: *\nAllow: /\n"
              "Sitemap: https://ordnet.dk/sitemaps/ddo/index.xml\n")

    def __init__(self, n_urls=12, n_affix=200):
        self.request_count = 0
        self.n_urls = n_urls
        self.n_affix = n_affix

    class _R:
        def __init__(self, text):
            self.text = text
            self.content = text.encode("utf-8")

    def get(self, url):
        self.request_count += 1
        if url.endswith("robots.txt"):
            return self._R(self.ROBOTS)
        if url.endswith("index.xml"):
            locs = "".join("<loc>https://x/shard_%d.xml</loc>" % i
                           for i in range(5))
            return self._R("<sitemapindex>%s</sitemapindex>" % locs)
        words = ["ord%d" % i for i in range(self.n_urls)]
        words += ["-affiks%d" % i for i in range(self.n_affix)]
        # The slug is the LAST path segment: stage 10 reads it with
        # unquote(loc.rsplit("/", 1)[-1]).
        locs = "".join(
            "<url><loc>https://ordnet.dk/ddo/%s</loc>"
            "<lastmod>2026-01-01</lastmod></url>" % w for w in words)
        return self._R("<urlset>%s</urlset>" % locs)


def test_the_sitemap_url_total_is_a_gate_row_not_a_bare_raise(cfg):
    """R4 m1 / R3 v3: the URL total was `raise FatalError(total < 80_000)` --
    a source constant extrapolated from a partial measurement, and a stop that
    never reached gates_report.json."""
    from ankidkdeck.util import read_json
    from ankidkdeck.stages import s10_sitemap
    net = _FakeNet(n_urls=12, n_affix=200)
    # shipped default: sitemap_total_range is null -> report only, no stop,
    # even though 12 * 5 shards is nowhere near 80k.
    report = s10_sitemap.run(cfg, net, {"sitemap_total_range": None,
                                        "affix_count_range": [150, 600]})
    assert report["total_urls"] == 5 * (12 + 200)
    assert "sitemap_total_range" in report["baseline_hint"]
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    inv = [r for r in rows if r["id"] == "G-SITEMAP-INV"]
    assert inv and inv[0]["ok"] is True and inv[0]["stage"] == "10"
    assert "report-only" in inv[0]["detail"]["total_note"]


def test_a_baselined_sitemap_range_fails_as_a_recorded_gate(cfg):
    from ankidkdeck.util import FatalError, read_json
    from ankidkdeck.stages import s10_sitemap
    with pytest.raises(FatalError) as exc:
        s10_sitemap.run(cfg, _FakeNet(), {"sitemap_total_range": [80000, 120000],
                                          "affix_count_range": [150, 600]})
    assert "G-SITEMAP-INV" in str(exc.value)
    rows = read_json(cfg.report_dir / "gates_report.json")["results"]
    inv = [r for r in rows if r["id"] == "G-SITEMAP-INV"][0]
    assert inv["ok"] is False
    assert "total_urls_outside_range" in inv["detail"]["violations"]
    # ...and sitemap.json was NOT written: the stage stops before the inventory
    assert not (cfg.json_dir / "sitemap.json").exists()


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
