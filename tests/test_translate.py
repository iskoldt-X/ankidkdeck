"""Stage 42: what gets billed, what does not, and what the dry path may touch.

Four round-2 findings live here:

  * THE ARCHIVE IS READ BACK (R3 M3). gc() moves a dead row into archive.json and
    its docstring has always claimed "a sense that comes back after a DDO edit
    must not be paid for twice" (guide D7) -- but no code path anywhere read
    archive.json, so the property was not implemented. Remove a sense, translate,
    put it back, translate: the cell was billed again.
  * A DRY RUN DOES NOT CONSUME THE DRIFT LEDGER (R3 m2). It was rewritten on
    every run, so the second bill-only run reported "nothing changed" and
    `--lang German` then `--lang English` gave the second language an empty drift
    report. Report-only, but the report is the artifact.
  * THE SCOPE FALLBACK NEEDS A FLAG (R3 m4). With no words.json the scope was
    every parsed entry -- 3.4x the cells on the fixture corpus, all of it on
    articles the classifier rejected and the exporter never renders.
  * --lang IS VALIDATED (R4 M3). `german` is not `German`, none of the migrated
    cells are visible under the lowercase key, and export's G-COV protects the
    deck but not the money.
"""

import pytest
from conftest import make_entry, make_expression, make_sense

from ankidkdeck.extract import ARTICLE_SHA_SCHEMA
from ankidkdeck.stages import s42_translate as S42
from ankidkdeck.util import FatalError, read_json, write_json


def _entry(with_second_sense=True):
    senses = [make_sense("21000001", "bygning man bor i")]
    if with_second_sense:
        senses.append(make_sense("21000002", "husholdning"))
    return make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                      senses=senses,
                      expressions=[make_expression("21000003", "hus forbi",
                                                   "helt forkert")],
                      source_words=["hus"])


def _workspace(cfg, entry=None):
    e = entry or _entry()
    write_json(cfg.json_dir / "entries.json", {e["entry_id"]: e})
    write_json(cfg.json_dir / "words.json",
               {"11021722": {"family_id": "11021722",
                             "anchor_entry_id": "11021722",
                             "entry_ids": ["11021722"], "freq_rank": 1}})
    return e


def _tdir(cfg, lang="German"):
    return cfg.json_dir / "translations" / lang


# ---------------------------------------------------------- the bill scope

def test_the_bill_quotes_the_renderable_scope(cfg, registry):
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=False)
    bill = report["bill"]["German"]
    assert bill["definitions"] == 2 and bill["expressions"] == 1
    assert bill["cells_total"] == 3
    assert report["scope"]["basis"] == "renderable families (words.json)"
    assert report["note"].startswith("bill only")


def test_the_registry_pos_table_is_not_billed(cfg, registry):
    """The 22 hand-written keys are already there; only a language the registry
    does not know is billed, which is what keeps adding a language cheap."""
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=False)
    assert report["bill"]["German"]["pos_keys"] == 0
    bill_file = read_json(cfg.report_dir / "translate_bill_German.json")
    assert bill_file["pos_keys_from_registry"] == ["sb."]


def test_a_language_the_registry_does_not_know_still_bills_its_pos_keys(cfg,
                                                                       registry):
    cfg.langs = list(cfg.langs) + ["Icelandic"]
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="Icelandic", confirm=False)
    assert report["bill"]["Icelandic"]["pos_keys"] == 1
    assert report["bill"]["Icelandic"]["pos_keys_list"] == ["sb."]


def test_the_scope_fallback_needs_an_explicit_flag(cfg, registry):
    e = _entry()
    write_json(cfg.json_dir / "entries.json", {e["entry_id"]: e})
    # no words.json
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=False)
    assert "--include-unused" in str(exc.value)
    report = S42.run(cfg, registry, lang="German", confirm=False,
                     include_unused=True)
    assert report["scope"]["basis"] == "all parsed entries"


def test_an_unconfigured_language_is_refused_before_anything_is_billed(cfg,
                                                                      registry):
    _workspace(cfg)
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="german", confirm=False)
    assert "german" in str(exc.value) and "German" in str(exc.value)
    assert not (cfg.json_dir / "translations" / "german").exists()


# ---------------------------------------------------------- the archive

def test_a_sense_that_comes_back_is_restored_not_billed_again(cfg, registry):
    """R3 M3, end to end: archive, then un-archive."""
    _workspace(cfg)
    tdir = _tdir(cfg)
    e = _entry()
    sha = {s["dannetid"]: s["src_sha"] for s in e["senses"]}
    write_json(tdir / "definitions.json", {
        "11021722:21000001": {"lemma": "Haus", "gloss": "Gebaeude",
                              "src_sha": sha["21000001"], "provenance": "p1"},
        "11021722:21000002": {"lemma": "Haushalt", "gloss": "Familie",
                              "src_sha": sha["21000002"], "provenance": "p2"},
    })
    write_json(tdir / "expressions.json", {})

    # DDO drops the second sense -> the row is archived
    _workspace(cfg, entry=_entry(with_second_sense=False))
    gone = S42.run(cfg, registry, lang="German", confirm=False)
    assert gone["gc"]["German"]["archived"]["definitions"] == 1
    archive = read_json(tdir / "archive.json")
    assert "11021722:21000002" in archive["definitions"]
    assert "11021722:21000002" not in read_json(tdir / "definitions.json")

    # ...and it comes back
    _workspace(cfg, entry=_entry(with_second_sense=True))
    back = S42.run(cfg, registry, lang="German", confirm=False)
    assert back["bill"]["German"]["definitions"] == 0, (
        "a returning sense was billed a second time")
    assert back["bill"]["German"]["restored_from_archive"] == 1
    live = read_json(tdir / "definitions.json")
    assert live["11021722:21000002"]["lemma"] == "Haushalt"
    assert live["11021722:21000002"]["provenance"] == "p2"


def test_an_archived_row_whose_source_text_changed_is_still_billed(cfg,
                                                                  registry):
    """The archived gloss was produced from a specific Danish string. If DDO
    rewrote the definition, restoring the old gloss would be worse than paying
    for a new one."""
    _workspace(cfg)
    tdir = _tdir(cfg)
    write_json(tdir / "definitions.json", {})
    write_json(tdir / "expressions.json", {})
    write_json(tdir / "archive.json", {
        "definitions": {"11021722:21000001": {
            "lemma": "Haus", "gloss": "alt", "src_sha": "a-stale-sha",
            "provenance": "p"}},
        "expressions": {}})
    report = S42.run(cfg, registry, lang="German", confirm=False, do_gc=False)
    cells = read_json(
        cfg.report_dir / "translate_bill_German.json")["cells_by_key"]
    assert cells["11021722:21000001"] == "missing"
    assert report["bill"]["German"]["restored_from_archive"] == 0


# ------------------------------------------------------- the drift ledger

def _ledger(cfg):
    return cfg.json_dir / "ledger" / "content_hashes.json"


def test_a_bill_only_run_does_not_write_the_drift_ledger(cfg, registry):
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=False)
    assert not _ledger(cfg).exists()
    # ...and a second dry run still reports the same thing
    r2 = S42.run(cfg, registry, lang="German", confirm=False)
    assert r2["drift"]["entries_new"] == 1
    assert r2["drift"]["ledger_written"] is False


def test_the_drift_ledger_carries_a_schema_and_reports_a_bump_as_such(cfg,
                                                                     registry):
    _workspace(cfg)
    write_json(_ledger(cfg), {"schema": 1, "hashes": {"11021722": "old-sha"}})
    report = S42.run(cfg, registry, lang="German", confirm=False)
    d = report["drift"]
    assert d["schema_changed"] is True
    assert d["ledger_schema"] == 1
    assert d["parser_schema"] == ARTICLE_SHA_SCHEMA
    assert d["entries_changed_since_last_run"] is None
    assert "schema changed" in d["note"]


def test_a_real_content_change_is_reported_as_drift(cfg, registry):
    e = _workspace(cfg)
    write_json(_ledger(cfg), {"schema": ARTICLE_SHA_SCHEMA,
                              "hashes": {"11021722": "a-different-sha"}})
    report = S42.run(cfg, registry, lang="German", confirm=False)
    assert report["drift"]["entries_changed_since_last_run"] == 1
    assert report["drift"]["changed_sample"] == ["11021722"]
    # the unchanged case, for contrast
    write_json(_ledger(cfg), {"schema": ARTICLE_SHA_SCHEMA,
                              "hashes": {"11021722": e["article_sha"]}})
    clean = S42.run(cfg, registry, lang="German", confirm=False)
    assert clean["drift"]["entries_changed_since_last_run"] == 0


# ------------------------------------------------- article_sha content-only

def test_article_sha_ignores_derived_and_provenance_fields():
    """R4 m4 / issue 7: hashing headword_glued, paradigm_index, form_index,
    source_words and the registry-supplied slot_label meant a parser or registry
    refactor reported all 3,812 articles as edited by DDO."""
    from ankidkdeck.stages.s20_parse import content_sha
    base = make_entry("11021722", "hus", pos_key="sb.",
                      paradigm_rows=[{"table": 0, "row": 0, "cells": ["huset"],
                                      "slot_label": "definite singular"}],
                      senses=[make_sense("21000001", "bygning")],
                      source_words=["hus"])
    before = content_sha(base)
    noise = dict(base)
    noise["headword_glued"] = "hus9"
    noise["paradigm_index"] = ["something", "else"]
    noise["form_index"] = ["another", "thing"]
    noise["source_words"] = ["hus", "huset", "husene"]
    noise["paradigm"] = {"short": base["paradigm"]["short"],
                         "rows": [{"table": 0, "row": 0, "cells": ["huset"],
                                   "slot_label": "RENAMED BY THE REGISTRY"}]}
    assert content_sha(noise) == before
    # ...but a real content change still moves it
    changed = dict(base)
    changed["senses"] = [make_sense("21000001", "en helt anden definition")]
    assert content_sha(changed) != before
    cells = dict(base)
    cells["paradigm"] = {"short": None,
                         "rows": [{"table": 0, "row": 0, "cells": ["husene"],
                                   "slot_label": "definite singular"}]}
    assert content_sha(cells) != before


# ------------------------------------------------------------ the ladders

def test_the_retry_budgets_are_the_ones_that_survived_d08():
    """One flaky response used to abort a paid multi-hour run (64 calls spent
    before the contamination FATAL), so the count-lock and transport ladders
    stay at 5.

    MAX_CORRECTION_ATTEMPTS does NOT: it was the outer loop of the automatic
    review pass, the only 10x amplifier in the design, and it left with the
    automatic pass (owner decision D-08). Its absence is asserted, not assumed
    -- re-adding it would quietly restore a 10x worst case.
    """
    assert S42.MAX_COUNT_LOCK_ATTEMPTS == 5
    assert S42.MAX_RETRIES == 5
    assert S42.MAX_503_RETRIES == 1
    assert not hasattr(S42, "MAX_CORRECTION_ATTEMPTS")


def test_requests_max_is_true_ceiling():
    """The bill's request ceiling is computed from the ladders that exist, and
    it says whether transport retries are in it.

    Before: def_calls was not multiplied at all, expressions were multiplied by
    2 (generate + review) and by MAX_CORRECTION_ATTEMPTS, and no layer counted
    _generate's own five attempts -- so the only number a human saw before
    pressing --confirm-spend was wrong in both directions at once.
    """
    todo = [{"kind": "expression", "entry_id": "1", "text": "a", "hint": "",
             "reason": "missing"},
            {"kind": "definition", "entry_id": "1", "text": "b", "hint": "",
             "grammar": "", "reason": "missing"}]
    row = S42.bill_row(todo, ["sb."])
    # one expression batch + one definition batch + one POS call
    assert row["requests_min"] == 3
    assert row["definition_requests"] == 1 and row["expression_requests"] == 1
    assert row["pos_requests"] == 1
    assert S42.REQUEST_CEILING_FACTOR == (S42.MAX_COUNT_LOCK_ATTEMPTS
                                          * S42.MAX_RETRIES)
    assert row["requests_max"] == 3 * S42.REQUEST_CEILING_FACTOR
    assert "INCLUDES transport retries" in row["requests_max_basis"]


# -------------------------------------------------- the confirmed path (FAKE)

@pytest.fixture
def translator(fake_genai, no_sleep, probe_stats):
    """A stand-in translator that always honours the count lock.

    probe_stats is a requirement, not decoration: a confirmed run reads the
    measured output fit off disk and refuses to size a paid request without it.
    """
    @fake_genai.respond
    def _answer(call):
        props = call["config"].kwargs["response_schema"]["properties"]
        if "definitions" in props:
            n = props["definitions"]["minItems"]
            return {"headword": "hus",
                    "definitions": [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                                    for i in range(n)]}
        if "fixed_expressions" in props:
            n = props["fixed_expressions"]["minItems"]
            return {"fixed_expressions": [{"lemma": "X%d" % i, "gloss": "Y%d" % i}
                                          for i in range(n)]}
        return {k: "POS-%s" % k for k in props}
    return fake_genai


def test_a_confirmed_run_records_usage_and_stamps_full_provenance(cfg, registry,
                                                                 translator):
    """1.4 + 1.10. Not one byte of usageMetadata used to be read, and provenance
    was the model and the date -- which cannot tell two runs apart when the
    prompt pack or the thinking level changed between them."""
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=True)
    # 2 definitions in one batch + 1 expression batch = 2 calls
    assert len(translator.calls) == 2
    assert report["usage"]["requests"] == 2
    assert report["usage"]["prompt_tokens"] > 0
    assert report["usage"]["thinking_tokens"] == 0
    assert report["usage"]["finish_reasons"] == ["STOP"]
    rows = read_json(cfg.report_dir / "translate_usage.json")
    assert len(rows) == 2
    assert all(r["cached_tokens"] <= r["prompt_tokens"] for r in rows)
    assert all(r["prompt_id"] == cfg.prompt_id for r in rows)
    prov = report["languages"]["German"]["provenance"]
    assert prov.startswith("gemini:gemini-3.7-flash+v4-frozen+LOW@")
    assert prov.isascii()
    assert read_json(_tdir(cfg) / "definitions.json")[
        "11021722:21000001"]["provenance"] == prov


def test_the_expression_wave_is_one_request_per_batch(cfg, registry, translator):
    """N-02(a): the generate -> review -> correct loop is gone. It cost two calls
    per batch minimum and up to ten, on every batch, whether or not anything was
    wrong."""
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=True)
    schemas = [c["config"].kwargs["response_schema"]["properties"]
               for c in translator.calls]
    assert sum(1 for s in schemas if "fixed_expressions" in s) == 1
    # ...and no reviewer call was placed at all
    assert not any("contains_other_languages" in s for s in schemas)


def test_the_drift_ledger_is_consumed_after_the_calls_not_before(cfg, registry,
                                                                 translator):
    """1.11. It used to be written in the first few lines of run(), before a
    single call: one crash consumed entries_changed_since_last_run, and under the
    batch transport (submit and ingest are both --confirm-spend) it was consumed
    twice per wave."""
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=True)
    assert _ledger(cfg).exists()
    ledger_written = read_json(cfg.report_dir / "translate_report.json")["drift"]
    assert ledger_written["ledger_written"] is True
    assert ledger_written["ledger_written_at_phase"] == "all"

    # a submit-only wave does NOT consume it
    _ledger(cfg).unlink()
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert not _ledger(cfg).exists()


def test_a_crash_before_the_last_call_leaves_the_drift_ledger_alone(
        cfg, registry, translator):
    _workspace(cfg)

    @translator.respond
    def _short(call):
        return {"headword": "hus", "definitions": []}

    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True)
    assert not _ledger(cfg).exists()


# ------------------------------------------------------- clean retranslation

def _existing_rows(cfg, entry, lang="German"):
    tdir = _tdir(cfg, lang)
    sha = {s["dannetid"]: s["src_sha"] for s in entry["senses"]}
    write_json(tdir / "definitions.json", {
        "11021722:21000001": {"lemma": "alt-1", "gloss": "alt-1",
                              "src_sha": sha["21000001"],
                              "provenance": "gemini:gemini-2.0-flash@2025-01-01"},
        "11021722:21000002": {"lemma": "alt-2", "gloss": "alt-2",
                              "src_sha": sha["21000002"],
                              "provenance": "gemini:gemini-2.0-flash@2025-01-01"},
    })
    write_json(tdir / "expressions.json", {})
    return tdir


def test_retranslate_all_archives_with_reason(cfg, registry, translator):
    """N-01. This is the path the v3 clean redo actually runs on.

    Deviation from the task list, deliberate: the DRY path is allowed to quote a
    clean-redo bill (it archives nothing and writes no row), because otherwise
    the only way to see the price of the redo would be to delete
    definitions.json by hand -- which is exactly the workaround that only works
    once. Everything destructive still needs --confirm-spend.
    """
    e = _workspace(cfg)
    tdir = _existing_rows(cfg, e)

    dry = S42.run(cfg, registry, lang="German", confirm=False,
                  retranslate_all=True)
    assert dry["bill"]["German"]["cells_total"] == 3
    assert dry["bill"]["German"]["definitions_clean_redo"] == 2
    assert read_json(tdir / "definitions.json")["11021722:21000001"]["lemma"] \
        == "alt-1"
    assert not (tdir / "archive.json").exists() or not read_json(
        tdir / "archive.json")["definitions"]

    done = S42.run(cfg, registry, lang="German", confirm=True,
                   retranslate_all=True)
    archive = read_json(tdir / "archive.json")
    assert set(archive["definitions"]) == {"11021722:21000001",
                                          "11021722:21000002"}
    assert all(r["reason"] == "clean_redo"
               for r in archive["definitions"].values())
    assert done["languages"]["German"]["archived_for_redo"]["definitions"] == 2
    live = read_json(tdir / "definitions.json")
    assert live["11021722:21000001"]["lemma"] == "L0"
    assert "3.7-flash" in live["11021722:21000001"]["provenance"]


def test_retranslate_all_does_not_let_the_archive_restore_what_it_archived(
        cfg, registry, translator):
    """The second half of N-01, and the reason `rm definitions.json` only ever
    worked once: the archive is read back on every normal run, so the rows come
    straight home."""
    e = _workspace(cfg)
    tdir = _existing_rows(cfg, e)
    S42.run(cfg, registry, lang="German", confirm=True, retranslate_all=True)
    live = read_json(tdir / "definitions.json")
    assert live["11021722:21000001"]["lemma"] == "L0"     # not "alt-1"
    # a NORMAL run afterwards restores nothing either: the rows are current
    again = S42.run(cfg, registry, lang="German", confirm=False)
    assert again["bill"]["German"]["cells_total"] == 0


def _archive_one(cfg, entry, reason, lang="German"):
    """One archived definition row whose sha still matches the live source."""
    tdir = _tdir(cfg, lang)
    sha = {s["dannetid"]: s["src_sha"] for s in entry["senses"]}
    write_json(tdir / "definitions.json", {})
    write_json(tdir / "expressions.json", {})
    row = {"lemma": "Haus", "gloss": "G", "src_sha": sha["21000001"],
           "provenance": "gemini:gemini-2.0-flash@2025-01-01"}
    if reason is not None:
        row["reason"] = reason
    write_json(tdir / "archive.json",
               {"definitions": {"11021722:21000001": row}, "expressions": {}})
    return tdir


def test_a_restored_row_does_not_carry_the_archiving_reason(cfg, registry):
    """`reason` says why a row was archived, which is not a property of the
    translation. definitions.json rows are compared and exported; a stray field
    there is a diff nobody can explain."""
    e = _workspace(cfg)
    tdir = _archive_one(cfg, e, "sense_gone")
    S42.run(cfg, registry, lang="German", confirm=False, do_gc=False)
    row = read_json(tdir / "definitions.json")["11021722:21000001"]
    assert "reason" not in row and row["lemma"] == "Haus"


def test_a_row_the_redo_retired_is_never_restored(cfg, registry):
    """F2(a). archive_everything stamps reason="clean_redo" and leaves src_sha
    alone, so every retired row still LOOKS restorable -- and the restore path
    only ever compared shas.

    Measured consequence: crash a confirmed clean redo, then run a plain
    `translate` (the natural "carry on"), and every gemini-2.0-flash row came
    home, the bill said 0 cells, and nothing warned. A row we retired on
    purpose is not a row that came back from DDO.
    """
    e = _workspace(cfg)
    tdir = _archive_one(cfg, e, "clean_redo")
    report = S42.run(cfg, registry, lang="German", confirm=False, do_gc=False)
    assert read_json(tdir / "definitions.json") == {}          # not restored
    assert report["bill"]["German"]["restored_from_archive"] == 0
    # ...and it is billed as work to do, at the CURRENT model, not resurrected
    keys = read_json(cfg.report_dir
                     / "translate_bill_German.json")["cells_by_key"]
    assert keys["11021722:21000001"] == "missing"
    # a renamed reason is refused too: the redo's reason is configurable
    cfg.retranslate_reason = "rebuild-2026"
    _archive_one(cfg, e, "rebuild-2026")
    S42.run(cfg, registry, lang="German", confirm=False, do_gc=False)
    assert read_json(tdir / "definitions.json") == {}


# ------------------------------------------- the redo is resumable (F1 + F2)

def _two_entries(cfg):
    """Two one-sense entries, so a redo can die between them."""
    a = make_entry("11000001", "hus", pos_key="sb.", pos_text="substantiv",
                   senses=[make_sense("21000001", "bygning man bor i")],
                   source_words=["hus"])
    b = make_entry("11000002", "bil", pos_key="sb.", pos_text="substantiv",
                   senses=[make_sense("21000002", "koeretoej med fire hjul")],
                   source_words=["bil"])
    entries = {e["entry_id"]: e for e in (a, b)}
    write_json(cfg.json_dir / "entries.json", entries)
    write_json(cfg.json_dir / "words.json",
               {eid: {"family_id": eid, "anchor_entry_id": eid,
                      "entry_ids": [eid], "freq_rank": i + 1}
                for i, eid in enumerate(sorted(entries))})
    tdir = _tdir(cfg)
    write_json(tdir / "definitions.json", {
        "11000001:21000001": {
            "lemma": "alt-1", "gloss": "alt-1",
            "src_sha": a["senses"][0]["src_sha"],
            "provenance": "gemini:gemini-2.0-flash@2025-01-01"},
        "11000002:21000002": {
            "lemma": "alt-2", "gloss": "alt-2",
            "src_sha": b["senses"][0]["src_sha"],
            "provenance": "gemini:gemini-2.0-flash@2025-01-01"}})
    write_json(tdir / "expressions.json", {})
    return entries, tdir


def _dies_on_bil(translator):
    """Answers the `hus` batch, starves the `bil` batch until the lock fires."""
    @translator.respond
    def _answer(call):
        props = call["config"].kwargs["response_schema"]["properties"]
        n = props["definitions"]["minItems"]
        if "bil" in call["contents"][0]:
            return {"headword": "bil", "definitions": []}
        return {"headword": "hus",
                "definitions": [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                                for i in range(n)]}
    return _answer


def test_a_fatal_run_leaves_its_spend_records_on_disk(cfg, registry, translator):
    """F1. Measured before the fix: five paid calls, a count-lock FatalError,
    and reports/translate_usage.json, review/count_lock_violations_German.json
    and reports/translate_report.json ALL ABSENT -- while _generate's own
    docstring promised "a crash mid-run cannot leave a paid call unaccounted
    for". It was true in memory and false on disk, and a crash is the only
    occasion the property is for."""
    _workspace(cfg)

    @translator.respond
    def _short(call):
        return {"headword": "hus", "definitions": []}

    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True)

    n = S42.MAX_COUNT_LOCK_ATTEMPTS
    assert len(translator.calls) == n                     # n paid calls placed
    lines = (cfg.report_dir / "translate_usage.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == n                                # ...n lines, fsync'd
    rows = read_json(cfg.report_dir / "translate_usage.json")
    assert len(rows) == n
    # every one of them is attempt 1 of its own call: the count-lock ladder
    # places a NEW request each time, it does not retry inside the transport
    assert [r["attempt"] for r in rows] == [1] * n
    violations = read_json(cfg.review_dir / "count_lock_violations_German.json")
    assert len(violations) == n
    crashed = read_json(cfg.report_dir / "translate_report.json")["crashed"]
    assert crashed["error"] == "FatalError"
    assert "after 5 attempts" in crashed["message"]
    # ...and the drift ledger is still NOT consumed by a failed run
    assert not _ledger(cfg).exists()


def test_count_lock_violation_records_finish_reason(cfg, registry, translator):
    """N-08, the spec's named test. "The model dropped a sense" and "the cap
    truncated the JSON" used to be the same log line and they need opposite
    fixes, so the violation row carries the finishReason, the cap that produced
    it and the candidate tokens that came back."""
    _workspace(cfg)

    @translator.respond
    def _short(call):
        return {"headword": "hus", "definitions": []}

    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True)
    rows = read_json(cfg.review_dir / "count_lock_violations_German.json")
    assert rows and all(r["finish_reason"] == "STOP" for r in rows)
    assert all(r["max_output_tokens"] >= 1024 for r in rows)
    assert all(r["candidates_tokens"] is not None for r in rows)
    assert all(r["kind"] == "definition" and r["expected"] == 2 for r in rows)


def test_the_usage_sink_sees_every_call(cfg, registry, translator):
    """The money stack's only hook into this stage (report 6.3), and its own
    ledger has to be append-per-call for the same reason ours is."""
    _workspace(cfg)
    seen = []
    report = S42.run(cfg, registry, lang="German", confirm=True,
                     usage_sink=seen.append)
    assert len(seen) == len(translator.calls) == report["usage"]["requests"]
    fields = {"label", "kind", "model", "mode", "prompt_id", "cache_name",
              "finish_reason", "n_expected", "prompt_tokens", "cached_tokens",
              "uncached_prompt_tokens", "candidates_tokens", "tool_use_tokens",
              "thinking_tokens", "total_tokens", "attempt",
              "max_output_tokens", "prompt_sha256"}
    assert fields <= set(seen[0])
    assert all(r["cached_tokens"] <= r["prompt_tokens"] for r in seen)
    # the sha on the ledger row IS the sha on the bill
    bill = read_json(cfg.report_dir / "translate_bill_German.json")
    assert bill["prompt_sha256"]["definition"] == next(
        r["prompt_sha256"] for r in seen if r["kind"] == "definition")


def test_a_crashed_clean_redo_resumes_without_paying_twice(cfg, registry,
                                                            translator):
    """F2(b)(c). A clean redo of one language is 5,565 requests and several
    hours; before this, a crash halfway left exactly two options, and both were
    wrong: run it plain (silently roll the whole redo back to 2.0-flash) or run
    it flagged (pay for the finished half again)."""
    entries, tdir = _two_entries(cfg)
    _dies_on_bil(translator)
    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True, retranslate_all=True)
    live = read_json(tdir / "definitions.json")
    assert live["11000001:21000001"]["lemma"] == "L0"       # hus was redone
    assert "11000002:21000002" not in live                  # bil was retired
    archive = read_json(tdir / "archive.json")
    assert archive["definitions"]["11000002:21000002"]["reason"] == "clean_redo"

    # (a) the plain "carry on" restores NOTHING and bills only the remainder
    plain = S42.run(cfg, registry, lang="German", confirm=False)
    assert plain["bill"]["German"]["restored_from_archive"] == 0
    assert plain["bill"]["German"]["cells_total"] == 1
    assert read_json(tdir / "definitions.json")[
        "11000001:21000001"]["lemma"] == "L0"

    # (b) the flagged resume shows the arithmetic and bills only the remainder
    dry = S42.run(cfg, registry, lang="German", confirm=False,
                  retranslate_all=True)
    resume = dry["bill"]["German"]["resume"]
    assert resume["definitions_already_redone"] == 1
    assert dry["bill"]["German"]["cells_total"] == 1
    assert resume["done_provenance"].startswith(
        "gemini:gemini-3.7-flash+v4-frozen+LOW@")

    # (c) ...and the confirmed resume pays for one cell, not two
    before = len(translator.calls)

    @translator.respond
    def _healthy(call):
        n = call["config"].kwargs["response_schema"][
            "properties"]["definitions"]["minItems"]
        return {"headword": "x",
                "definitions": [{"lemma": "N%d" % i, "gloss": "M%d" % i}
                                for i in range(n)]}

    done = S42.run(cfg, registry, lang="German", confirm=True,
                   retranslate_all=True)
    assert len(translator.calls) - before == 1
    moved = done["languages"]["German"]["archived_for_redo"]
    assert moved["definitions"] == 0 and moved["kept"]["definitions"] == 1
    final = read_json(tdir / "definitions.json")
    assert len(final) == 2
    assert final["11000001:21000001"]["lemma"] == "L0"      # kept, not re-paid
    assert final["11000002:21000002"]["lemma"] == "N0"      # the remainder
    assert all("3.7-flash" in r["provenance"] for r in final.values())


# ------------------------------------------------------ the review subcommand

def test_review_never_runs_by_itself_and_needs_a_flag_file(cfg, registry,
                                                           translator):
    """N-02(c). The correction pass is hand-run: its work list is a file a human
    or an offline gate put there, not a model's opinion."""
    _workspace(cfg)
    dry = S42.review(cfg, registry, lang="German", confirm=False)
    assert dry["flags_on_file"] == 0 and translator.calls == []
    assert dry["note"].startswith("dry run")


def test_review_puts_the_correction_in_the_user_message(cfg, registry,
                                                        translator):
    """The correction used to be PREPENDED to the system prompt, which changes
    the cached prefix and so forfeits the discount on exactly the requests being
    redone."""
    _workspace(cfg)
    write_json(cfg.review_dir / "script_violations_German.json", [
        {"kind": "definition", "key": "11021722:21000001",
         # Two of the ten most frequent traditional-character hits in the
         # measured contamination (325 definition cells across 63 entries),
         # written as escapes so the source file stays ASCII.
         "reason": "traditional characters",
         "detected": ["\u500b", "\u52d5"]}])
    report = S42.review(cfg, registry, lang="German", confirm=True)
    assert report["redone"]["definitions"] == 1
    assert len(translator.calls) == 1
    kwargs = translator.calls[0]["config"].kwargs
    user = translator.calls[0]["contents"][0]
    assert "IMPORTANT CORRECTION" in user and "\u500b" in user
    assert "IMPORTANT CORRECTION" not in kwargs["system_instruction"]
    assert kwargs["system_instruction"] == S42.definition_prompt("German")


def test_the_bill_counts_the_grammar_field_it_now_sends(cfg, registry):
    """N-04: 38.8% of the senses on file carry a DDO grammar note and not a byte
    of it used to reach the model. It is billable input, so it is counted."""
    e = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                   senses=[make_sense("21000001", "bygning man bor i")],
                   source_words=["hus"])
    e["senses"][0]["grammar"] = "NOGET er et hus"
    _workspace(cfg, entry=e)
    report = S42.run(cfg, registry, lang="German", confirm=False)
    row = report["bill"]["German"]
    assert row["source_chars"] == len("bygning man bor i") + len("NOGET er et hus")
    bill = read_json(cfg.report_dir / "translate_bill_German.json")
    # ...and the bill file names the cell without quoting the DDO text
    assert bill["cells_by_key"]["11021722:21000001"] == "missing"
    assert "NOGET er et hus" not in (cfg.report_dir
                                     / "translate_bill_German.json").read_text(
        encoding="utf-8")


# --------------------------------------------- the bill file itself (F7/O-7)

def test_the_bill_file_carries_sizes_and_shas_not_ddo_text(cfg, registry):
    """O-7 at 28x. On a clean redo the cells block quoted all 22,282 Danish
    definitions in scope: 7.6 MB of DDO source text in a report artifact, to
    say what the counts already said. A bill needs sizes and shas."""
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=False,
            retranslate_all=True)
    path = cfg.report_dir / "translate_bill_German.json"
    raw = path.read_text(encoding="utf-8")
    for ddo_text in ("bygning man bor i", "husholdning", "hus forbi",
                     "helt forkert"):
        assert ddo_text not in raw, ddo_text
    bill = read_json(path)
    assert set(bill["cells_by_key"].values()) == {"clean_redo"}
    assert len(bill["cells_by_key"]) == bill["cells_total"]
    # per request, which is what the money math actually consumes (spec 2.2)
    reqs = bill["requests"]
    assert {r["kind"] for r in reqs} == {"definition", "expression"}
    assert sum(r["n"] for r in reqs) == bill["cells_total"]
    assert all(len(r["prompt_sha256"]) == 64 for r in reqs)
    assert {r["prompt_sha256"] for r in reqs if r["kind"] == "definition"} == \
        {bill["prompt_sha256"]["definition"]}


def test_the_bill_shows_three_dollar_figures_and_the_ceiling(cfg, registry,
                                                             probe_stats,
                                                             monkeypatch):
    """Spec 4.2(1): cache-works / LEAN-uncached / RICH-uncached, offline, with
    the accepted ceiling, four languages. The tokens are computed here from the
    measured constants; the RATE CARD is the money stack's.

    ONE LINE CHANGED BY CREW B: this used to assert that the dollar figures were
    None because ankidkdeck/prices.py did not exist yet. It exists now, so the
    figures are real, and the no-rate-card branch is asserted in
    tests/test_money.py (dollar_figures with rates=None) -- where it does not
    depend on a module being absent.

    TWO LINES CHANGED BY THE CREW-B FIXER (review round, both reviewers): the
    bill used to book thinking at 0 for EVERY kind and charge the cached input
    rate to EVERY kind. Both were wrong in the direction that under-states the
    only figure a human reads before pressing --confirm-spend, and correcting
    them moved the four-language program from $9.78 to over the cap. The old
    assertions are kept as the per-kind statements they were true of."""
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=False)
    tokens = report["bill"]["German"]["tokens"]
    assert tokens["available"] is True
    assert tokens["cache_works"]["cached_input_tokens"] > 0
    assert tokens["lean_uncached"]["cached_input_tokens"] == 0
    assert (tokens["rich_uncached"]["uncached_input_tokens"]
            > tokens["lean_uncached"]["uncached_input_tokens"])
    # thinking on the DEFINITION wave is a MEASURED zero, read off disk
    assert tokens["thinking_per_request_p95"] == 0.0
    assert tokens["thinking_basis"]["definition"] == "measured_p95"
    # ...and on a kind nobody measured it is a labelled PRIOR, not a zero. The
    # measured 0 belongs to the definition PROMPT, not to thinkingLevel=LOW: the
    # ranking prompt thought 236-275 tokens at the same level.
    assert tokens["thinking_basis"]["expression"] == \
        "unmeasured_conservative_prior"
    assert tokens["thinking_per_request_unmeasured_kinds"] > 0
    assert tokens["cache_works"]["thinking_tokens"] > 0
    # only the definition prompt clears the measured 1,024-token cache floor, so
    # the cached input rate may not be quoted for anything else
    assert tokens["cacheable_kinds"] == ["definition"]
    n_def = report["bill"]["German"]["definition_requests"]
    assert (tokens["cache_works"]["cached_input_tokens"]
            == n_def * tokens["system_tokens_lean"])
    money = report["bill"]["German"]["dollars"]
    assert money["cache_works"] < money["lean_uncached"] < money["rich_uncached"]
    assert money["rates"]["cached_input_usd_per_mtok"] == 0.075
    assert money["ceiling_usd"] == cfg.spend_cap_usd
    assert money["forbidden"] == "rich_uncached"

    # ...and a DIFFERENT rate card moves every figure: the bill reads the card,
    # it does not carry one
    import sys
    import types

    import ankidkdeck
    prices = types.ModuleType("ankidkdeck.prices")
    prices.rate_card = lambda model, mode: {"input_usd_per_mtok": 0.30,
                                            "cached_input_usd_per_mtok": 0.075,
                                            "output_usd_per_mtok": 2.50}
    monkeypatch.setitem(sys.modules, "ankidkdeck.prices", prices)
    monkeypatch.setattr(ankidkdeck, "prices", prices, raising=False)
    priced = S42.run(cfg, registry, lang="German", confirm=False)
    m = priced["bill"]["German"]["dollars"]
    assert m["cache_works"] < m["lean_uncached"] < m["rich_uncached"]
    assert m["over_ceiling"] == []
    assert m["rates"]["cached_input_usd_per_mtok"] == 0.075


def test_the_bill_sha_follows_the_prompt_that_is_actually_sent(cfg, registry,
                                                               monkeypatch):
    """F5, on the real artifact. The bill file used to read the prompt through
    its own call site, so replacing the builder -- which is exactly what the
    prompt-pack work does -- would have left G-PROMPT comparing a stale sha to
    itself and calling it agreement."""
    _workspace(cfg)
    monkeypatch.setitem(S42._SYSTEM_PROMPTS, "definition",
                        lambda lang: "PROMPT-X for %s" % lang)
    S42.run(cfg, registry, lang="German", confirm=False)
    bill = read_json(cfg.report_dir / "translate_bill_German.json")
    expected = S42.prompt_sha256("PROMPT-X for German")
    assert bill["prompt_sha256"]["definition"] == expected
    assert all(r["prompt_sha256"] == expected for r in bill["requests"]
               if r["kind"] == "definition")


def test_the_first_paid_run_freezes_the_bind_audit(cfg, registry, translator):
    """Spec 5.9. bind's n_bound / n_dropped / bind_rate are the audit of the
    2025 asset and they are computed against the LIVE tables, so a paid
    translate changes bind's answer while leaving its name alone. The last clean
    report is copied once, before the first paid call, and never rewritten."""
    _workspace(cfg)
    write_json(cfg.report_dir / "bind_report.json",
               {"per_language": {"German": {"bind_rate": 0.99, "n_bound": 7}}})
    S42.run(cfg, registry, lang="German", confirm=True)
    snap = read_json(cfg.report_dir / S42.BIND_AUDIT_SNAPSHOT)
    assert snap["report"]["per_language"]["German"]["n_bound"] == 7
    # a second paid run does NOT move the boundary
    write_json(cfg.report_dir / "bind_report.json",
               {"per_language": {"German": {"bind_rate": 0.42, "n_bound": 1}}})
    S42.run(cfg, registry, lang="German", confirm=True)
    again = read_json(cfg.report_dir / S42.BIND_AUDIT_SNAPSHOT)
    assert again["report"]["per_language"]["German"]["n_bound"] == 7


# ------------------------------------------------------- the bill-only path

def test_the_dry_path_imports_no_llm_module(cfg, registry):
    """The bill cannot place a call even by accident: nothing under google.* is
    imported on that path, so the package also installs and runs with no LLM
    dependency at all. Asserted on the import graph, not on behaviour."""
    import sys
    for name in [m for m in list(sys.modules) if m.startswith("google")]:
        del sys.modules[name]
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=False)
    S42.review(cfg, registry, lang="German", confirm=False)
    assert [m for m in sys.modules if m.startswith("google")] == []


# ------------------------------- prompt/ledger fix round: F2 and F7 (cross-owner)

def test_a_failing_script_gate_still_leaves_the_paid_run_on_disk(cfg, registry,
                                                                translator):
    """F2. G-SCRIPT used to be evaluated between the drift ledger's irreversible
    consumption and the single write of translate_report.json, and run_gates
    raises. So a wave that had already been paid for lost its report entirely --
    and left the PREVIOUS run's file on disk, describing a different run.

    Order now: evaluate, record the verdict in the report, write the report,
    then raise. The drift ledger's success-only consumption (guide 1.11) is
    unchanged: it is still written before this, only on a run that got that far.
    """
    _workspace(cfg)
    # every cell this run writes carries gemini:* provenance, so a forbidden
    # class on one of them is BLOCK tier with no baseline -- reviewer A's
    # end-to-end repro, reproduced from the gate's own findings.
    from ankidkdeck import gates
    real = gates.script_findings

    def poisoned(cells, **kw):
        found = real(cells, **kw)
        if cells and kw.get("kind") == "definitions":
            key = sorted(cells)[0]
            found = found + [{"key": key, "lang": kw.get("lang"),
                              "kind": "definitions",
                              "class": "traditional_han", "tier": gates.BLOCK,
                              "legacy": False, "fields": ["gloss"], "chars": ""}]
        return found

    gates.script_findings = poisoned
    try:
        with pytest.raises(FatalError) as err:
            S42.run(cfg, registry, lang="German", confirm=True)
    finally:
        gates.script_findings = real
    assert "G-SCRIPT" in str(err.value)

    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["script_gate_ok"] is False
    assert report["usage"]["requests"] == 2
    assert report["waves"][0]["prompt_id"] == cfg.prompt_id
    assert report["drift"]["ledger_written"] is True
    # the per-call money records, and the gate's own detail, are on disk too
    assert (cfg.report_dir / "translate_usage.jsonl").exists()
    assert read_json(cfg.report_dir / "translate_usage.json")
    verdicts = {(v["lang"], v["kind"]): v for v in report["script_gate"]}
    assert verdicts[("German", "definitions")]["ok"] is False
    assert verdicts[("German", "definitions")]["block_tier_findings"] == 1
    gates_report = read_json(cfg.report_dir / "gates_report.json")
    assert any(r["id"] == "G-SCRIPT" and not r["ok"]
               for r in gates_report["results"])


def test_a_failing_script_gate_on_the_dry_path_still_writes_the_bill(cfg,
                                                                    registry):
    """F2, the other half. The dry path exists so a human can read the bill
    before spending. One drifted cell made the gate raise ahead of the report,
    so the bill was unreadable until somebody bumped a baseline in
    registry/gates.json -- refusing to spend is right, refusing to show the
    bill is not."""
    _workspace(cfg)
    S42.run(cfg, registry, lang="German", confirm=False)
    tdir = cfg.json_dir / "translations" / "German"
    write_json(tdir / "definitions.json", {
        # a Traditional character (U+52D5) in a cell this pipeline wrote
        "11021722:21000001": {"lemma": "Haus", "gloss": "Ein \u52d5.",
                              "src_sha": "a" * 64,
                              "provenance": "gemini:x+v4-frozen+LOW@2026-08-27"}})
    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=False, do_gc=False)
    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["script_gate_ok"] is False
    assert report["bill"]["German"]["cells_total"] >= 1
    assert (cfg.report_dir / "translate_bill_German.json").exists()


def test_the_consumption_rules_actually_refuse_a_confirmed_run(cfg, registry,
                                                              translator):
    """F7 (cross-owner: this file does not own billing.py). billing.assert_ready_to_spend
    and billing.consumption_rules had NO production caller -- the only importer
    was tests/test_money.py, s42 did not import billing at all, and the paid
    path's pre-flight was transport_guard + probe_stats + cfg.validate, none of
    which evaluates rule 6. So "the artifact was measured on v4-frozen and the
    config says rich-core-1" was a rule that computed correctly, reported
    blocking=True, and was never asked at the moment it exists for.

    The refusals come first, because a run that succeeds leaves nothing to
    translate and the later cases would pass on an empty todo.
    """
    _workspace(cfg)
    stats = read_json(cfg.probe_stats_path)

    # 1. a rich prompt_id against an artifact measured on the frozen prompt.
    #    This is the rule the whole prompt-pack exercise depends on.
    cfg.prompt_id = "rich-core-1"
    with pytest.raises(FatalError) as err:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "R6-prompt-id" in str(err.value)
    cfg.prompt_id = "v4-frozen"

    # 2. constants measured before the model was published. probe_stats does not
    #    look at the date at all, so this refusal is new.
    write_json(cfg.probe_stats_path,
               dict(stats, measured_at="2026-08-12T23:59+02:00"))
    with pytest.raises(FatalError) as err:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "R2-measured-at" in str(err.value)

    # 3. the guard tools/backfill_probe_stats.py sets when the artifact is not
    #    fit to authorise a spend. doctor read it; the spend path did not.
    write_json(cfg.probe_stats_path,
               dict(stats, CONSUMPTION_GUARD="basis relabelled by hand"))
    with pytest.raises(FatalError) as err:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "R1-guard" in str(err.value)

    # 4. a missing measured constant is refused too -- by probe_stats, which
    #    R1-constants duplicates on purpose (see assert_ready_to_spend).
    write_json(cfg.probe_stats_path,
               {k: v for k, v in stats.items() if k != "EXPECTED_OUTPUT"})
    with pytest.raises(FatalError) as err:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "EXPECTED_OUTPUT" in str(err.value)

    # 5. and on the default path every blocking rule passes, with the rows in
    #    the report so a reader can see what authorised the spend.
    write_json(cfg.probe_stats_path, stats)
    report = S42.run(cfg, registry, lang="German", confirm=True)
    rules = {r["rule"]: r for r in report["consumption_rules"]}
    assert rules["R6-prompt-id"]["ok"] is True
    assert all(r["ok"] for r in rules.values() if r["blocking"]), rules
    # LEAN reads no pack, so the pack rule cannot block there
    assert rules["R6-pack-version"]["blocking"] is False
    assert report["usage"]["requests"] == 2


def test_the_bill_and_the_provenance_carry_the_pack_version(cfg, registry,
                                                           translator):
    """F6. `prompt_id` names the block family; the pack is the rest of the
    prompt text. Under the frozen prompt the effective id is the bare
    prompt_id -- LEAN reads no pack, so folding a version in would relabel cells
    whose text did not change."""
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=True)
    bill = read_json(cfg.report_dir / "translate_bill_German.json")
    assert bill["pack_version"] == "de-1"
    assert bill["effective_prompt_id"] == "v4-frozen"
    assert report["waves"][0]["effective_prompt_id"] == {"German": "v4-frozen"}
    prov = report["languages"]["German"]["provenance"]
    assert prov.startswith("gemini:gemini-3.7-flash+v4-frozen+LOW@")


# ==========================================================================
# FIXER round -- the money gates now have production call sites
# ==========================================================================
#
# Both acceptance reviewers found the same two wiring holes, and neither was a
# defect in a gate: `grep -rn pre_spend_gates src/` matched only the definition
# and its own docstring, so G-SCOPE-FROZEN and G-BUDGET could not refuse
# anything anywhere; and post_wave_gates had exactly one caller, inside the batch
# transport, so a confirmed standard or flex wave placed real calls and
# adjudicated none of them.


def test_a_confirmed_run_refuses_until_the_scope_is_refrozen(
        cfg, registry, translator, not_refrozen):
    """G-SCOPE-FROZEN, on an unsigned package (`not_refrozen`).

    Paying to translate a scope that is about to change is paying twice, so an
    unsigned package must refuse every confirmed run. That was the real state of
    a checkout until the 2026-08-27 release refreeze -- packaged card_keys `{}`,
    no stamp -- and it is the state the program returns to the next time the
    scope moves, which is why the refusal is pinned rather than deleted. What
    was wrong before this gate existed was that nothing asked. The dry path is
    deliberately unaffected: a human still has to be able to read the bill.
    """
    _workspace(cfg)
    dry = S42.run(cfg, registry, lang="German", confirm=False)
    assert dry["bill"]["German"]["cells_total"] > 0

    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "G-SCOPE-FROZEN" in str(exc.value)
    assert "no refreeze stamp" in str(exc.value)
    # and it refused BEFORE a single call was placed
    assert translator.calls == []


def test_the_review_pass_asks_the_pre_spend_gates_too(cfg, registry,
                                                      translator,
                                                      not_refrozen):
    """`review --fix` is a smaller paid path, not a free one: it draws on the same
    monthly cap and it redoes cells inside the same scope."""
    entry = _workspace(cfg)
    write_json(cfg.review_dir / "review_flags_German.json",
               [{"key": "11021722:21000001", "reasons": ["wrong gloss"]}])
    del entry
    with pytest.raises(FatalError) as exc:
        S42.review(cfg, registry, lang="German",
                   keys=["11021722:21000001"], confirm=True)
    assert "G-SCOPE-FROZEN" in str(exc.value)
    assert translator.calls == []


def test_the_review_pass_quotes_itself_so_the_budget_gate_has_a_number(
        cfg, registry, translator):
    """G-BUDGET refuses an unpriced run on purpose ("an unpriced run is not a
    cheap run"), so the review path has to carry a bill of its own."""
    _workspace(cfg)
    write_json(cfg.review_dir / "review_flags_German.json",
               [{"key": "11021722:21000001", "reasons": ["wrong gloss"]}])
    report = S42.review(cfg, registry, lang="German",
                        keys=["11021722:21000001"], confirm=True)
    quote = report["bill"]["German"]["dollars"]
    assert quote["lean_uncached"] is not None and quote["lean_uncached"] > 0
    assert report["redone"]["definitions"] == 1
    ids = {row["id"] for row in report["pre_spend_gates"]}
    assert {"G-SCOPE-FROZEN", "G-BUDGET"} == ids
    assert all(row["ok"] for row in report["pre_spend_gates"])


def test_g_budget_sums_the_languages_before_the_first_call(cfg, registry,
                                                           translator):
    """The bill's own ceiling check is PER LANGUAGE ($4.09 against $10 on the real
    corpus); four times $4.09 is $16.37, which is 164% of the cap. The only thing
    in the package that sums across languages is billing.forecast, whose only
    consumer is G-BUDGET -- so until it was wired, the cap was a number in a toml
    file."""
    _workspace(cfg)
    cfg.spend_cap_usd = 0.0001
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "G-BUDGET" in str(exc.value)
    assert "the only thing that does" in str(exc.value) \
        or "cap" in str(exc.value)
    assert translator.calls == []
    # ...and with room, the same run goes through and records the verdict
    cfg.spend_cap_usd = 10.0
    report = S42.run(cfg, registry, lang="German", confirm=True)
    budget = [r for r in report["pre_spend_gates"] if r["id"] == "G-BUDGET"][0]
    assert budget["ok"] is True
    assert budget["detail"]["forecast_usd"] > 0
    assert budget["detail"]["cap_usd"] == 10.0


def test_the_money_gates_run_on_the_interactive_surface_too(cfg, registry,
                                                            translator):
    """G-BILL / G-THINK / G-PROMPT / G-CACHE had one call site and it was inside
    the batch transport. `--mode standard --confirm-spend` adjudicated nothing at
    all, which made spec 4.2(6) false on the interactive path."""
    _workspace(cfg)
    report = S42.run(cfg, registry, lang="German", confirm=True)
    ids = {row["id"] for row in report["wave_gates"]["rows"]}
    assert {"G-BILL", "G-THINK", "G-PROMPT", "G-CACHE"} == ids
    assert report["wave_gates"]["ok"] is True
    assert report["wave_gates"]["usage_rows_by_language"]["German"][1] > 0
    # G-CACHE reports n/a rather than passing quietly: nothing on this surface
    # creates a cache, so there is no denominator and no discount was claimed
    cache = [r for r in report["wave_gates"]["rows"] if r["id"] == "G-CACHE"][0]
    assert cache["ok"] is True
    recorded = read_json(cfg.report_dir / "gates_report.json")["results"]
    assert {"G-BILL", "G-CACHE"} <= {r["id"] for r in recorded}


def test_an_interactive_wave_that_costs_more_than_its_quote_is_refused(
        cfg, registry, fake_genai, no_sleep, probe_stats):
    """The gate has to be able to fail, and the report has to reach disk first.

    The order is the one the review round asked for: a failing money gate that raised ahead
    of write_json left translate_report.json describing the PREVIOUS run.
    """
    _workspace(cfg)

    @fake_genai.respond
    def _expensive(call):
        from conftest import FakeResponse, FakeUsage
        props = call["config"].kwargs["response_schema"]["properties"]
        if "definitions" in props:
            n = props["definitions"]["minItems"]
            body = {"headword": "hus",
                    "definitions": [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                                    for i in range(n)]}
        elif "fixed_expressions" in props:
            n = props["fixed_expressions"]["minItems"]
            body = {"fixed_expressions": [{"lemma": "X%d" % i,
                                           "gloss": "Y%d" % i}
                                          for i in range(n)]}
        else:
            body = {k: "POS-%s" % k for k in props}
        import json as _json
        return FakeResponse(_json.dumps(body),
                            usage=FakeUsage(prompt=1135, candidates=4000))

    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "G-BILL" in str(exc.value)
    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["confirmed"] is True
    assert report["wave_gates"]["ok"] is False
    assert report["usage"]["candidates_tokens"] > 0, \
        "the paid wave's own numbers reached disk before the failure continued"
