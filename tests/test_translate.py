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
    rows = [r for r in read_json(
        cfg.report_dir / "translate_bill_German.json")["cells"]
        if r["key"] == "11021722:21000001"]
    assert rows and rows[0]["reason"] == "missing"
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

def test_the_retry_budgets_match_v2_1():
    """One flaky response used to abort a paid multi-hour run: 64 calls were
    spent before the contamination FATAL."""
    assert S42.MAX_CORRECTION_ATTEMPTS == 5
    assert S42.MAX_COUNT_LOCK_ATTEMPTS == 5
    assert S42.MAX_RETRIES == 5


def test_the_bill_quotes_the_worst_case_request_count():
    todo = [{"kind": "expression", "entry_id": "1", "text": "a", "hint": "",
             "reason": "missing"}]
    row = S42.bill_row(todo, [])
    assert row["requests_min"] == 2
    assert row["requests_max"] == 2 * S42.MAX_CORRECTION_ATTEMPTS
