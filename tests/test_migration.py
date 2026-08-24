"""Stage 40's re-key, and stage 41's accounting: the two files whose silent
drift picks a different translation for 40-50% of multi-file cells.

The guide's requirement (4.14) is three properties, and none of them was tested:

  1. THE TIE-BREAK IS DETERMINISTIC. Two runs over a SHUFFLED input produce
     identical output. Without a written rule the winner among 2-6 candidate
     translations comes from dict iteration order, and the build is simply not
     reproducible -- while looking perfectly healthy.
  2. THE DISCARD FILE IS COMPLETE. Every loser is written out, or "we picked one"
     is unauditable.
  3. n_bound + n_dropped == n_legacy, with every drop carrying a reason code from
     the closed set (stage 41's G-BIND).

These run on synthetic dicts: the rule is what is under test, not the corpus. The
full-corpus reproduction of the published digits (5,259 bridged / 3,812 ids /
22,734 texts) needs the recovered 2025 workspace and lives in the release
checklist, not in CI.
"""

import random

import pytest
from conftest import (make_entry, make_expression, make_sense, write_workspace)

from ankidkdeck.stages.s40_migrate import _order_key, rekey
from ankidkdeck.util import FatalError, canonical_json, read_json, write_json


# --------------------------------------------------------- the tie-break

def _download_map():
    """Three files for one entry_id. hus.html is the lemma's own file, so it
    wins clause 1; the other two tie on clause 1 and are separated by rank."""
    return {
        "hus__2.html": {"query_word": "huset", "query_rank": 900},
        "hus.html": {"query_word": "hus", "query_rank": 1},
        "hus__3.html": {"query_word": "husene", "query_rank": 40},
    }


def _headwords():
    return {fn: "hus" for fn in _download_map()}


def _keyfn():
    dm, hw = _download_map(), _headwords()
    return lambda fn: _order_key(fn, dm, hw)


def _files_by_eid(order=None):
    files = list(order or _download_map())
    files.sort(key=_keyfn())
    return {"11021722": files}


def test_the_written_tie_break_prefers_the_lemmas_own_file_then_frequency():
    keyfn = _keyfn()
    ordered = sorted(_download_map(), key=keyfn)
    # clause 1: the file whose query_word IS the headword
    assert ordered[0] == "hus.html"
    # clause 2: then the lower wiktionary rank
    assert ordered[1:] == ["hus__3.html", "hus__2.html"]


def test_two_runs_over_a_shuffled_input_produce_identical_output():
    """The property the guide names. dict order in Python follows insertion
    order, so a shuffled input is exactly how the non-determinism would show."""
    old_base = {
        "hus.html": {"bygning": {"lemma": "Haus", "gloss": "aus dem Lemma"}},
        "hus__3.html": {"bygning": {"lemma": "Gebaeude", "gloss": "aus 40"}},
        "hus__2.html": {"bygning": {"lemma": "Bau", "gloss": "aus 900"}},
    }
    outs, discards, stats = [], [], []
    for seed in range(6):
        names = list(old_base)
        random.Random(seed).shuffle(names)
        old = {n: old_base[n] for n in names}
        fbe = _files_by_eid(names)
        o, d, s = rekey(old, fbe, _keyfn())
        outs.append(canonical_json(o))
        discards.append(canonical_json(d))
        stats.append(canonical_json(s))
    assert len(set(outs)) == 1, "rekey() is not order-independent"
    assert len(set(discards)) == 1
    assert len(set(stats)) == 1
    # ...and the winner is the one the WRITTEN rule names, not the first key
    won = read_back(outs[0])
    assert won["11021722"]["bygning"]["lemma"] == "Haus"
    assert won["11021722"]["bygning"]["provenance"] == "migrated:2025:hus.html"


def read_back(json_text):
    import json
    return json.loads(json_text)


def test_every_loser_is_written_to_the_discard_file():
    old = {
        "hus.html": {"bygning": {"lemma": "Haus", "gloss": "a"}},
        "hus__3.html": {"bygning": {"lemma": "Gebaeude", "gloss": "b"}},
        "hus__2.html": {"bygning": {"lemma": "Bau", "gloss": "c"}},
    }
    out, discarded, stats = rekey(old, _files_by_eid(), _keyfn())
    assert stats["cells_in"] == 3 and stats["cells_out"] == 1
    assert stats["multi_candidate"] == 1 and stats["conflicts"] == 1
    assert len(discarded) == 1
    row = discarded[0]
    assert row["winner_file"] == "hus.html" and row["conflict"] is True
    # the two losers, both of them, with what they said
    assert sorted(l["file"] for l in row["losers"]) == ["hus__2.html",
                                                        "hus__3.html"]
    assert sorted(l["lemma"] for l in row["losers"]) == ["Bau", "Gebaeude"]


def test_identical_candidates_are_not_a_conflict_but_are_still_recorded():
    same = {"lemma": "Haus", "gloss": "ein Gebaeude"}
    old = {"hus.html": {"bygning": dict(same)},
           "hus__2.html": {"bygning": dict(same)},
           "hus__3.html": {"bygning": dict(same)}}
    out, discarded, stats = rekey(old, _files_by_eid(), _keyfn())
    assert stats["multi_candidate"] == 1
    assert stats["conflicts"] == 0          # nothing to disagree about
    assert len(discarded) == 1              # still auditable


def test_a_conflict_the_rule_cannot_separate_is_counted_as_byte_order():
    """The 4 same-word re-downloads (hiv, gloria, ram, rom): identical
    query_word AND identical query_rank, so clauses 1 and 2 both tie and only
    byte order remains. That is reproducible, so it is BASELINED (G-TIE's
    tie_break_byte_order_max) rather than forbidden -- but it must be counted."""
    dm = {"gloria__b.html": {"query_word": "gloria", "query_rank": 2000},
          "gloria__c.html": {"query_word": "gloria", "query_rank": 2000}}
    hw = {fn: "gloria" for fn in dm}
    keyfn = lambda fn: _order_key(fn, dm, hw)      # noqa: E731
    files = sorted(dm, key=keyfn)
    old = {"gloria__b.html": {"aere": {"lemma": "Ehre", "gloss": "b"}},
           "gloria__c.html": {"aere": {"lemma": "Ruhm", "gloss": "c"}}}
    out, discarded, stats = rekey(old, {"11009999": files}, keyfn)
    assert stats["conflicts"] == 1
    assert stats["unresolved_conflicts"] == 1
    assert out["11009999"]["aere"]["provenance"] == \
        "migrated:2025:gloria__b.html"


def test_cells_in_unbridged_files_are_counted_not_silently_lost():
    old = {"hus.html": {"bygning": {"lemma": "Haus", "gloss": "a"}},
           "ghost.html": {"noget": {"lemma": "X", "gloss": "y"}}}
    out, discarded, stats = rekey(old, {"11021722": ["hus.html"]}, _keyfn())
    assert stats["cells_in_unbridged_files"] == 1
    assert "11021722" in out and len(out) == 1


def test_the_src_sha_is_the_sha_of_the_danish_key_text():
    from ankidkdeck.util import NFC, sha256_str
    old = {"hus.html": {"bygning": {"lemma": "Haus", "gloss": "a"}}}
    out, _, _ = rekey(old, {"11021722": ["hus.html"]}, _keyfn())
    assert out["11021722"]["bygning"]["src_sha"] == sha256_str(NFC("bygning"))


# ------------------------------------------------- stage 41's accounting

def _bind_workspace(cfg, registry, legacy_defs, legacy_exprs, entries,
                    classification):
    cfg.langs = ["German"]
    write_workspace(cfg, entries, [(1, "hus")], classification=classification)
    write_json(cfg.json_dir / "legacy" / "legacy_German_definitions.json",
               legacy_defs)
    write_json(cfg.json_dir / "legacy" / "legacy_German_expressions.json",
               legacy_exprs)
    write_json(cfg.json_dir / "words.json",
               {"11021722": {"family_id": "11021722",
                             "anchor_entry_id": "11021722",
                             "entry_ids": list(entries)}})
    from ankidkdeck.stages.s41_bind import run as bind_run
    return bind_run(cfg, registry)


def test_n_bound_plus_n_dropped_equals_n_legacy(cfg, registry):
    e = make_entry("11021722", "hus", pos_key="sb.",
                   senses=[make_sense("21000001", "bygning man bor i")],
                   expressions=[make_expression("21000002", "hus forbi",
                                                "helt forkert")],
                   source_words=["hus"])
    entries = {e["entry_id"]: e}
    classification = {"hus": {"members": [{"entry_id": "11021722",
                                           "bucket": "exact_cs",
                                           "demoted": False}],
                              "xrefs": [], "rejected": [],
                              "resolved_by": "forward"}}
    legacy_defs = {"11021722": {
        "bygning man bor i": {"lemma": "Haus", "gloss": "Gebaeude"},
        "en tekst DDO har rettet": {"lemma": "X", "gloss": "y"},
    }}
    legacy_exprs = {"11021722": {
        "hus forbi": {"lemma": "voll daneben", "gloss": "ganz falsch"},
    }}
    report = _bind_workspace(cfg, registry, legacy_defs, legacy_exprs, entries,
                             classification)
    s = report["per_language"]["German"]
    assert s["n_legacy"] == 3
    assert s["n_bound"] + s["n_dropped"] == s["n_legacy"]
    assert s["n_unexplained"] == 0
    assert set(s["reasons"]) <= set(report["drop_reason_codes"])
    # the surviving rows are keyed on dannetid now
    defs = read_json(cfg.json_dir / "translations" / "German" / "definitions.json")
    assert defs["11021722:21000001"]["lemma"] == "Haus"


def test_a_changed_sense_text_is_split_by_reason_code(cfg, registry):
    """Guide 4.10 lists an expression-sense fallback for DEFINITIONS that was
    adjudicated out on measurement (0 of 22,728 English keys). The split is the
    PROOF, run on every corpus instead of inherited: a non-zero
    expression_sense_match is the signal to reinstate the fallback."""
    x = make_expression("21000012", "hus forbi", "helt forkert")
    e = make_entry("11021722", "hus", pos_key="sb.",
                   senses=[make_sense("21000011", "bygning")],
                   expressions=[x], source_words=["hus"])
    entries = {e["entry_id"]: e}
    classification = {"hus": {"members": [{"entry_id": "11021722",
                                           "bucket": "exact_cs",
                                           "demoted": False}],
                              "xrefs": [], "rejected": [],
                              "resolved_by": "forward"}}
    legacy_defs = {"11021722": {
        # matches nothing at all
        "en tekst DDO har rettet": {"lemma": "X", "gloss": "y"},
        # matches an EXPRESSION's sub-definition, not an article sense
        "helt forkert": {"lemma": "ganz falsch", "gloss": "voll daneben"},
    }}
    report = _bind_workspace(cfg, registry, legacy_defs, {}, entries,
                             classification)
    split = report["sense_text_changed_split"]["German"]
    assert split["no_match"] == 1
    assert split["expression_sense_match"] == 1
    dropped = read_json(cfg.json_dir / "translations" / "German" / "dropped.json")
    subs = {d["text"]: d.get("sub_reason") for d in dropped}
    assert subs["helt forkert"] == "expression_sense_match"
    assert subs["en tekst DDO har rettet"] == "no_match"


def test_a_rejected_article_is_a_reason_code_not_an_unexplained_loss(cfg,
                                                                    registry):
    e = make_entry("11022726", "I", pos_key="symbol",
                   senses=[make_sense("21000020", "kemisk tegn for jod")],
                   source_words=["i"])
    entries = {e["entry_id"]: e}
    # every word that saw it rejected it -> rejected_everywhere
    classification = {"i": {"members": [], "xrefs": [],
                            "rejected": [{"entry_id": "11022726",
                                          "headword": "I",
                                          "pos_key": "symbol",
                                          "reason": "case_only_demoted_pos"}],
                            "resolved_by": None}}
    legacy_defs = {"11022726": {"kemisk tegn for jod": {"lemma": "Jod",
                                                        "gloss": "Element"}}}
    write_json(cfg.json_dir / "words.json", {})
    report = _bind_workspace(cfg, registry, legacy_defs, {}, entries,
                             classification)
    s = report["per_language"]["German"]
    assert s["reasons"] == {"rejected_article": 1}
    assert s["n_bound"] == 0 and s["n_dropped"] == 1 and s["n_unexplained"] == 0


def test_bind_refuses_to_run_before_migrate(cfg, registry):
    from ankidkdeck.stages.s41_bind import run as bind_run
    write_workspace(cfg, {"11021722": make_entry("11021722", "hus")},
                    [(1, "hus")], classification={})
    with pytest.raises(FatalError) as exc:
        bind_run(cfg, registry)
    assert "migrate" in str(exc.value)
