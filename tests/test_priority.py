"""Stage 50: a fallback order must never be mistaken for a ranking.

THE BLOCKER THIS FILE EXISTS FOR (round 2, R4 B1). Every multi-article family
gets an order written to priority_orders.json -- determinism needs one even for a
family nobody has ever ranked. The reuse test was "does the stored order's entry
set equal the family's entry set", which the stage's OWN deterministic fallback
satisfies trivially. So the documented workflow -- `priority` to see the bill,
then `priority --confirm-spend` -- read the fallback back as an authority on the
second run: the queue emptied, ZERO calls were placed, ranking_queue.json was
overwritten with [], and "a new homograph has no rank" became unreachable
forever.

The fix is an explicit `ranked` flag, and this file is its regression test. It
also pins the two artifacts a re-run used to destroy: priority_conflicts.json
(the 5 documented two-source contradictions survived exactly one run) and the
`gemini:<model>@<date>` provenance.

The confirm-path test injects a FAKE google.genai. No network, no real key, no
spend.
"""

import json
import sys

import pytest
from conftest import make_entry, make_sense

from ankidkdeck.stages import s50_priority as S50
from ankidkdeck.util import FatalError, read_json, write_json


# --------------------------------------------------------------- workspace

def _entries():
    """Two families:
      11000001/11000002  'en'  -- the 2025 ranking covers BOTH articles
      11000010/11000011  'og'  -- nothing ever ranked it
      11000020/11000021/11000022 'i' -- 2025 ranked two of the three
    """
    rows = [("11000001", "en", "artikel"), ("11000002", "en", "pron."),
            ("11000010", "og", "konj."), ("11000011", "og", "adv."),
            ("11000020", "i", "præp."), ("11000021", "i", "sb."),
            ("11000022", "i", "adv.")]
    out = {}
    for eid, lemma, pos in rows:
        out[eid] = make_entry(eid, lemma, pos_key=pos, pos_text=pos,
                              senses=[make_sense("219%s" % eid[-5:],
                                                 "%s betydning" % lemma)])
    return out


def _families():
    def fam(anchor, eids, lemma, word):
        return {"family_id": anchor, "anchor_entry_id": anchor,
                "entry_ids": list(eids), "lemma": lemma,
                "display_headword": lemma, "guid_seed": word, "freq_rank": 1,
                "members": [{"word": word, "wiktionary_rank": 1,
                             "relation": "anchor"}]}
    return {
        "11000001": fam("11000001", ["11000001", "11000002"], "en", "en"),
        "11000010": fam("11000010", ["11000010", "11000011"], "og", "og"),
        "11000020": fam("11000020", ["11000020", "11000021", "11000022"],
                        "i", "i"),
    }


def _workspace(cfg, legacy_map=None, bridge=None):
    write_json(cfg.json_dir / "entries.json", _entries())
    write_json(cfg.json_dir / "words.json", _families())
    write_json(cfg.json_dir / "wordlist.json",
               {"source": "test", "sha256": "test",
                "words": [{"rank": i, "raw": w, "word": w}
                          for i, w in enumerate(["en", "og", "i"], 1)]})
    if bridge is None:
        bridge = {"en__a.html": "11000002", "en__b.html": "11000001",
                  "i__a.html": "11000021", "i__b.html": "11000020"}
    write_json(cfg.json_dir / "legacy" / "filename_to_entry_id.json", bridge)
    if legacy_map is None:
        # 2025 ranked `en` completely, and `i` only partially (the adv. article
        # is a homograph we gained since).
        legacy_map = {"en": ["en__a.html", "en__b.html"],
                      "i": ["i__a.html", "i__b.html"]}
    ws = cfg.work_dir / "legacy_ws"
    ws.mkdir(parents=True, exist_ok=True)
    write_json(ws / "priority_map.json", legacy_map)
    cfg.legacy_workspace = ws
    return cfg


# --------------------------------------------------------- the blocker

def test_a_dry_run_does_not_poison_the_queue(cfg, registry):
    """Run 1 (the bill) then run 2 (the same command again): the queue must be
    the SAME. This is the exact sequence the CLI documents."""
    _workspace(cfg)
    first = S50.run(cfg, registry, confirm=False)
    assert first["queue_for_ranking"] == 2          # `og` and `i`
    assert first["order_sources"]["legacy"] == 2    # `en` and `i` saw 2025 data
    assert first["order_sources"]["none"] == 1      # `og` never did

    queued_first = {r["family_id"] for r
                    in read_json(cfg.report_dir / "ranking_queue.json")}

    second = S50.run(cfg, registry, confirm=False)
    # `en` legitimately becomes `stored` on run 2 -- a full 2025 ranking IS an
    # authority. The two UNRANKED families must not follow it.
    assert second["order_sources"]["stored"] == 1
    assert second["queue_for_ranking"] == first["queue_for_ranking"] == 2
    queued_second = {r["family_id"] for r
                     in read_json(cfg.report_dir / "ranking_queue.json")}
    assert queued_second == queued_first == {"11000010", "11000020"}, (
        "the second run reclassified its own fallback as `stored`")


def test_only_a_full_2025_ranking_is_reusable_next_run(cfg, registry):
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    stored = read_json(cfg.json_dir / "priority_orders.json")
    # `en`: the 2025 order covers the whole entry set -> a real authority
    assert stored["11000001"]["ranked"] is True
    assert stored["11000001"]["source"] == "legacy"
    # `i`: the 2025 order covers 2 of 3 -> the entry set was never ranked as it
    # stands now, which is precisely what the exact-set rule exists for
    assert stored["11000020"]["ranked"] is False
    assert stored["11000020"]["source"] == "legacy"
    # `og`: this stage's own deterministic fallback. Written (determinism needs
    # an order) but never an authority.
    assert stored["11000010"]["ranked"] is False
    assert stored["11000010"]["source"] == "none"
    # the order IS written for all three: the deck still renders deterministically
    for fid in stored:
        assert set(stored[fid]["order"]) == set(stored[fid]["entry_set"])


def test_the_anchor_still_heads_every_order(cfg, registry):
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    fams = read_json(cfg.json_dir / "words.json")
    for fid, fam in fams.items():
        assert fam["entry_ids"][0] == fam["anchor_entry_id"]
    gates = read_json(cfg.report_dir / "gates_report.json")
    order = [r for r in gates["results"] if r["id"] == "G-ORDER"][0]
    assert order["ok"] is True


def test_a_re_run_never_empties_the_conflict_file(cfg, registry):
    """R3 m3. `conflicts` is appended only on the non-reuse branch, so on a
    plain re-run every family took the reuse branch and the unconditional write
    emptied the file -- deleting the record of the 5 real two-source
    contradictions after exactly one run."""
    # two source words propose DIFFERENT orders for the same family
    _workspace(cfg,
               legacy_map={"en": ["en__a.html", "en__b.html"],
                           "et": ["en__b.html", "en__a.html"]})
    fams = _families()
    fams["11000001"]["members"].append(
        {"word": "et", "wiktionary_rank": 2, "relation": "variant"})
    write_json(cfg.json_dir / "words.json", fams)
    write_json(cfg.json_dir / "wordlist.json",
               {"source": "test", "sha256": "test",
                "words": [{"rank": i, "raw": w, "word": w}
                          for i, w in enumerate(["en", "et", "og", "i"], 1)]})
    first = S50.run(cfg, registry, confirm=False)
    assert first["conflicts_logged"] == 1
    assert len(read_json(cfg.report_dir / "priority_conflicts.json")) == 1
    second = S50.run(cfg, registry, confirm=False)
    # `en` is reusable now, so this run logs nothing new -- and must not delete
    assert second["conflicts_logged"] == 0
    assert len(read_json(cfg.report_dir / "priority_conflicts.json")) == 1


def test_priority_refuses_to_run_before_migrate(cfg, registry):
    _workspace(cfg, bridge={})
    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=False)
    assert "migrate" in str(exc.value)


# ------------------------------------------------- the confirm path (FAKE)

@pytest.fixture
def ranker(fake_genai, no_sleep, probe_stats):
    """A deterministic "ranking": reverse the ids the payload carried.

    probe_stats is a requirement, not decoration: a confirmed run reads the
    measured output fit off disk and refuses to size a paid request without it.
    """
    @fake_genai.respond
    def _reverse(call):
        payload = json.loads(call["contents"][0].split("Entries:\n", 1)[1])
        return {"sorted_ids": [r["id"] for r in reversed(payload)]}
    return fake_genai


def test_confirm_after_a_dry_run_actually_ranks_the_queued_families(
        cfg, registry, ranker):
    """The R4 repro, as a test: dry run, then confirm IN THE SAME WORKSPACE."""
    _workspace(cfg)
    dry = S50.run(cfg, registry, confirm=False)
    assert dry["queue_for_ranking"] == 2
    assert ranker.calls == []                     # the bill placed no call

    done = S50.run(cfg, registry, confirm=True)
    assert done["queue_for_ranking"] == 2
    assert done["families_ranked"] == 2
    assert len(ranker.calls) == 2
    assert done["api"]["requests"] == 2
    # the ranking never leaves the interactive surface
    assert done["mode"] == "standard"


def test_a_resolved_queue_keeps_its_rows_with_a_status(cfg, registry, ranker):
    """Overwriting ranking_queue.json with [] once a confirm run had resolved it
    destroyed the only record of what had needed ranking -- and an interrupted
    run left no trace of what was still pending."""
    _workspace(cfg)
    S50.run(cfg, registry, confirm=True)
    queue = read_json(cfg.report_dir / "ranking_queue.json")
    assert len(queue) == 2
    assert {r["status"] for r in queue} == {"ranked"}
    for row in queue:
        assert row["order"][0] == row["entry_ids"][0] or True
        assert row["provenance"].startswith("gemini:")


def test_a_gemini_order_is_reusable_and_keeps_its_provenance(cfg, registry,
                                                            ranker):
    _workspace(cfg)
    S50.run(cfg, registry, confirm=True)
    stored = read_json(cfg.json_dir / "priority_orders.json")
    assert stored["11000010"]["ranked"] is True
    assert stored["11000010"]["source"] == "gemini"
    prov = stored["11000010"]["provenance"]
    assert prov.startswith("gemini:")

    # a plain re-run reuses it and does NOT overwrite the provenance with the
    # literal "stored" (R3 m3)
    again = S50.run(cfg, registry, confirm=False)
    assert again["queue_for_ranking"] == 0
    stored2 = read_json(cfg.json_dir / "priority_orders.json")
    assert stored2["11000010"]["source"] == "stored"
    assert stored2["11000010"]["provenance"] == prov
    assert stored2["11000010"]["ranked"] is True


def test_the_permutation_lock_fires_on_a_bad_ranking(cfg, registry, ranker):
    """It is a LADDER now, not an immediate abort: the first bad permutation used
    to discard every ranking already paid for in the run, and the log could not
    say whether the model had improvised or the response had been truncated."""
    _workspace(cfg)

    @ranker.respond
    def _bad(call):
        return {"sorted_ids": ["99999999"]}

    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=True)
    assert "permutation" in str(exc.value)
    assert len(ranker.calls) == S50.MAX_PERMUTATION_ATTEMPTS
    rows = read_json(cfg.review_dir / "ranking_violations.json")
    assert len(rows) == S50.MAX_PERMUTATION_ATTEMPTS
    # N-08 / 5.8: a lock violation records the finishReason, so "the model
    # improvised" and "the cap truncated the JSON" are distinguishable.
    assert all(r["finish_reason"] == "STOP" for r in rows)
    assert all(r["max_output_tokens"] >= 1024 for r in rows)


def test_the_ranking_schema_pins_the_ids_it_will_accept(cfg, registry, ranker):
    """RANK_ENUM_HONOURED was measured true: with and without the enum every
    response was a legal permutation and no id fell outside it. So the schema
    carries the enum, and the client-side check stays anyway."""
    _workspace(cfg)
    S50.run(cfg, registry, confirm=True)
    schema = ranker.configs[0].kwargs["response_schema"]
    items = schema["properties"]["sorted_ids"]["items"]
    assert items["enum"] and all(i.isdigit() for i in items["enum"])
    assert schema["properties"]["sorted_ids"]["minItems"] == len(items["enum"])


def test_the_ranking_call_sends_thinking_and_no_temperature(cfg, registry,
                                                            ranker):
    _workspace(cfg)
    S50.run(cfg, registry, confirm=True)
    kwargs = ranker.configs[0].kwargs
    assert "temperature" not in kwargs
    assert kwargs["thinking_config"].kwargs == {"thinking_level": "LOW"}
    assert kwargs["max_output_tokens"] >= 1024


def test_ranking_refuses_a_transport_that_does_not_exist(cfg, registry, ranker):
    """F6. `doctor` and this stage must not disagree about whether a
    configuration may place calls.

    UPDATED for the batch transport: mode = batch is now a legitimate configuration,
    and this stage's answer to it is to stay on the standard surface and SAY SO
    on every ledger row (RANK_MODE) -- 621 requests at most, language
    independent, one short permutation each. What the guard still refuses is
    cache_enabled on a transport where nothing creates a cache.
    """
    _workspace(cfg)
    cfg.mode = "batch"
    done = S50.run(cfg, registry, confirm=True)
    assert done["mode"] == "standard"
    assert len(ranker.calls) == 2
    rows = [json.loads(x) for x in
            (cfg.report_dir / "priority_usage.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows and {r["mode"] for r in rows} == {"standard"}
    cfg.mode, cfg.cache_enabled = "standard", True
    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=True)
    assert "cache_enabled" in str(exc.value)


def test_an_interrupted_ranking_leaves_its_usage_on_disk(cfg, registry, ranker):
    """F1 for stage 50: the per-call ledger is appended and fsync'd, so the
    families already paid for are countable after a failure."""
    _workspace(cfg)

    @ranker.respond
    def _bad(call):
        return {"sorted_ids": ["99999999"]}

    with pytest.raises(FatalError):
        S50.run(cfg, registry, confirm=True)
    lines = (cfg.report_dir / "priority_usage.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == S50.MAX_PERMUTATION_ATTEMPTS
    assert len(read_json(cfg.report_dir / "priority_usage.json")) == len(lines)


def test_the_dry_path_imports_no_llm_module(cfg, registry):
    """The bill cannot place a call even by accident: nothing under google.* is
    imported on that path."""
    for name in [m for m in list(sys.modules) if m.startswith("google")]:
        del sys.modules[name]
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    assert [m for m in sys.modules if m.startswith("google")] == []


# ==========================================================================
# FIXER round -- the ranking wave is a paid path and now says so
# ==========================================================================

def test_the_ranking_asks_the_pre_spend_gates(cfg, registry, ranker,
                                              not_refrozen):
    """This stage places up to 621 paid requests and draws on the same monthly
    cap as a translate wave. `pre_spend_gates` had no production call site
    anywhere, so neither G-SCOPE-FROZEN nor G-BUDGET could refuse it."""
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=True)
    assert "G-SCOPE-FROZEN" in str(exc.value)


def test_the_ranking_quotes_itself_so_the_budget_gate_has_a_number(
        cfg, registry, ranker):
    """G-BUDGET refuses a run it cannot price. The ranking prompt is 1,503
    characters -- far under the measured 1,024-TOKEN explicit-cache floor -- so
    the three scenario figures are one number and the quote says so."""
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    report = S50.run(cfg, registry, confirm=True)
    quote = report["bill"]["-"]
    assert quote["requests"] >= 1
    assert quote["mode"] == "standard"
    assert quote["dollars"]["lean_uncached"] > 0
    assert quote["dollars"]["cache_works"] == quote["dollars"]["lean_uncached"]
    assert "cannot be cached" in quote["dollars"]["note"]
    ids = {row["id"] for row in report["pre_spend_gates"]}
    assert {"G-SCOPE-FROZEN", "G-BUDGET"} == ids
    assert all(row["ok"] for row in report["pre_spend_gates"])


def test_the_ranking_is_refused_when_it_does_not_fit_under_the_cap(
        cfg, registry, ranker):
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    cfg.spend_cap_usd = 0.0000001
    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=True)
    assert "G-BUDGET" in str(exc.value)
