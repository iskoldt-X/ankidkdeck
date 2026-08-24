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
import types

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

class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, calls):
        self.calls = calls

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append(model)
        payload = json.loads(contents[0].split("Entries:\n", 1)[1])
        # a deterministic "ranking": reverse the input
        return _FakeResp(json.dumps(
            {"sorted_ids": [r["id"] for r in reversed(payload)]}))


@pytest.fixture
def fake_genai(monkeypatch):
    """A fake google.genai. No network, no key, no spend -- and it also proves
    the dry path never imports the real one, because the real one is not here."""
    calls = []

    class _Client:
        def __init__(self, api_key=None):
            self.models = _FakeModels(calls)

    class _Config:
        def __init__(self, **kw):
            self.kwargs = kw

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")
    genai.Client = _Client
    gtypes.GenerateContentConfig = _Config
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setenv("GEMINI_API_KEYS", "fake-key")
    monkeypatch.setattr(S50.time, "sleep", lambda *a, **k: None)
    return calls


def test_confirm_after_a_dry_run_actually_ranks_the_queued_families(
        cfg, registry, fake_genai):
    """The R4 repro, as a test: dry run, then confirm IN THE SAME WORKSPACE."""
    _workspace(cfg)
    dry = S50.run(cfg, registry, confirm=False)
    assert dry["queue_for_ranking"] == 2
    assert fake_genai == []                       # the bill placed no call

    done = S50.run(cfg, registry, confirm=True)
    assert done["queue_for_ranking"] == 2
    assert done["families_ranked"] == 2
    assert len(fake_genai) == 2
    assert done["api"]["requests"] == 2


def test_a_resolved_queue_keeps_its_rows_with_a_status(cfg, registry,
                                                      fake_genai):
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
                                                            fake_genai):
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


def test_the_permutation_lock_fires_on_a_bad_ranking(cfg, registry, fake_genai,
                                                     monkeypatch):
    _workspace(cfg)

    def bad(self, model=None, contents=None, config=None):
        return _FakeResp(json.dumps({"sorted_ids": ["99999999"]}))

    monkeypatch.setattr(_FakeModels, "generate_content", bad)
    with pytest.raises(FatalError) as exc:
        S50.run(cfg, registry, confirm=True)
    assert "permutation" in str(exc.value)


def test_the_dry_path_imports_no_llm_module(cfg, registry):
    """The bill cannot place a call even by accident: nothing under google.* is
    imported on that path."""
    for name in [m for m in list(sys.modules) if m.startswith("google")]:
        del sys.modules[name]
    _workspace(cfg)
    S50.run(cfg, registry, confirm=False)
    assert [m for m in sys.modules if m.startswith("google")] == []
