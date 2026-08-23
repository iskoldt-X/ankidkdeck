"""The merge: connected components, refusal, unique assignment, dense rank.

The merge unit is a connected component of the (word, entry_id) graph, not a
"duplicate group" loop: 37 query words sit in two groups at once (`have` heads
both have-1 sb. and have-2 vb.) and a group loop emits two cards for them and
then collides on the family id.

This runs the real stage in a tmp workspace, so the gates run too -- a failure
here is a FatalError from run_gates, not a soft assertion.
"""

import pytest
from conftest import make_entry, make_expression, make_sense, write_workspace

from ankidkdeck.stages.s30_merge import UnionFind, run as merge_run
from ankidkdeck.util import read_json


def _members(entry_id, bucket):
    return {"entry_id": entry_id, "bucket": bucket, "demoted": False}


@pytest.fixture
def workspace(cfg, registry):
    hus = make_entry("11021722", "hus", pos_key="sb.", pos_text="substantiv",
                     forms=["huse", "huset", "husene"],
                     senses=[make_sense("2100030%d" % i, "hus sense %d" % i)
                             for i in range(3)],
                     source_words=["hus", "huse"])
    hav = make_entry("11000010", "hav", pos_key="sb.", pos_text="substantiv",
                     senses=[make_sense("21000400", "stort saltvand")],
                     source_words=["have"])
    have = make_entry("11000011", "have", pos_key="vb.", pos_text="verbum",
                      forms=["har", "havde", "haft"],
                      senses=[make_sense("2100050%d" % i, "have sense %d" % i)
                              for i in range(5)],
                      source_words=["have"])
    tom = make_entry("11000012", "tom", pos_key="adj.", source_words=["tom"])
    entries = {e["entry_id"]: e for e in (hus, hav, have, tom)}
    classification = {
        "hus": {"members": [_members("11021722", "exact_cs")], "xrefs": [],
                "rejected": [], "resolved_by": "forward"},
        "huse": {"members": [_members("11021722", "form")], "xrefs": [],
                 "rejected": [], "resolved_by": "forward"},
        "have": {"members": [_members("11000011", "exact_cs"),
                             _members("11000010", "form")],
                 "xrefs": [], "rejected": [], "resolved_by": "forward"},
        "tom": {"members": [_members("11000012", "exact_cs")], "xrefs": [],
                "rejected": [], "resolved_by": "forward"},
    }
    write_workspace(cfg, entries, [(1, "hus"), (40, "huse"), (60, "have"),
                                   (70, "tom")],
                    classification=classification,
                    v2_querywords={"hus": 1})
    report = merge_run(cfg, registry)
    return {"cfg": cfg, "registry": registry, "report": report,
            "families": read_json(cfg.json_dir / "words.json"),
            "assignments": read_json(cfg.json_dir / "assignments.json")}


def test_union_find_keeps_the_two_sides_apart():
    uf = UnionFind()
    uf.union(("W", "hus"), ("E", "11021722"))
    uf.union(("W", "huse"), ("E", "11021722"))
    comps = uf.components()
    assert comps == [(["hus", "huse"], ["11021722"])]


def test_two_words_sharing_an_entry_become_one_family(workspace):
    fams = workspace["families"]
    hus = fams["11021722"]
    assert sorted(m["word"] for m in hus["members"]) == ["hus", "huse"]
    assert hus["merge_state"] == "merged"
    assert hus["entry_ids"] == ["11021722"]
    assert hus["rank"] == 1
    assert hus["freq_rank"] == 1


def test_the_inflection_is_searchable_from_the_card(workspace):
    hus = workspace["families"]["11021722"]
    for form in ("hus", "huse", "huset", "husene"):
        assert form in hus["searchable_forms"]


def test_multi_headword_component_is_refused_not_guessed(workspace):
    cfg = workspace["cfg"]
    conflicts = read_json(cfg.review_dir / "merge_conflicts.json")
    assert conflicts, "hav + have in one component must be refused"
    assert sorted(conflicts[0]["heads"]) == ["hav", "have"]
    refused = [f for f in workspace["families"].values()
               if f["merge_state"] == "refused"]
    assert len(refused) == 1
    # the fallback keeps the word's own articles, anchored on the richer one
    assert refused[0]["anchor_entry_id"] == "11000011"
    assert refused[0]["lemma"] == "have"


def test_every_word_lands_in_exactly_one_family(workspace):
    assignments = workspace["assignments"]
    assert set(assignments) == {"hus", "huse", "have", "tom"}
    seen = {}
    for fid, fam in workspace["families"].items():
        for m in fam["members"]:
            seen.setdefault(m["word"], []).append(fid)
    assert all(len(v) == 1 for v in seen.values())
    for word, row in assignments.items():
        assert seen[word] == [row["family_id"]]


def test_freq_rank_is_dense_over_renderable_families_only(workspace):
    fams = workspace["families"]
    ranks = sorted(f["freq_rank"] for f in fams.values()
                   if f["freq_rank"] is not None)
    assert ranks == [1, 2]
    # the 0-sense, 0-expression family exists but never becomes a card
    assert fams["11000012"]["freq_rank"] is None
    assert workspace["report"]["cards"] == 2


def test_guid_seed_is_frozen_from_the_v2_queryword(workspace):
    fams = workspace["families"]
    card_keys = read_json(workspace["cfg"].registry_local / "card_keys.json")
    assert fams["11021722"]["guid_seed"] == "hus"
    assert card_keys["11021722"]["carried_from_v2"] is True
    # a family with no v2 member falls back to the DDO lemma and is marked new
    have = [f for f in fams.values() if f["lemma"] == "have"][0]
    assert have["guid_seed"] == "have"
    assert card_keys[have["family_id"]]["carried_from_v2"] is False


def test_all_gates_recorded_and_green(workspace):
    report = read_json(workspace["cfg"].report_dir / "gates_report.json")
    ids = {r["id"] for r in report["results"]}
    assert {"G-RANK", "G-ASSIGN", "G-ANCHOR", "G-SEED"} <= ids
    assert report["failed"] == []


def test_expressions_alone_make_a_family_renderable(cfg, registry):
    """A family whose article has no senses but does have fixed expressions is
    still a card (stage 30's rule). Stage 70 then has to keep Content
    non-empty; test_gates.py pins that half."""
    e = make_entry("11000020", "krig", pos_key="sb.",
                   expressions=[make_expression("21000600", "kold krig",
                                                "spaendt tilstand")],
                   source_words=["krig"])
    write_workspace(cfg, {"11000020": e}, [(1, "krig")],
                    classification={"krig": {"members": [_members("11000020",
                                                                 "exact_cs")],
                                             "xrefs": [], "rejected": [],
                                             "resolved_by": "forward"}},
                    v2_querywords={})
    merge_run(cfg, registry)
    fam = read_json(cfg.json_dir / "words.json")["11000020"]
    assert fam["freq_rank"] == 1
