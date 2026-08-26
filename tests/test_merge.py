"""The merge: connected components, refusal, unique assignment, dense rank.

The merge unit is a connected component of the (word, entry_id) graph, not a
"duplicate group" loop: 37 query words sit in two groups at once (`have` heads
both have-1 sb. and have-2 vb.) and a group loop emits two cards for them and
then collides on the family id.

This runs the real stage in a tmp workspace, so the gates run too -- a failure
here is a FatalError from run_gates, not a soft assertion.
"""

import re

import pytest
from conftest import make_entry, make_expression, make_sense, write_workspace

from ankidkdeck.stages.s30_merge import UnionFind, run as merge_run
from ankidkdeck.util import FatalError, read_json

FAMILY_ID = re.compile(r"^[0-9]{6,}$")


def _members(entry_id, bucket, demoted=False):
    return {"entry_id": entry_id, "bucket": bucket, "demoted": demoted}


def _c(*members, xrefs=(), rejected=()):
    return {"members": list(members), "xrefs": list(xrefs),
            "rejected": list(rejected), "resolved_by": "forward"}


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


def test_every_family_id_is_a_bare_entry_id(workspace):
    """family_id IS the anchor's entry_id. The old refused-merge fallback minted
    `11028611#kunne`, putting a wordlist form into the append-only registry
    whose whole purpose is to make wordlist changes GUID-neutral."""
    card_keys = read_json(workspace["cfg"].registry_local / "card_keys.json")
    for fid in list(workspace["families"]) + list(card_keys):
        assert FAMILY_ID.match(fid), fid


def test_refused_component_splits_by_head_and_emits_no_duplicate_family(
        cfg, registry_empty_card_keys):
    """kan/kunne/khan, the measured duplicate-card case (R2's micro30b).

    Two words in one refused component used to produce TWO families with
    identical anchor_entry_id, identical entry_ids and identical lemma --
    `11028611` and `11028611#kunne`, i.e. two byte-identical cards with two
    GUIDs, all four gates passing. The old fallback handed each word back ALL of
    its members, which re-created exactly the component it had just refused.

    The split gives one family per distinct non-demoted head lemma, each owning
    only that head's own articles. The entry sets are therefore disjoint: no two
    families can share an anchor, no family holds two heads, and family_id stays
    a bare entry_id. WHICH head a word joins when it matched both at the same
    bucket is the pipeline's standing pick rule (lowest bucket, own lemma, then
    lowest entry_id) -- the refusal itself is what review/merge_conflicts.json
    exists to put in front of a human.
    """
    kunne = make_entry("11028611", "kunne", pos_key="vb.", pos_text="verbum",
                       forms=["kan", "kunne", "kunnet"],
                       senses=[make_sense("2100070%d" % i, "kunne %d" % i)
                               for i in range(3)],
                       source_words=["kan", "kunne"])
    khan = make_entry("11025847", "khan", pos_key="sb.", pos_text="substantiv",
                      forms=["khanen", "khaner"],
                      senses=[make_sense("21000710", "mongolsk fyrste")],
                      source_words=["kan"])
    entries = {e["entry_id"]: e for e in (kunne, khan)}
    # the pre-F1 classifier's output, replayed: both articles are members of kan
    classification = {
        "kan": _c(_members("11028611", "form"), _members("11025847", "form")),
        "kunne": _c(_members("11028611", "exact_cs")),
    }
    write_workspace(cfg, entries, [(23, "kan"), (400, "kunne")],
                    classification=classification, v2_querywords={"kan": 23})
    merge_run(cfg, registry_empty_card_keys)
    fams = read_json(cfg.json_dir / "words.json")

    # one family per head lemma, and NO family holds both heads
    assert sorted(fams) == ["11025847", "11028611"]
    assert fams["11028611"]["entry_ids"] == ["11028611"]
    assert fams["11025847"]["entry_ids"] == ["11025847"]
    assert {f["lemma"] for f in fams.values()} == {"kunne", "khan"}
    for fam in fams.values():
        assert fam["merge_state"] == "refused"
    # no duplicate family: distinct anchors, distinct entry sets, no shared entry
    anchors = [f["anchor_entry_id"] for f in fams.values()]
    assert len(anchors) == len(set(anchors))
    all_eids = [e for f in fams.values() for e in f["entry_ids"]]
    assert len(all_eids) == len(set(all_eids))
    # every family_id is a bare entry_id, in words.json AND in the registry
    card_keys = read_json(cfg.registry_local / "card_keys.json")
    for fid in list(fams) + list(card_keys):
        assert FAMILY_ID.match(fid), fid
    # and each word still lands in exactly one family
    seen = {}
    for fid, fam in fams.items():
        for m in fam["members"]:
            seen.setdefault(m["word"], []).append(fid)
    assert all(len(v) == 1 for v in seen.values()), seen
    # the refusal is recorded with what the split did
    conflicts = read_json(cfg.review_dir / "merge_conflicts.json")
    assert conflicts[0]["heads"] == ["khan", "kunne"]
    assert conflicts[0]["fallback_heads"] == ["khan", "kunne"]


def test_a_variant_or_case_only_member_is_not_refused_as_a_second_head(cfg,
                                                                      registry):
    """The head test is an ORTHOGRAPHIC IDENTITY, not a raw lemma string.

    Buckets 3 and 4 exist to admit `uden for` alongside `udenfor` and `I`
    alongside `i` -- and 2025 shipped `udenfor -> {udenfor, uden for}`, plus the
    same shape for indenfor, overfor, ovenpaa, bagefter, bagved, nedenunder,
    indeni, udenom, udover. Comparing raw lemmas refuses every one of those
    components, so the classifier fix would buy nothing downstream: the member
    is admitted at stage 22 and thrown away at stage 30. It is also the only
    reading under which the guide's two claims agree (those words must ship AND
    the post-classifier corpus has 0 multi-headword components).
    """
    solid = make_entry("12003753", "udenfor", pos_key="adv.", pos_text="adverbium",
                       senses=[make_sense("2100130%d" % i, "udenfor %d" % i)
                               for i in range(3)],
                       source_words=["udenfor"])
    spaced = make_entry("12003754", "uden for", pos_key="præp.",
                        pos_text="præposition",
                        senses=[make_sense("2100131%d" % i, "uden for %d" % i)
                                for i in range(4)],
                        source_words=["udenfor"])
    lower = make_entry("11022727", "i", pos_key="præp.", pos_text="præposition",
                       senses=[make_sense("21001320", "inde i")],
                       source_words=["i"])
    upper = make_entry("11022728", "I", pos_key="pron.", pos_text="pronomen",
                       senses=[make_sense("21001321", "2. person flertal")],
                       source_words=["i"])
    entries = {e["entry_id"]: e for e in (solid, spaced, lower, upper)}
    classification = {
        "udenfor": _c(_members("12003753", "exact_cs"),
                      _members("12003754", "variant")),
        "i": _c(_members("11022727", "exact_cs"), _members("11022728", "exact_ci")),
    }
    write_workspace(cfg, entries, [(5, "i"), (585, "udenfor")],
                    classification=classification,
                    v2_querywords={"i": 5, "udenfor": 585})
    report = merge_run(cfg, registry)

    assert report["refused_components"] == 0
    assert report["dropped_components"] == 0
    assert read_json(cfg.review_dir / "merge_conflicts.json") == []
    fams = read_json(cfg.json_dir / "words.json")
    # THE BUCKET DECIDES THE HEADLINE (guide 4.8). `uden for` is the variant
    # bucket with 4 senses and `udenfor` is exact_cs with 3; sense count used to
    # win, so the card face showed a spelling no wordlist word has.
    uf = fams["12003753"]
    assert uf["lemma"] == "udenfor"
    assert sorted(uf["entry_ids"]) == ["12003753", "12003754"]
    assert [m["word"] for m in uf["members"]] == ["udenfor"]
    assert "12003754" not in fams
    # ...and I(pron.) reaches the `i` card instead of no card at all, anchored on
    # the exact_cs preposition rather than the exact_ci pronoun.
    fi = fams["11022727"]
    assert fi["lemma"] == "i"
    assert sorted(fi["entry_ids"]) == ["11022727", "11022728"]
    # a genuinely different lemma is STILL refused
    hav = make_entry("11000600", "hav", pos_key="sb.",
                     senses=[make_sense("21001400", "saltvand")],
                     source_words=["have"])
    have = make_entry("11000601", "have", pos_key="vb.",
                      senses=[make_sense("21001401", "besidde")],
                      source_words=["have"])
    write_workspace(cfg, {e["entry_id"]: e for e in (hav, have)},
                    [(1, "have")],
                    classification={"have": _c(_members("11000601", "exact_cs"),
                                               _members("11000600", "form"))},
                    v2_querywords={})
    report2 = merge_run(cfg, registry)
    assert report2["refused_components"] == 1


def test_the_bucket_decides_the_anchor_before_the_sense_count(cfg, registry):
    """Guide 4.8: "bucket ORDER decides the card headline, Variants and
    Etymology, because 09 uses primary_entry = sorted_entries[0]".

    Measured defect: `uden for` (praep., variant bucket, 4 senses) headed the
    card for `udenfor` (exact_cs, 3 senses), so the card face showed a spelling
    no wordlist word has -- and Etymology came from the wrong article.
    """
    solid = make_entry("12003753", "udenfor", pos_key="adv.",
                       senses=[make_sense("2100150%d" % i, "udenfor %d" % i)
                               for i in range(3)],
                       source_words=["udenfor"])
    spaced = make_entry("12003754", "uden for", pos_key="præp.",
                        senses=[make_sense("2100151%d" % i, "uden for %d" % i)
                                for i in range(4)],
                        source_words=["udenfor"])
    entries = {e["entry_id"]: e for e in (solid, spaced)}
    write_workspace(cfg, entries, [(585, "udenfor")],
                    classification={"udenfor": _c(
                        _members("12003753", "exact_cs"),
                        _members("12003754", "variant"))},
                    v2_querywords={"udenfor": 585})
    merge_run(cfg, registry)
    fams = read_json(cfg.json_dir / "words.json")
    assert list(fams) == ["12003753"]
    assert fams["12003753"]["lemma"] == "udenfor"
    assert fams["12003753"]["display_headword"] == "udenfor"


def test_the_wordlist_spelling_breaks_a_tie_inside_a_bucket(cfg, registry):
    """Third key: two exact_cs articles with the same sense count, one of which
    IS the family's lowest-rank member word."""
    other = make_entry("11000801", "have", pos_key="sb.",
                       senses=[make_sense("21001600", "en have")],
                       source_words=["haven"])
    own = make_entry("11000802", "haven", pos_key="sb.",
                     senses=[make_sense("21001601", "haven")],
                     source_words=["haven"])
    entries = {e["entry_id"]: e for e in (other, own)}
    write_workspace(cfg, entries, [(50, "haven")],
                    classification={"haven": _c(_members("11000802", "exact_cs"),
                                                _members("11000801", "form"))},
                    v2_querywords={})
    merge_run(cfg, registry)
    fams = read_json(cfg.json_dir / "words.json")
    assert list(fams) == ["11000802"]
    assert fams["11000802"]["lemma"] == "haven"


def test_case_only_memberships_are_written_out_and_baselined(cfg, registry):
    """Bucket 4 is card membership on purpose -- making it xref-only would delete
    `I`(pron., a real word) from the deck. What remains is a content change a
    human should see once, so the rows are an artifact and the COUNT is
    baselined."""
    lower = make_entry("11022727", "i", pos_key="præp.",
                       senses=[make_sense("21001700", "inde i")],
                       source_words=["i"])
    upper = make_entry("11022728", "I", pos_key="pron.",
                       senses=[make_sense("21001701", "2. person flertal")],
                       source_words=["i"])
    entries = {e["entry_id"]: e for e in (lower, upper)}
    write_workspace(cfg, entries, [(5, "i")],
                    classification={"i": _c(_members("11022727", "exact_cs"),
                                            _members("11022728", "exact_ci"))},
                    v2_querywords={"i": 5})
    report = merge_run(cfg, registry)
    rows = read_json(cfg.review_dir / "case_only_members.json")
    assert report["case_only_members"] == len(rows) == 1
    row = rows[0]
    assert row["kind"] == "case_only_member"
    assert (row["word"], row["member_lemma"]) == ("i", "I")
    assert row["family_id"] == "11022727"
    gates = read_json(cfg.report_dir / "gates_report.json")
    case = [r for r in gates["results"] if r["id"] == "G-CASE"][0]
    assert case["ok"] is True and case["detail"]["rows"] == 1


def test_the_case_only_population_cannot_grow_past_its_baseline(cfg, registry):
    lower = make_entry("11022727", "i", pos_key="præp.",
                       senses=[make_sense("21001800", "inde i")],
                       source_words=["i"])
    upper = make_entry("11022728", "I", pos_key="pron.",
                       senses=[make_sense("21001801", "2. person flertal")],
                       source_words=["i"])
    write_workspace(cfg, {e["entry_id"]: e for e in (lower, upper)},
                    [(5, "i")],
                    classification={"i": _c(_members("11022727", "exact_cs"),
                                            _members("11022728", "exact_ci"))},
                    v2_querywords={})
    from ankidkdeck.registry import Registry
    from ankidkdeck.util import write_json
    write_json(cfg.registry_local / "gates.json", {"case_only_members_max": 0})
    with pytest.raises(FatalError) as exc:
        merge_run(cfg, Registry(cfg))
    assert "G-CASE" in str(exc.value)


def test_an_anchor_whose_spelling_differs_only_by_case_is_recorded(cfg, registry):
    """The var/VAR class: the family's content belongs to a different
    capitalisation from the wordlist word. Information for a human, not a
    tie-break -- forcing the empty article to anchor would put an article with
    nothing to render in charge of family_id and Etymology."""
    acronym = make_entry("49002989", "VAR", pos_key="sb.",
                         senses=[make_sense("21001900", "video assistant "
                                                        "referee")],
                         source_words=["var"])
    entries = {acronym["entry_id"]: acronym}
    write_workspace(cfg, entries, [(10, "var")],
                    classification={"var": _c(_members("49002989", "exact_ci"))},
                    v2_querywords={"var": 10})
    merge_run(cfg, registry)
    rows = read_json(cfg.review_dir / "case_only_members.json")
    kinds = {r["kind"] for r in rows}
    assert kinds == {"case_only_member", "anchor_spelling"}
    spelling = [r for r in rows if r["kind"] == "anchor_spelling"][0]
    assert (spelling["anchor_lemma"], spelling["word"]) == ("VAR", "var")


def test_anchor_prefers_a_non_demoted_article_over_sense_count(cfg, registry):
    """Cm(symbol, 4 senses) + centimeter(sb., 1 sense), both reachable from cm.

    With sense count ahead of demotion the symbol won and G-ANCHOR then failed
    the whole build -- a self-inflicted stop on a tie-break accident rather than
    on a defect. Demotion first makes the gate unfalsifiable by construction.
    """
    cm = make_entry("10000001", "Cm", pos_key="symbol", pos_text="symbol",
                    senses=[make_sense("2100080%d" % i, "curium %d" % i)
                            for i in range(4)],
                    source_words=["cm"])
    centimeter = make_entry("10000002", "centimeter", pos_key="sb.",
                            pos_text="substantiv", forms=["cm"],
                            senses=[make_sense("21000810", "laengdeenhed")],
                            source_words=["cm"])
    entries = {e["entry_id"]: e for e in (cm, centimeter)}
    classification = {"cm": _c(_members("10000001", "form", demoted=True),
                               _members("10000002", "form"))}
    write_workspace(cfg, entries, [(1, "cm")], classification=classification,
                    v2_querywords={"cm": 1})
    merge_run(cfg, registry)      # no FatalError from G-ANCHOR
    fams = read_json(cfg.json_dir / "words.json")
    fam = fams["10000002"]
    assert fam["anchor_entry_id"] == "10000002"
    assert fam["lemma"] == "centimeter"
    gates = read_json(cfg.report_dir / "gates_report.json")
    anchor = [r for r in gates["results"] if r["id"] == "G-ANCHOR"][0]
    assert anchor["ok"] is True
    assert anchor["detail"]["demoted_anchored_with_alternative"] == []


def test_a_failed_gate_leaves_the_registry_untouched(cfg, registry, monkeypatch):
    """card_keys.json IS the users' study progress.

    The freeze used to run before the gates, so a build that failed G-ANCHOR
    still wrote a seed -- and the next run's `if fid in card_keys: continue`
    treats that as reviewed truth and never revisits it. A failed build must
    leave the file absent.
    """
    path = cfg.registry_local / "card_keys.json"
    assert not path.exists()
    e = make_entry("11000040", "hus", pos_key="sb.",
                   senses=[make_sense("21000900", "bygning")],
                   source_words=["hus"])
    write_workspace(cfg, {"11000040": e}, [(1, "hus")],
                    classification={"hus": _c(_members("11000040", "exact_cs"))},
                    v2_querywords={})
    # force one gate to fail without touching the merge itself
    import ankidkdeck.stages.s30_merge as S30
    monkeypatch.setattr(S30, "dense_unique_ranks",
                        lambda *a, **k: (False, {"forced": "failure"}))
    with pytest.raises(FatalError):
        merge_run(cfg, registry)
    assert not path.exists(), "a failed build froze a GUID seed"
    assert not (cfg.json_dir / "words.json").exists()
    # and the gate report still records everything, which is the point of it
    gates = read_json(cfg.report_dir / "gates_report.json")
    assert "G-RANK" in gates["failed"]
    assert {"G-RANK", "G-ASSIGN", "G-ANCHOR", "G-SEED", "G-REGKEY"} <= {
        r["id"] for r in gates["results"]}


def test_a_phase_b_lemma_word_joins_the_family(cfg, registry):
    """R1's micro3 scenario: `have` is NOT on the wordlist -- it was fetched by
    phase B because `har` resolved to the lemma `have`.

    Non-wordlist words used to be `continue`d out of the union-find, so the
    homographs the mandatory phase-B fetch exists to obtain never joined any
    family: have(sb., garden, 5 senses) landed on no card and the ~275 phase-B
    requests bought nothing. They join now, with wiktionary_rank = null so they
    never drive rank.
    """
    have_vb = make_entry("11020037", "have", pos_key="vb.", pos_text="verbum",
                         forms=["har", "havde", "haft"],
                         senses=[make_sense("2100100%d" % i, "have vb %d" % i)
                                 for i in range(4)],
                         source_words=["har", "have"])
    have_sb = make_entry("11020036", "have", pos_key="sb.", pos_text="substantiv",
                         forms=["haven", "haver"],
                         senses=[make_sense("2100101%d" % i, "have sb %d" % i)
                                 for i in range(2)],
                         source_words=["have"])
    entries = {e["entry_id"]: e for e in (have_vb, have_sb)}
    classification = {
        "har": _c(_members("11020037", "form")),
        # `have` is off the wordlist; both articles are its own exact matches
        "have": _c(_members("11020037", "exact_cs"),
                   _members("11020036", "exact_cs")),
    }
    write_workspace(cfg, entries, [(30, "har")], classification=classification,
                    v2_querywords={"har": 30})
    report = merge_run(cfg, registry)
    fams = read_json(cfg.json_dir / "words.json")
    assert len(fams) == 1
    fam = list(fams.values())[0]
    # BOTH articles are in the family, and the off-wordlist word is a member
    assert sorted(fam["entry_ids"]) == ["11020036", "11020037"]
    assert sorted(m["word"] for m in fam["members"]) == ["har", "have"]
    ranks = {m["word"]: m["wiktionary_rank"] for m in fam["members"]}
    assert ranks == {"har": 30, "have": None}
    assert fam["rank"] == 30            # the null-rank member never sets it
    assert fam["freq_rank"] == 1
    assert report["dropped_components"] == 0
    assert read_json(cfg.report_dir / "merge_report.json")[
        "words_off_wordlist"] == ["have"]
    # assignments stays the wordlist's own map
    assert set(read_json(cfg.json_dir / "assignments.json")) == {"har"}


def test_a_component_no_wordlist_word_claims_is_dropped_not_fatal(cfg, registry):
    """The replacement for the `memberless` FATAL, which is what forced
    non-wordlist words out of the graph in the first place. Owner ruling: a
    resolution problem is recorded, never a stop."""
    wanted = make_entry("11000050", "hus", pos_key="sb.",
                        senses=[make_sense("21001100", "bygning")],
                        source_words=["hus"])
    stray = make_entry("11000051", "zebra", pos_key="sb.",
                       senses=[make_sense("21001101", "stribet dyr")],
                       source_words=["zebra"])
    entries = {e["entry_id"]: e for e in (wanted, stray)}
    classification = {"hus": _c(_members("11000050", "exact_cs")),
                      "zebra": _c(_members("11000051", "exact_cs"))}
    write_workspace(cfg, entries, [(1, "hus")], classification=classification,
                    v2_querywords={})
    report = merge_run(cfg, registry)
    fams = read_json(cfg.json_dir / "words.json")
    assert set(fams) == {"11000050"}
    assert report["dropped_components"] == 1
    dropped = read_json(cfg.report_dir / "merge_report.json")[
        "dropped_components_sample"]
    assert dropped[0]["entry_ids"] == ["11000051"]
    assert dropped[0]["off_wordlist_words"] == ["zebra"]


def test_searchable_forms_never_carry_the_glued_homograph_form(cfg, registry):
    """`al2` / `udenfor1` / `i5` is not what DDO shows and was a junk Anki
    search token on every homograph card."""
    e = make_entry("11000060", "al", super_="2", pos_key="pron.",
                   forms=["alt", "alle"],
                   senses=[make_sense("21001200", "hele")],
                   source_words=["al"])
    write_workspace(cfg, {"11000060": e}, [(1, "al")],
                    classification={"al": _c(_members("11000060", "exact_cs"))},
                    v2_querywords={})
    merge_run(cfg, registry)
    fam = read_json(cfg.json_dir / "words.json")["11000060"]
    assert fam["display_headword"] == "al"
    assert fam["super"] == "2"
    assert "al2" not in fam["searchable_forms"]
    assert fam["searchable_forms"][0] == "al"


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


# --------------------------------------------------------------------------
# The alias branch of the head relation
# --------------------------------------------------------------------------

def _tjek_workspace(cfg, quarantine, alias_pairs=None):
    """The measured shape of the alias/merge disagreement.

    `check`(11007687) and `tjek`(12001518) are one dictionary word registered in
    alias_pairs.json. The classifier admitted the pair through the ALIAS branch,
    and this stage then compared heads with squash() alone -- which is only the
    FIRST branch -- so it called the component two dictionary words and split it
    back into two cards of the same word.
    """
    check = make_entry("11007687", "check", pos_key="sb.",
                       senses=[make_sense("21007687", "kontrol")],
                       source_words=["check"])
    tjek = make_entry("12001518", "tjek", pos_key="sb.",
                      senses=[make_sense("2200151%d" % i, "tjek %d" % i)
                              for i in range(2)],
                      source_words=["tjek"])
    entries = {e["entry_id"]: e for e in (check, tjek)}
    classification = {
        "check": _c(_members("11007687", "exact_cs"),
                    _members("12001518", "variant")),
        "tjek": _c(_members("12001518", "exact_cs"),
                   _members("11007687", "variant")),
    }
    write_workspace(cfg, entries, [(1126, "tjek"), (1707, "check")],
                    classification=classification,
                    v2_querywords={"tjek": 1126, "check": 1707})
    from ankidkdeck.registry import Registry
    reg = Registry(cfg)
    # Set in place rather than through a work/registry overlay: a registry file
    # that IS a list is APPENDED to by the overlay (that is what makes
    # alias_pairs and demoted_pos_keys extensible), so an overlay can never
    # shorten this one. Lifting the quarantine is therefore an edit to the
    # shipped src/ankidkdeck/registry/alias_merge_pending.json -- reviewed and
    # committed, which is the right weight for a decision that retires a GUID.
    reg.data["alias_merge_pending"] = list(quarantine)
    if alias_pairs is not None:
        reg.data["alias_pairs"] = list(alias_pairs)
    return reg


def test_an_alias_pair_merges_into_one_family(cfg):
    """The fix. Heads are grouped by is_variant(), the classifier's own
    relation, so a pair the classifier admitted is no longer refused here."""
    reg = _tjek_workspace(cfg, quarantine=[])
    report = merge_run(cfg, reg)
    assert report["refused_components"] == 0
    assert report["dropped_components"] == 0
    assert report["families"] == 1
    assert read_json(cfg.review_dir / "merge_conflicts.json") == []
    fams = read_json(cfg.json_dir / "words.json")
    # `tjek` heads it: it is the lower-ranked wordlist word, so anchor_of's
    # wordlist-spelling tie-break picks its article, and min(carried) picks its
    # seed. `check`'s family_id -- and its frozen v2.1 GUID -- retires.
    fam = fams["12001518"]
    assert sorted(fam["entry_ids"]) == ["11007687", "12001518"]
    assert sorted(m["word"] for m in fam["members"]) == ["check", "tjek"]
    assert fam["guid_seed"] == "tjek"
    assert "11007687" not in fams
    # both spellings still reach the card through Anki search
    assert {"check", "tjek"} <= set(fam["searchable_forms"])


def test_a_quarantined_alias_pair_is_still_refused(cfg):
    """The switch that keeps this round's rerun free of GUID churn.

    Both sides of the three pre-existing pairs already hold a frozen
    card_keys.json row, so landing the merge retires one of two frozen seeds --
    a release decision with a deadline, not a build step. Quarantining keeps the
    CLASSIFIER admitting the pair (or `naeh` and `o.k.` lose their only edge)
    while the merge keeps two heads, i.e. exactly the pre-fix behaviour.
    """
    reg = _tjek_workspace(cfg, quarantine=[["check", "tjek"]])
    report = merge_run(cfg, reg)
    assert report["refused_components"] == 1
    assert report["families"] == 2
    conflicts = read_json(cfg.review_dir / "merge_conflicts.json")
    assert conflicts[0]["heads"] == ["check", "tjek"]
    assert conflicts[0]["fallback_heads"] == ["check", "tjek"]
    fams = read_json(cfg.json_dir / "words.json")
    assert set(fams) == {"11007687", "12001518"}
    assert report["alias_pairs_quarantined_from_merge"] == [["check", "tjek"]]


def test_a_cased_alias_pair_is_normalised_on_both_sides_of_the_quarantine(cfg):
    """The quarantine key and the alias registry key must be normalised the SAME
    way, because this comparison fails OPEN.

    `quarantined` was built with nk() while registry.alias_pairs hands back the
    raw strings, so the set difference only matched because all three live pairs
    happen to be lowercase NFC already. A pair carrying one uppercase letter
    would not match its own quarantine row -- and the failure direction is
    "merge lands, a frozen v2.1 GUID retires, no error, no report line". Both
    halves are asserted here: the quarantine holds through a case difference,
    and (the discriminating half) a cased pair is still a live alias when it is
    NOT quarantined, which the raw comparison got wrong in the other direction
    -- is_variant() builds its lookup key with nk(), so a raw cased pair was
    inert everywhere.
    """
    reg = _tjek_workspace(cfg, quarantine=[["check", "Tjek"]],
                          alias_pairs=[["Check", "tjek"]])
    report = merge_run(cfg, reg)
    assert report["refused_components"] == 1
    assert report["families"] == 2
    assert set(read_json(cfg.json_dir / "words.json")) == {"11007687",
                                                           "12001518"}
    assert report["alias_pairs_quarantined_from_merge"] == [["check", "tjek"]]


def test_a_cased_alias_pair_that_is_not_quarantined_still_merges(cfg):
    """The other half of the normalisation: lifting the quarantine on a cased
    pair has to actually land the merge, or the switch is a no-op that reads
    like a decision."""
    reg = _tjek_workspace(cfg, quarantine=[],
                          alias_pairs=[["Check", "tjek"]])
    report = merge_run(cfg, reg)
    assert report["refused_components"] == 0
    assert report["families"] == 1
    fams = read_json(cfg.json_dir / "words.json")
    assert set(fams) == {"12001518"}
    assert fams["12001518"]["guid_seed"] == "tjek"


def test_a_genuinely_different_lemma_is_still_two_heads(cfg, registry):
    """The guard the alias branch must not weaken: `hav`/`have` and
    `kan`/`khan` squash apart AND are not alias-registered, so they stay
    refused."""
    hav = make_entry("11000600", "hav", pos_key="sb.",
                     senses=[make_sense("21001400", "saltvand")],
                     source_words=["have"])
    have = make_entry("11000601", "have", pos_key="vb.",
                      senses=[make_sense("21001401", "besidde")],
                      source_words=["have"])
    write_workspace(cfg, {e["entry_id"]: e for e in (hav, have)}, [(1, "have")],
                    classification={"have": _c(_members("11000601", "exact_cs"),
                                               _members("11000600", "form"))},
                    v2_querywords={})
    assert merge_run(cfg, registry)["refused_components"] == 1


def test_a_stale_guid_seed_is_reported_and_never_rewritten(cfg, registry):
    """card_keys.json is append-only because those bytes are the users' study
    progress -- so a seed frozen from a SMALLER member set (the freeze ran before
    the unresolved list was curated) can never be corrected by a rerun. 22
    families locked in the less frequent of two spellings that way, with no
    error, no warning and no report line, because the freeze loop skipped an
    already-frozen family before it computed a seed at all.
    """
    from ankidkdeck.util import write_json
    hus = make_entry("11021722", "hus", pos_key="sb.", forms=["huse"],
                     senses=[make_sense("21000300", "bygning")],
                     source_words=["hus"])
    write_workspace(cfg, {"11021722": hus}, [(1, "hus"), (40, "huse")],
                    classification={
                        "hus": _c(_members("11021722", "exact_cs")),
                        "huse": _c(_members("11021722", "form"))},
                    v2_querywords={"hus": 1, "huse": 40})
    # A previous build froze this family on `huse` -- a real v2.1 QueryWord, so
    # G-SEED is satisfied -- back when `hus` had not yet been resolved into the
    # family. Today's data would choose `hus` (v2 rank 1 beats 40).
    write_json(cfg.registry_local / "card_keys.json",
               {"11021722": {"guid_seed": "huse", "lemma_at_freeze": "hus",
                             "since": "3.0", "carried_from_v2": True}})
    from ankidkdeck.registry import Registry
    report = merge_run(cfg, Registry(cfg))
    rows = read_json(cfg.review_dir / "stale_guid_seeds.json")
    assert rows == [{"family_id": "11021722", "frozen_seed": "huse",
                     "seed_today": "hus", "frozen_since": "3.0",
                     "lemma_at_freeze": "hus"}]
    assert report["registry_freeze"]["stale_seeds"] == 1
    assert report["registry_freeze"]["added"] == 0
    # unchanged on disk: reporting is not rewriting
    assert read_json(cfg.registry_local / "card_keys.json")[
        "11021722"]["guid_seed"] == "huse"
    assert read_json(cfg.json_dir / "words.json")["11021722"]["guid_seed"] == "huse"


def test_the_freeze_report_says_what_is_uncommitted_when_added_is_zero(
        cfg, monkeypatch, shipped_card_keys_empty):
    """`added: 0` is correct on a rerun and useless as evidence.

    Round 2 appended 4 rows, then two idempotent reruns overwrote
    registry_freeze_report.json with `added: 0` -- so the file an owner opens
    said this workspace had frozen nothing, and the append-only invariant could
    only be shown by diffing a snapshot taken outside the pipeline. The count
    that survives a rerun is the overlay's diff against the COMMITTED registry
    (src/ankidkdeck/registry/card_keys.json), which is also the diff a human has
    to review before release.

    `shipped_card_keys_empty` is what keeps the arithmetic here readable: the
    committed registry was `{}` when this test was written and ships 2,927 rows
    since the release refreeze -- one of them this test's own `11021722`.
    """
    from ankidkdeck.registry import Registry
    from ankidkdeck.util import write_json
    hus = make_entry("11021722", "hus", pos_key="sb.", forms=["huse"],
                     senses=[make_sense("21000300", "bygning")],
                     source_words=["hus"])
    write_workspace(cfg, {"11021722": hus}, [(1, "hus"), (40, "huse")],
                    classification={
                        "hus": _c(_members("11021722", "exact_cs")),
                        "huse": _c(_members("11021722", "form"))},
                    v2_querywords={"hus": 1, "huse": 40})
    # first run: the overlay does not exist yet, so this row is appended here
    first = merge_run(cfg, Registry(cfg))["registry_freeze"]
    assert first["added"] == 1
    assert first["rows_not_in_the_committed_registry"] == 1
    # the idempotent rerun: nothing to append, and the row is still uncommitted
    again = merge_run(cfg, Registry(cfg))["registry_freeze"]
    assert again["added"] == 0
    assert again["stale_seeds"] == 0
    assert again["rows_not_in_the_committed_registry"] == 1
    assert again["total"] == 1
    # and after release, when the row IS committed, the diff is empty again
    from ankidkdeck import registry as R
    committed = {"11021722": {"guid_seed": "hus", "lemma_at_freeze": "hus",
                              "since": "3.0", "carried_from_v2": True}}
    orig = R._package_default
    monkeypatch.setattr(R, "_package_default",
                        lambda name: (dict(committed)
                                      if name == "card_keys.json"
                                      else orig(name)))
    write_json(cfg.registry_local / "card_keys.json", {})
    shipped = merge_run(cfg, Registry(cfg))["registry_freeze"]
    assert shipped["added"] == 0
    assert shipped["rows_not_in_the_committed_registry"] == 0


def test_an_abbreviation_member_is_not_a_variant_on_the_card_face(cfg, registry):
    """The abbreviation card, end to end (owner policy B, 2026-08-26).

    Four properties, each one a decision that could have gone the other way:

      * the card is HEADLINED by DDO's own dotted spelling (`hr.`), because the
        dotted article is the family's only entry and therefore its anchor;
      * the member's relation is `abbreviation`, not the `alias` fall-through
        _relation() used to give any unknown bucket -- s70.alt_forms_html renders
        variant/alias members, so the fall-through would have printed `hr` as a
        variant spelling on a card already headlined `hr.`;
      * the dotless wordlist spelling is still in searchable_forms, which is what
        makes typing `hr` in Anki find the card;
      * the guid_seed is the v2.1 QueryWord, so the retired 2.1 card comes back
        with its own GUID instead of a new one.
    """
    from ankidkdeck.stages.s70_export import alt_forms_html, headword_html
    hr = make_entry("11021497", "hr.", pos_key="sb.", source_words=["hr"],
                    senses=[make_sense("21000400", "titel")])
    write_workspace(cfg, {"11021497": hr}, [(179, "hr")],
                    classification={"hr": {"members": [
                        {"entry_id": "11021497", "bucket": "abbreviation",
                         "demoted": False, "evidence": "abbreviation",
                         "why": "dotted_abbreviation_entry"}],
                        "xrefs": [], "rejected": [],
                        "resolved_by": "abbreviation"}},
                    v2_querywords={"hr": 179})
    merge_run(cfg, registry)
    fam = read_json(cfg.json_dir / "words.json")["11021497"]
    assert fam["lemma"] == "hr."
    assert [(m["word"], m["relation"]) for m in fam["members"]] == [
        ("hr", "abbreviation")]
    assert fam["searchable_forms"] == ["hr.", "hr"]
    assert fam["guid_seed"] == "hr"
    assert headword_html(hr) == "hr."
    # the whole point: NOT on the Variants line
    assert alt_forms_html(fam, [hr]) == ""
    # ...and it would have been, under the alias fall-through
    aliased = dict(fam, members=[{"word": "hr", "wiktionary_rank": 179,
                                  "relation": "alias"}])
    assert "hr" in alt_forms_html(aliased, [hr])


def test_an_abbreviation_article_never_outranks_a_real_one_for_the_anchor(
        cfg, registry):
    """BUCKET_ORDER puts `abbreviation` last, after exact_ci, so a mixed family
    is still headlined by the real word. Unreachable on the 2026 corpus -- layer
    4 only fires for a word with no other member at all -- and asserted anyway,
    because the table is what stage 30 reads for both best_member() and the
    anchor rule."""
    from ankidkdeck.stages.s22_classify import BUCKET_ORDER
    assert BUCKET_ORDER["abbreviation"] == max(BUCKET_ORDER.values())
    real = make_entry("11033715", "min", pos_key="pron.", source_words=["min"],
                      senses=[make_sense("21000401", "tilhoerende mig")])
    abbrev = make_entry("11033713", "min.", pos_key="fork.",
                        source_words=["min"],
                        senses=[make_sense("21000402", "minut")])
    write_workspace(cfg, {"11033715": real, "11033713": abbrev},
                    [(53, "min")],
                    classification={"min": _c(
                        _members("11033715", "exact_cs"),
                        _members("11033713", "abbreviation", demoted=True))},
                    v2_querywords={"min": 53})
    merge_run(cfg, registry)
    fams = read_json(cfg.json_dir / "words.json")
    assert list(fams) == ["11033715"]
    assert fams["11033715"]["lemma"] == "min"
    assert fams["11033715"]["members"][0]["relation"] == "anchor"
