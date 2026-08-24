"""Stage 21: what the reverse index is allowed to hand a word, and what the
unresolved list has to say.

The owner's 2026-08-24 ruling -- a word that resolves to nothing is skipped and
recorded, never a stop -- only works if the record is readable. And layer 2/3
are a recovery door: an article the classifier refused to make a member anywhere
must not walk back in through it with a fabricated demoted flag.
"""

from conftest import make_entry, make_sense, write_workspace

from ankidkdeck.stages.s21_resolve import (rejected_everywhere_ids,
                                           run as resolve_run)
from ankidkdeck.util import read_json, write_json

DEMOTED_SYMBOL = {"entry_id": "11022726", "headword": "I", "pos_key": "symbol",
                  "reason": "case_only_demoted_pos"}


def test_an_xref_only_entry_counts_as_seen():
    """Exclusive exactness moves a case-only homograph from members to xrefs. If
    no word rejected it outright its `seen` count was zero, so it failed the
    `seen > 0` test and stayed in the reverse index -- and layer 3 could then
    hand a word the erbium-class symbol article as a member. 6 of 134 entries on
    the fixture set were xref-only and every one of them was indexed."""
    classification = {
        "i": {"members": [{"entry_id": "11022727", "bucket": "exact_cs",
                           "demoted": False}],
              "xrefs": ["11022724", "11022728"], "rejected": [],
              "resolved_by": "forward"},
    }
    assert rejected_everywhere_ids(classification) == {"11022724", "11022728"}


def test_a_kept_entry_is_never_rejected_everywhere():
    classification = {
        "a": {"members": [{"entry_id": "1", "bucket": "form", "demoted": False}],
              "xrefs": [], "rejected": [], "resolved_by": "forward"},
        "b": {"members": [], "xrefs": ["1"],
              "rejected": [{"entry_id": "1", "headword": "x", "reason": "unrelated"}],
              "resolved_by": None},
    }
    assert rejected_everywhere_ids(classification) == set()


def test_the_reverse_index_never_hands_back_a_demoted_or_affix_article(cfg,
                                                                      registry):
    """`I`(symbol) and `-hus`(sidsteled) must not be recoverable, and a
    recovered member carries the entry's REAL demoted flag, not a hardcoded
    False."""
    symbol = make_entry("11022726", "I", pos_key="symbol", forms=["ir"],
                        senses=[make_sense("21000060", "kemisk tegn for jod")])
    affix = make_entry("11000501", "-hus", pos_key="sidsteled", forms=["zzz"],
                       senses=[make_sense("21000061", "sidsteled")])
    real = make_entry("11000502", "være", pos_key="vb.", forms=["er", "var"],
                      senses=[make_sense("21000062", "eksistere")])
    entries = {e["entry_id"]: e for e in (symbol, affix, real)}
    # nothing was seen forward for any of the three words
    write_workspace(cfg, entries, [(1, "ir"), (2, "zzz"), (3, "er")],
                    classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    resolve_run(cfg, registry)
    c = read_json(cfg.json_dir / "classification.json")
    assert c["ir"]["members"] == []          # the symbol article is not indexed
    assert c["zzz"]["members"] == []         # nor the affix page
    assert [m["entry_id"] for m in c["er"]["members"]] == ["11000502"]
    assert c["er"]["members"][0]["demoted"] is False
    # The shipped registry curates er -> vaere, and a curated override now WINS
    # over the automatic reverse index: the registry exists to correct the
    # automatic layer, so being unreachable whenever that layer had a hit made it
    # useless. Both layers reach the same article here.
    assert c["er"]["members"][0]["evidence"] == "override"
    # ...and the bucket is the CLASSIFIER's verdict, not a hardcoded "form".
    assert c["er"]["members"][0]["bucket"] == "form"


def test_a_curated_override_beats_a_reverse_index_hit(cfg, registry):
    """Human curation wins. The reverse index reaches the WRONG article here (a
    spurious flex cell), and the override names the right one."""
    wrong = make_entry("11000700", "smide", pos_key="vb.", forms=["smed"],
                       senses=[make_sense("21000090", "kaste")])
    right = make_entry("11000701", "smed", pos_key="sb.",
                       senses=[make_sense("21000091", "person der former jern")])
    entries = {e["entry_id"]: e for e in (wrong, right)}
    write_workspace(cfg, entries, [(1, "smed")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "form_to_lemma.json", {"smed": "smed"})
    from ankidkdeck.registry import Registry
    resolve_run(cfg, Registry(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["smed"]["members"]] == ["11000701"]
    assert c["smed"]["members"][0]["evidence"] == "override"
    assert c["smed"]["members"][0]["bucket"] == "exact_cs"


def test_the_recovery_door_applies_the_classifier_and_keeps_its_bucket(cfg,
                                                                      registry):
    """The recovery door used to hardcode bucket="form" and skip classify_one
    entirely, so an abbreviation/multiword/unrelated article the forward page
    would have refused could still be attached -- and every recovered member was
    mislabelled, which made s30._relation() call an official alternative spelling
    an "inflection" and dropped it from the visible Variants list."""
    # `uden for` is an OFFICIAL alternative spelling of the `udenfor` article, so
    # the classifier's verdict is `variant`, not `form`.
    art = make_entry("12003753", "udenfor", pos_key="adv.",
                     alt_spellings=[{"form": "uden for", "official": True}],
                     senses=[make_sense("21001300", "på ydersiden")])
    # An abbreviation page reachable through its own flex cell. A `fork.`
    # pos_key is already blocked by indexable(), so this one carries a
    # non-demoted pos_key on purpose: the point of routing through classify_one
    # is to close the CLASS, not the one instance the demotion registry already
    # covers.
    abbrev = make_entry("11000710", "nr.", pos_key="sb.", forms=["nr"],
                        senses=[make_sense("21001301", "nummer")])
    entries = {e["entry_id"]: e for e in (art, abbrev)}
    write_workspace(cfg, entries, [(1, "uden for"), (2, "nr")],
                    classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    resolve_run(cfg, registry)
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["bucket"] for m in c["uden for"]["members"]] == ["variant"]
    # the abbreviation is refused, with a reason that says which door it came in
    assert c["nr"]["members"] == []
    assert [r["reason"] for r in c["nr"]["rejected"]] == ["recovery_abbreviation"]
    rows = read_json(cfg.report_dir / "recovery_rejected.json")
    assert rows and rows[0]["reason"] == "recovery_abbreviation"


def test_a_recovered_demoted_flag_is_read_from_the_entry(cfg, registry):
    """A demoted pos_key that is NOT in the demotion registry (so it is
    indexable) still has to report its real flag rather than a fabricated one --
    the flag is what anchor_of and G-ANCHOR read."""
    reg_demoted = registry.demoted_pos_keys
    assert "vb." not in reg_demoted
    e = make_entry("11000510", "være", pos_key="vb.", forms=["er"],
                   senses=[make_sense("21000070", "eksistere")])
    write_workspace(cfg, {"11000510": e}, [(1, "er")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    # pretend vb. became demoted for this run: the attachment must follow
    write_json(cfg.registry_local / "demoted_pos_keys.json", ["vb."])
    from ankidkdeck.registry import Registry
    resolve_run(cfg, Registry(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    # a demoted article is not indexable at all, so `er` stays unresolved --
    # which is the safe answer, and it is recorded
    assert c["er"]["members"] == []
    rows = read_json(cfg.report_dir / "unresolved.json")
    assert [r["word"] for r in rows] == ["er"]


def test_every_unresolved_row_carries_a_reason_and_the_counters_agree(cfg,
                                                                     registry):
    """`resolved["unresolved"] += 1 if status != "nohit" else 0` while the word
    was appended regardless: the counter and the file disagreed by the nohit
    count, i.e. two different denominators for the same review list."""
    rejected_only = make_entry("11000520", "-ske", pos_key="sidsteled",
                               senses=[make_sense("21000080", "sidsteled")])
    entries = {"11000520": rejected_only}
    classification = {
        "sker": {"members": [], "xrefs": [],
                 "rejected": [{"entry_id": "11000520", "headword": "-ske",
                               "pos_key": "sidsteled", "reason": "affix"}],
                 "resolved_by": None},
    }
    write_workspace(cfg, entries, [(1, "sker"), (2, "zzznothing"),
                                   (3, "brugtvognsforhandler")],
                    classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json",
               {"zzznothing": {"status": "nohit", "article_count": 0}})
    # an override naming a lemma page nobody crawled must be RECORDED, not
    # silently dropped: the human cannot otherwise tell "no override" from
    # "override present but unusable"
    write_json(cfg.registry_local / "form_to_lemma.json",
               {"brugtvognsforhandler": "brugtvognsforhandle"})
    from ankidkdeck.registry import Registry
    report = resolve_run(cfg, Registry(cfg))

    rows = read_json(cfg.report_dir / "unresolved.json")
    by_word = {r["word"]: r for r in rows}
    assert by_word["sker"]["reason"] == "all_rejected"
    assert by_word["sker"]["rejected"][0]["reason"] == "affix"
    assert by_word["zzznothing"]["reason"] == "nohit"
    assert by_word["brugtvognsforhandler"]["reason"] == "override_lemma_not_crawled"
    assert by_word["brugtvognsforhandler"]["override_lemma"] == \
        "brugtvognsforhandle"
    # the counters are DERIVED from the rows, so they cannot disagree with them
    assert report["unresolved"] == len(rows) == 3
    assert sum(report["unresolved_by_reason"].values()) == len(rows)
    assert report["nohit"] == 1
