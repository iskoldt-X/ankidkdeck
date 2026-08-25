"""Stage 21: what the reverse index is allowed to hand a word, and what the
unresolved list has to say.

The owner's 2026-08-24 ruling -- a word that resolves to nothing is skipped and
recorded, never a stop -- only works if the record is readable. And layer 2/3
are a recovery door: an article the classifier refused to make a member anywhere
must not walk back in through it with a fabricated demoted flag.
"""

import pytest
from conftest import make_entry, make_sense, write_workspace

from ankidkdeck.stages.s21_resolve import (rejected_everywhere_ids,
                                           run as resolve_run)
from ankidkdeck.util import FatalError, read_json, write_json


def _reg(cfg):
    from ankidkdeck.registry import Registry
    return Registry(cfg)


def _allow_override_problems(cfg, n: int) -> None:
    """G-OVERRIDE is baselined at 0: a curated mapping that binds nothing stops
    the build. A test that CONSTRUCTS that state has to say so out loud."""
    write_json(cfg.registry_local / "gates.json", {"override_problems_max": n})

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
    resolve_run(cfg, _reg(cfg))
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
    # the shipped registry curates er -> vaere, and demoting the target makes
    # that mapping unusable -- which G-OVERRIDE is entitled to stop the build for
    _allow_override_problems(cfg, 1)
    resolve_run(cfg, _reg(cfg))
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
    _allow_override_problems(cfg, 1)
    report = resolve_run(cfg, _reg(cfg))

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


# --------------------------------------------------------------------------
# The override channel. Every test above this line passed while the channel
# was 100% dead, because the only positive override test mapped a form to
# ITSELF (`smed` -> `smed`), which classify_one answers with exact_cs no matter
# what the recovery door does. These cover the case the registry exists for:
# form != lemma, and the form is NOT in the article's flex table.
# --------------------------------------------------------------------------

def _vente(cfg, *, extra_forms=(), wordlist=((159, "vent"),)):
    """DDO's 2026 flex table for `vente` -- ventede / venter / ventet and NO
    imperative. That is the real, measured shape: the imperative, the genitive,
    the middle-voice -s, the present participle and the definite superlative are
    all absent from the 2026 tables, so the reverse index cannot reach them and
    form_to_lemma.json is the only door."""
    e = make_entry("12006356", "vente", pos_key="vb.",
                   forms=["ventede", "venter", "ventet", *extra_forms],
                   senses=[make_sense("21006356", "afvente")])
    write_workspace(cfg, {"12006356": e}, list(wordlist), classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    return e


def test_a_curated_override_binds_a_form_the_flex_table_does_not_have(cfg):
    """FIX-A. `vent` is the imperative of `vente`; the 2026 flex table has no
    imperative, so classify_one walks all the way to its `unrelated` fallback.
    On the CURATED path that fallback means "the automatic layers have nothing to
    say", and the human mapping wins."""
    _vente(cfg)
    write_json(cfg.registry_local / "form_to_lemma.json", {"vent": "vente"})
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    m = c["vent"]["members"]
    assert [x["entry_id"] for x in m] == ["12006356"]
    assert m[0]["evidence"] == "override"
    # bucket `form`, NOT `variant`: s30._relation turns `form` into "inflection",
    # and s70.alt_forms_html only renders variant/alias -- so the imperative
    # becomes a hidden Anki search token and never card-face text.
    assert m[0]["bucket"] == "form"
    assert m[0]["why"] == "curated_override"
    assert report["override"] == 1
    assert report["override_problems"] == 0
    assert report["unresolved"] == 0
    assert read_json(cfg.report_dir / "unresolved.json") == []


def test_the_override_channel_writes_an_audit_row_for_every_mapping_it_admits(cfg):
    """Accepting `unrelated` means the classifier is no longer a second opinion
    on these edges, so the only remaining check is a human reading them."""
    _vente(cfg)
    write_json(cfg.registry_local / "form_to_lemma.json", {"vent": "vente"})
    resolve_run(cfg, _reg(cfg))
    rows = read_json(cfg.review_dir / "override_accepted.json")
    assert [(r["word"], r["override_lemma"]) for r in rows] == [("vent", "vente")]
    assert rows[0]["in_paradigm_index"] is False
    assert rows[0]["senses"] == 1


def test_an_override_that_the_flex_table_already_covers_is_not_an_audit_row(cfg):
    """`er` IS in vaere's flex table, so it binds on real evidence (flex_table)
    and must not be listed as something the human waved through. This is the
    whole payload the registry had before: one redundant mapping."""
    _vente(cfg, extra_forms=["vent"], wordlist=((159, "vent"),))
    write_json(cfg.registry_local / "form_to_lemma.json", {"vent": "vente"})
    resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert c["vent"]["members"][0]["why"] == "flex_table"
    assert read_json(cfg.review_dir / "override_accepted.json") == []


def test_the_override_channel_still_refuses_the_other_four_rejections(cfg):
    """`unrelated` is the classifier declining to have an opinion. The other
    rejections are positive claims about the article and still stand, or the
    curated door becomes a way to put `-ske`, `nr.` and `en bloc` on a card."""
    abbrev = make_entry("11000710", "nr.", pos_key="sb.",
                        senses=[make_sense("21001301", "nummer")])
    multi = make_entry("11000711", "en bloc", pos_key="adv.",
                       senses=[make_sense("21001302", "samlet")])
    affix = make_entry("11000712", "-ske", pos_key="sidsteled",
                       senses=[make_sense("21001303", "sidsteled")])
    entries = {e["entry_id"]: e for e in (abbrev, multi, affix)}
    write_workspace(cfg, entries,
                    [(1, "nr"), (2, "en"), (3, "sker")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    # `en` -> `en bloc` is the multiword shape that matters: squash("en") is not
    # squash("en bloc"), so the variant branch does not save it -- whereas
    # `udenfor`/`uden for` DO squash together and are still admitted.
    write_json(cfg.registry_local / "form_to_lemma.json",
               {"nr": "nr.", "en": "en bloc", "sker": "-ske"})
    _allow_override_problems(cfg, 3)
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert c["nr"]["members"] == [] and c["en"]["members"] == []
    assert c["sker"]["members"] == []
    reasons = {r["word"]: r["reason"]
               for r in read_json(cfg.report_dir / "recovery_rejected.json")}
    assert reasons["nr"] == "recovery_abbreviation"
    assert reasons["en"] == "recovery_multiword_neighbour"
    # the affix page never even reaches judged(): indexable() still blocks it on
    # the curated path, so it leaves no recovery row -- only an override problem
    assert "sker" not in reasons
    assert report["override_problems"] == 3
    assert read_json(cfg.review_dir / "override_accepted.json") == []


def test_an_override_target_rejected_only_by_its_own_word_is_still_reachable(cfg):
    """FIX-B, the self-referential lockout. `indstil` is the imperative of
    `indstille`; nothing else on the wordlist reaches that article, so the ONLY
    verdict on it is `indstil`'s own `unrelated` -- which put it in
    rejected_everywhere and made indexable() withhold it from the very word the
    mapping was written for. The candidate was dropped BEFORE judged(), so it
    left no `recovery_` audit row either: four articles (planlaegge, oversaette,
    fascinere, indstille -- 10 senses) were unreachable and invisible."""
    e = make_entry("11023568", "indstille", pos_key="vb.",
                   forms=["indstiller", "indstillede", "indstillet"],
                   senses=[make_sense("21023568", "justere")])
    classification = {
        "indstil": {"members": [], "xrefs": [],
                    "rejected": [{"entry_id": "11023568",
                                  "headword": "indstille", "pos_key": "vb.",
                                  "reason": "unrelated"}],
                    "resolved_by": None},
    }
    write_workspace(cfg, {"11023568": e}, [(4825, "indstil")],
                    classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    assert rejected_everywhere_ids(classification) == {"11023568"}
    write_json(cfg.registry_local / "form_to_lemma.json",
               {"indstil": "indstille"})
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["indstil"]["members"]] == ["11023568"]
    assert c["indstil"]["members"][0]["evidence"] == "override"
    assert report["override_problems"] == 0
    # ...and the automatic layer is NOT relaxed: rejected_everywhere still keeps
    # that article out of the reverse index built for every other word.
    assert report["entries_in_reverse_index"] == 0


def test_the_curated_path_still_blocks_a_demoted_or_affix_target(cfg):
    """The filters that FIX-B did NOT relax. rejected_everywhere exists to keep
    the erbium-class symbol article out of the recovery door; symbol articles are
    demoted, and demotion (plus the affix shape) is what actually guards it."""
    sym = make_entry("46000779", "Ta", pos_key="symbol",
                     senses=[make_sense("21000779", "kemisk tegn for tantal")])
    classification = {"ta": {"members": [], "xrefs": [],
                             "rejected": [{"entry_id": "46000779",
                                           "headword": "Ta", "pos_key": "symbol",
                                           "reason": "case_only_demoted_pos"}],
                             "resolved_by": None}}
    write_workspace(cfg, {"46000779": sym}, [(1, "ta")],
                    classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "form_to_lemma.json", {"ta": "Ta"})
    _allow_override_problems(cfg, 1)
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert c["ta"]["members"] == []
    assert report["override_problems"] == 1
    rows = read_json(cfg.report_dir / "unresolved.json")
    assert [r["reason"] for r in rows] == ["no_survivor"]


def test_no_survivor_is_the_reason_when_every_candidate_is_refused(cfg):
    """The reason code the 140-mapping failure actually surfaced as. It was
    reported by a counter no gate read: all_rejected 207 -> 56 and no_survivor
    0 -> 140 with every gate row green."""
    abbrev = make_entry("11000710", "nr.", pos_key="sb.",
                        senses=[make_sense("21001301", "nummer")])
    write_workspace(cfg, {"11000710": abbrev}, [(1, "nr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "form_to_lemma.json", {"nr": "nr."})
    _allow_override_problems(cfg, 1)
    report = resolve_run(cfg, _reg(cfg))
    rows = read_json(cfg.report_dir / "unresolved.json")
    assert [(r["word"], r["reason"]) for r in rows] == [("nr", "no_survivor")]
    assert report["unresolved_by_reason"]["no_survivor"] == 1
    assert report["unresolved_by_reason"]["override_lemma_not_crawled"] == 0


def test_g_override_stops_the_build_and_writes_the_evidence_first(cfg):
    """The gate stage 21 shipped without. `override_problems` went 0 -> 140 --
    every mapping in the registry failing -- and `ankidkdeck gates` printed
    11 PASS / 0 FAIL. It also has to leave the rows behind when it fires: a
    curator handed a count and no rows cannot act on it."""
    _vente(cfg, wordlist=((159, "vent"),))
    write_json(cfg.registry_local / "form_to_lemma.json", {"vent": "vent"})
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-OVERRIDE" in str(exc.value)
    # written BEFORE the gate ran, on purpose
    rows = read_json(cfg.report_dir / "unresolved.json")
    assert [r["word"] for r in rows] == ["vent"]
    assert read_json(cfg.report_dir / "resolve_report.json")["override_problems"] == 1
    report = read_json(cfg.report_dir / "gates_report.json")
    assert report["failed"] == ["G-OVERRIDE"]
    assert "G-OVERRIDE" in report["gate_ids_with_a_verdict"]
