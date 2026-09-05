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


# --------------------------------------------------------------------------
# Layer 4: DDO's dotted abbreviation entry (owner policy B, 2026-08-26)
# --------------------------------------------------------------------------

def _baselines(cfg, **kw) -> None:
    """G-SUPPRESS is baselined at the SHIPPED population size, so any test that
    layers a local known_no_entry / wordlist_invalid row on top has to declare
    the new size out loud. That is the gate working, not a test annoyance."""
    write_json(cfg.registry_local / "gates.json", dict(kw))


def _shipped(cfg, name: str) -> int:
    from ankidkdeck.registry import Registry
    return len(getattr(Registry(cfg), name))


def _gate_row(cfg, gate_id: str) -> dict:
    return next(r for r in read_json(cfg.report_dir / "gates_report.json")
                ["results"] if r["id"] == gate_id)


def test_a_dotless_wordlist_word_adopts_ddos_dotted_abbreviation_entry(cfg):
    """Owner policy B. `hr`(rank 179) had a v2.1 card; DDO answers /hr with
    `hr.`, `herre` and `HR`, and every one of them was rejected -- the
    abbreviation rule by design, `herre` as unrelated (an abbreviation is not an
    inflection of the word it abbreviates) and `HR` as a demoted case-only
    match. The adopted member carries bucket `abbreviation`, which stage 30
    turns into relation `abbreviation`: a hidden searchable form, never a
    card-face variant."""
    hr = make_entry("11021497", "hr.", pos_key="sb.", source_words=["hr"],
                    senses=[make_sense("21000001", "titel")])
    herre = make_entry("11020561", "herre", pos_key="sb.", source_words=["hr"],
                       forms=["herren", "herrer"],
                       senses=[make_sense("21000002", "mand")])
    hr_fork = make_entry("40001843", "HR", pos_key="fork.", source_words=["hr"],
                         senses=[make_sense("21000003", "human resources")])
    entries = {e["entry_id"]: e for e in (hr, herre, hr_fork)}
    write_workspace(cfg, entries, [(179, "hr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [(m["entry_id"], m["bucket"], m["evidence"])
            for m in c["hr"]["members"]] == [("11021497", "abbreviation",
                                              "abbreviation")]
    assert c["hr"]["resolved_by"] == "abbreviation"
    assert report["abbreviation_accepted"] == 1
    assert report["abbreviation_words"] == ["hr"]
    assert read_json(cfg.report_dir / "unresolved.json") == []
    rows = read_json(cfg.review_dir / "abbreviation_accepted.json")
    assert [(r["word"], r["abbreviation_lemma"], r["demoted_pos"])
            for r in rows] == [("hr", "hr.", False)]
    # G-ADMIT has a verdict and it is the admitted word, not just a count.
    admit = _gate_row(cfg, "G-ADMIT")
    assert admit["ok"] is True
    assert admit["detail"]["words"] == ["hr"]
    assert admit["detail"]["admitted_beyond_the_baseline"] == []


def test_the_abbreviation_layer_is_case_insensitive(cfg):
    """`Dr`(358) and `Mrs`(598) are capitalised in the wordlist and lower case
    in DDO. The raw-NFC rule missed both, which is why round 2 recorded the
    case-sensitivity defect and the abbreviation policy as ONE piece of work:
    relaxing the policy alone would still have left these two out."""
    dr = make_entry("11009566", "dr.", pos_key="fork.", source_words=["Dr"],
                    senses=[make_sense("21000004", "doktor")])
    write_workspace(cfg, {"11009566": dr}, [(358, "Dr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["Dr"]["members"]] == ["11009566"]
    assert c["Dr"]["members"][0]["bucket"] == "abbreviation"
    # the REAL demoted flag, and `fork.` IS demoted: layer 4 drops that filter
    # on purpose and is the only layer that does.
    assert c["Dr"]["members"][0]["demoted"] is True


def test_a_word_that_owns_a_card_never_adopts_a_dotted_lookalike(cfg):
    """The negative case, and the reason layer 4 is LAST. 22 words reach a dotted
    article on their own page while already owning a card by exact match; `min`
    (pron.) must not grow a `min.` = `minut` meaning block. Layer 1 returns
    first, so exclusive exactness is protected by the layer ORDER."""
    real = make_entry("11033715", "min", pos_key="pron.", source_words=["min"],
                      senses=[make_sense("21000005", "tilhoerende mig")])
    abbrev = make_entry("11033713", "min.", pos_key="fork.", source_words=["min"],
                        senses=[make_sense("21000006", "minut")])
    entries = {e["entry_id"]: e for e in (real, abbrev)}
    classification = {"min": {"members": [{"entry_id": "11033715",
                                           "bucket": "exact_cs",
                                           "demoted": False}],
                              "xrefs": [],
                              "rejected": [{"entry_id": "11033713",
                                            "headword": "min.",
                                            "pos_key": "fork.",
                                            "reason": "abbreviation"}],
                              "resolved_by": "forward"}}
    write_workspace(cfg, entries, [(53, "min")], classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["min"]["members"]] == ["11033715"]
    assert report["abbreviation_accepted"] == 0
    assert report["forward"] == 1


def test_the_abbreviation_layer_runs_after_the_override(cfg):
    """`vaer` reaches `vaer.`(fork.) on its own page AND is the imperative of
    `vaere`, which a curated override binds at layer 2. An abbreviation layer
    placed before the override -- or before the reverse index -- would hand the
    rank-500 imperative to an abbreviation card."""
    vaere = make_entry("12007519", "vaere", pos_key="vb.",
                       source_words=["vaer"], forms=["er", "var"],
                       senses=[make_sense("21000007", "eksistere")])
    abbrev = make_entry("12007490", "vaer.", pos_key="fork.",
                        source_words=["vaer"],
                        senses=[make_sense("21000008", "vaerelse")])
    entries = {e["entry_id"]: e for e in (vaere, abbrev)}
    write_workspace(cfg, entries, [(500, "vaer")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "form_to_lemma.json", {"vaer": "vaere"})
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["vaer"]["members"]] == ["12007519"]
    assert c["vaer"]["resolved_by"] == "override"
    assert report["abbreviation_accepted"] == 0


def test_the_abbreviation_layer_cannot_reach_a_symbol_homograph(cfg):
    """`th` adopts `th.`(fork.) and still rejects `Th`(thorium). Layer 4 drops
    the demoted filter, so this is the one that matters: the guard is
    is_dotted_abbreviation(), which admits only the query plus periods, and an
    element symbol carries no period."""
    abbrev = make_entry("12000873", "th.", pos_key="fork.", source_words=["th"],
                        senses=[make_sense("21000009", "thorium-forkortelse")])
    symbol = make_entry("46001345", "Th", pos_key="symbol", source_words=["th"],
                        senses=[make_sense("21000010", "kemisk tegn")])
    entries = {e["entry_id"]: e for e in (abbrev, symbol)}
    write_workspace(cfg, entries, [(3905, "th")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert [m["entry_id"] for m in c["th"]["members"]] == ["12000873"]


def test_the_abbreviation_layer_only_looks_at_the_words_own_page(cfg):
    """A dotted lemma somewhere else in the corpus is not evidence. The relation
    means "DDO answered THIS query with the abbreviation entry", so the candidate
    set is source_words, exactly as stage 22 builds it."""
    abbrev = make_entry("11026086", "kl.", pos_key="fork.",
                        source_words=["klokken"],
                        senses=[make_sense("21000011", "klokken")])
    write_workspace(cfg, {"11026086": abbrev}, [(581, "kl")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {"kl": {"status": "nohit"}})
    report = resolve_run(cfg, _reg(cfg))
    assert report["abbreviation_accepted"] == 0
    assert [r["reason"] for r in
            read_json(cfg.report_dir / "unresolved.json")] == ["nohit"]


def test_an_abbreviation_beats_known_no_entry(cfg):
    """Layer order, the other end: known_no_entry stays LAST so that a word DDO
    starts covering is discovered by the rescue layers rather than suppressed by
    the registry (owner note on decision 10.45)."""
    abbrev = make_entry("11036233", "nr.", pos_key="fork.", source_words=["nr"],
                        senses=[make_sense("21000012", "nummer")])
    write_workspace(cfg, {"11036233": abbrev}, [(1523, "nr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "known_no_entry.json", {"nr": "test"})
    _baselines(cfg, known_no_entry_max=_shipped(cfg, "known_no_entry") + 1)
    report = resolve_run(cfg, _reg(cfg))
    assert report["abbreviation_accepted"] == 1
    assert report["known_no_entry"] == 0


def _hr_workspace(cfg) -> None:
    hr = make_entry("11021497", "hr.", pos_key="sb.", source_words=["hr"],
                    senses=[make_sense("21000015", "titel")])
    write_workspace(cfg, {"11021497": hr}, [(179, "hr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})


def test_the_abbreviation_audit_table_survives_a_second_resolve(cfg):
    """review/abbreviation_accepted.json is the sheet a human signs for the 19
    edges owner policy B admits against the classifier's verdict -- and a second
    standalone `resolve` used to write `[]` over it, with no gate looking.

    Mechanism: this stage rewrites classification.json with the members layer 4
    attached, so the retry returns at LAYER 1; a table appended to while layer 4
    ran therefore ends up empty while the cards stay (reviewer B, round 4,
    MINOR-4). It is derived from the members instead, which also makes it a pure
    function of what is on disk."""
    _hr_workspace(cfg)
    first = resolve_run(cfg, _reg(cfg))
    rows_1 = read_json(cfg.review_dir / "abbreviation_accepted.json")
    assert first["abbreviation"] == 1 and first["abbreviation_accepted"] == 1
    second = resolve_run(cfg, _reg(cfg))
    # Layer 1 answers this time -- that part is NOT idempotent and is not what
    # this test is about.
    assert second["forward"] == 1 and second["abbreviation"] == 0
    # The audit table and the gate's population are.
    assert second["abbreviation_accepted"] == 1
    assert second["abbreviation_words"] == ["hr"]
    assert read_json(cfg.review_dir / "abbreviation_accepted.json") == rows_1
    assert _gate_row(cfg, "G-ADMIT")["detail"]["words"] == ["hr"]


def test_g_admit_fails_when_a_word_outside_the_baseline_is_admitted(cfg):
    """The gate round 4 forgot. Layer 4 is the only channel in the pipeline that
    turns an article the classifier REJECTED into a card, so its population is
    baselined by WORD -- gates.json:abbreviation_accepted_words -- exactly as the
    two suppression registries added in the same change are baselined by count.
    A 20th admission stops the build until the registry says who signed for it.

    By word and not only by count because a SWAP -- one admission lost, another
    gained -- leaves the total at 19."""
    zz = make_entry("11099998", "zz.", pos_key="fork.", source_words=["zz"],
                    senses=[make_sense("21000017", "noget")])
    write_workspace(cfg, {"11099998": zz}, [(4999, "zz")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-ADMIT" in str(exc.value)
    detail = _gate_row(cfg, "G-ADMIT")["detail"]
    assert detail["admitted_beyond_the_baseline"] == ["zz"]
    # The gate runs after the writes, so the curator gets the rows, not just a
    # count -- the same rule G-OVERRIDE follows.
    assert [r["word"] for r in
            read_json(cfg.review_dir / "abbreviation_accepted.json")] == ["zz"]


def test_g_admit_fails_when_one_baselined_word_admits_two_articles(cfg):
    """The second assertion, and it is not theoretical: `pr` reaches BOTH
    `pr.`(fork., 1 sense) and `pr.`(praep., 6 senses) on its own page, and layer
    4 does not choose between them -- is_dotted_abbreviation() is a shape test,
    not a pos test. `pr` is out of reach only because layer 1 binds it to `PR`
    first, so a word already on the baseline list could still double its cards
    without changing the word set."""
    a = make_entry("11021497", "hr.", pos_key="sb.", source_words=["hr"],
                   senses=[make_sense("21000018", "titel")])
    b = make_entry("11021498", "hr.", pos_key="fork.", source_words=["hr"],
                   senses=[make_sense("21000019", "en anden hr.")])
    write_workspace(cfg, {a["entry_id"]: a, b["entry_id"]: b},
                    [(179, "hr")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    _baselines(cfg, abbreviation_accepted_max=1)
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-ADMIT" in str(exc.value)
    detail = _gate_row(cfg, "G-ADMIT")["detail"]
    # the WORD is signed for; the edge count is what fires
    assert detail["words"] == ["hr"]
    assert detail["admitted_beyond_the_baseline"] == []
    assert detail["rows"] == 2 and detail["over_max_rows"] is True


def test_g_admit_is_fail_closed_when_the_baseline_key_is_missing(cfg):
    """A deleted registry line must not silently un-baseline the channel. An
    absent abbreviation_accepted_words is an EMPTY signed list, so every
    admission is beyond it -- the same fail-closed default G-SUPPRESS uses for
    its two `_max` keys. (The overlay empties the list here; a missing key
    reaches the gate as the same empty tuple.)"""
    _hr_workspace(cfg)
    write_json(cfg.registry_local / "gates.json",
               {"abbreviation_accepted_words": []})
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-ADMIT" in str(exc.value)
    assert _gate_row(cfg, "G-ADMIT")["detail"][
        "admitted_beyond_the_baseline"] == ["hr"]


# --------------------------------------------------------------------------
# Layer 0: registry/wordlist_invalid.json (owner decision 10.5)
# --------------------------------------------------------------------------

def test_an_invalid_wordlist_row_never_binds_and_never_reaches_unresolved(cfg):
    """The registry route for the OCR damage in the sha-pinned wordlist. The row
    is skipped before layer 1, so it cannot bind by any route, and it leaves
    unresolved.json -- while the wordlist FILE stays byte-identical, which is
    the whole point: wordlist_sha256 is the foundation of every GUID, so no
    row's rank may move."""
    real = make_entry("11000100", "til", pos_key="praep.", source_words=["til"],
                      senses=[make_sense("21000013", "retning")])
    write_workspace(cfg, {"11000100": real}, [(11, "til"), (459, "tii")],
                    classification={})
    write_json(cfg.json_dir / "fetch_ledger.json",
               {"til": {"status": "ok", "article_count": 1},
                "tii": {"status": "nohit"}})
    write_json(cfg.registry_local / "wordlist_invalid.json",
               {"tii": {"correct": "til", "reason": "ocr"}})
    _baselines(cfg, wordlist_invalid_max=_shipped(cfg, "wordlist_invalid") + 1)
    report = resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert c["tii"]["members"] == []
    assert c["tii"]["resolved_by"] == "wordlist_invalid"
    assert report["wordlist_invalid"] == 1
    assert report["invalid_rows_that_bound"] == 0
    assert [r["word"] for r in
            read_json(cfg.report_dir / "unresolved.json")] == []


def test_g_suppress_fails_when_an_invalid_row_would_delete_a_binding(cfg):
    """The failure this gate exists for. An invalid row that DID bind means the
    registry is deleting a real card edge, and the only visible symptom would be
    a card quietly disappearing between two builds."""
    real = make_entry("11000101", "tii", pos_key="sb.", source_words=["tii"],
                      senses=[make_sense("21000014", "noget")])
    classification = {"tii": {"members": [{"entry_id": "11000101",
                                           "bucket": "exact_cs",
                                           "demoted": False}],
                              "xrefs": [], "rejected": [],
                              "resolved_by": "forward"}}
    write_workspace(cfg, {"11000101": real}, [(459, "tii")],
                    classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "wordlist_invalid.json",
               {"tii": {"correct": "til", "reason": "ocr"}})
    _baselines(cfg, wordlist_invalid_max=_shipped(cfg, "wordlist_invalid") + 1)
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-SUPPRESS" in str(exc.value)
    rows = read_json(cfg.report_dir / "resolve_report.json")
    assert rows["invalid_rows_that_bound"] == 1
    # The clearing line itself, which nothing covered: layer 0 empties the
    # members AND records what it emptied. Removing either half is a silent
    # behaviour change today (reviewer A, round 4, mutation 6).
    c = read_json(cfg.json_dir / "classification.json")
    assert c["tii"]["members"] == []
    assert c["tii"]["suppressed_by_wordlist_invalid"] == ["11000101"]


def test_g_suppress_fails_when_the_two_registries_disagree_about_a_word(cfg):
    """"DDO has no such word" and "this row is not a word" are contradictory
    claims, and a row in both files is the shape a copy-paste mistake takes: the
    53 OCR rows were carved OUT of the 545-row nohit population."""
    write_workspace(cfg, {}, [(459, "tii")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "wordlist_invalid.json",
               {"tii": {"correct": "til", "reason": "ocr"}})
    write_json(cfg.registry_local / "known_no_entry.json", {"tii": "test"})
    _baselines(cfg, wordlist_invalid_max=_shipped(cfg, "wordlist_invalid") + 1,
               known_no_entry_max=_shipped(cfg, "known_no_entry") + 1)
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-SUPPRESS" in str(exc.value)


def test_g_suppress_fails_when_a_suppression_list_grows_past_its_baseline(cfg):
    """The condition round 1 (section 4.1) and round 2 (section 9.4) both put on
    filling known_no_entry at all: unresolved.json is the entire enforcement
    surface for the owner's skip-and-record ruling, so a registry that subtracts
    from it needs the baseline case_only_members_max and
    all_demoted_families_max already have."""
    write_workspace(cfg, {}, [(1, "zzz")], classification={})
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    _baselines(cfg, known_no_entry_max=_shipped(cfg, "known_no_entry") - 1)
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-SUPPRESS" in str(exc.value)
    detail = next(r for r in read_json(cfg.report_dir / "gates_report.json")
                  ["results"] if r["id"] == "G-SUPPRESS")["detail"]
    assert detail["over_baseline"]["known_no_entry"] is True


def test_a_deleted_binding_still_fails_g_suppress_on_a_second_run(cfg):
    """G-SUPPRESS's third assertion used to SELF-HEAL on a retry.

    Layer 0 cleared `members`, and this stage rewrites classification.json
    BEFORE the gates run -- so a second standalone `resolve` read back an
    already-emptied row, had nothing to report, and the fatal gate PASSED with
    the binding still deleted. After that second run overwrote
    resolve_report.json the count was gone from every artifact too: a card had
    quietly disappeared and no file on disk said so (reviewer A, round 4,
    MAJOR-1, reproduced end to end).

    The fix is that layer 0 persists what it suppressed into the classification
    row, so the condition is re-derivable on every later run from the artifact
    itself rather than from the members it just destroyed."""
    real = make_entry("11000101", "tii", pos_key="sb.", source_words=["tii"],
                      senses=[make_sense("21000016", "noget")])
    classification = {"tii": {"members": [{"entry_id": "11000101",
                                           "bucket": "exact_cs",
                                           "demoted": False}],
                              "xrefs": [], "rejected": [],
                              "resolved_by": "forward"}}
    write_workspace(cfg, {"11000101": real}, [(459, "tii")],
                    classification=classification)
    write_json(cfg.json_dir / "fetch_ledger.json", {})
    write_json(cfg.registry_local / "wordlist_invalid.json",
               {"tii": {"correct": "til", "reason": "ocr"}})
    _baselines(cfg, wordlist_invalid_max=_shipped(cfg, "wordlist_invalid") + 1)
    with pytest.raises(FatalError):
        resolve_run(cfg, _reg(cfg))
    c = read_json(cfg.json_dir / "classification.json")
    assert c["tii"]["members"] == []
    assert c["tii"]["suppressed_by_wordlist_invalid"] == ["11000101"]
    # RUN 2. Nothing was fixed, the registry is unchanged, and the input is now
    # the classification.json run 1 wrote. It must still be fatal.
    with pytest.raises(FatalError) as exc:
        resolve_run(cfg, _reg(cfg))
    assert "G-SUPPRESS" in str(exc.value)
    assert read_json(cfg.report_dir / "resolve_report.json"
                     )["invalid_rows_that_bound"] == 1
    detail = _gate_row(cfg, "G-SUPPRESS")["detail"]
    assert [r["word"] for r in
            detail["invalid_rows_that_bound_an_article"]] == ["tii"]
