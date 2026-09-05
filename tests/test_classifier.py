"""The four-bucket classifier, on synthetic articles.

The named 2025 casualties are the specification: `kan` must keep `kunne` and
reject `khan`; `kvinder` must keep `kvinde` and reject `-kvinde`; `godt` (15
senses of its own, rank 53) must keep its own card and only cross-reference
`god`; `udenfor` must keep `uden for` as a variant. Real saved pages for each
case are the job of test_parse_pages.py; these are the decision-table tests,
which run everywhere.
"""

import pytest
from conftest import make_entry, make_sense, write_workspace

from ankidkdeck.stages.s22_classify import BUCKET_ORDER, classify_one
from ankidkdeck.stages.s22_classify import run as classify_run

DEMOTED = {"symbol", "fork.", "forkortelse", "førsteled", "sidsteled",
           "suffiks", "præfiks"}
ALIASES = {("ok", "o.k."), ("næ", "næh"), ("check", "tjek")}


def test_bucket_order_decides_the_headline():
    # exact_cs > form > variant > exact_ci. 09 takes sorted_entries[0] as the
    # primary entry, so this ordering is what keeps erbium off the rank-1 card.
    assert (BUCKET_ORDER["exact_cs"] < BUCKET_ORDER["form"]
            < BUCKET_ORDER["variant"] < BUCKET_ORDER["exact_ci"])


def test_exact_codepoint_match():
    e = make_entry("11018364", "godt", pos_key="adv.",
                   senses=[make_sense("21000001", "i hoej grad")])
    assert classify_one("godt", e, DEMOTED, ALIASES) == ("exact_cs", "exact")


def test_form_match_comes_from_the_flex_table():
    god = make_entry("11018326", "god", pos_key="adj.",
                     forms=["godt", "gode", "bedre", "bedst"])
    assert classify_one("godt", god, DEMOTED, ALIASES) == ("form", "flex_table")
    # kan is in kunne's flex table and NOT in khan's: tab labels are refuted,
    # containment is the signal.
    kunne = make_entry("11036129", "kunne", pos_key="vb.",
                       forms=["kan", "kunne", "kunnet"])
    khan = make_entry("11034911", "khan", pos_key="sb.", forms=["khanen", "khaner"])
    assert classify_one("kan", kunne, DEMOTED, ALIASES)[0] == "form"
    assert classify_one("kan", khan, DEMOTED, ALIASES) == ("reject", "unrelated")


def test_affix_pages_are_rejected_by_shape():
    # -kvinde would otherwise be merged into the kvinde card and vanish.
    for lemma in ("-kvinde", "-ske", "for-"):
        e = make_entry("11000001", lemma, pos_key="sidsteled")
        assert classify_one("kvinder", e, DEMOTED, ALIASES) == ("reject", "affix")


def test_multiword_neighbour_is_rejected():
    e = make_entry("11000002", "en bloc", pos_key="adv.")
    assert classify_one("bloc", e, DEMOTED, ALIASES) == ("reject",
                                                         "multiword_neighbour")


def test_abbreviation_page_is_rejected():
    e = make_entry("11000003", "min.", pos_key="fork.")
    assert classify_one("min", e, DEMOTED, ALIASES) == ("reject", "abbreviation")


def test_variant_via_squash_ignores_hyphens():
    e = make_entry("11000004", "e-mail", pos_key="sb.")
    assert classify_one("email", e, DEMOTED, ALIASES) == ("variant",
                                                          "orthographic_variant")


def test_variant_via_alias_pair():
    e = make_entry("11000005", "o.k.", pos_key="adj.")
    assert classify_one("ok", e, DEMOTED, ALIASES) == ("variant",
                                                       "orthographic_variant")


def test_variant_via_alternative_spelling_row():
    e = make_entry("11000006", "hyggelig", pos_key="adj.",
                   alt_spellings=[{"form": "hyggeligt", "official": True}])
    assert classify_one("hyggeligt", e, DEMOTED, ALIASES)[0] in ("form", "variant")


def test_a_demoted_case_only_article_is_rejected():
    """Guide 1.5(a): `Er`, the erbium symbol, must be REJECTED for the word
    `er`, not admitted as an inflection of itself.

    This is the defect that made exact_ci and case_only_demoted_pos unreachable
    for every possible input: nk(lemma) was unconditionally in form_index, so
    the form bucket -- tested first -- matched every case-only pair. 2025's own
    download_map proves the pair is real (er__0.html final_url
    ?select=Er&query=er, i.e. the shipped rank-1 card is erbium).
    """
    e = make_entry("11000008", "Er", pos_key="symbol",
                   senses=[make_sense("21000002", "kemisk tegn for erbium")])
    # the lemma key is in form_index (stage 21's reverse index wants it) but NOT
    # in paradigm_index, which is what the form bucket now tests
    assert "er" in e["form_index"]
    assert "er" not in e["paradigm_index"]
    assert classify_one("er", e, DEMOTED, ALIASES) == ("reject",
                                                       "case_only_demoted_pos")


def test_a_real_case_only_homograph_is_kept_as_exact_ci():
    """`I` (pron., 2nd person plural -- a real word with 2 senses and 7 fixed
    expressions) must reach the card for `i` as an exact_ci MEMBER. It used to
    be classified `form`, then stripped into xrefs by exclusive exactness, so
    none of the three /i articles reached any family."""
    e = make_entry("11022728", "I", pos_key="pron.",
                   senses=[make_sense("21000010", "2. person flertal"),
                           make_sense("21000011", "hoeflig tiltaleform")])
    assert classify_one("i", e, DEMOTED, ALIASES) == ("exact_ci", "case_only")
    # and the symbol homograph on the same page is still rejected
    sym = make_entry("11022726", "I", pos_key="symbol",
                     senses=[make_sense("21000012", "kemisk tegn for jod")])
    assert classify_one("i", sym, DEMOTED, ALIASES) == ("reject",
                                                        "case_only_demoted_pos")


def test_a_deprecated_spelling_is_not_a_classification_signal():
    """`khan` carries `<span class="diskret">kan</span>` -- 'nu uofficiel
    stavemaade'. That deprecated form is searchable data, never evidence: it is
    the sole reason the kan/khan component existed, and with the old
    refused-merge fallback it put the Mongol ruler on the rank-23 `kunne`
    card."""
    khan = make_entry("11025847", "khan", pos_key="sb.",
                      forms=["khanen", "khaner", "khanerne"],
                      alt_spellings=[{"form": "kan", "official": False}],
                      senses=[make_sense("21000020", "mongolsk fyrste")])
    assert "kan" not in khan["form_index"]
    assert "kan" not in khan["paradigm_index"]
    assert classify_one("kan", khan, DEMOTED, ALIASES) == ("reject", "unrelated")
    # ...while a real inflection cell still wins bucket 2
    kunne = make_entry("11028611", "kunne", pos_key="vb.",
                       forms=["kan", "kunne", "kunnet"],
                       senses=[make_sense("21000021", "vaere i stand til")])
    assert classify_one("kan", kunne, DEMOTED, ALIASES) == ("form", "flex_table")


def test_solid_vs_spaced_variant_is_kept():
    """Guide 1.5(d) / F02: /udenfor keeps `udenfor` as exact AND `uden for`
    (praep., 4 senses, no flex table) as a VARIANT -- which is what 2025
    shipped, and the same shape for indenfor, overfor, ovenpaa, bagefter,
    bagved, nedenunder, indeni, udenom, udover. The multiword guard used to
    fire first and drop all four senses."""
    e = make_entry("11000009", "uden for", pos_key="præp.",
                   senses=[make_sense("21000003", "paa ydersiden af")])
    assert classify_one("udenfor", e, DEMOTED, ALIASES) == (
        "variant", "orthographic_variant")


def test_genuine_multiword_neighbours_still_reject():
    """The multiword reject moved AFTER the variant test, so it has to be shown
    that it still fires: these squash to something other than the query."""
    for q, lemma in (("bloc", "en bloc"), ("alle", "alle sammen"),
                     ("alle", "alle tiders"), ("gør", "gør det selv"),
                     ("en", "en gros")):
        e = make_entry("11000030", lemma, pos_key="adv.")
        assert classify_one(q, e, DEMOTED, ALIASES) == (
            "reject", "multiword_neighbour"), (q, lemma)


def test_exclusive_exactness_keeps_godt_as_its_own_card(cfg, registry):
    """godt has 15 senses of its own AND sits in god's flex table. Under the
    old rule it was absorbed into god and its card was deleted."""
    godt = make_entry("11018364", "godt", pos_key="adv.", pos_text="adverbium",
                      senses=[make_sense("2100010%d" % i, "sense %d" % i)
                              for i in range(3)],
                      source_words=["godt"])
    god = make_entry("11018326", "god", pos_key="adj.", pos_text="adjektiv",
                     forms=["godt", "gode", "bedre", "bedst"],
                     senses=[make_sense("21000200", "af hoej kvalitet")],
                     source_words=["godt", "god"])
    godte = make_entry("11018367", "godte", pos_key="vb.", source_words=["godt"])
    entries = {e["entry_id"]: e for e in (godt, god, godte)}
    write_workspace(cfg, entries, [(53, "godt"), (100, "god")])

    classify_run(cfg, registry)
    from ankidkdeck.util import read_json
    c = read_json(cfg.json_dir / "classification.json")

    assert [m["entry_id"] for m in c["godt"]["members"]] == ["11018364"]
    assert c["godt"]["xrefs"] == ["11018326"]        # god is a cross-reference
    assert any(r["entry_id"] == "11018367" for r in c["godt"]["rejected"])
    # god keeps its own exact article
    assert [m["entry_id"] for m in c["god"]["members"]] == ["11018326"]


def test_zero_sense_article_is_kept_but_marked():
    godte = make_entry("11018367", "godte", pos_key="vb.")
    assert godte["empty"] is True


@pytest.mark.parametrize("word,lemma,expected", [
    ("hus", "hus", "exact_cs"),
    ("huse", "hus", "form"),
    # case-only against a NON-demoted pos_key is bucket 4, not bucket 2: the
    # word is not in this article's flex table, it IS this article's headword
    # spelled differently.
    ("HUS", "hus", "exact_ci"),
    ("nothus", "hus", "reject"),
])
def test_table(word, lemma, expected):
    e = make_entry("11021722", lemma, forms=["huse", "huset", "husene"])
    assert classify_one(word, e, DEMOTED, ALIASES)[0] == expected


def test_the_form_bucket_tests_paradigm_cells_not_the_lemma_key():
    """The split that makes all four buckets reachable.

    paradigm_index = real inflection cells. form_index = cells + lemma key +
    official alternative spellings, because stage 21's reverse index needs the
    lemma key to answer an override lookup. Testing form_index in the form
    bucket is what shadowed exact_ci, case_only_demoted_pos AND variant.
    """
    e = make_entry("11018326", "god", pos_key="adj.",
                   forms=["godt", "gode", "bedre", "bedst"],
                   alt_spellings=[{"form": "goed", "official": True},
                                  {"form": "gaad", "official": False}])
    assert e["paradigm_index"] == sorted(["godt", "gode", "bedre", "bedst"])
    assert "god" in e["form_index"] and "god" not in e["paradigm_index"]
    assert "goed" in e["form_index"]        # official alternative: indexable
    assert "gaad" not in e["form_index"]    # deprecated: searchable data only
    # every bucket, and both rejections that used to be dead, are reachable
    reached = set()
    for word, entry in (
            ("god", e),
            ("godt", e),
            ("goed", e),
            ("GOD", make_entry("1", "god", pos_key="adj.")),
            ("er", make_entry("2", "Er", pos_key="symbol")),
            ("kvinder", make_entry("3", "-kvinde", pos_key="sidsteled")),
            ("min", make_entry("4", "min.", pos_key="fork.")),
            ("bloc", make_entry("5", "en bloc", pos_key="adv.")),
            ("xyzzy", make_entry("6", "hus", pos_key="sb."))):
        bucket, why = classify_one(word, entry, DEMOTED, ALIASES)
        reached.add(bucket if bucket != "reject" else "reject:" + why)
    assert reached == {"exact_cs", "form", "variant", "exact_ci",
                       "reject:case_only_demoted_pos", "reject:affix",
                       "reject:abbreviation", "reject:multiword_neighbour",
                       "reject:unrelated"}


# --------------------------------------------------------------------------
# The abbreviation relation (owner policy B, 2026-08-26)
# --------------------------------------------------------------------------

def test_the_abbreviation_rule_is_case_insensitive():
    """reviewer B N-1, round 2. The rule compared NFC bytes, so DDO's lower-case
    `dr.` did not match the wordlist's `Dr`(rank 358) or `mrs.` the wordlist's
    `Mrs`(598): both landed in `unrelated`, the classifier's "no opinion"
    fallback, while a rule for exactly that shape sat two lines above. Measured
    effect of the fix: 2 reason codes in review/rejected.json, 0 verdicts."""
    from ankidkdeck.stages.s22_classify import is_dotted_abbreviation
    dr = make_entry("11009566", "dr.", pos_key="fork.")
    mrs = make_entry("11034528", "mrs.", pos_key="fork.")
    assert classify_one("Dr", dr, DEMOTED, ALIASES) == ("reject", "abbreviation")
    assert classify_one("Mrs", mrs, DEMOTED, ALIASES) == ("reject", "abbreviation")
    assert is_dotted_abbreviation("Dr", dr)
    assert is_dotted_abbreviation("dr", dr)


def test_is_dotted_abbreviation_admits_only_the_query_plus_periods():
    """The guard that lets stage 21 drop the demoted filter safely.

    The element-symbol articles that put `symbol` in demoted_pos_keys are
    CASE-ONLY matches and carry no period, so none of them can pass -- which is
    what keeps `th`/`no`/`ca`/`kr`/`ma` from adopting thorium, nobelium, calcium,
    krypton and the milliampere while they adopt `th.`, `no.`, `ca.`, `kr.` and
    `ma.`. `o.k.` cannot pass either: nk("o.k.").rstrip(".") is "o.k".
    """
    from ankidkdeck.stages.s22_classify import is_dotted_abbreviation
    yes = [("th", "th."), ("no", "no."), ("ca", "ca."), ("kr", "kr."),
           ("ma", "ma."), ("hr", "hr."), ("st", "st."), ("Dr", "dr.")]
    no = [("th", "Th"), ("no", "No"), ("ca", "Ca"), ("kr", "Kr"), ("ma", "mA"),
          ("ok", "o.k."), ("min", "min"), ("hr", "herre"), ("no", "no-go"),
          ("on", "on the rocks"), ("nr", "nummer")]
    for word, lemma in yes:
        assert is_dotted_abbreviation(word, make_entry("1", lemma)), (word, lemma)
    for word, lemma in no:
        assert not is_dotted_abbreviation(word, make_entry("1", lemma)), (word, lemma)


def test_the_forward_page_still_rejects_an_abbreviation_entry():
    """Policy B is a stage-21 LAST-RESORT layer, not a relaxation here.

    22 words in the corpus already own a card through an exact match AND reach a
    dotted article on their own page (min/min., med/med., to/to., ti, tv, par,
    port, red, art, da, den, do, eks, el, fa, man, pr, sen, net, soe, soen,
    aarh). Accepting the abbreviation in the classifier would staple a second
    dictionary word's meaning block onto all 22 -- `min.` is `minut` -- so the
    forward verdict must stay a rejection and exclusive exactness must keep
    being protected by the LAYER ORDER, which no later edit to a filter can
    weaken.
    """
    assert classify_one("min", make_entry("11033713", "min.", pos_key="fork."),
                        DEMOTED, ALIASES) == ("reject", "abbreviation")
    assert classify_one("min", make_entry("11033715", "min", pos_key="pron."),
                        DEMOTED, ALIASES) == ("exact_cs", "exact")
