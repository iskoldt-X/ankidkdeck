"""The four-bucket classifier, on synthetic articles.

The named 2025 casualties are the specification: `kan` must keep `kunne` and
reject `khan`; `kvinder` must keep `kvinde` and reject `-kvinde`; `godt` (15
senses of its own, rank 53) must keep its own card and only cross-reference
`god`. Real saved pages for each case are the job of test_parse_pages.py; these
are the decision-table tests, which run everywhere.

Two of them assert CURRENT behaviour that the final guide would decide
differently. Both are flagged in-line for the reviewers rather than quietly
"fixed" here, because changing the classifier changes the card set.
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


def test_the_case_only_bucket_is_unreachable_today():
    """CURRENT BEHAVIOUR, flagged for review. Two independent shadows.

    Guide 1.5(a) splits bucket 1 into exact_cs (codepoint-equal) and exact_ci
    (equal only after casefold) so that `Er`, the erbium symbol, is demoted and
    can never own the rank-1 card. Neither the exact_ci bucket nor the
    case_only_demoted_pos rejection can be reached on real parser output:

      1. stage 20 always puts nk(lemma) into form_index, and classify_one()
         tests the form bucket first -> ("form", "flex_table");
      2. even with an empty form_index, _squash() casefolds before comparing,
         so _is_variant() is already true -> ("variant",
         "orthographic_variant").

    Both answers ACCEPT the erbium article as a member of `er`. The demotion
    still bites later (stage 30's anchor_of prefers a non-demoted article and
    G-ANCHOR fails a demoted anchor that had an alternative), so this is not a
    live card defect -- but the bucket split the guide asked for is not
    implemented. Reviewers decide; these assertions make the change visible.
    """
    e = make_entry("11000008", "Er", pos_key="symbol",
                   senses=[make_sense("21000002", "kemisk tegn for erbium")])
    assert "er" in e["form_index"]
    assert classify_one("er", e, DEMOTED, ALIASES) == ("form", "flex_table")

    e2 = make_entry("11000007", "Er", pos_key="symbol")
    e2["form_index"] = []
    assert classify_one("er", e2, DEMOTED, ALIASES) == ("variant",
                                                        "orthographic_variant")


def test_solid_vs_spaced_variant_currently_rejected():
    """CURRENT BEHAVIOUR, flagged for review.

    Guide 1.5(d) wants /udenfor to keep `udenfor` as exact AND `uden for`
    (praep., 4 senses, no flex table) as a VARIANT, which is what 2025 shipped.
    The multiword guard fires first, so today the article is rejected as a
    multiword neighbour and its 4 senses do not reach the card.
    """
    e = make_entry("11000009", "uden for", pos_key="præp.",
                   senses=[make_sense("21000003", "paa ydersiden af")])
    assert classify_one("udenfor", e, DEMOTED, ALIASES) == ("reject",
                                                            "multiword_neighbour")


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
    ("HUS", "hus", "form"),      # see test_case_only_match_currently_lands...
])
def test_table(word, lemma, expected):
    e = make_entry("11021722", lemma, forms=["huse", "huset", "husene"])
    assert classify_one(word, e, DEMOTED, ALIASES)[0] == expected
