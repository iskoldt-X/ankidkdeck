"""The extraction contract: the per-field separator table.

A one-character change in extract.SEP silently invalidates 22,734 translation
cells x 4 languages -- that is how 2,007 English cards shipped with bare Danish
definitions. Three layers guard it: this file pins the table itself,
test_separators.py compares it byte-for-byte against the real 2025 corpus
(fixtures required), and G-SEP re-runs that comparison on the EXPORT path so a
release cannot be built with a drifted table.

The flex-cell split -- the one piece of logic no separator can express -- moved
to test_flex_table.py.
"""

import pytest

from ankidkdeck.extract import ARTICLE_SHA_SCHEMA, SEP, xt

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 is a runtime dependency")


def soup(html):
    return bs4.BeautifulSoup(html, "html.parser")


def td(html):
    return soup("<table><tr>%s</tr></table>" % html).select_one("td")


# The measured table (guide 1.1): space-joined fields reproduce the 2025 strings
# 100% and "" reproduces 0-85% of them; for the no-separator fields it is the
# other way round and " " actively corrupts them.
SPACE_FIELDS = ["definition", "grammar", "example", "etymology_raw",
                "expr_definition", "expr_example", "ipa", "udtale_label"]
NOSEP_FIELDS = ["expression", "pos_text", "headword", "sense_number",
                "example_source", "etymology_form", "onym_link", "orddannelse",
                "wordform_cell"]


@pytest.mark.parametrize("field", SPACE_FIELDS)
def test_space_joined_rows_exist(field):
    assert SEP[field] == " ", field


@pytest.mark.parametrize("field", NOSEP_FIELDS)
def test_no_separator_rows_exist(field):
    assert SEP[field] == "", field


def test_every_field_is_one_of_the_two_separators():
    assert set(SEP.values()) == {" ", ""}


def test_definition_and_expression_disagree_on_purpose():
    # The single most expensive pair in the table: 388/388 definitions need
    # " " and 689/689 expressions need "".
    assert SEP["definition"] == " " and SEP["expression"] == ""


def test_wordform_cell_glues_the_flex_marker():
    # <td>hus<span class="mark-flex">et</span></td> -> "huset", never "hus et"
    cell = td('<td>hus<span class="mark-flex">et</span></td>')
    assert xt(cell, "wordform_cell") == "huset"


def test_definition_separator_keeps_words_apart():
    node = soup('<span class="modern-definition">et <span class="ordform">'
                'stort</span> hus</span>').select_one("span")
    assert xt(node, "definition") == "et stort hus"


def test_the_article_sha_schema_is_stamped_and_positive():
    """The drift ledger records this number so a change in WHAT article_sha
    hashes prints "parser schema changed" instead of reporting all 3,812
    articles as edited by DDO."""
    assert isinstance(ARTICLE_SHA_SCHEMA, int) and ARTICLE_SHA_SCHEMA >= 2
