"""The extraction contract: the per-field separator table and the flex cell split.

A one-character change in extract.SEP silently invalidates 22,734 translation
cells x 4 languages -- that is how 2,007 English cards shipped with bare Danish
definitions. The byte-for-byte golden comparison against the real corpus lives
in test_separators.py (fixtures required); this file pins the table itself and
the one piece of logic that no separator can express.
"""

import pytest

from ankidkdeck.extract import SEP, cell_alternatives, xt

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


def test_cell_alternatives_splits_the_diskret_glue():
    # The reproduced bug: a naive get_text("") returns
    # 'engroshandelenellerengroshandlen'.
    cell = td('<td>engroshandelen<span class="diskret">eller</span>'
              'engroshandlen</td>')
    assert xt(cell, "wordform_cell") == "engroshandelenellerengroshandlen"
    assert cell_alternatives(cell) == ["engroshandelen", "engroshandlen"]


def test_cell_alternatives_drops_unofficial_form_prose():
    cell = td('<td>gotte<span class="diskret">eller</span>'
              'uofficiel form: gotter</td>')
    assert cell_alternatives(cell) == ["gotte"]
    cell2 = td('<td>servicer<span class="diskret">eller</span>'
               'form: services</td>')
    assert cell_alternatives(cell2) == ["servicer"]


def test_cell_alternatives_strips_unbalanced_parens():
    # Measured shape: 'dag til dag-servicer(' + 'dag til dag-services)'.
    # NOTE for reviewers: here the "uofficiel form:" prose sits INSIDE the
    # span.diskret separator, which is discarded by the split, so BOTH
    # spellings survive -- the drop rule only fires when the prose is in a
    # part. This is the current behaviour, asserted so a change is visible.
    cell = td('<td>dag til dag-servicer(<span class="diskret">eller uofficiel '
              'form:</span>dag til dag-services)</td>')
    assert cell_alternatives(cell) == ["dag til dag-servicer",
                                       "dag til dag-services"]


def test_cell_alternatives_falls_back_to_the_whole_cell():
    cell = td("<td>huset</td>")
    assert cell_alternatives(cell) == ["huset"]


def test_multi_cell_row_is_alternatives_of_one_slot():
    # 331 of 539 tbody rows have more than one <td>; they are spellings of ONE
    # slot, not different slots.
    row = soup('<table><tr><td>gå hjem-mødet</td><td>gå-hjem-mødet</td></tr>'
               "</table>").select_one("tr")
    cells = [c for cell in row.select("td") for c in cell_alternatives(cell)]
    assert cells == ["gå hjem-mødet", "gå-hjem-mødet"]
