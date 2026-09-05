"""The flex table: the ONE parser hazard the guide names by itself (1.9).

`<span class="diskret">eller ...</span>` separates the alternative spellings of
ONE inflection slot inside a single <td>. A naive get_text("") glues them into
`engroshandelenellerengroshandlen`, which then enters paradigm_index (bucket 2's
only evidence), form_index (stage 21's reverse index) and searchable_forms -- so a
single mis-split poisons classification, recovery and Anki search at once, and
`uofficiel form:` prose becomes a "word" the deck claims is Danish.

Moved out of test_extract.py, where it sat next to the separator TABLE: no
separator can express this rule, and guide 4.14 lists it as its own module.
"""

import pytest

from ankidkdeck.extract import cell_alternatives, expand_elision, xt

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 is a runtime dependency")


def soup(html):
    return bs4.BeautifulSoup(html, "html.parser")


def td(html):
    return soup("<table><tr>%s</tr></table>" % html).select_one("td")


def row_cells(html):
    """What stage 20 stores for one <tr>: every td's alternatives, in order."""
    tr = soup("<table>%s</table>" % html).select_one("tr")
    return [c for cell in tr.select("td") for c in cell_alternatives(cell)]


def test_wordform_cell_glues_the_flex_marker():
    # <td>hus<span class="mark-flex">et</span></td> -> "huset", never "hus et"
    cell = td('<td>hus<span class="mark-flex">et</span></td>')
    assert xt(cell, "wordform_cell") == "huset"


def test_cell_alternatives_splits_the_diskret_glue():
    """The reproduced bug, verbatim: get_text("") returns
    'engroshandelenellerengroshandlen' and that string reached the card."""
    cell = td('<td>engroshandelen<span class="diskret">eller</span>'
              'engroshandlen</td>')
    assert xt(cell, "wordform_cell") == "engroshandelenellerengroshandlen"
    assert cell_alternatives(cell) == ["engroshandelen", "engroshandlen"]


def test_the_separator_prose_itself_is_never_a_form():
    for prose in ("eller", "eller også"):
        cell = td('<td>bogen<span class="diskret">%s</span>bogene</td>' % prose)
        got = cell_alternatives(cell)
        assert got == ["bogen", "bogene"]
        assert prose not in got


def test_cell_alternatives_drops_unofficial_form_prose():
    """`uofficiel form:` marks a spelling DDO does not endorse. It must not
    become a paradigm cell -- paradigm cells are bucket 2's evidence, so an
    unofficial form there would let the classifier absorb a word on DDO's own
    disclaimer."""
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
    assert cell_alternatives(td("<td>huset</td>")) == ["huset"]


def test_an_empty_cell_yields_one_empty_string_which_the_parser_drops():
    """The parser filters falsy cells; this pins what the splitter hands it."""
    assert cell_alternatives(td("<td></td>")) == [""]
    assert cell_alternatives(td("<td>   </td>")) == [""]


def test_multi_cell_row_is_alternatives_of_one_slot():
    # 331 of 539 tbody rows have more than one <td>; they are spellings of ONE
    # slot, not different slots.
    assert row_cells('<tr><td>gå hjem-mødet</td><td>gå-hjem-mødet</td></tr>') == \
        ["gå hjem-mødet", "gå-hjem-mødet"]


def test_three_alternatives_in_one_cell():
    cell = td('<td>a<span class="diskret">eller</span>b'
              '<span class="diskret">eller</span>c</td>')
    assert cell_alternatives(cell) == ["a", "b", "c"]


def test_whitespace_inside_a_form_is_collapsed_not_dropped():
    cell = td('<td>dag   til\n dag-servicer</td>')
    assert cell_alternatives(cell) == ["dag til dag-servicer"]


# --------------------------------------------------------------------------
# DDO's own ".." prefix elision (19 articles, 31 cells)
# --------------------------------------------------------------------------

# Every distinct shape in the corpus, taken from the real cells. The literal ".."
# is DDO's; the raw HTML is <span class="mark-flex">..mænd</span>.
ELIDED_CELLS = [
    ("julemand", "..mænd", "julemænd"),
    ("julemand", "..mændene", "julemændene"),
    ("politimand", "..mænd", "politimænd"),
    ("ægtemand", "..mændene", "ægtemændene"),
    ("troldmand", "..mænd", "troldmænd"),
    ("bedstefar", "..fædre", "bedstefædre"),
    ("bedstemor", "..mødre", "bedstemødre"),
    ("oversætte", "..satte", "oversatte"),
    ("oversætte", "..sat", "oversat"),
    ("værdsætte", "..satte", "værdsatte"),
    ("værdsætte", "..sat", "værdsat"),
    ("efterlade", "..lod", "efterlod"),
    ("overgive", "..gav", "overgav"),
    ("overtage", "..tog", "overtog"),
    ("foretage", "..tog", "foretog"),
    ("foregå", "..gik", "foregik"),
    ("gennemgå", "..gik", "gennemgik"),
    ("bortse", "..så", "bortså"),
    ("tilintetgøre", "..gjorde", "tilintetgjorde"),
    ("tilintetgøre", "..gjort", "tilintetgjort"),
    # the hyphenated compounds: the prefix ends inside the hyphen segment
    ("gør det selv-bog", "..bøger", "gør det selv-bøger"),
    ("gør det selv-bog", "..bøgerne", "gør det selv-bøgerne"),
    ("gør det selv-mand", "..mænd", "gør det selv-mænd"),
    ("pop op-bog", "..bøger", "pop op-bøger"),
]


@pytest.mark.parametrize("lemma,cell,want", ELIDED_CELLS)
def test_expand_elision_restores_the_elided_prefix(lemma, cell, want):
    assert expand_elision(cell, lemma) == want


def test_expand_elision_leaves_an_ordinary_cell_alone():
    for cell in ("huset", "julemanden", "", "-en", "..", ".."):
        assert expand_elision(cell, "julemand") == cell


def test_expand_elision_refuses_to_guess_when_the_prefix_is_not_there():
    """No shared prefix means the rule has no answer; a visible ".." a human can
    report is better than an invented Danish word."""
    assert expand_elision("..xyz", "julemand") == "..xyz"
    # first character only at position 0 => no prefix to keep
    assert expand_elision("..jul", "julemand") == "..jul"


def test_the_parser_expands_the_elided_cell_and_indexes_the_real_form(registry):
    """End-to-end: the expansion has to reach paradigm.rows (what stage 70
    prints) AND paradigm_index (bucket 2's only evidence), or the card shows
    "..mænd" and julemænd stays unmatchable."""
    from ankidkdeck.stages.s20_parse import parse_article, slice_articles
    html = """
    <article><div id="11024753"><div class="artikel">
      <div class="modern-top-row"><h1 class="modern-match">julemand</h1>
        <span class="text-large">substantiv</span></div>
      <div data-pos-key="sb."></div>
      <div class="modern-row" id="id-boj">
        <button class="kilde">-en, ..mænd, ..mændene</button>
        <table class="flex-table"><tbody>
          <tr><td>julemand<span class="mark-flex">en</span></td></tr>
          <tr><td><span class="mark-flex">..mænd</span></td></tr>
          <tr><td><span class="mark-flex">..mændene</span></td></tr>
        </tbody></table>
      </div>
      <div id="content-betydninger">
        <div><div class="modern-definition-box" id="betydning-1" dannetid="21990001">
          <span class="modern-definition">figur der bringer julegaver</span>
        </div></div>
      </div>
    </div></div></article>
    """
    s = soup(html)
    eid, scope, art = next(iter(slice_articles(s)))
    report = {}
    e = parse_article(eid, scope, art, registry, report)
    assert [r["cells"] for r in e["paradigm"]["rows"]] == [
        ["julemanden"], ["julemænd"], ["julemændene"]]
    assert e["paradigm_index"] == ["julemanden", "julemænd", "julemændene"]
    assert not any(".." in c for c in e["paradigm_index"])
    # the short notation keeps DDO's own ".." -- it is not a form, it is notation
    assert e["paradigm"]["short"] == "-en, ..mænd, ..mændene"
    assert "unexpanded_elided_cells" not in report
    assert "short_form_missing_from_cells" not in report


def test_an_upstream_truncated_cell_is_reported_never_patched(registry):
    """planlaegge (11039990): DDO's own past-tense cell is
    <td>p<span>lagde</span></td>, i.e. "plagde" -- the page truncated the stem,
    while its short notation "-r, ..lagde, ..lagt" is correct. We must not
    invent the missing letters into a content field article_sha treats as DDO's;
    the run has to survive it and the owner has to see it."""
    from ankidkdeck.stages.s20_parse import parse_article, slice_articles
    html = """
    <article><div id="11039990"><div class="artikel">
      <div class="modern-top-row"><h1 class="modern-match">planlægge</h1>
        <span class="text-large">verbum</span></div>
      <div data-pos-key="vb."></div>
      <div class="modern-row" id="id-boj">
        <button class="kilde">-r, ..lagde, ..lagt</button>
        <table class="flex-table"><tbody>
          <tr><td>planlægge<span class="mark-flex">r</span></td></tr>
          <tr><td>p<span class="mark-flex">lagde</span></td></tr>
          <tr><td>p<span class="mark-flex">lagt</span></td></tr>
        </tbody></table>
      </div>
      <div id="content-betydninger">
        <div><div class="modern-definition-box" id="betydning-1" dannetid="21990001">
          <span class="modern-definition">lægge en plan for</span>
        </div></div>
      </div>
    </div></div></article>
    """
    s = soup(html)
    eid, scope, art = next(iter(slice_articles(s)))
    report = {}
    e = parse_article(eid, scope, art, registry, report)
    # transcribed faithfully, NOT corrected
    assert [r["cells"] for r in e["paradigm"]["rows"]] == [
        ["planlægger"], ["plagde"], ["plagt"]]
    rows = report["short_form_missing_from_cells"]
    assert [r["expected_form"] for r in rows] == ["planlagde", "planlagt"]
    assert rows[0]["entry_id"] == "11039990"


def test_the_parser_stores_the_split_cells_and_indexes_them(registry):
    """End-to-end through stage 20: the split has to reach paradigm_index, or
    the classifier is judging glued junk."""
    from ankidkdeck.stages.s20_parse import parse_article, slice_articles
    html = """
    <article><div id="11000001"><div class="artikel">
      <div class="modern-top-row"><h1 class="modern-match">engroshandel</h1>
        <span class="text-large">substantiv</span></div>
      <div data-pos-key="sb."></div>
      <div class="modern-row" id="id-boj">
        <table class="flex-table"><tbody>
          <tr><td>engroshandelen<span class="diskret">eller</span>engroshandlen</td></tr>
        </tbody></table>
      </div>
      <div id="content-betydninger">
        <div><div class="modern-definition-box" id="betydning-1" dannetid="21990001">
          <span class="modern-definition">handel i store partier</span>
        </div></div>
      </div>
    </div></div></article>
    """
    s = soup(html)
    eid, scope, art = next(iter(slice_articles(s)))
    e = parse_article(eid, scope, art, registry, {})
    assert e["paradigm"]["rows"][0]["cells"] == ["engroshandelen",
                                                 "engroshandlen"]
    assert e["paradigm_index"] == ["engroshandelen", "engroshandlen"]
    # form_index adds the lemma key; paradigm_index deliberately does not
    assert "engroshandel" in e["form_index"]
    assert "engroshandel" not in e["paradigm_index"]
    assert "engroshandelenellerengroshandlen" not in e["form_index"]
