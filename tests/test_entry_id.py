"""entry_id is the pipeline's primary key. It gets THREE witnesses.

family_id is an entry_id, card_keys.json is keyed by it, the audio URL is derived
from it and stage 40's whole bridge is filename -> entry_id. A slice that carries
the wrong id therefore mis-attaches translations AND audio, silently.

So stage 20 reads the id off the immediate parent div (measured 216/216) and then
demands that the two independent witnesses embedded in the same slice agree:

    fejlrapport mailto:...(<entry_id>)     present in 5,259 of 5,267 pages
    <audio id="<entry_id>_<slot>">         the fallback

A disagreement is FATAL. There is no repair for a mis-keyed article after
release, so guessing is not on the menu.

Synthetic HTML on purpose: the failure this file pins has never been seen in the
corpus, which is exactly why it needs a test rather than a fixture.
"""

import pytest

from ankidkdeck.util import FatalError

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 is a runtime dependency")

from ankidkdeck.stages.s20_parse import (entry_id_of,  # noqa: E402
                                         slice_articles)

MAILTO = ('<a href="mailto:ordbog@dsl.dk?subject=Fejl i DDO: hus (%s)">'
          "fejlrapport</a>")
AUDIO = '<audio id="%s_1" src="x.mp3"></audio>'


def page(eid="11021722", mailto_id=None, audio_id=None, wrapper=None):
    """One <article> holding exactly one div.artikel, with chosen witnesses."""
    inner = '<div class="modern-top-row"><h1 class="modern-match">hus</h1></div>'
    if mailto_id is not None:
        inner += MAILTO % mailto_id
    if audio_id is not None:
        inner += AUDIO % audio_id
    art = '<div class="artikel">%s</div>' % inner
    if wrapper is None:
        wrapper = '<div id="%s">%%s</div>' % eid
    return bs4.BeautifulSoup("<article>%s</article>" % (wrapper % art),
                             "html.parser")


def test_the_id_comes_from_the_immediate_parent_div():
    soup = page()
    eid, scope, art = next(iter(slice_articles(soup)))
    assert eid == "11021722"


def test_all_three_witnesses_agreeing_is_accepted():
    soup = page(mailto_id="11021722", audio_id="11021722")
    eid, scope, art = next(iter(slice_articles(soup)))
    assert eid == "11021722"


def test_a_disagreeing_mailto_witness_is_fatal():
    soup = page(mailto_id="99999999")
    with pytest.raises(FatalError) as exc:
        next(iter(slice_articles(soup)))
    assert "witnesses disagree" in str(exc.value)
    assert "11021722" in str(exc.value) and "99999999" in str(exc.value)


def test_a_disagreeing_audio_witness_is_fatal():
    soup = page(audio_id="99999999")
    with pytest.raises(FatalError) as exc:
        next(iter(slice_articles(soup)))
    assert "witnesses disagree" in str(exc.value)


def test_an_apostrophe_headword_still_bridges():
    """The mailto character class excludes " and > ONLY, never the apostrophe:
    the fejlrapport subject embeds the headword, and d'herrer (11008184) carries
    one. With an apostrophe-excluding class that page loses its witness -- which
    is the difference between the published 5,259/3,812/8 counts and 5,258 with a
    9th unbridged file that is then NOT set-equal to the no-entry files."""
    art = ('<div class="artikel"><div class="modern-top-row">'
           '<h1 class="modern-match">d\'herrer</h1></div>'
           '<a href="mailto:ordbog@dsl.dk?subject=Fejl i DDO: d\'herrer '
           '(11008184)">fejlrapport</a></div>')
    soup = bs4.BeautifulSoup(
        '<article><div id="11008184">%s</div></article>' % art, "html.parser")
    eid, scope, _ = next(iter(slice_articles(soup)))
    assert eid == "11008184"


def test_a_slice_with_no_numeric_id_anywhere_is_fatal():
    art = ('<div class="artikel"><div class="modern-top-row">'
           '<h1 class="modern-match">hus</h1></div></div>')
    soup = bs4.BeautifulSoup(
        '<article><div id="not-a-number">%s</div></article>' % art,
        "html.parser")
    with pytest.raises(FatalError) as exc:
        next(iter(slice_articles(soup)))
    assert "numeric entry_id" in str(exc.value)


def test_the_id_is_found_on_an_ancestor_when_the_parent_has_none():
    """Defence in depth: the parent div is where it lives (216/216), but the
    walk up the ancestors is what keeps a markup reshuffle from being silent."""
    art = ('<div class="artikel"><div class="modern-top-row">'
           '<h1 class="modern-match">hus</h1></div></div>')
    soup = bs4.BeautifulSoup(
        '<article><div id="11021722"><div class="wrap">%s</div></div></article>'
        % art, "html.parser")
    eid, _, _ = next(iter(slice_articles(soup)))
    assert eid == "11021722"


def test_a_scope_holding_two_artikel_divs_is_fatal():
    """One <article> owns exactly one div.artikel plus its faste-udtryk sibling.
    Two would mean the fixed expressions of one article are attributed to
    another."""
    art = ('<div class="artikel"><div class="modern-top-row">'
           '<h1 class="modern-match">hus</h1></div></div>')
    soup = bs4.BeautifulSoup(
        '<article><div id="11021722">%s%s</div></article>' % (art, art),
        "html.parser")
    with pytest.raises(FatalError) as exc:
        list(slice_articles(soup))
    assert "more than one div.artikel" in str(exc.value)


def test_entry_id_of_is_callable_on_its_own_with_a_narrow_scope():
    """The witness check reads the SCOPE, not the whole page: a mailto for a
    different article elsewhere on a multi-article page must not fail the
    slice."""
    soup = page(eid="11021722")
    art = soup.select_one("div.artikel")
    # a foreign witness OUTSIDE the scope is irrelevant
    other = bs4.BeautifulSoup(MAILTO % "99999999", "html.parser")
    soup.append(other)
    assert entry_id_of(art, art.parent) == "11021722"
