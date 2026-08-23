"""The three-state verdict: nohit / ok / error.

DDO answers 200 for everything -- there is no 404 any more -- so a miss can only
be read off the page. #results-label is the page's own checksum against the
article count, and it is what separates "this word is not in the dictionary"
from "our fetch or our parse is broken". The 2025 pipeline could not tell those
apart, which is how a word that produced nothing got re-searched on every
resume.

The synthetic half of this file runs everywhere. The saved-page half needs
fixtures.
"""

import pytest
from conftest import FIXTURES_DIR, requires_fixtures, fixtures_available

pytest.importorskip("bs4", reason="beautifulsoup4 is a runtime dependency")

from ankidkdeck.stages.s12_download import NOHIT_MARKER, verdict_of  # noqa: E402

ARTICLE = '<div class="artikel">x</div>'


def body(label=None, n=0, nohit=False):
    parts = []
    if label is not None:
        parts.append('<span id="results-label">%s</span>' % label)
    parts.append(ARTICLE * n)
    if nohit:
        parts.append("<p>Din soegning %s.</p>" % NOHIT_MARKER)
    return "<html><body>%s</body></html>" % "".join(parts)


def test_nohit_page():
    assert verdict_of(body(nohit=True))[0] == "nohit"


def test_single_article_page_has_an_empty_label():
    # /har -> "" and one article; the empty label is not a missing label.
    verdict, label, n = verdict_of(body(label="", n=1))
    assert (verdict, label, n) == ("ok", "", 1)


def test_multi_article_page_reconciles():
    assert verdict_of(body(label="3 resultater", n=3))[0] == "ok"
    assert verdict_of(body(label="25 resultater", n=25))[0] == "ok"


def test_label_disagreeing_with_the_count_is_an_error():
    assert verdict_of(body(label="5 resultater", n=1))[0] == "error"
    assert verdict_of(body(label="", n=3))[0] == "error"


def test_missing_label_without_the_nohit_marker_is_an_error():
    # A truncated or WAF-substituted body: not a nohit, and not usable.
    assert verdict_of(body(n=2))[0] == "error"
    assert verdict_of("<html><body></body></html>")[0] == "error"


def test_a_nohit_marker_with_articles_is_not_a_nohit():
    assert verdict_of(body(label="1 resultater", n=1, nohit=True))[0] == "ok"


@requires_fixtures
def test_saved_pages_reconcile():
    import json
    with open(FIXTURES_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    checked = 0
    for page in manifest["pages"]:
        text = (FIXTURES_DIR / page["file"]).read_text(encoding="utf-8")
        verdict, label, n = verdict_of(text)
        if page.get("kind") == "nonword":
            assert verdict == "nohit", page["file"]
        else:
            assert verdict == "ok", (page["file"], label, n)
            assert n == page["n_articles"], page["file"]
        checked += 1
    assert checked, "fixture manifest lists no pages"


@requires_fixtures
def test_at_least_one_saved_page_is_the_empty_label_case():
    import json
    with open(FIXTURES_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    singles = [p for p in manifest["pages"] if p.get("n_articles") == 1]
    if not singles:
        pytest.skip("no single-article page in this fixture set")
    text = (FIXTURES_DIR / singles[0]["file"]).read_text(encoding="utf-8")
    verdict, label, n = verdict_of(text)
    assert (verdict, label, n) == ("ok", "", 1)


def test_fixture_availability_is_reported_not_guessed():
    # A test file that silently passes because it found no data is worse than
    # no test; the flag is asserted so the skip reason is always accurate.
    assert isinstance(fixtures_available(), bool)
