"""URL construction: the path form, the apostrophe words, and NFD.

Why this is a test and not a comment: DDO no longer answers 404, so an NFD word
returns HTTP 200 with zero articles -- a silent non-word. `?query=` is banned
because the CloudFront edge function double-encodes non-ASCII in its redirect
(hj%C3%A6lp becomes hj%25C3%25A6lp).
"""

import unicodedata

import pytest

from ankidkdeck.urls import DDO_BASE, entry_id_url, roundtrips, word_url
from ankidkdeck.util import FatalError

# 4 apostrophe words and 10 capitalised words are in the shipped wordlist;
# lemmas can also hold spaces and dots, so a word is never a filename.
SAMPLE_WORDS = [
    "bli'r", "ta'r", "gi'r", "d'herrer",
    "gå", "hjælp", "lån", "være", "å", "øje", "æble",
    "Jack", "Boston", "April",
    "a cappella", "a c.", "Alzheimers sygdom",
    "hus", "er", "og",
]


@pytest.mark.parametrize("word", SAMPLE_WORDS)
def test_quote_unquote_roundtrip(word):
    assert roundtrips(word)


@pytest.mark.parametrize("word", SAMPLE_WORDS)
def test_word_url_is_path_form_and_never_query_form(word):
    url = word_url(word)
    assert url.startswith(DDO_BASE + "/ddo/ordbog/")
    assert "?query=" not in url
    assert "?" not in url


def test_known_encodings():
    # Measured live: /ddo/ordbog/g%C3%A5 -> 200, 0 redirects, 2 articles;
    # /ddo/ordbog/bli%27r -> 200, 1 article.
    assert word_url("gå") == DDO_BASE + "/ddo/ordbog/g%C3%A5"
    assert word_url("bli'r") == DDO_BASE + "/ddo/ordbog/bli%27r"
    assert word_url("hjælp") == DDO_BASE + "/ddo/ordbog/hj%C3%A6lp"


def test_nfd_word_is_refused_before_it_becomes_a_request():
    nfd = unicodedata.normalize("NFD", "lån")
    assert nfd != "lån"
    with pytest.raises(FatalError):
        word_url(nfd)


def test_nfd_hazard_is_invisible_to_a_non_ascii_smoke_test():
    # ae/oe have NO canonical decomposition, so "contains non-ASCII" passes
    # while aa/e-acute/u-umlaut still break. This is why the check is NFC, not
    # "is it ASCII".
    for w in ("æble", "øje"):
        assert unicodedata.normalize("NFD", w) == w
    for w in ("lån", "café", "über"):
        assert unicodedata.normalize("NFD", w) != w


def test_entry_id_url_carries_the_id():
    assert entry_id_url("11021722").endswith("entry_id=11021722")
