"""URL construction for the 2026 DDO site.

Path form ONLY: the CloudFront edge function double-encodes non-ASCII in
?query= redirects (hj%C3%A6lp becomes hj%25C3%25A6lp). An NFD word is a
request-level hazard: it returns HTTP 200 with zero articles, i.e. a silent
non-word, because DDO no longer answers 404 for anything.
"""

from urllib.parse import quote, unquote

from .util import NFC, FatalError

DDO_BASE = "https://ordnet.dk"


def word_url(word: str) -> str:
    if NFC(word) != word:
        raise FatalError(f"word is not NFC-normalized: {word!r}")
    return f"{DDO_BASE}/ddo/ordbog/{quote(word, safe='')}"


def entry_id_url(entry_id: str) -> str:
    # The path segment is decorative when entry_id is supplied (measured:
    # /ddo/ordbog/vaere?entry_id=... returns the correct article regardless).
    return f"{DDO_BASE}/ddo/ordbog/x?entry_id={entry_id}"


def roundtrips(word: str) -> bool:
    w = NFC(word)
    return unquote(quote(w, safe="")) == w
