"""The JSONL row key (patch plan 5.2).

A batch result file is reconciled BY KEY -- the output order is not the input
order past ~1000 rows (measured 2026-08-27 on the first real wave), so the key
is the attribution and not a decoration on it. A key that collides therefore
does not weaken a cross-check, it makes the wave unattributable. Three key
schemes were tried against the real corpus and two of them collide:

    def__{entry_id}                  42 collisions (3,623 requests, 3,581 keys)
    expr__{entry_id}                 77 collisions (69 entries, up to 64 items)
    {kind}__{label}   (lemma+pos)   186 collisions on the definition side,
                                    108 on the expression side

The cause is the chunker, not the identifier: _group_by_entry splits an entry
with more than 20 rows into several REQUESTS, so a key that names the entry
names all of them. 38 entries have more than 20 senses (max 71) and 69 have more
than 20 expressions (max 64). The key therefore has to carry the chunk index,
and it has to carry the language, because one wave file is per language but the
registry, the fingerprints and the retry bookkeeping are not.

    {kind}__{lang}__{entry_id}__{chunk:02d}          e.g. def__German__11000142__00

All ASCII, and validated against a pinned regex before anything is uploaded:
the failure this prevents (a duplicate key reconciled to the wrong request) is
invisible in the output file.
"""

import re

from ..util import FatalError

# `[A-Za-z]+` for the language and `\d+` for the entry id are the shapes the
# real corpus has: all 4,543 entry ids in json/entries.json are digit strings,
# and the four configured languages are plain ASCII words. A language or an
# entry id that does not fit is a REFUSAL rather than a silently different key
# shape -- see language_tag().
KEY_RE = re.compile(r"^[a-z]+__[A-Za-z]+__\d+__\d{2}$")

# The short tag for each request kind. "definition"/"expression" are the kind
# names the rest of the pipeline uses; the tags are what goes on the wire.
KIND_TAG = {"definition": "def", "expression": "expr"}


def language_tag(lang: str) -> str:
    """The language segment of a key: ASCII letters only.

    A language name with a space or a hyphen ("Brazilian Portuguese") would
    produce a key the pinned regex rejects, and the reconciliation would then
    fail on the whole wave rather than on the one language. Non-letters are
    dropped, and an empty result is fatal here rather than at upload time.
    """
    tag = "".join(c for c in str(lang) if c.isascii() and c.isalpha())
    if not tag:
        raise FatalError(
            "language %r has no ASCII letters, so it cannot appear in a batch "
            "row key (%s). Batch keys are ASCII by construction: the result "
            "file is joined on the key, so a key that cannot be formed is a "
            "wave that cannot be attributed." % (lang, KEY_RE.pattern))
    return tag


def make_key(kind: str, lang: str, entry_id: str, chunk: int) -> str:
    """One key per REQUEST. `chunk` is the index within the entry, not global."""
    tag = KIND_TAG.get(kind)
    if tag is None:
        raise FatalError(
            "no batch key tag for request kind %r (known: %s)"
            % (kind, ", ".join(sorted(KIND_TAG))))
    return "%s__%s__%s__%02d" % (tag, language_tag(lang), entry_id, int(chunk))


def validate_keys(keys) -> list:
    """Uniqueness, ASCII and the pinned shape, before a single byte is uploaded.

    Returns the keys so this can wrap the list it checks. Raises on the first
    problem with the offending keys named: a duplicate key is not a warning, it
    is a wave whose results cannot be attributed.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys):
        seen, dupes = set(), []
        for key in keys:
            if key in seen and key not in dupes:
                dupes.append(key)
            seen.add(key)
        raise FatalError(
            "%d duplicate batch row key(s) in a wave of %d: %s. The result "
            "file is joined on the echoed key, so two requests sharing a key "
            "leave no way to tell which answer belongs to which."
            % (len(dupes), len(keys), ", ".join(dupes[:5])))
    bad = [k for k in keys if not k.isascii() or not KEY_RE.match(k)]
    if bad:
        raise FatalError(
            "%d batch row key(s) do not match the pinned shape %s: %s"
            % (len(bad), KEY_RE.pattern, ", ".join(bad[:5])))
    return keys
