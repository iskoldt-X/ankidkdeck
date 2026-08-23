"""Stage 00: pin the wordlist artifact.

v3.0 deliberately locks the 2025 wordlist file (the one the released decks were
built from): GUID continuity depends on the word set, so changing the wordlist
is an explicit decision (--accept-new-wordlist), never a side effect.
"""

from pathlib import Path

from ..config import Config
from ..urls import roundtrips
from ..util import NFC, FatalError, canonical_json, sha256_str, write_json


def run(cfg: Config, accept_new: bool = False) -> dict:
    if not cfg.wordlist_file or not Path(cfg.wordlist_file).exists():
        raise FatalError(
            "wordlist_file not configured or missing. Point it at the pinned "
            "2025 wordlist (one word per line, frequency order)."
        )
    words = []
    seen = set()
    with open(cfg.wordlist_file, encoding="utf-8") as f:
        for rank, line in enumerate((l.strip() for l in f if l.strip()), 1):
            w = NFC(line)
            if w in seen:
                raise FatalError(f"duplicate wordlist entry: {w!r} at rank {rank}")
            seen.add(w)
            if not roundtrips(w):
                raise FatalError(f"word does not URL-roundtrip: {w!r}")
            words.append({"rank": rank, "raw": line, "word": w})
    if not 4900 <= len(words) <= 5100:
        raise FatalError(f"wordlist has {len(words)} words; expected ~5000")
    sha = sha256_str(canonical_json([w["word"] for w in words]))
    if cfg.wordlist_sha256 and sha != cfg.wordlist_sha256 and not accept_new:
        raise FatalError(
            "wordlist changed vs the pinned sha256 -- the whole deck would "
            "silently re-rank. Re-run with --accept-new-wordlist to confirm."
        )
    out = {"source": str(cfg.wordlist_file), "sha256": sha, "words": words}
    write_json(cfg.json_dir / "wordlist.json", out)
    return {"n_words": len(words), "sha256": sha}
