"""Shared primitives: normalization, hashing, atomic JSON I/O, fatal errors."""

import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path


class FatalError(RuntimeError):
    """Raised for conditions where continuing would corrupt data or hammer DDO.

    Deliberately NOT raised for a wordlist word that resolves to zero entries:
    those are skipped and recorded (owner decision, 2026-08-24).
    """


def NFC(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nk(s: str) -> str:
    """Normalized key: NFC + casefold. An index value, never an identity."""
    return NFC(s).casefold()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj) -> None:
    atomic_write_text(Path(path), json.dumps(obj, ensure_ascii=False, indent=1))


def read_json(path: Path, default=None):
    p = Path(path)
    if not p.exists():
        if default is not None:
            return default
        raise FatalError(f"required file missing: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collapse_ws(s: str) -> str:
    return " ".join(s.split())


def read_text_nfc_tolerant(path: Path) -> str:
    """Open a file whose on-disk name may be NFD while the key is NFC (macOS legacy).

    The 2025 corpus has 204 NFD filenames on disk vs 0 NFD map keys; a plain
    open() by key silently misses them on Linux.
    """
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    alt = p.parent / unicodedata.normalize("NFD", p.name)
    if alt.exists():
        return alt.read_text(encoding="utf-8", errors="replace")
    raise FatalError(f"file not found under NFC or NFD name: {p}")
