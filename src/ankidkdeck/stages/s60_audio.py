"""Stage 60: audio cache -- reuse the 4,629 files of 2025, download only the delta.

static.ordnet.dk is a separate plain nginx: no CloudFront, no WAF, and the 2025
URLs are still byte-valid. So this stage is cheap, but it is also the one place
where a wrong slice would attach the wrong sound to a card, and the URL shape is
a free proof that it did not happen:

    https://static.ordnet.dk/mp3/{entry_id[:5]}/{entry_id}_{n}.mp3

All 4,629 legacy URLs match it [measured], so the path is derivable -- but n is
NOT enumerable (hus has slots 1 and 2, hu has only 1, and 640 of 7,400 udtale
rows have no audio at all). Therefore: keep parsing a[href$=".mp3"], and use the
derivation as an assertion.

--seed-legacy hardlinks (or copies) from the recovered 2025 workspace instead of
downloading, which saves ~4,629 requests. The legacy corpus has no sha manifest,
so every seeded file is hashed after the copy and the hash is recorded as ours.
"""

import os
import re
import shutil
from pathlib import Path

from ..config import Config
from ..gates import G_MEDIA, Gate, run_gates
from ..util import FatalError, read_json, sha256_bytes, write_json

# G_MEDIA comes from gates.py. Stage 70 re-runs the same gate over the notes
# that were actually written; this run covers the cache itself and the merged
# report keeps the later result. Two local copies of a gate id is how a typo
# creates a second, invisible gate.

AUDIO_URL_RE = re.compile(
    r"^https://static\.ordnet\.dk/mp3/(\d{5})/(\d{6,})_(\d+)\.mp3$")
ORPHAN_DIR = "_orphans"
MANIFEST = "manifest.json"


def audio_filename(entry_id: str, slot_n) -> str:
    return "%s_%s.mp3" % (entry_id, slot_n)


def assert_url_belongs(url: str, entry_id: str, slot_n) -> int:
    """Both halves of the assertion: the URL shape, and that the shape names the
    entry we parsed it out of."""
    m = AUDIO_URL_RE.match(url)
    if m is None:
        raise FatalError("audio URL does not match the DDO pattern: %r" % url)
    n = int(m.group(3))
    if slot_n is not None and int(slot_n) != n:
        raise FatalError("audio slot mismatch for %s: parsed %s, URL says %d"
                         % (entry_id, slot_n, n))
    expected = "/mp3/%s/%s_%d.mp3" % (entry_id[:5], entry_id, n)
    if not url.endswith(expected):
        raise FatalError("audio URL %r does not belong to entry %s" % (url, entry_id))
    return n


def want_set(entries: dict) -> dict:
    """{url: (entry_id, slot_n)}, deterministic."""
    want: dict[str, tuple] = {}
    for eid in sorted(entries):
        for u in entries[eid].get("udtale", []):
            url = u.get("audio_url")
            if not url:
                continue
            n = assert_url_belongs(url, eid, u.get("slot_n"))
            want[url] = (eid, n)
    return want


def _legacy_index(cfg: Config) -> dict:
    """{url: absolute path} from the recovered 2025 workspace.

    Two sources, in order: audio_map.json (url -> 'audio/<eid>_<n>.mp3') and,
    for anything it misses, the derived filename inside <workspace>/audio.
    """
    if not cfg.legacy_workspace:
        raise FatalError(
            "--seed-legacy needs --legacy-workspace: it is the directory holding "
            "audio_map.json and audio/ from the recovered 2025 run.")
    root = Path(cfg.legacy_workspace)
    amap = read_json(root / "audio_map.json", default={})
    out = {}
    for url, rel in amap.items():
        p = root / rel
        if p.exists():
            out[url] = p
    return out


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        # Different filesystem, or a filesystem without hard links.
        shutil.copy2(src, dest)


def _sha_of(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _media_gate(missing: list, zero_byte: list, n_want: int):
    """G-MEDIA (cache half). Every non-null audio_url must resolve to a
    non-empty file; a zero-byte mp3 imports into Anki as a silent card, which no
    later gate would notice."""
    ok = not missing and not zero_byte
    return ok, {"urls_wanted": n_want, "missing": missing[:20],
                "n_missing": len(missing), "zero_byte": zero_byte[:20],
                "n_zero_byte": len(zero_byte)}


def run(cfg: Config, net=None, seed_legacy: bool = False,
        sweep_orphans: bool = False) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    audio_dir = Path(cfg.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(audio_dir / MANIFEST, default={})
    known_before = set(manifest)
    want = want_set(entries)
    legacy = _legacy_index(cfg) if seed_legacy else {}

    stats = {"wanted": len(want), "already_cached": 0, "rehashed": 0,
             "seeded_from_legacy": 0, "downloaded": 0, "legacy_index": len(legacy)}
    missing, zero_byte = [], []

    for url in sorted(want):
        eid, n = want[url]
        dest = audio_dir / audio_filename(eid, n)
        row = manifest.get(url)
        if dest.exists() and dest.stat().st_size > 0:
            size = dest.stat().st_size
            # SIZE FIRST. `ankidkdeck audio` re-runs on every resume, and hashing
            # every already-cached file meant a full sha256 sweep of 4,629+ files
            # per invocation. The manifest already records the size, so the hash
            # is only computed when the cheap check disagrees (or when there is no
            # manifest row at all) -- which is precisely when it carries
            # information.
            if row and row.get("sha256") and row.get("bytes") == size:
                stats["already_cached"] += 1
                continue
            sha = _sha_of(dest)
            if row and row.get("sha256") == sha:
                # Right content, stale/absent size: repair the row, do not
                # re-download.
                if row.get("bytes") != size:
                    manifest[url] = {**row, "bytes": size}
                    write_json(audio_dir / MANIFEST, manifest)
                stats["already_cached"] += 1
                continue
            # Present but unaccounted for (a seeded file, or a manifest that was
            # lost). Adopt it and record the hash rather than re-downloading.
            manifest[url] = {"url": url, "file": dest.name, "sha256": sha,
                             "bytes": size, "entry_id": eid,
                             "slot_n": n, "source": (row or {}).get("source", "adopted")}
            stats["rehashed"] += 1
            write_json(audio_dir / MANIFEST, manifest)
            continue
        src = legacy.get(url)
        if src is None and legacy:
            cand = Path(cfg.legacy_workspace) / "audio" / audio_filename(eid, n)
            src = cand if cand.exists() else None
        if src is not None:
            _link_or_copy(src, dest)
            data_len = dest.stat().st_size
            if data_len == 0:
                zero_byte.append(url)
                continue
            manifest[url] = {"url": url, "file": dest.name, "sha256": _sha_of(dest),
                             "bytes": data_len, "entry_id": eid, "slot_n": n,
                             "source": "legacy:%s" % src.name}
            stats["seeded_from_legacy"] += 1
            write_json(audio_dir / MANIFEST, manifest)
            continue
        if net is None:
            missing.append(url)
            continue
        r = net.get_audio(url)  # 1s sleep, no WAF on this host
        if not r.content:
            zero_byte.append(url)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        manifest[url] = {"url": url, "file": dest.name,
                         "sha256": sha256_bytes(r.content),
                         "bytes": len(r.content), "entry_id": eid, "slot_n": n,
                         "source": "download"}
        stats["downloaded"] += 1
        write_json(audio_dir / MANIFEST, manifest)  # checkpoint after every file

    # ---- quarantine anything the entries no longer reference -----------------
    # BEHIND A FLAG. "Not referenced by entries.json" and "not wanted any more"
    # are the same predicate only when entries.json is complete: run `audio`
    # against a partial parse -- a pilot, an interrupted crawl -- and the sweep
    # moved the entire 2025 cache into _orphans/, which the next full run then
    # re-downloads. The unreferenced files are counted on every run, so the sweep
    # is a decision made on a number rather than a surprise.
    referenced = {audio_filename(eid, n) for eid, n in want.values()}
    orphans, unreferenced = [], []
    for p in sorted(audio_dir.iterdir()):
        if p.is_dir() or p.name == MANIFEST:
            continue
        if p.name in referenced:
            continue
        unreferenced.append(p.name)
        if not sweep_orphans:
            continue
        target = audio_dir / ORPHAN_DIR / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))
        orphans.append(p.name)
    stats["quarantined_orphans"] = len(orphans)
    stats["unreferenced_files"] = len(unreferenced)
    stats["swept"] = bool(sweep_orphans)

    # ---- final verification -------------------------------------------------
    for url in sorted(want):
        eid, n = want[url]
        dest = audio_dir / audio_filename(eid, n)
        if not dest.exists():
            if url not in missing:
                missing.append(url)
        elif dest.stat().st_size == 0 and url not in zero_byte:
            zero_byte.append(url)

    report = {**stats,
              "new_vs_known": {"known_urls_before": len(known_before),
                               "new_urls_this_run": len(set(want) - known_before),
                               "known_urls_now": len(manifest),
                               "gone_from_ddo": len(known_before - set(want))},
              "orphans_sample": orphans[:20],
              "unreferenced_sample": unreferenced[:20],
              "sweep_hint": (
                  "%d cached file(s) are not referenced by entries.json; pass "
                  "--sweep-orphans to quarantine them into %s once you are sure "
                  "the parse is complete" % (len(unreferenced), ORPHAN_DIR)
                  if unreferenced and not sweep_orphans else None),
              "hint": ("run with --seed-legacy --legacy-workspace <path> to "
                       "hardlink the 2025 files instead of downloading them"
                       if missing and not seed_legacy else None)}
    write_json(cfg.report_dir / "audio_report.json", report)
    run_gates([Gate(G_MEDIA, "every non-null audio_url resolves to a non-empty "
                             "file in the audio cache",
                    lambda: _media_gate(missing, zero_byte, len(want)),
                    stage="60")], cfg, stage="60")
    return report
