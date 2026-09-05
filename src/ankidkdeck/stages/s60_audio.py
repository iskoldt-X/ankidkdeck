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
from ..registry import Registry
from ..util import (AudioUnavailable, FatalError, read_json, sha256_bytes,
                    write_json)

# G_MEDIA comes from gates.py. Stage 70 re-runs the same gate over the notes
# that were actually written; this run covers the cache itself and the merged
# report keeps the later result. Two local copies of a gate id is how a typo
# creates a second, invisible gate.

# EIGHT digits, as guide 4.11 asserts and as the corpus shows: all 4,629 legacy
# audio URLs carry an 8-digit entry_id (measured 4,629/4,629). `\d{6,}` was looser
# than both the spec and the data, so a 6- or 7-digit id -- a malformed slice, or a
# re-shaped DDO URL -- would have walked past the one free integrity check this
# stage has, and the wrong sound on a card is silent by construction.
AUDIO_URL_RE = re.compile(
    r"^https://static\.ordnet\.dk/mp3/(\d{5})/(\d{8})_(\d+)\.mp3$")
ORPHAN_DIR = "_orphans"
MANIFEST = "manifest.json"


def audio_filename(entry_id: str, slot_n) -> str:
    return "%s_%s.mp3" % (entry_id, slot_n)


def assert_url_belongs(url: str, entry_id: str, slot_n) -> int:
    """Both halves of the assertion: the URL shape, and that the shape names the
    entry we parsed it out of."""
    m = AUDIO_URL_RE.match(url)
    if m is None:
        raise FatalError(
            "audio URL does not match the DDO pattern "
            "https://static.ordnet.dk/mp3/<eid[:5]>/<8-digit eid>_<n>.mp3: %r"
            % url)
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


def known_missing_audio_status(known: dict, want: dict, audio_dir: Path) -> dict:
    """Every row of registry/known_missing_audio.json, checked against THIS
    workspace: still dead upstream, recovered, or no longer declared.

    Read off the DISK, not out of the download loop: a slot also "recovers" by
    being seeded from the legacy workspace or by already sitting in the cache
    from an earlier run, and neither of those goes through net.get_audio. A row
    whose URL the corpus no longer declares at all (DDO dropped the slot, or the
    entry left the scope) is reported too -- it is a row the registry should stop
    carrying, and nothing else in the pipeline would ever mention it.
    """
    still_missing, recovered, not_declared = [], [], []
    for url in sorted(known):
        if url not in want:
            not_declared.append(url)
            continue
        eid, n = want[url]
        dest = audio_dir / audio_filename(eid, n)
        if dest.exists() and dest.stat().st_size > 0:
            recovered.append(url)
        else:
            still_missing.append(url)
    return {"registry_rows": len(known), "still_missing": still_missing,
            "recovered": recovered, "no_longer_declared": not_declared}


def _media_gate(missing: list, zero_byte: list, n_want: int,
                known: dict | None = None, known_max: int | None = None):
    """G-MEDIA (cache half), BASELINED against registry/known_missing_audio.json.

    Every non-null audio_url must resolve to a non-empty file; a zero-byte mp3
    imports into Anki as a silent card, which no later gate would notice.

    Four of the 5,893 declared slots cannot be made to resolve by anything on
    this side of the wire: DDO answers them with HTTP 200, content-length 0 and
    content-type text/html -- one shared zero-byte placeholder, same etag on all
    four -- while sibling slots of the same entries serve real mp3 bodies. So the
    gate is baselined the way G-SUPPRESS and G-ADMIT are, rather than switched
    off or hidden behind a nulled audio_url:

      * a slot the registry names, and only while it is STILL dead, is reported
        as known_missing_upstream instead of counted as a failure;
      * a missing or zero-byte slot the registry does NOT name still fails, which
        is the whole population this gate was written for -- a lost cache, a
        failed download, a skipped seed;
      * the registry's own size is checked against gates.json:
        known_missing_audio_max, so a row cannot be added without a human
        editing that number in the same commit. Fail-closed on a missing key,
        exactly like G-SUPPRESS: no baseline means every row is over it;
      * a row that RECOVERED, or that the corpus no longer declares, is reported
        and never fails. That asymmetry is deliberate (G-ADMIT has the same one):
        the day DDO repairs the file is the mechanism working, and the report is
        how the release notes find out the row should be deleted.
    """
    known = known or {"registry_rows": 0, "still_missing": [], "recovered": [],
                      "no_longer_declared": []}
    excused = set(known["still_missing"])
    unexpected_missing = [u for u in missing if u not in excused]
    unexpected_zero = [u for u in zero_byte if u not in excused]
    over = known_max is not None and known["registry_rows"] > known_max
    ok = not unexpected_missing and not unexpected_zero and not over
    return ok, {"urls_wanted": n_want, "missing": unexpected_missing[:20],
                "n_missing": len(unexpected_missing),
                "zero_byte": unexpected_zero[:20],
                "n_zero_byte": len(unexpected_zero),
                "known_missing_upstream": {
                    "registry_rows": known["registry_rows"], "max": known_max,
                    "over_baseline": bool(over),
                    "n_still_missing": len(known["still_missing"]),
                    "still_missing": known["still_missing"][:20],
                    "recovered": known["recovered"][:20],
                    "no_longer_declared": known["no_longer_declared"][:20]}}


def run(cfg: Config, net=None, seed_legacy: bool = False,
        sweep_orphans: bool = False, registry=None) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    audio_dir = Path(cfg.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(audio_dir / MANIFEST, default={})
    known_before = set(manifest)
    want = want_set(entries)
    legacy = _legacy_index(cfg) if seed_legacy else {}
    registry = registry if registry is not None else Registry(cfg)
    known_missing = registry.known_missing_audio

    stats = {"wanted": len(want), "already_cached": 0, "rehashed": 0,
             "seeded_from_legacy": 0, "downloaded": 0, "legacy_index": len(legacy)}
    missing, zero_byte = [], []
    # The host's own answer, per slot it refused to serve. `missing` says a file
    # is absent; this says WHY, which is the difference between an upstream
    # defect and a download this run got wrong.
    no_audio_from_host = []

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
        # A slot already recorded upstream-dead is still RE-PROBED, once, with
        # the retry and the circuit-breaker record suppressed: that probe is the
        # only thing in the pipeline that can notice DDO repairing the file, and
        # four known-dead failures in a row would otherwise trip the breaker on
        # an otherwise fully cached rerun.
        try:
            r = net.get_audio(url,  # 1s sleep, no WAF on this host
                              expected_missing=url in known_missing)
        except AudioUnavailable as exc:
            # Nothing was returned, so nothing can be written: a text/html
            # placeholder or an empty body never becomes an mp3 on a card. The
            # slot is left absent and classified after the loop, against the
            # registry -- known rows are reported, unknown ones fail G-MEDIA.
            no_audio_from_host.append(exc.as_row())
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

    known_status = known_missing_audio_status(known_missing, want, audio_dir)
    stats["known_missing_upstream"] = len(known_status["still_missing"])
    stats["recovered_upstream"] = len(known_status["recovered"])
    # The hint is about OUR gaps: "pass --seed-legacy" is no advice at all for a
    # slot the upstream host serves empty.
    unexplained_missing = [u for u in missing
                           if u not in set(known_status["still_missing"])]
    report = {**stats,
              "new_vs_known": {"known_urls_before": len(known_before),
                               "new_urls_this_run": len(set(want) - known_before),
                               "known_urls_now": len(manifest),
                               "gone_from_ddo": len(known_before - set(want))},
              "known_missing_audio": known_status,
              "no_audio_from_host": no_audio_from_host[:20],
              "n_no_audio_from_host": len(no_audio_from_host),
              "orphans_sample": orphans[:20],
              "unreferenced_sample": unreferenced[:20],
              "sweep_hint": (
                  "%d cached file(s) are not referenced by entries.json; pass "
                  "--sweep-orphans to quarantine them into %s once you are sure "
                  "the parse is complete" % (len(unreferenced), ORPHAN_DIR)
                  if unreferenced and not sweep_orphans else None),
              "recovered_hint": (
                  "%d slot(s) in registry/known_missing_audio.json now serve real "
                  "audio; delete those rows and bump gates.json:"
                  "known_missing_audio_max down in the same commit"
                  % len(known_status["recovered"])
                  if known_status["recovered"] else None),
              "hint": ("run with --seed-legacy --legacy-workspace <path> to "
                       "hardlink the 2025 files instead of downloading them"
                       if unexplained_missing and not seed_legacy else None)}
    write_json(cfg.report_dir / "audio_report.json", report)
    run_gates([Gate(G_MEDIA, "every non-null audio_url resolves to a non-empty "
                             "file in the audio cache, except the slots "
                             "registry/known_missing_audio.json declares dead "
                             "upstream -- and that registry is inside its "
                             "baseline",
                    lambda: _media_gate(
                        missing, zero_byte, len(want), known_status,
                        int(registry.gates.get("known_missing_audio_max", 0))),
                    stage="60")], cfg, stage="60")
    return report
