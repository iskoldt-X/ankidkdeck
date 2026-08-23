"""Stage 40: OFFLINE re-key of the 2025 translation assets. Zero network, zero LLM.

Four steps, all measured against the recovered 2025 production workspace:

  (a) filename -> entry_id, from the fejlrapport mailto in the saved HTML.
      5,259 of 5,267 files yield an id, 3,812 distinct; the 8 that do not are
      exactly the 8 files that produced no entry in 2025 (set-equal, both
      directions), and they occupy the deck's 8 long-standing FrequencyRank
      holes.
  (b) the v2.1 QueryWord set (4,442 words) with its rank -- the input stage 30
      needs to freeze GUID seeds without guessing.
  (c) the offline sense address. `dannetid` is a 2026-only attribute (0 of the
      5,267 2025 files carry it), so the 2025 side can only be addressed by
      `id="betydning-1-2"`. Stage 41 promotes those to dannetid after the
      first 2026 parse. This is why "track 1, runnable today" has to be split
      in two.
  (d) re-key each language's translations from {filename: {danish_text: ...}}
      to {entry_id: {danish_text: ...}}, applying the WRITTEN tie-break. For
      40-50% of the cells inside multi-file buckets there are 2-6 candidate
      translations; with no written rule the winner comes from dict iteration
      order and the build is not reproducible. Every loser is written out.

Runtime note: step (c) parses every bridged 2025 page with html.parser. Expect
tens of minutes on the full corpus. The stage is restartable -- it writes only
at the end of each step and reads nothing it wrote.
"""

import re

from ..config import Config
from ..extract import xt
from ..gates import G_TIE, Gate, run_gates, tie_break_resolved
from ..util import (NFC, FatalError, nk, read_json, read_text_nfc_tolerant,
                    sha256_str, write_json)

# NOTE the character class: only " and > are excluded, NOT the apostrophe.
# The guide prints [^"'>] and its own measurement contradicts it: d'herrer
# (entry 11008184) carries the apostrophe inside the mailto subject, so the
# apostrophe-excluding form bridges 5,258/3,811 and leaves 9 files unbridged,
# which then is NOT set-equal to the 8 no-entry files. With this form the
# published 5,259/3,812/8 reproduces exactly. [verified on the 2025 corpus]
MAILTO_RE = re.compile(r"mailto:[^\">]*?\((\d{6,})\)")
AUDIO_ID_RE = re.compile(r'<audio[^>]*id="(\d{6,})_\d+"')

LEGACY_MODEL = "gemini-2.0-flash"
FULL_CORPUS_FILES = 5267
FULL_CORPUS_BRIDGED = 5259
FULL_CORPUS_ENTRY_IDS = 3812
FULL_CORPUS_QUERYWORDS = 4442

# Guide section 1.3, reproduced digit-for-digit from the 2025 assets. Printed
# next to what this run measures; deliberately NOT asserted, because a
# re-measurement that differs is information, not a crash.
EXPECTED_CONFLICTS = {
    "definitions": {"English": 2174, "German": 2158, "Chinese": 2744, "Spanish": 2337},
    "expressions": {"English": 1663, "German": 4072, "Chinese": 4703, "Spanish": 5129},
}


def _legacy_dir(cfg: Config):
    if not cfg.legacy_workspace:
        raise FatalError(
            "legacy_workspace is not configured. Point it at the recovered "
            "2025 production workspace (the directory holding download_map.json, "
            "ddo_entries.json, ddo_html_all_versions/ and the "
            "*_translations_*.json files)."
        )
    p = cfg.legacy_workspace
    if not p.exists():
        raise FatalError(f"legacy_workspace does not exist: {p}")
    return p


def _legacy_json(cfg: Config, name: str, pattern: str | None = None):
    """Load a legacy artifact by exact name, falling back to a glob so a
    re-run of the 2025 translators under a different model name still loads."""
    root = _legacy_dir(cfg)
    p = root / name
    if p.exists():
        return read_json(p)
    if pattern:
        hits = sorted(root.glob(pattern))
        if len(hits) == 1:
            return read_json(hits[0])
        if len(hits) > 1:
            raise FatalError(f"ambiguous legacy artifact {pattern!r}: {[h.name for h in hits]}")
    raise FatalError(f"legacy artifact not found: {p}")


def bridge_filenames(cfg: Config, download_map: dict, entry_filenames: set) -> tuple[dict, dict]:
    """T1. 2025 filename -> entry_id, NFC/NFD tolerant."""
    html_dir = _legacy_dir(cfg) / "ddo_html_all_versions"
    if not html_dir.is_dir():
        raise FatalError(f"2025 HTML corpus not found: {html_dir}")
    bridge, unbridged, via_audio = {}, [], 0
    for fn in download_map:
        html = read_text_nfc_tolerant(html_dir / fn)
        m = MAILTO_RE.search(html)
        if m is None:
            m = AUDIO_ID_RE.search(html)
            if m is not None:
                via_audio += 1
        if m is None:
            unbridged.append(fn)
            continue
        bridge[fn] = m.group(1)

    no_entry = sorted(set(download_map) - entry_filenames)
    # Structural invariant, scale-free: a file bridges if and only if it
    # produced an entry in 2025. This one is fatal at any corpus size.
    if sorted(unbridged) != no_entry:
        raise FatalError(
            "bridge is not set-equal to the no-entry files: unbridged=%s "
            "no_entry=%s" % (sorted(unbridged)[:12], no_entry[:12]))
    stats = {"files": len(download_map), "bridged": len(bridge),
             "distinct_entry_ids": len(set(bridge.values())),
             "via_audio_fallback": via_audio, "unbridged": sorted(unbridged)}
    if len(download_map) == FULL_CORPUS_FILES:
        expect = (FULL_CORPUS_BRIDGED, FULL_CORPUS_ENTRY_IDS)
        got = (len(bridge), len(set(bridge.values())))
        if got != expect:
            raise FatalError(
                f"full 2025 corpus bridged {got} (bridged, distinct ids); "
                f"expected {expect}")
        stats["full_corpus_counts_verified"] = True
    else:
        # A fixture or a partial copy: report rather than crash, so the offline
        # tests can run this stage on a handful of pages.
        stats["full_corpus_counts_verified"] = False
        stats["note"] = (f"corpus holds {len(download_map)} files, not "
                         f"{FULL_CORPUS_FILES}; absolute counts not asserted")
    return bridge, stats


def v2_querywords(entries_2025: list) -> dict:
    """(b) The QueryWords that actually shipped as v2.1 notes, with rank."""
    out: dict[str, int] = {}
    for e in entries_2025:
        w = NFC(e["query_word"])
        r = int(e["query_rank"])
        if w not in out or r < out[w]:
            out[w] = r
    return out


def sense_paths(cfg: Config, files_by_eid: dict) -> tuple[dict, dict]:
    """(c) {entry_id: {sense_path: definition_text}} from the 2025 markup.

    Files are visited in tie-break order, so when two downloads of the same
    article disagree the winner is the same file whose translations win in
    step (d). Disagreements are counted, never silently overwritten.
    """
    # Imported here, not at module scope: steps (a), (b) and (d) are pure
    # dict work, and keeping them importable without a parser lets the
    # re-key be verified on a host that has no bs4 installed.
    from bs4 import BeautifulSoup

    html_dir = _legacy_dir(cfg) / "ddo_html_all_versions"
    out: dict[str, dict] = {}
    disagreements, n_parsed = [], 0
    for eid in sorted(files_by_eid):
        for fn in files_by_eid[eid]:
            soup = BeautifulSoup(read_text_nfc_tolerant(html_dir / fn), "html.parser")
            n_parsed += 1
            for box in soup.select('div.definitionBox[id^="betydning-"]'):
                span = box.select_one("span.definition")
                if span is None:
                    continue
                text = xt(span, "definition")  # SPACE separator (extract.SEP)
                path = box.get("id")
                prev = out.setdefault(eid, {}).get(path)
                if prev is None:
                    out[eid][path] = text
                elif prev != text and len(disagreements) < 200:
                    disagreements.append({"entry_id": eid, "sense_path": path,
                                          "file": fn})
    return out, {"files_parsed": n_parsed, "entry_ids": len(out),
                 "sense_paths": sum(len(v) for v in out.values()),
                 "cross_file_disagreements": len(disagreements),
                 "disagreement_sample": disagreements[:20]}


def _order_key(fn: str, download_map: dict, headword_of: dict):
    """The written tie-break (guide 1.3): the lemma's own file first, then
    frequency, then byte order. Explicit, so it is reproducible across
    machines -- never JSON insertion order."""
    d = download_map[fn]
    qw = NFC(d.get("query_word") or "")
    hw = NFC(headword_of.get(fn) or "")
    return (0 if hw and nk(qw) == nk(hw) else 1,
            int(d.get("query_rank") or 10 ** 9),
            fn)


def rekey(old: dict, files_by_eid: dict, keyfn) -> tuple[dict, list, dict]:
    """(d) {filename: {text: {lemma, gloss}}} -> {entry_id: {text: row}}."""
    out: dict[str, dict] = {}
    discarded: list = []
    stats = {"cells_in": sum(len(v) for v in old.values()),
             "cells_out": 0, "multi_candidate": 0, "conflicts": 0,
             "unresolved_conflicts": 0, "cells_in_unbridged_files": 0}
    bridged_files = {f for fs in files_by_eid.values() for f in fs}
    for fn, inner in old.items():
        if fn not in bridged_files:
            stats["cells_in_unbridged_files"] += len(inner)

    for eid in sorted(files_by_eid):
        files = files_by_eid[eid]
        texts, seen = [], set()
        for f in files:
            for t in old.get(f, {}):
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
        for text in texts:
            hits = [(f, old[f][text]) for f in files if text in old.get(f, {})]
            winner_file, winner = hits[0]
            out.setdefault(eid, {})[text] = {
                "lemma": winner.get("lemma"),
                "gloss": winner.get("gloss"),
                "src_sha": sha256_str(NFC(text)),
                "provenance": f"migrated:2025:{winner_file}",
            }
            stats["cells_out"] += 1
            if len(hits) == 1:
                continue
            stats["multi_candidate"] += 1
            conflict = len({(v.get("lemma"), v.get("gloss")) for _, v in hits}) > 1
            if conflict:
                stats["conflicts"] += 1
                if keyfn(hits[0][0]) == keyfn(hits[1][0]):
                    # Two candidates the written rule cannot separate. This is
                    # what G-TIE forbids; filenames are unique so it is
                    # unreachable, and the check is what proves that.
                    stats["unresolved_conflicts"] += 1
            discarded.append({
                "entry_id": eid, "key": text, "winner_file": winner_file,
                "conflict": conflict,
                "losers": [{"file": f, "lemma": v.get("lemma"), "gloss": v.get("gloss")}
                           for f, v in hits[1:]],
            })
    return out, discarded, stats


def run(cfg: Config, registry=None) -> dict:
    root = _legacy_dir(cfg)
    legacy_out = cfg.json_dir / "legacy"
    download_map = read_json(root / "download_map.json")
    entries_2025 = read_json(root / "ddo_entries.json")
    if not isinstance(entries_2025, list):
        raise FatalError("legacy ddo_entries.json is expected to be a list of entries")
    headword_of = {e["filename"]: e.get("headword") for e in entries_2025}

    report: dict = {}

    # ---- (a) bridge --------------------------------------------------------
    bridge, bridge_stats = bridge_filenames(cfg, download_map, set(headword_of))
    write_json(legacy_out / "filename_to_entry_id.json", bridge)
    report["bridge"] = bridge_stats

    def keyfn(fn):
        return _order_key(fn, download_map, headword_of)

    files_by_eid: dict[str, list] = {}
    for fn, eid in bridge.items():
        files_by_eid.setdefault(eid, []).append(fn)
    for eid in files_by_eid:
        files_by_eid[eid].sort(key=keyfn)

    # ---- (b) v2 QueryWords -------------------------------------------------
    v2 = v2_querywords(entries_2025)
    write_json(legacy_out / "v2_querywords.json", v2)
    report["v2_querywords"] = {"n": len(v2), "expected": FULL_CORPUS_QUERYWORDS,
                               "matches_expected": len(v2) == FULL_CORPUS_QUERYWORDS}

    # ---- (c) offline sense addresses ---------------------------------------
    paths, path_stats = sense_paths(cfg, files_by_eid)
    write_json(legacy_out / "sense_paths.json", paths)
    # Self-check on the separator contract: every 2025 definition text the
    # translators were keyed on should be recoverable from the markup we just
    # extracted. A low rate here means extract.SEP["definition"] drifted, and
    # stage 41's sense_path fallback would silently stop firing.
    have = 0
    total = 0
    for e in entries_2025:
        eid = bridge.get(e["filename"])
        if eid is None:
            continue
        texts = set(paths.get(eid, {}).values())
        for d in e.get("definitions", []):
            total += 1
            if d.get("definition") in texts:
                have += 1
    path_stats["legacy_definition_texts"] = total
    path_stats["recovered_from_markup"] = have
    path_stats["recovery_rate"] = round(have / total, 4) if total else None
    report["sense_paths"] = path_stats

    # ---- (d) re-key each language ------------------------------------------
    per_lang_tie: dict[str, dict] = {}
    report["languages"] = {}
    for lang in cfg.langs:
        defs_old = _legacy_json(cfg, f"definition_translations_{lang}_{LEGACY_MODEL}.json",
                                f"definition_translations_{lang}_*.json")
        # The 2025 expression files have NO underscore before the language.
        exprs_old = _legacy_json(cfg, f"expression_translations{lang}_{LEGACY_MODEL}.json",
                                 f"expression_translations{lang}_*.json")
        d_out, d_disc, d_stats = rekey(defs_old, files_by_eid, keyfn)
        e_out, e_disc, e_stats = rekey(exprs_old, files_by_eid, keyfn)
        write_json(legacy_out / f"legacy_{lang}_definitions.json", d_out)
        write_json(legacy_out / f"legacy_{lang}_expressions.json", e_out)
        write_json(legacy_out / f"discarded_variants_{lang}.json",
                   {"definitions": d_disc, "expressions": e_disc})
        per_lang_tie[lang] = {
            "unresolved_conflicts": (d_stats["unresolved_conflicts"]
                                     + e_stats["unresolved_conflicts"]),
            "discard_file_written": True,
            "definition_conflicts": d_stats["conflicts"],
            "expression_conflicts": e_stats["conflicts"],
        }
        report["languages"][lang] = {
            "definitions": {**d_stats,
                            "entry_ids": len(d_out),
                            "conflicts_expected_from_guide":
                                EXPECTED_CONFLICTS["definitions"].get(lang)},
            "expressions": {**e_stats,
                            "entry_ids": len(e_out),
                            "conflicts_expected_from_guide":
                                EXPECTED_CONFLICTS["expressions"].get(lang)},
        }

    run_gates([
        Gate(G_TIE, "every multi-candidate migration cell was resolved by the "
                    "written tie-break and every loser was written out",
             lambda: tie_break_resolved(per_lang_tie), stage="40"),
    ], cfg, stage="40")

    write_json(cfg.report_dir / "migrate_report.json", report)
    write_json(cfg.report_dir / "guid_dryrun.json", {
        "n_v2_querywords": len(v2),
        "n_2025_files": bridge_stats["files"],
        "n_bridged": bridge_stats["bridged"],
        "n_2025_entry_ids": bridge_stats["distinct_entry_ids"],
        "note": "family / kept / retired accounting is emitted by stage 30 "
                "(registry_freeze_report.json) and by tools/guid_diff.py (T5)",
    })
    return report
