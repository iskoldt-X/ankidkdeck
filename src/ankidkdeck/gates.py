"""The gate framework: named, reported, blocking checks.

A gate is an id from the final guide's gate table (section 4.12), one human
sentence, and a zero-argument function returning (ok, detail). run_gates()
executes every gate, records every result in reports/gates_report.json, and
only then raises FatalError listing the failures. Two properties make that
report worth reading:

  1. ALL gates run before anything raises, so one build shows every failure
     instead of only the first.
  2. Results accumulate in one file, merged by (gate id, stage, extra), so a
     later stage appends to the same report rather than overwriting it. `extra`
     is what makes the per-LANGUAGE export gates independent rows: G-COV,
     G-RATE, G-MEDIA and G-DET are verdicts about one language's deck, and
     merging them on the bare id let a passing German export erase a failing
     Chinese one -- i.e. `ankidkdeck gates` certified a release all-green while
     the Chinese deck's coverage failure had been overwritten.

A gate that cannot fail is not a gate: every helper below returns the measured
detail alongside the verdict, so a passing gate still leaves evidence.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import FatalError, NFC, canonical_json, read_json, write_json

# EVERY gate id in the pipeline lives here, spelled exactly as in the final
# guide's table (section 4.12). Segment 3 used to declare its own copies at the
# top of s42/s50/s70; one list means a typo in a gate id cannot silently create
# a second, invisible gate.
G_RANK = "G-RANK"
G_SEED = "G-SEED"
G_GUID = "G-GUID"
G_ANCHOR = "G-ANCHOR"
G_AFFIX = "G-AFFIX"
G_BIND = "G-BIND"
G_TIE = "G-TIE"
G_SITEMAP = "G-SITEMAP"
G_ASSIGN = "G-ASSIGN"  # "every word in at most one family" (guide 4.9 step 4)
G_REGKEY = "G-REGKEY"  # family_id IS an entry_id; no wordlist form may leak in
# Export-time and LLM-stage ids (previously local to s42/s50/s70).
G_ORPH = "G-ORPH"
G_ORDER = "G-ORDER"       # 4.11 spells the assertion out but names no id
G_EMPTY_C = "G-EMPTY-C"
G_RATE = "G-RATE"
G_COV = "G-COV"
G_MEDIA = "G-MEDIA"
G_NOTE = "G-NOTE"
G_DET = "G-DET"
# The separator table (guide 4.12 row 1): "wire it into the export gate (G-SEP)
# so an .apkg can never be built by a parser with a corrupted separator table".
# It used to exist only under pytest, which is not on the export path.
G_SEP = "G-SEP"
G_LABEL = "G-LABEL"          # #results-label reconciles against the article count
G_REL = "G-REL"              # the release-note churn numbers are computed, not estimated
G_SITEMAP_INV = "G-SITEMAP-INV"   # the sitemap inventory is inside its declared range
G_CASE = "G-CASE"            # the case-only membership population is baselined
# The guide names no id for stage 21, which is how override_problems went 0 ->
# 140 -- every curated mapping in the registry failing to bind -- with
# gates_report.json still reporting every row ok. Stage 21 had no gate at all.
G_OVERRIDE = "G-OVERRIDE"
# Round 4 added two registries (owner decisions 10.45 and 10.5) that SILENTLY
# REMOVE words from reports/unresolved.json -- the only artifact that shows them.
# known_no_entry went 0 -> 479 rows and wordlist_invalid 0 -> 53 in one step, and
# round 1 section 4.1 and round 2 section 9.4 both made a baseline the explicit
# precondition for doing that, for the same reason all_demoted_families_max and
# case_only_members_max exist: an unbaselined suppression list is how a
# population grows with nobody noticing.
G_SUPPRESS = "G-SUPPRESS"
# Round 4's OTHER new channel, and the one it forgot to baseline. Layer 4 is an
# ADMISSION channel: it mints cards from articles the classifier rejects, on the
# strength of an owner policy rather than of automatic evidence. Every other
# human-policy channel here is baselined -- G-OVERRIDE + override_problems_max,
# case_only_members_max, all_demoted_families_max, and G-SUPPRESS for the two
# registries added in the same change -- so the argument that produced those
# applies verbatim to this one (reviewer A, round 4, MAJOR-2). Indirect cover was
# partial and would have missed the interesting case: 18 of the 19 admissions are
# demoted and land in G-ANCHOR's all_demoted_families, but `hr.` is `sb.` and is
# invisible there.
G_ADMIT = "G-ADMIT"

# The declared set, so a report can say what it did NOT check. "11 gates PASS"
# was read as a release verdict when it meant "11 rows are on file, 8 of them
# written by this run, and 12 of the 24 declared gates have never executed on
# this workspace at all" -- G-SITEMAP among them, which is the one that
# adjudicates merge_report.sitemap_shortfall_families. A gate that never ran is
# not a gate that passed.
ALL_GATE_IDS = (
    G_ADMIT, G_AFFIX, G_ANCHOR, G_ASSIGN, G_BIND, G_CASE, G_COV, G_DET,
    G_EMPTY_C, G_GUID, G_LABEL, G_MEDIA, G_NOTE, G_ORDER, G_ORPH, G_OVERRIDE,
    G_RANK, G_RATE, G_REGKEY, G_REL, G_SEED, G_SEP, G_SITEMAP, G_SITEMAP_INV,
    G_SUPPRESS, G_TIE,
)

# Gates that are a HUMAN signature, not a script; they can never appear in
# gates_report.json, so they are named here rather than silently missing.
MANUAL_GATE_IDS = ("G-IMPORT", "G-REVIEW")

# (report path, row_key) for every row a gate in THIS invocation actually
# produced. The report is a merged ledger by design -- stage 20 through stage 41
# all append to it, and rows survive across runs -- so "12 rows recorded, 0
# failing" was read as "12 gates just passed" when two of those rows (G-TIE at
# stage 40, G-SITEMAP-INV at stage 10) were left by an earlier run that touched
# stages this build never executed. Nothing else in the file can tell those two
# apart from the 10 that ran, so the running stage marks its own rows here.
#
# Deliberately process-scoped rather than a timestamp or a run counter: the
# value is a function of WHICH stages the invocation executed, so two identical
# `build` runs still write byte-identical bytes, and it only changes when the
# set of executed stages changes -- which is exactly when it should.
_ROWS_EXECUTED_THIS_RUN: set[tuple[str, str]] = set()

# The closed set of translation drop reason codes. A drop carrying anything
# else is an unexplained loss, which is what G-BIND exists to forbid.
DROP_REASONS = frozenset({
    "article_gone_from_ddo",
    "rejected_article",
    "sense_text_changed",
    "expression_text_changed",
    "shared_dannetid_conflict",
    "source_gap",
})

AFFIX_POS_KEYS = frozenset({"førsteled", "sidsteled", "suffiks", "præfiks"})

# family_id is the anchor article's entry_id and nothing else: it is the
# permanent key of card_keys.json, so a wordlist-derived token in it would be
# orphaned the day that word joins another family or leaves the list.
FAMILY_ID_RE = re.compile(r"^[0-9]{6,}$")


def is_affix_entry(entry: dict) -> bool:
    """Affix pages are detected by headword shape AND data-pos-key, never by
    sitemap shard: `-kvinde` (kvinder), `-ske` (sker), `for-`."""
    lemma = entry.get("lemma") or ""
    return (entry.get("pos_key") in AFFIX_POS_KEYS
            or lemma.startswith("-") or lemma.endswith("-"))


@dataclass
class Gate:
    """id: the guide's gate id. description: why a human should care.
    fn: () -> (ok, detail). detail is JSON-serialisable and always recorded.

    extra: the SCOPE of this verdict, and part of the report's merge key. A
    per-language export gate passes extra={"lang": lang} so that two languages
    keep two rows instead of the later run silently overwriting the earlier
    one's failure.
    """

    id: str
    description: str
    fn: Callable[[], tuple[bool, Any]]
    stage: str = ""
    extra: dict = field(default_factory=dict)


def row_key(row: dict) -> str:
    """The report's merge key: (id, stage, extra). Canonical JSON so the key is
    stable across runs and machines regardless of dict order."""
    return canonical_json([row.get("id"), str(row.get("stage") or ""),
                           row.get("extra") or {}])


def row_label(row: dict) -> str:
    """`G-COV` or `G-COV[lang=Chinese]` -- what a human needs to see in the
    failure list to know WHICH deck failed."""
    extra = row.get("extra") or {}
    if not extra:
        return str(row.get("id"))
    inner = ",".join("%s=%s" % (k, extra[k]) for k in sorted(extra))
    return "%s[%s]" % (row.get("id"), inner)


def run_gates(gates: Iterable[Gate], cfg, stage: str = "") -> list[dict]:
    """Run every gate, write the merged report, then raise on any failure."""
    results = []
    for g in gates:
        ok, detail = g.fn()
        results.append({"id": g.id, "description": g.description,
                        "stage": g.stage or stage, "extra": dict(g.extra),
                        "ok": bool(ok), "detail": detail})
    _write_report(cfg, results)
    failed = [r for r in results if not r["ok"]]
    if failed:
        lines = ["  %s: %s -> %s" % (row_label(r), r["description"], r["detail"])
                 for r in failed]
        raise FatalError(
            "%d gate(s) failed; no output is valid until they pass:\n%s"
            % (len(failed), "\n".join(lines))
        )
    return results


def _write_report(cfg, results: list[dict]) -> None:
    path = cfg.report_dir / "gates_report.json"
    prev = read_json(path, default={"results": []})
    for row in results:
        row.setdefault("extra", {})
        _ROWS_EXECUTED_THIS_RUN.add((str(path), row_key(row)))
    rows = list(prev.get("results", [])) + results
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        row.setdefault("extra", {})
        key = row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row      # a later run of the SAME scope wins
    # Provenance, per row, recomputed from this process every time -- never
    # trusted from the file, or an earlier run's `true` would be carried forward
    # as if this run had produced it.
    for key in order:
        merged[key]["executed_this_run"] = (
            (str(path), key) in _ROWS_EXECUTED_THIS_RUN)
    out = {"results": [merged[k] for k in order]}
    # `failed` stays a list of bare ids (that is what the release checklist and
    # the tests read); `failed_rows` carries the scope, so a Chinese-only
    # failure is not reported as if German had failed too.
    out["failed"] = sorted({r["id"] for r in out["results"] if not r["ok"]})
    out["failed_rows"] = [row_label(r) for r in out["results"] if not r["ok"]]
    out["n_gates"] = len(out["results"])
    # RAN vs NEVER-RUN. `n_gates` counts ROWS ON FILE, which is not the same
    # question as "which gates have a verdict on this workspace" -- the report
    # accumulates across stages and across runs by design, so 11 passing rows
    # coexisted with 12 declared gates that had never executed. No timestamps
    # and no run counters: this file has to stay byte-stable across a repeated
    # run for the determinism check to mean anything.
    have = {r["id"] for r in out["results"]}
    out["gates_declared"] = len(ALL_GATE_IDS)
    out["gate_ids_with_a_verdict"] = sorted(have)
    out["gate_ids_never_run"] = sorted(set(ALL_GATE_IDS) - have)
    out["stages_reported"] = sorted({str(r.get("stage") or "?")
                                     for r in out["results"]})
    # Three states, not two: RAN here, CARRIED from an earlier run, and NEVER
    # RUN on this workspace (gate_ids_never_run above). Row LABELS, not bare
    # ids, so a per-language gate that ran for German and carries a stale
    # Chinese row is not summarised as if both were fresh.
    out["gate_rows_executed_this_run"] = [
        row_label(r) for r in out["results"] if r["executed_this_run"]]
    out["gate_rows_carried_from_an_earlier_run"] = [
        row_label(r) for r in out["results"] if not r["executed_this_run"]]
    out["stages_executed_this_run"] = sorted(
        {str(r.get("stage") or "?") for r in out["results"]
         if r["executed_this_run"]})
    out["manual_gates_not_recorded_here"] = list(MANUAL_GATE_IDS)
    write_json(path, out)


# --------------------------------------------------------------------------
# Reusable gate bodies. Each returns (ok, detail); bind them into a Gate with
# a lambda or functools.partial at the call site.
# --------------------------------------------------------------------------

def dense_unique_ranks(ranks: Iterable[int], expected_n: int | None = None):
    """G-RANK. FrequencyRank is the Anki sort field, stored as a STRING and
    ordered only by SQLite integer affinity: a duplicate makes the deck order
    undefined, and a hole makes it non-reproducible."""
    vals = list(ranks)
    n = expected_n if expected_n is not None else len(vals)
    uniq = set(vals)
    ok = len(vals) == n and uniq == set(range(1, n + 1))
    dupes = sorted({v for v in vals if vals.count(v) > 1}) if len(uniq) != len(vals) else []
    return ok, {"n": len(vals), "expected_n": n, "unique": len(uniq),
                "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                "duplicates": dupes[:20],
                "missing": sorted(set(range(1, n + 1)) - uniq)[:20]}


def registry_seed_bytes(card_keys: dict, v2_querywords: dict, family_ids=None):
    """G-SEED. Every carried guid_seed must be NFC and byte-equal to the v2.1
    QueryWord it claims to carry -- 55 (headword, wordlist-word) pairs differ
    only by case, `Er`/`er` at rank 1, and guid_for() hashes the bytes."""
    scope = set(family_ids) if family_ids is not None else set(card_keys)
    bad_nfc, not_in_v2, carried, fresh = [], [], 0, 0
    for fid in sorted(scope):
        row = card_keys.get(fid)
        if row is None:
            not_in_v2.append({"family_id": fid, "why": "no registry row"})
            continue
        seed = row.get("guid_seed", "")
        if NFC(seed) != seed:
            bad_nfc.append(fid)
        if row.get("carried_from_v2"):
            carried += 1
            if seed not in v2_querywords:
                not_in_v2.append({"family_id": fid, "guid_seed": seed})
        else:
            fresh += 1
    seeds = [card_keys[f]["guid_seed"] for f in scope if f in card_keys]
    dupes = sorted({s for s in seeds if seeds.count(s) > 1})
    ok = not bad_nfc and not not_in_v2 and not dupes
    return ok, {"families": len(scope), "carried": carried, "new": fresh,
                "non_nfc_seeds": bad_nfc[:20],
                "carried_seed_not_a_v2_queryword": not_in_v2[:20],
                "duplicate_seeds": dupes[:20]}


def unique_assignment(assignments: dict, families: dict):
    """G-ASSIGN. Each wordlist form belongs to exactly one family: /mit returns
    jeg AND min, /have returns have AND hav, and a form counted twice makes
    FrequencyRank non-unique."""
    seen: dict[str, list[str]] = {}
    for fid, fam in families.items():
        for m in fam.get("members", []):
            seen.setdefault(m["word"], []).append(fid)
    multi = {w: fids for w, fids in seen.items() if len(fids) > 1}
    mismatched = [w for w, a in assignments.items()
                  if seen.get(w, [None])[0] != a.get("family_id")]
    ok = not multi and not mismatched
    return ok, {"words_assigned": len(assignments),
                "words_in_more_than_one_family": dict(list(multi.items())[:20]),
                "assignment_vs_membership_mismatch": mismatched[:20]}


def anchors_not_demoted_or_affix(families: dict, entries: dict, demoted_pos: set,
                                 all_demoted_max: int | None = None):
    """G-ANCHOR + G-AFFIX. A family must never be headed by an affix page
    (`-kvinde`, `-ske`) and must never be headed by a demoted article while a
    real one is available in the same family.

    Deviation from the guide, deliberate and reported: a family every one of
    whose articles is demoted (a `cm`/`Cm` abbreviation, a chemical symbol
    reached through its own flex table) has no non-demoted anchor to pick, and
    stopping the build for it would contradict the owner's 2026-08-24 ruling
    that word-level resolution problems are recorded, not fatal. Those are
    listed as all_demoted -- but their COUNT is baselined
    (registry/gates.json:all_demoted_families_max) so the population cannot
    grow silently, which is the condition the reviewers put on the softening.

    The other half of the invariant lives in stage 30's anchor_of(), which sorts
    demotion FIRST: with that key demoted_anchored_with_alternative is
    unreachable by construction, so a failure here is a real defect rather than
    a tie-break accident.
    """
    affix_anchored, demoted_anchored, all_demoted = [], [], []
    for fid, fam in families.items():
        a = entries[fam["anchor_entry_id"]]
        lemma = a.get("lemma", "")
        if is_affix_entry(a):
            affix_anchored.append({"family_id": fid, "lemma": lemma,
                                   "pos_key": a.get("pos_key")})
        if a.get("pos_key") in demoted_pos:
            alt = [e for e in fam["entry_ids"]
                   if entries[e].get("pos_key") not in demoted_pos
                   and not entries[e].get("empty")]
            row = {"family_id": fid, "lemma": lemma, "pos_key": a.get("pos_key"),
                   "non_demoted_alternatives": alt}
            (demoted_anchored if alt else all_demoted).append(row)
    over = all_demoted_max is not None and len(all_demoted) > all_demoted_max
    ok = not affix_anchored and not demoted_anchored and not over
    return ok, {"affix_anchored": affix_anchored[:20],
                "demoted_anchored_with_alternative": demoted_anchored[:20],
                "all_demoted_families": len(all_demoted),
                "all_demoted_families_max": all_demoted_max,
                "all_demoted_over_baseline": bool(over),
                "all_demoted_sample": all_demoted[:10]}


def no_affix_members(classification: dict, entries: dict):
    """G-AFFIX. No accepted member may be an affix-page article.

    5 of the spec's 30 new lemmas are affix pages, and `-kvinde` absorbed into
    the `kvinde` card is how the real word vanishes. This used to be a bare
    `raise AssertionError` AFTER classification.json was written; as a gate it
    runs before the outputs are final and shows up in gates_report.json.
    """
    bad = []
    for word in sorted(classification):
        for m in (classification[word].get("members") or []):
            e = entries.get(m["entry_id"])
            if e is not None and is_affix_entry(e):
                bad.append({"word": word, "entry_id": m["entry_id"],
                            "lemma": e.get("lemma"), "pos_key": e.get("pos_key"),
                            "bucket": m.get("bucket"),
                            "evidence": m.get("evidence")})
    return not bad, {"words": len(classification), "affix_members": len(bad),
                     "sample": bad[:20]}


def registry_family_ids(card_keys: dict):
    """G-REGKEY. Every card_keys.json key is a bare DDO entry_id.

    The v1 refused-merge fallback minted `11028611#kunne`, putting a wordlist
    form into the registry whose whole purpose is to make wordlist changes
    GUID-neutral. A gate makes that impossible to commit rather than merely
    unlikely."""
    bad = sorted(k for k in card_keys if not FAMILY_ID_RE.match(str(k)))
    return not bad, {"rows": len(card_keys), "malformed": len(bad),
                     "sample": bad[:20]}


def bind_accounting(per_lang: dict):
    """G-BIND. n_bound + n_dropped == n_legacy, every drop carries a reason
    code from the closed set, n_unexplained == 0. This replaces the 2025 gate
    "coverage == 2025 coverage, not one row lost", which was unsatisfiable and
    therefore never enforced."""
    bad = {}
    for lang, s in per_lang.items():
        problems = {}
        if s["n_bound"] + s["n_dropped"] != s["n_legacy"]:
            problems["accounting"] = {"bound": s["n_bound"], "dropped": s["n_dropped"],
                                      "legacy": s["n_legacy"]}
        unknown = sorted(set(s.get("reasons", {})) - DROP_REASONS)
        if unknown:
            problems["unknown_reason_codes"] = unknown
        if s.get("n_unexplained", 0):
            problems["n_unexplained"] = s["n_unexplained"]
        if problems:
            bad[lang] = problems
    return not bad, {"per_language": per_lang, "violations": bad}


def tie_break_resolved(per_lang: dict, byte_order_max: int = 0):
    """G-TIE. Every multi-candidate migration cell was resolved by the written
    tie-break and every loser was written out. 40-50% of the cells in
    multi-file buckets carry 2-6 candidate translations, so without this the
    build picks one by dict iteration order and is not reproducible.

    `unresolved_conflicts` counts conflicts the RULE could not separate -- the
    key compared WITHOUT its filename component. Comparing the full key made
    the number structurally zero (filenames are unique), i.e. a gate that could
    not fail. Falling through to byte order is still reproducible, so the count
    is baselined (registry/gates.json:tie_break_byte_order_max) instead of
    being flatly forbidden: the gate fires when the population GROWS.
    """
    bad = {}
    for lang, s in per_lang.items():
        problems = {}
        n = s.get("unresolved_conflicts", 0)
        if n > byte_order_max:
            problems["conflicts_resolved_only_by_byte_order"] = n
        if not s.get("discard_file_written"):
            problems["discard_file_written"] = False
        if problems:
            bad[lang] = problems
    return not bad, {"per_language": per_lang, "violations": bad,
                     "byte_order_max": byte_order_max}


def sitemap_shortfall(rows: list, n_families: int, max_rate: float):
    """G-SITEMAP. Per-family homograph shortfall against the sitemap
    inventory: how many DDO articles for this lemma we never saw. Reported by
    stage 30, enforced at export time -- it recovers the articles the 2025
    crawl missed (vinge, uanset, vove, zone)."""
    rate = (len(rows) / n_families) if n_families else 0.0
    return rate <= max_rate, {"families_short": len(rows), "n_families": n_families,
                              "rate": round(rate, 5), "max_rate": max_rate,
                              "sample": rows[:20]}


def sitemap_inventory(total: int, total_range, affix_slugs: int, affix_range):
    """G-SITEMAP-INV. The inventory's own size, as a recorded verdict.

    The URL total used to be a bare `raise FatalError(total < 80_000)` with the
    threshold as a source constant, so (a) it never reached
    gates_report.json and (b) the bound was extrapolated from nothing. An
    absolute lower bound guessed from a partial measurement is the same class of
    vacuous gate the guide rejects: `sitemap_total_range` therefore ships as
    null, meaning REPORT ONLY, and a human copies the first real 9-request run's
    total into registry/gates.json as a band. A 3x jump is as suspicious as a
    collapse -- it means the shard set changed -- which is why it is a range and
    not a floor.

    The affix range is already baselined from a real measurement, so it stays
    enforced; it just lands in the report now instead of raising inline.
    """
    detail = {"total_urls": total, "total_range": total_range,
              "affix_slugs": affix_slugs, "affix_range": affix_range}
    problems = {}
    lo, hi = (list(total_range) + [None, None])[:2] if total_range else (None, None)
    if lo is None and hi is None:
        detail["total_note"] = (
            "sitemap_total_range is null: report-only. Copy this total into "
            "registry/gates.json as a band (+/-25%) after the first real run.")
    else:
        if (lo is not None and total < lo) or (hi is not None and total > hi):
            problems["total_urls_outside_range"] = {"total": total,
                                                    "range": [lo, hi]}
    a_lo, a_hi = (list(affix_range) + [None, None])[:2] if affix_range else (None, None)
    if a_lo is not None and a_hi is not None and not a_lo <= affix_slugs <= a_hi:
        problems["affix_slugs_outside_range"] = {"unique_affix_slugs": affix_slugs,
                                                 "range": [a_lo, a_hi]}
    detail["violations"] = problems
    return not problems, detail


def curated_overrides_bind(problems: list, n_edges_admitted: int,
                           max_problems: int = 0):
    """G-OVERRIDE. A registry/form_to_lemma.json row that binds nothing is a
    CURATION BUG and must stop the build.

    Measured failure this replaces: 140 hand-verified mappings were added, every
    one of them was refused by the recovery door, `override_problems` went from 0
    to 140, `no_survivor` went from 0 to 140 -- and the gate report stayed
    all-green, because stage 21 shipped without a single gate. The registry that
    exists to correct the automatic layer could therefore be 100% dead and the
    release checklist could not tell.

    Baselined (registry/gates.json:override_problems_max) rather than hard-zero
    for one honest reason: a mapping whose lemma page was never crawled
    (override_lemma_not_crawled) is a fetch problem, not a bad mapping, and an
    owner may knowingly carry a few. The default is 0.

    The two counts are NOT the same unit, which is why they are named apart. One
    mapping can admit several (word, entry_id) EDGES -- `vent` binds both
    homograph articles of `vente` -- so 140 mappings admitted 177 edges. Calling
    that number `mappings_that_bound` in a file the release checklist reads
    invited "140 mappings, 177 of which bound", which is not a sentence.
    `mappings_that_bound_nothing` really is per mapping: it is one row per
    registry mapping that reached no article at all.
    """
    by_reason: dict[str, int] = {}
    for p in problems:
        r = str(p.get("reason"))
        by_reason[r] = by_reason.get(r, 0) + 1
    over = len(problems) > max_problems
    return not over, {"edges_admitted_by_the_curated_path": n_edges_admitted,
                      "mappings_that_bound_nothing": len(problems),
                      "max": max_problems, "by_reason": dict(sorted(by_reason.items())),
                      "sample": problems[:20]}


def suppression_registries(known_no_entry: dict, wordlist_invalid: dict,
                           invalid_rows_that_bound: list,
                           known_max: int | None = None,
                           invalid_max: int | None = None):
    """G-SUPPRESS. The two registries that DELETE words from unresolved.json are
    baselined, disjoint, and cannot hide a binding.

    Three assertions, each with a measured failure behind it:

      1. Both populations are inside a baseline. reports/unresolved.json is the
         whole enforcement surface for the owner's skip-and-record ruling -- G-ZERO
         was downgraded to a report on the condition that the report lists every
         unresolved word. A registry that silently subtracts from it needs the same
         baseline `case_only_members_max` and `all_demoted_families_max` have, and
         round 1 section 4.1 made that the explicit precondition for filling
         known_no_entry at all.
      2. The two files are DISJOINT. They make contradictory claims about a word --
         "DDO has no such word" versus "this row is not a word" -- and a row in
         both means one of the two is wrong. It is also the shape a copy-paste
         mistake takes: the OCR rows were carved OUT of the nohit population.
      3. No invalidated row bound anything. wordlist_invalid is consumed before
         every resolve layer, so a row appearing in it should have had no members
         to begin with; if it did, the registry is deleting a real card edge and
         the only visible symptom would be a card quietly disappearing.
    """
    both = sorted(set(known_no_entry) & set(wordlist_invalid))
    over_known = known_max is not None and len(known_no_entry) > known_max
    over_invalid = invalid_max is not None and len(wordlist_invalid) > invalid_max
    ok = (not both and not invalid_rows_that_bound
          and not over_known and not over_invalid)
    return ok, {"known_no_entry": len(known_no_entry),
                "known_no_entry_max": known_max,
                "wordlist_invalid": len(wordlist_invalid),
                "wordlist_invalid_max": invalid_max,
                "over_baseline": {"known_no_entry": bool(over_known),
                                  "wordlist_invalid": bool(over_invalid)},
                "in_both_registries": both[:20],
                "invalid_rows_that_bound_an_article": invalid_rows_that_bound[:20]}


def abbreviation_admissions(rows: list, baseline_words=None,
                            max_rows: int | None = None):
    """G-ADMIT. The layer-4 abbreviation admission channel is BASELINED.

    Layer 4 (owner policy B, 2026-08-26) is the only place in the pipeline where
    an article the classifier REJECTED becomes a card anyway. Round 4 baselined
    the two suppression registries it added in the same change (G-SUPPRESS) and
    left this channel unbaselined, so a future admission -- a new dotted DDO
    entry, a wordlist swap, a relaxed guard -- would mint a card with nothing to
    fire (reviewer A, round 4, MAJOR-2).

    Two assertions:

      1. Every admitted WORD is one the owner signed for. The baseline is the 19
         words themselves (registry/gates.json:abbreviation_accepted_words), not
         just a count, so a swap -- one admission lost, another gained -- cannot
         pass on an unchanged total. Growing the channel means editing that list
         in the same commit as the change that grew it, which is the whole point.
      2. The EDGE count is inside its max. One word can admit more than one
         dotted article -- `pr` reaches both `pr.`(fork., 1 sense) and
         `pr.`(praep., 6 senses), and layer 4 does not pick between them -- so a
         word already on the baseline list could still double its cards.

    SHRINKING is reported, not failed, and the asymmetry is deliberate: the day
    DDO gives one of these 19 words a real entry, layers 1-3 bind it first and
    this channel correctly stops firing for it. That is the mechanism working,
    not a regression, and it is exactly the case known_no_entry's layer order was
    designed for. The rows that disappeared are named in the detail so the
    release notes can pick them up.
    """
    words = sorted({str(r.get("word")) for r in rows})
    baseline = sorted({str(x) for x in (baseline_words or ())})
    beyond = [w for w in words if w not in set(baseline)]
    gone = [w for w in baseline if w not in set(words)]
    over = max_rows is not None and len(rows) > max_rows
    ok = not beyond and not over
    return ok, {"rows": len(rows), "max_rows": max_rows,
                "over_max_rows": bool(over),
                "words": words, "baseline_words": len(baseline),
                "admitted_beyond_the_baseline": beyond,
                "baseline_words_not_admitted": gone}


def case_only_members(rows: list, max_n: int | None = None):
    """G-CASE. The population of case-only family memberships is BASELINED.

    Bucket 4 (exact_ci) is deliberately a card-membership bucket, not an
    xref-only one: `I` (pron., 2 senses + 7 expressions) is not a wordlist word
    and would otherwise be deleted from the deck entirely. The anchor rule keeps
    it from heading the card, but ~55 (headword, wordlist word) pairs still
    differ only by case, and `var`/`VAR` is the case where the family's whole
    content belongs to a different spelling -- information for a human, not a
    tie-break. So the rows are written to review/case_only_members.json and the
    COUNT is baselined (registry/gates.json:case_only_members_max), exactly as
    round 1 did for all_demoted_families_max: the gate fires when the population
    GROWS, never on the existing, reviewed population.
    """
    over = max_n is not None and len(rows) > max_n
    return not over, {"rows": len(rows), "max": max_n,
                      "over_baseline": bool(over), "sample": rows[:20]}


_RESULTS_RE = re.compile(r"^(\d+) resultater$")


def _label_reconciles(label, n_articles) -> bool:
    """s12.verdict_of's rule, applied to what the ledger STORED."""
    if label == "" and n_articles == 1:
        return True
    m = _RESULTS_RE.fullmatch(label or "")
    return bool(m) and int(m.group(1)) == n_articles


def ledger_label_reconciliation(ledger: dict, parsed_counts: dict,
                                error_max_rate: float = 0.01):
    """G-LABEL. DDO answers 200 for everything, so a miss can only be read off
    the page: #results-label is the page's own checksum against the article
    count. The rule ran inside stage 12 and in pytest, and its verdict never
    reached gates_report.json -- so a release could not show that it held.

    Three measurements: every ok page's stored label still reconciles with its
    stored article_count, every ok page parsed to that same number, and the
    error-status population is inside its baseline (an `error` page is skipped
    by stage 20 by design, and it is the page a human is asked about).
    """
    rows = sorted(ledger.items())
    n = len(rows)
    errors, bad_label, parse_mismatch = [], [], []
    n_ok = 0
    for word, row in rows:
        status = row.get("status")
        if status == "error":
            errors.append({"word": word, "results_label": row.get("results_label"),
                           "article_count": row.get("article_count")})
            continue
        if status != "ok":
            continue
        n_ok += 1
        label, n_art = row.get("results_label"), row.get("article_count")
        if not _label_reconciles(label, n_art):
            bad_label.append({"word": word, "results_label": label,
                              "article_count": n_art})
        got = parsed_counts.get(word)
        if got is not None and got != n_art:
            parse_mismatch.append({"word": word, "ledger": n_art, "parsed": got})
    rate = (len(errors) / n) if n else 0.0
    ok = (not bad_label and not parse_mismatch and rate <= error_max_rate)
    return ok, {"words_in_ledger": n, "ok_pages": n_ok,
                "error_pages": len(errors), "error_rate": round(rate, 5),
                "error_max_rate": error_max_rate,
                "label_does_not_reconcile": bad_label[:20],
                "parsed_count_disagrees_with_ledger": parse_mismatch[:20],
                "error_sample": errors[:20]}


def guid_diff_reconciles(report: dict, n_notes: int, lang: str):
    """G-REL. tools/guid_diff.py computes kept / new / retired against the
    released .apkg; nothing used to compare those numbers to the deck actually
    being shipped, so the release note's churn figure was an estimate (the
    original spec's was off by up to +22%).

    The assertion is narrow on purpose: the summary row must describe THIS
    language and the same number of cards the exporter is about to write. The
    kept/retired split is a human-review number, not a machine-checkable one.
    """
    summary = (report or {}).get("summary") or {}
    counts = (report or {}).get("counts") or {}
    card_count = summary.get("card_count")
    problems = {}
    if card_count is None:
        problems["no_summary_row"] = ("reports/guid_diff.json predates the "
                                      "summary row; re-run tools/guid_diff.py")
    elif card_count != n_notes:
        problems["card_count_mismatch"] = {"guid_diff": card_count,
                                           "notes_being_written": n_notes}
    if summary.get("language") not in (None, lang):
        problems["language_mismatch"] = {"guid_diff": summary.get("language"),
                                         "export": lang}
    return not problems, {"summary": summary, "counts": counts,
                          "notes_being_written": n_notes,
                          "violations": problems}


# --------------------------------------------------------------------------
# G-SEP: the separator table, as an EXPORT gate
# --------------------------------------------------------------------------

# Guide 1.1, measured: definitions reproduce 388/388 under " " and 330/388
# under ""; expressions 689/689 under "" and 499/689 under " ".
SEP_MIN_CORRECT_RATE = 0.98


def _fixtures_root(explicit=None) -> Path | None:
    """$ANKIDKDECK_FIXTURES, then <work>/fixtures. Returns None when neither
    holds a manifest -- and the CALLER must treat that as a FAILURE, not a
    skip: a release host with no fixtures has not checked the separator table."""
    import os
    cands = []
    env = os.environ.get("ANKIDKDECK_FIXTURES")
    if env:
        cands.append(Path(env))
    if explicit:
        cands.append(Path(explicit))
    for c in cands:
        if (c / "manifest.json").exists():
            return c
    return None


def _pages_to_parse(manifest: dict, joinable: set) -> list:
    """Only the pages that carry a JOINABLE entry_id.

    A page with nothing on both sides contributes to neither reproduction rate,
    so parsing it changes no measurement -- and this gate runs on the export
    path, twice per call, over a fixture set that is meant to grow to the whole
    crawl corpus. The bound is what keeps it a gate rather than a tax.
    """
    out = []
    for page in manifest.get("pages", []):
        if set(page.get("entry_ids") or ()) & joinable:
            out.append(page)
    return out


def _parse_fixture_pages(root: Path, pages: list, registry) -> dict:
    # Imported lazily: gates.py is imported by every stage, and the parser
    # pulls in bs4.
    from bs4 import BeautifulSoup

    from .stages.s20_parse import parse_article, slice_articles
    entries: dict = {}
    for page in pages:
        p = root / page.get("file", "")
        if not p.exists():
            continue
        soup = BeautifulSoup(p.read_text(encoding="utf-8"), "html.parser")
        report: dict = {}
        for eid, scope, art in slice_articles(soup):
            entries[eid] = parse_article(eid, scope, art, registry, report)
    return entries


def _reproduction_rate(entries: dict, expected: dict, kind: str) -> tuple:
    hit = total = 0
    misses = []
    for eid, wanted in expected.items():
        e = entries.get(eid)
        if e is None:
            continue
        if kind == "definitions":
            have = {s["definition"] for s in e["senses"]}
            have |= {s["definition"] for x in e["expressions"]
                     for s in x.get("senses", [])}
        else:
            have = {x["expression"] for x in e["expressions"]}
            have |= {v for x in e["expressions"] for v in x.get("variants", [])}
        for text in wanted:
            total += 1
            if text in have:
                hit += 1
            elif len(misses) < 10:
                misses.append({"entry_id": eid, "text": text})
    return (hit / total if total else None), total, misses


def separator_golden(registry, fixtures_dir=None):
    """G-SEP. Two-sided: the shipped extract.SEP table reproduces the 2025
    Danish strings the 22,734 x 4 translation cells are keyed by, AND a
    deliberately wrong table provably does not.

    A one-character change here silently invalidates the whole translation
    asset -- that is how 2,007 bare English cards shipped -- and the observable
    symptom is a collapsed bind rate plus a very large translate bill, not a
    blocked build. So this runs on the EXPORT path, not only under pytest.

    Fixtures absent is a FAILURE, never a silent pass: a release host that
    cannot run this check has not run it.
    """
    from . import extract

    root = _fixtures_root(fixtures_dir)
    if root is None:
        return False, {"checked": False, "reason": "fixtures unavailable",
                       "hint": "build them with tools/build_fixtures.py "
                               "--work <workspace> and point "
                               "ANKIDKDECK_FIXTURES at the result, or place "
                               "them in <work>/fixtures"}
    manifest = read_json(root / "manifest.json")
    joinable = set(manifest.get("joinable_entry_ids") or [])
    if not joinable:
        return False, {"checked": False,
                       "reason": "no entry_id joins the 2025 and 2026 sides in "
                                 "this fixture set",
                       "pages": len(manifest.get("pages") or [])}
    exp = manifest.get("expected") or {}
    want_defs = {k: v for k, v in
                 read_json(root / exp.get("definitions", "x"), default={}).items()
                 if k in joinable}
    want_exprs = {k: v for k, v in
                  read_json(root / exp.get("expressions", "x"), default={}).items()
                  if k in joinable}
    pages = _pages_to_parse(manifest, joinable)
    entries = _parse_fixture_pages(root, pages, registry)
    r_def, n_def, miss_def = _reproduction_rate(entries, want_defs, "definitions")
    r_expr, n_expr, miss_expr = _reproduction_rate(entries, want_exprs, "expressions")

    saved = dict(extract.SEP)
    try:
        extract.SEP["definition"] = ""
        extract.SEP["expr_definition"] = ""
        extract.SEP["expression"] = " "
        wrong = _parse_fixture_pages(root, pages, registry)
        w_def, _, _ = _reproduction_rate(wrong, want_defs, "definitions")
        w_expr, _, _ = _reproduction_rate(wrong, want_exprs, "expressions")
    finally:
        extract.SEP.clear()
        extract.SEP.update(saved)

    def _under(rate) -> bool:
        return rate is not None and rate < SEP_MIN_CORRECT_RATE

    problems = {}
    if _under(r_def):
        problems["definitions_do_not_reproduce"] = r_def
    if _under(r_expr):
        problems["expressions_do_not_reproduce"] = r_expr
    if n_def and (w_def is None or not w_def < r_def):
        problems["wrong_definition_separator_still_reproduces"] = w_def
    if n_expr and (w_expr is None or not w_expr < r_expr):
        problems["wrong_expression_separator_still_reproduces"] = w_expr
    if not n_def and not n_expr:
        problems["nothing_to_compare"] = True
    return not problems, {
        "checked": True, "fixtures": str(root),
        "joinable_entry_ids": len(joinable),
        "pages_parsed": len(pages),
        "pages_in_fixture_set": len(manifest.get("pages") or []),
        "definitions": {"texts": n_def, "rate": r_def, "rate_wrong_sep": w_def,
                        "misses": miss_def},
        "expressions": {"texts": n_expr, "rate": r_expr, "rate_wrong_sep": w_expr,
                        "misses": miss_expr},
        "min_correct_rate": SEP_MIN_CORRECT_RATE,
        "violations": problems}
