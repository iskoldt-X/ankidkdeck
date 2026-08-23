"""The gate framework: named, reported, blocking checks.

A gate is an id from the final guide's gate table (section 4.12), one human
sentence, and a zero-argument function returning (ok, detail). run_gates()
executes every gate, records every result in reports/gates_report.json, and
only then raises FatalError listing the failures. Two properties make that
report worth reading:

  1. ALL gates run before anything raises, so one build shows every failure
     instead of only the first.
  2. Results accumulate in one file, merged by gate id, so a later stage
     appends to the same report rather than overwriting it. The export-time
     gate list (segment 3) plugs in here with no changes to this module.

A gate that cannot fail is not a gate: every helper below returns the measured
detail alongside the verdict, so a passing gate still leaves evidence.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .util import FatalError, NFC, read_json, write_json

# Gate ids, spelled exactly as in the final guide's table (section 4.12). The
# ids segment 2 owns are listed here; segment 3 adds the export-time ones.
G_RANK = "G-RANK"
G_SEED = "G-SEED"
G_GUID = "G-GUID"
G_ANCHOR = "G-ANCHOR"
G_AFFIX = "G-AFFIX"
G_BIND = "G-BIND"
G_TIE = "G-TIE"
G_SITEMAP = "G-SITEMAP"
G_ASSIGN = "G-ASSIGN"  # "every word in at most one family" (guide 4.9 step 4)

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


@dataclass
class Gate:
    """id: the guide's gate id. description: why a human should care.
    fn: () -> (ok, detail). detail is JSON-serialisable and always recorded."""

    id: str
    description: str
    fn: Callable[[], tuple[bool, Any]]
    stage: str = ""
    extra: dict = field(default_factory=dict)


def run_gates(gates: Iterable[Gate], cfg, stage: str = "") -> list[dict]:
    """Run every gate, write the merged report, then raise on any failure."""
    results = []
    for g in gates:
        ok, detail = g.fn()
        results.append({"id": g.id, "description": g.description,
                        "stage": g.stage or stage, "ok": bool(ok),
                        "detail": detail})
    _write_report(cfg, results)
    failed = [r for r in results if not r["ok"]]
    if failed:
        lines = [f"  {r['id']}: {r['description']} -> {r['detail']}" for r in failed]
        raise FatalError(
            "%d gate(s) failed; no output is valid until they pass:\n%s"
            % (len(failed), "\n".join(lines))
        )
    return results


def _write_report(cfg, results: list[dict]) -> None:
    path = cfg.report_dir / "gates_report.json"
    prev = read_json(path, default={"results": []})
    merged: dict[str, dict] = {}
    for row in list(prev.get("results", [])) + results:
        merged[row["id"]] = row  # later stage wins; first-seen order kept below
    order = []
    for row in list(prev.get("results", [])) + results:
        if row["id"] not in order:
            order.append(row["id"])
    out = {"results": [merged[i] for i in order]}
    out["failed"] = sorted(r["id"] for r in out["results"] if not r["ok"])
    out["n_gates"] = len(out["results"])
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


def anchors_not_demoted_or_affix(families: dict, entries: dict, demoted_pos: set):
    """G-ANCHOR + G-AFFIX. A family must never be headed by an affix page
    (`-kvinde`, `-ske`) and must never be headed by a demoted article while a
    real one is available in the same family.

    Deviation from the guide, deliberate and reported: a family every one of
    whose articles is demoted (a `cm`/`Cm` abbreviation, a chemical symbol
    reached through its own flex table) has no non-demoted anchor to pick, and
    stopping the build for it would contradict the owner's 2026-08-24 ruling
    that word-level resolution problems are recorded, not fatal. Those are
    listed as all_demoted, not failed.
    """
    affix_anchored, demoted_anchored, all_demoted = [], [], []
    for fid, fam in families.items():
        a = entries[fam["anchor_entry_id"]]
        lemma = a.get("lemma", "")
        if a.get("pos_key") in AFFIX_POS_KEYS or lemma.startswith("-") or lemma.endswith("-"):
            affix_anchored.append({"family_id": fid, "lemma": lemma,
                                   "pos_key": a.get("pos_key")})
        if a.get("pos_key") in demoted_pos:
            alt = [e for e in fam["entry_ids"]
                   if entries[e].get("pos_key") not in demoted_pos
                   and not entries[e].get("empty")]
            row = {"family_id": fid, "lemma": lemma, "pos_key": a.get("pos_key"),
                   "non_demoted_alternatives": alt}
            (demoted_anchored if alt else all_demoted).append(row)
    ok = not affix_anchored and not demoted_anchored
    return ok, {"affix_anchored": affix_anchored[:20],
                "demoted_anchored_with_alternative": demoted_anchored[:20],
                "all_demoted_families": len(all_demoted),
                "all_demoted_sample": all_demoted[:10]}


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


def tie_break_resolved(per_lang: dict):
    """G-TIE. Every multi-candidate migration cell was resolved by the written
    tie-break and every loser was written out. 40-50% of the cells in
    multi-file buckets carry 2-6 candidate translations, so without this the
    build picks one by dict iteration order and is not reproducible."""
    bad = {lang: s for lang, s in per_lang.items()
           if s.get("unresolved_conflicts", 0) or not s.get("discard_file_written")}
    return not bad, {"per_language": per_lang, "violations": bad}


def sitemap_shortfall(rows: list, n_families: int, max_rate: float):
    """G-SITEMAP. Per-family homograph shortfall against the sitemap
    inventory: how many DDO articles for this lemma we never saw. Reported by
    stage 30, enforced at export time -- it recovers the articles the 2025
    crawl missed (vinge, uanset, vove, zone)."""
    rate = (len(rows) / n_families) if n_families else 0.0
    return rate <= max_rate, {"families_short": len(rows), "n_families": n_families,
                              "rate": round(rate, 5), "max_rate": max_rate,
                              "sample": rows[:20]}
