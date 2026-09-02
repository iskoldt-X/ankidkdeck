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

import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import resources
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
# The offline CONTENT gate over the translation cells (patch plan 1.9 / N-05).
# The one gate here that adjudicates what the model WROTE rather than what the
# pipeline counted, and the only reason it can be trusted is that it was
# calibrated against 85,259 shipped cells before it was allowed to have an
# opinion:
#
#   naive rule ("any character outside the target script")  17 FINDINGS on 15
#     cells, 4 real defects, 13 false positives (76.5% per finding, 73.3% per
#     cell -- two Greek-letter cells hit twice, once in the lemma and once in
#     the gloss) -- and the 13 are all cells of the three entries whose SUBJECT
#     is a Greek letter, which have to carry one.
#   this gate                                               0 false positives,
#     because the discriminator is mechanical: a Greek letter INSIDE a run of
#     Latin letters is contamination (one cell, a Greek beta written for a
#     German sharp s), a standalone one is a mention.
#
# The class it exists for was invisible to every audit: 325 Chinese definition
# cells (2.41%) and 22 expression cells carry Traditional characters mixed into
# Simplified text, 28 entries have every translated sense contaminated, and the
# failure is PER REQUEST -- one batch whose system prompt never said
# "Simplified" and the whole batch drifts. The 2025 prompts never said it.
G_SCRIPT = "G-SCRIPT"
# The MONEY gates. Before these, ALL_GATE_IDS had 26 ids and not one of them was
# about money, tokens, the cache, thinking or the prompt version -- i.e. the
# whole gate framework adjudicated the deck and nothing adjudicated the only
# irreversible act in the pipeline. Every one of the five has a specific
# measured failure behind it:
#
#   G-BILL    the old bill counted the Danish payload and nothing else, which
#             under-stated a clean redo by ~17x. A quote nobody checks against
#             the invoice is a wish.
#   G-THINK   thinkingLevel defaults to MEDIUM, measured at mean 578.7 (p95
#             1,042) thought tokens per request against an n=20 batch's entire
#             1,115-token cap, and the derived output cap has no thinking term.
#             The wave's thinking DISTRIBUTION per kind -- mean tokens/request
#             and the share of rows that thought at all -- against the measured
#             one is how "we accidentally ran at MEDIUM" becomes visible. It is
#             deliberately not a per-row ceiling: a real 3,644-request definition
#             wave has a 1.2% non-zero tail reaching 797 tokens, all healthy, so
#             a gate that failed on any non-zero row could not be passed by
#             correct output. See thinking_is_at_the_measured_level.
#   G-PROMPT  the constants, the cache and the bill are all properties of ONE
#             prompt. Rows written with a second prompt are unbilled work.
#   G-CACHE   the discount is the entire economic case for the explicit cache,
#             and an expired or misplaced cache is a full-price wave.
#   G-BUDGET  Google's project-level cap does NOT stop an already-submitted
#             batch wave (measured: the same job submitted twice, both accepted,
#             both billed), so the $10/month ceiling can only be enforced here.
#   G-SCOPE-FROZEN  the refreeze reselects 22 guid_seeds and merges three alias
#             pairs. Paying to translate a scope that is about to change is
#             paying twice.
G_BILL = "G-BILL"
G_THINK = "G-THINK"
G_PROMPT = "G-PROMPT"
G_CACHE = "G-CACHE"
G_BUDGET = "G-BUDGET"
G_SCOPE_FROZEN = "G-SCOPE-FROZEN"

# The declared set, so a report can say what it did NOT check. "11 gates PASS"
# was read as a release verdict when it meant "11 rows are on file, 8 of them
# written by this run, and 12 of the 24 declared gates have never executed on
# this workspace at all" -- G-SITEMAP among them, which is the one that
# adjudicates merge_report.sitemap_shortfall_families. A gate that never ran is
# not a gate that passed.
ALL_GATE_IDS = (
    G_ADMIT, G_AFFIX, G_ANCHOR, G_ASSIGN, G_BILL, G_BIND, G_BUDGET, G_CACHE,
    G_CASE, G_COV, G_DET, G_EMPTY_C, G_GUID, G_LABEL, G_MEDIA, G_NOTE, G_ORDER,
    G_ORPH, G_OVERRIDE, G_PROMPT, G_RANK, G_RATE, G_REGKEY, G_REL, G_SCOPE_FROZEN,
    G_SCRIPT, G_SEED, G_SEP, G_SITEMAP, G_SITEMAP_INV, G_SUPPRESS, G_THINK,
    G_TIE,
)

# The money gates, as a set, so a caller can ask "has anything adjudicated this
# spend?" without hard-coding six strings.
MONEY_GATE_IDS = (G_BILL, G_BUDGET, G_CACHE, G_PROMPT, G_SCOPE_FROZEN, G_THINK)

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


def failure_message(results: list[dict]) -> str:
    """The FatalError text for a set of gate results, or "" when they all pass.

    Factored out of run_gates so a caller that has to record the verdict before
    the failure continues -- write the run's report first, then raise -- says
    the same thing run_gates would have said.
    """
    failed = [r for r in results if not r["ok"]]
    if not failed:
        return ""
    lines = ["  %s: %s -> %s" % (row_label(r), r["description"], r["detail"])
             for r in failed]
    return ("%d gate(s) failed; no output is valid until they pass:\n%s"
            % (len(failed), "\n".join(lines)))


def run_gates(gates: Iterable[Gate], cfg, stage: str = "",
              raise_on_failure: bool = True) -> list[dict]:
    """Run every gate, write the merged report, then raise on any failure.

    `raise_on_failure=False` still evaluates and still writes the report; it
    returns the results and leaves the raising to the caller. That exists for
    one shape: a caller whose OWN report has to reach disk before the failure
    propagates, because the alternative is a paid wave whose report was never
    written while the previous run's file stays on disk describing a different
    run.
    """
    results = []
    for g in gates:
        ok, detail = g.fn()
        results.append({"id": g.id, "description": g.description,
                        "stage": g.stage or stage, "extra": dict(g.extra),
                        "ok": bool(ok), "detail": detail})
    _write_report(cfg, results)
    message = failure_message(results)
    if message and raise_on_failure:
        raise FatalError(message)
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


# --------------------------------------------------------------------------
# G-SCRIPT: the offline content gate over translations/<lang>/{definitions,
# expressions}.json. Patch plan 1.9 + N-05.
# --------------------------------------------------------------------------

# The script blocks a target language MAY forbid. Which of them it DOES forbid
# is derived per language in script_profile(), because this table is not a
# universal truth: it names Hiragana, Katakana and Hangul, so hard-coding it
# made Japanese or Korean a target language in which EVERY cell is a BLOCK-tier
# finding and run_gates raises -- against D-10's promise that one language word
# in the config runs the whole pipeline with no hand-prepared files.
#
# Greek is NOT here: three DDO entries ARE Greek letters, so their cells have to
# carry one, and Greek gets its own two-way test below.
_SCRIPT_BLOCKS = (
    ("cyrillic", 0x0400, 0x052F),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("devanagari", 0x0900, 0x097F),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("hangul", 0xAC00, 0xD7AF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
)

# How a pack's own prose names each block. `allowed_scripts`,
# `lemma_allowed_set` and `gloss_allowed_set` are a lexicographer's words, not a
# controlled vocabulary, so the match is on the names one would actually write.
_SCRIPT_NAMED_BY = {
    "cyrillic": ("cyrillic",),
    "hebrew": ("hebrew",),
    "arabic": ("arabic",),
    "devanagari": ("devanagari",),
    "thai": ("thai",),
    "hangul": ("hangul", "korean"),
    "hiragana": ("hiragana", "kana", "japanese"),
    "katakana": ("katakana", "kana", "japanese"),
}
# "Arabic DIGITS" appears in all four shipped packs and is not the Arabic
# script. One of the four defects three audits found by hand is an Arabic gloss
# in a Chinese cell, so reading that phrase as permission would switch off the
# check that catches it.
_NOT_A_SCRIPT_PHRASE = ("arabic digit",)

# The script a TARGET LANGUAGE writes in, for a language no pack describes yet.
# Cyrillic is the first entry of _SCRIPT_BLOCKS, so without this a pack-less
# Russian run flags every cell as a BLOCK-tier forbidden_script at ingest, with
# the money already spent -- the same defect `han_allowed` had for a Han-script
# target. One language word, nothing inferred: the other Cyrillic-script
# languages are out of scope until someone adds them here or ships them a pack,
# because this table is a claim about a target this project is enabling, not a
# script-to-language mapping.
_SCRIPTS_BY_LANGUAGE = {"russian": ("cyrillic",)}

_GREEK = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
_HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))

# Pinyin tone marks. A Han lemma carrying one of these is a romanisation -- 20
# such cells shipped in the 2025 Chinese expression corpus while pos_prompt had
# said "DO NOT include any pinyin" since 2025 and the definition and expression
# prompts had never mentioned it. The tone mark is the WHOLE discriminator now:
# the old rule also accepted a bare parenthesis as evidence of romanisation,
# which is how nine Latin-in-lemma cells of the 2026-08-27 Chinese wave were
# reported as `pinyin_in_lemma` while carrying no pinyin at all. Measured on the
# two populations: 20 of 20 archive expression cells have a tone mark, 0 of 9
# new definition cells do. See _latin_in_lemma_class.
_TONE_MARKS = frozenset(
    "\u0101\u00e1\u01ce\u00e0\u0113\u00e9\u011b\u00e8\u012b\u00ed"
    "\u01d0\u00ec\u014d\u00f3\u01d2\u00f2\u016b\u00fa\u01d4\u00f9"
    "\u01d6\u01d8\u01da\u01dc\u00fc")

# The Latin tokens that are a SYMBOL rather than a word, so a Han lemma built
# around one is naming its own subject instead of leaking foreign text. Solfege
# syllables (both the la/si and the la/ti naming, plus the historical `ut`) and
# `TA`, the standard Chinese gender-neutral third-person pronoun written in
# Latin capitals. Single Latin letters and short all-capital runs are admitted
# by shape in _is_symbol_token rather than listed here.
_SYMBOL_TOKENS = frozenset(
    ("do", "re", "mi", "fa", "sol", "so", "la", "si", "ti", "ut", "ta"))

# Longest all-capital Latin run still read as a symbol (a letter name, a
# playing-card rank, a note name, an acronym) rather than a word. Measured
# against the real defects it has to keep out: `lille` (5) and `undskylde` (9).
_SYMBOL_CAPS_MAX = 3

# Paired parentheses, ASCII and fullwidth. Both appear in the corpus: the 2025
# expression lemmas parenthesise pinyin with ASCII `(` 17 times out of 20, the
# 2026 definition lemmas use the fullwidth pair.
_PAREN_PAIRS = (("(", ")"), ("\uff08", "\uff09"))

# BLOCK: zero tolerance on a cell this pipeline wrote. BASELINE: the same class
# on a cell inherited from 2025, where the count is pinned in
# registry/gates.json and growing it means editing that file in the same commit
# (the G-SUPPRESS / G-ADMIT discipline). REVIEW: reported, never failed.
_BLOCK_CLASSES = ("empty_field", "forbidden_script", "greek_in_lemma",
                  "greek_latin_internal", "han_outside_the_target",
                  "traditional_han", "pinyin_in_lemma",
                  "foreign_text_in_lemma")
_REVIEW_CLASSES = ("greek_mention", "greek_subject_lemma",
                   "latin_in_han_lemma", "latin_subject_lemma")
SCRIPT_CLASSES = _BLOCK_CLASSES + _REVIEW_CLASSES

BLOCK, BASELINE, REVIEW = "BLOCK", "BASELINE", "REVIEW"

# What marks a cell as inherited rather than written by this pipeline. A cell
# whose provenance starts with any of these is 2025 material the baseline
# absorbs; everything else -- every `gemini:` row a clean redo writes -- is held
# to BLOCK. That is the whole mechanism: after the clean redo the baselines fall
# to zero on their own and every class becomes zero-tolerance, with no second
# edit and no human remembering to tighten anything.
LEGACY_PROVENANCE_PREFIXES = ("migrated:",)


def _in(ch: str, ranges) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ranges)


def _script_of(ch: str, blocks=_SCRIPT_BLOCKS):
    o = ord(ch)
    for name, a, b in blocks:
        if a <= o <= b:
            return name
    return None


def _is_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    return "LATIN" in unicodedata.name(ch, "")


def _greek_mentions(text: str) -> set:
    """The Greek characters `text` MENTIONS: present, and not hugged by Latin
    letters. The same discriminator the gloss check uses, factored out because
    the lemma's verdict now reads it."""
    out = set()
    for i, ch in enumerate(text):
        if not _in(ch, _GREEK):
            continue
        if ((i and _is_latin_letter(text[i - 1]))
                or (i + 1 < len(text) and _is_latin_letter(text[i + 1]))):
            continue
        out.add(ch)
    return out


def _names_a_letter(lemma: str, ch: str) -> bool:
    """Is `ch` part of a lemma that NAMES the letter, rather than the whole of
    it? A lemma that is only the Greek character is script leakage, not a
    lexicographic phrase, and stays a defect."""
    rest = [c for c in lemma.strip() if not _in(c, _GREEK) and not c.isspace()]
    return bool(rest)


TRADITIONAL_VARIANTS_FILE = "traditional_variants.json"
_TRADITIONAL_VARIANTS = None


def traditional_variants() -> dict:
    """{Traditional character: its distinct Simplified form(s)}.

    Package data (registry/traditional_variants.json), read once and cached. Not
    a Registry file and deliberately not overlayable from work/registry/: the
    overlayable numbers are the BASELINE COUNTS in gates.json, and what
    "Traditional" MEANS is not a per-run policy knob -- a run that could
    redefine it could silently unblock a real leak.
    """
    global _TRADITIONAL_VARIANTS
    if _TRADITIONAL_VARIANTS is None:
        ref = resources.files("ankidkdeck").joinpath(
            "registry", TRADITIONAL_VARIANTS_FILE)
        with ref.open("r", encoding="utf-8") as fh:
            _TRADITIONAL_VARIANTS = dict(json.load(fh)["variants"])
    return _TRADITIONAL_VARIANTS


def _is_traditional(ch: str) -> bool:
    """A Han character that HAS a distinct Simplified form.

    That is the only honest mechanical definition of "Traditional leaked into a
    Simplified cell": the character has a Simplified form that is a DIFFERENT
    character, so Simplified text would have used that other character.

    This replaces a GB2312-encodability test, and the replacement was measured
    on the 2026-08-27 Chinese wave, where the old test raised 9 findings and 8
    of them were rare SIMPLIFIED characters -- U+9CC0 and U+8137 are not in any
    Traditional charset at all, U+7947 and U+77AD are retained in Simplified,
    U+808F is the same character in both scripts. The old test's docstring
    called that over-reporting "the right direction for a gate", which was the
    wrong call twice over: a gate whose findings are 89% false gets ignored, and
    the same test also UNDER-reported -- 54 entries of the table it lacked,
    U+5F8C among them, are GB2312-encodable and were invisible to it. U+5F8C is
    one of the most common Traditional characters there is.

    See registry/traditional_variants.json for the source, the derivation rule
    and an honest statement of coverage.
    """
    return ch in traditional_variants()


def _latin_runs(text: str) -> list:
    """Every maximal run of Latin letters in `text`, as (start, end, run)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if not _is_latin_letter(text[i]):
            i += 1
            continue
        j = i
        while j < n and _is_latin_letter(text[j]):
            j += 1
        out.append((i, j, text[i:j]))
        i = j
    return out


def _is_symbol_token(run: str) -> bool:
    """Is this Latin run a SYMBOL rather than a word?

    Three shapes, and each was measured against the cells that have to pass and
    the cells that have to fail:

      * one Latin letter, any case -- the letter itself, a playing-card rank, a
        note name, a grade, a vitamin;
      * an all-capital run of at most _SYMBOL_CAPS_MAX letters -- `TA`, an
        acronym;
      * a solfege syllable, case-insensitive (_SYMBOL_TOKENS).

    `lille` and `undskylde`, the two real foreign-text leaks of the 2026-08-27
    wave, are none of those. This is a SHAPE test and it cannot be more than
    that without a Danish dictionary the gate is not allowed to read: a Danish
    word that is one letter long, or three capitals, would be admitted. The
    consequence of admitting one is a REVIEW row, never a silent pass.
    """
    if len(run) == 1:
        return True
    if run.isupper() and len(run) <= _SYMBOL_CAPS_MAX:
        return True
    return run.lower() in _SYMBOL_TOKENS


def _parenthesised_spans(text: str) -> list:
    """The (start, end) half-open spans INSIDE parentheses, either width.

    Unclosed opener: everything after it counts as inside. That is the safe
    direction -- it can only make the subject exemption harder to earn.
    """
    spans = []
    for op, cl in _PAREN_PAIRS:
        i = 0
        while True:
            a = text.find(op, i)
            if a < 0:
                break
            b = text.find(cl, a + 1)
            end = len(text) if b < 0 else b
            spans.append((a + 1, end))
            i = end + 1
    return spans


def _is_latin_subject_lemma(lemma: str) -> bool:
    """Is the Latin in this Han lemma the entry's own SUBJECT?

    The sibling of greek_subject_lemma, and built the same way: an entry whose
    subject IS a character has to be able to name it, or the pipeline cannot
    translate that entry at all without failing the gate, at ingest, after the
    money. Seven such cells were adjudicated innocent on the 2026-08-27 Chinese
    wave -- the neutral pronoun TA, the playing cards Q and J, the solfege
    syllable la.

    TWO conditions, both required:

      1. every Latin run in the lemma is a symbol token (_is_symbol_token), and
      2. no Latin run sits inside a parenthesis.

    (2) is what keeps the exemption off the shape the wave's real defects have:
    a Chinese translation with the Danish headword parenthesised after it,
    `U+5C0F` + `(lille` + `'s plural)`. The lemma is then a translation that
    mentions a foreign word, not an entry about a symbol -- and note that
    "the gloss mentions the same token" is NOT one of the conditions. It was the
    obvious candidate and the real data inverts it: none of the seven innocent
    glosses repeats its own token (the TA glosses quote Danish `de`, the Q gloss
    names J and K), while BOTH real defects do repeat theirs, because the
    parenthesised Danish word is the family headword the gloss has to explain.
    Requiring a gloss mention would have blocked all seven and excused both.
    """
    runs = _latin_runs(lemma)
    if not runs:
        return False
    if not all(_is_symbol_token(run) for _, _, run in runs):
        return False
    spans = _parenthesised_spans(lemma)
    return not any(a < end and start < b
                   for start, end, _ in runs for a, b in spans)


def _latin_in_lemma_class(lemma: str) -> str:
    """Which class a Latin-carrying Han lemma belongs to. Four truthful classes
    where there used to be two, split on discriminators the corpus supplies:

      * `pinyin_in_lemma` -- a pinyin TONE MARK is present. Real romanisation,
        the 2025 expression defect, 20 cells, all 20 tone-marked.
      * `latin_subject_lemma` -- the Latin is the entry's subject. REVIEW.
      * `foreign_text_in_lemma` -- a parenthesised Latin run with no tone mark
        and no subject claim. The 2026 definition defect: the Danish source word
        leaking into a shipped Chinese lemma, which is a DDO-text-in-the-deck
        problem and not only a script-purity one. BLOCK, baseline 0.
      * `latin_in_han_lemma` -- a bare Latin run, no parenthesis, not a symbol.
        Exactly the cross-reference population the note in gates.json describes,
        55 archive definition cells of the form "see <Danish word>". REVIEW, as
        before.

    The class NAME is load-bearing: whoever triages `pinyin_in_lemma` goes
    looking for romanisation, and for nine cells of the 2026-08-27 wave there
    was none to find.
    """
    if any(ch in _TONE_MARKS for ch in lemma):
        return "pinyin_in_lemma"
    if _is_latin_subject_lemma(lemma):
        return "latin_subject_lemma"
    if any(op in lemma for op, _ in _PAREN_PAIRS):
        return "foreign_text_in_lemma"
    return "latin_in_han_lemma"


def script_profile(pack: dict | None, lang: str | None = None) -> dict:
    """Which language-specific checks apply, read off the PACK.

    `allowed_scripts` and `lemma_allowed_set` are the same two pack fields the
    prompt's script contract interpolates. Reading them here rather than keeping
    a second table is the point: the 2025 pipeline shipped English lemmas on
    Chinese cards because the generator prompt and the reviewer prompt were two
    prose paragraphs that disagreed, and two prose paragraphs is exactly what a
    gate with its own table would recreate.

    A language with NO pack keeps only the universal checks. That is correct
    rather than lenient: a brand-new target language must run with zero
    hand-prepared files (D-10), and "no pack" means nobody has yet said what
    that language's letters are -- so the gate says nothing about them either,
    while still refusing Cyrillic in a Spanish cell.

    "Says nothing" is now literal on both halves that used to make a POSITIVE
    claim out of the pack's ABSENCE:

      * `forbidden_scripts` is DERIVED. A pack that names Hiragana or Hangul as
        one of its scripts removes that block from its own forbidden set, so a
        Japanese or Korean target is a pack away rather than a rewrite away.
      * `han_outside_the_target` is only asked when there IS a pack. Without
        one, `han_allowed` was False and every Han character in the cell became
        a BLOCK-tier finding -- so a Han-script target language with no pack
        failed every cell of its first wave, at ingest, after the money.

    `lang` is the SECOND source of permission, after the pack, and it exists
    because the pack's absence cannot excuse the target's own script: see
    _SCRIPTS_BY_LANGUAGE. What it permits is reported under its own key, because
    `scripts_the_pack_names` stays a statement about the pack.
    """
    pack = pack or {}
    allowed = str(pack.get("allowed_scripts") or "").lower()
    lemma_set = str(pack.get("lemma_allowed_set") or "").lower()
    gloss_set = str(pack.get("gloss_allowed_set") or "").lower()
    han_allowed = "han" in allowed
    haystack = " ".join((allowed, lemma_set, gloss_set))
    for phrase in _NOT_A_SCRIPT_PHRASE:
        haystack = haystack.replace(phrase, " ")
    named = {name for name, words in _SCRIPT_NAMED_BY.items()
             if any(w in haystack for w in words)}
    # .strip() because a miss here is silent and total: a language name that
    # arrives with a stray space re-blocks every cell of the target's own
    # script, at ingest, which is the whole failure this lookup exists to stop.
    by_lang = _SCRIPTS_BY_LANGUAGE.get(str(lang or "").strip().lower(), ())
    forbidden = tuple((name, a, b) for name, a, b in _SCRIPT_BLOCKS
                      if name not in named and name not in by_lang)
    return {
        "has_pack": bool(pack),
        "han_allowed": han_allowed,
        # A Han-based lemma charset does not admit Latin letters unless the pack
        # says so. For a Latin-script language the question does not arise.
        "latin_in_lemma_allowed": (not han_allowed) or ("latin" in lemma_set),
        "simplified_required": han_allowed,
        "forbidden_scripts": forbidden,
        "scripts_the_pack_names": tuple(sorted(named)),
        "scripts_the_language_uses": tuple(sorted(by_lang)),
    }


def script_findings(cells: dict, *, lang: str, kind: str, pack=None,
                    legacy_prefixes=LEGACY_PROVENANCE_PREFIXES) -> list:
    """One finding per (cell, class). Pure function of the cells and the pack.

    Reads only `lemma`, `gloss` and `provenance`. It never reads the Danish
    source text, so it cannot put DDO material into a report artifact.
    """
    profile = script_profile(pack, lang)
    blocks = profile["forbidden_scripts"]
    out = []
    for key in sorted(cells):
        row = cells[key] or {}
        lemma = str(row.get("lemma") or "")
        gloss = str(row.get("gloss") or "")
        prov = str(row.get("provenance") or "")
        legacy = any(prov.startswith(pre) for pre in legacy_prefixes)
        hits = {}

        def hit(cls, field, ch=""):
            slot = hits.setdefault(cls, {"fields": set(), "chars": set()})
            slot["fields"].add(field)
            if ch:
                slot["chars"].add(ch)

        # The characters this cell's GLOSS explains: a Greek letter that stands
        # clear of Latin letters in the gloss is the gloss talking ABOUT the
        # letter, which is the mechanical signature of an entry whose own
        # subject is that letter. Computed before the field loop because the
        # lemma's verdict depends on it -- see greek_subject_lemma below.
        gloss_mentions = _greek_mentions(gloss)
        if not lemma.strip() or not gloss.strip():
            hit("empty_field", "lemma" if not lemma.strip() else "gloss")
        for field, text in (("lemma", lemma), ("gloss", gloss)):
            for i, ch in enumerate(text):
                script = _script_of(ch, blocks)
                if script:
                    hit("forbidden_script", field, ch)
                    continue
                if _in(ch, _GREEK):
                    latin_hugged = ((i and _is_latin_letter(text[i - 1]))
                                    or (i + 1 < len(text)
                                        and _is_latin_letter(text[i + 1])))
                    if field == "gloss":
                        hit("greek_latin_internal" if latin_hugged
                            else "greek_mention", field, ch)
                    elif latin_hugged:
                        # A Greek letter inside a run of Latin letters is
                        # contamination wherever it sits: a beta typed for a
                        # German sharp s.
                        hit("greek_latin_internal", field, ch)
                    elif ch in gloss_mentions and _names_a_letter(lemma, ch):
                        # THE two-cell case the clean redo would otherwise fail
                        # on. The Chinese lemmas for the DDO entries `my` and
                        # `ny` are phrases meaning "Greek letter M" / "Greek
                        # letter N", and the gloss of each explains that same
                        # letter -- so the entry's subject IS the character, and
                        # the natural target lemma names it. The pipeline had no
                        # way to translate those two entries without failing the
                        # gate, at ingest, after the money.
                        hit("greek_subject_lemma", field, ch)
                    else:
                        hit("greek_in_lemma", field, ch)
                    continue
                if _in(ch, _HAN):
                    if not profile["han_allowed"]:
                        if profile["has_pack"]:
                            hit("han_outside_the_target", field, ch)
                    elif profile["simplified_required"] and _is_traditional(ch):
                        hit("traditional_han", field, ch)
        if profile["han_allowed"] and not profile["latin_in_lemma_allowed"]:
            if any(_is_latin_letter(ch) for ch in lemma):
                hit(_latin_in_lemma_class(lemma), "lemma")
        for cls, slot in hits.items():
            if cls in _REVIEW_CLASSES:
                tier = REVIEW
            else:
                tier = BASELINE if legacy else BLOCK
            out.append({"key": key, "lang": lang, "kind": kind, "class": cls,
                        "tier": tier, "legacy": legacy,
                        "fields": sorted(slot["fields"]),
                        "chars": "".join(sorted(slot["chars"]))})
    return out


def script_contract(findings: list, baseline=None, *, lang: str = "",
                    kind: str = "", examples: int = 5):
    """G-SCRIPT. Three tiers over one language's cells of one kind.

    Fails when either is true:

      1. ANY finding is BLOCK tier -- a class this dictionary forbids, on a cell
         this pipeline wrote. There is no tolerance and no baseline for those:
         the whole reason the baselines exist is so that this number can be
         zero from the first run instead of after a cleanup nobody schedules.
      2. A BASELINE class exceeds its pinned count, or has no pinned count at
         all. An unpinned population is how 325 contaminated cells became
         invisible for a year, and the fix is the same one G-SUPPRESS and
         G-ADMIT use: the number lives in registry/gates.json and it is edited
         in the same commit as whatever moved it.

    SHRINKING a baseline is reported, never failed, and the asymmetry is the
    point: the clean redo rewrites every cell, so every one of these baselines
    is supposed to go to zero. A gate that failed on that would fail on success.
    """
    pinned = dict(baseline or {})
    block = [f for f in findings if f["tier"] == BLOCK]
    counts, review, over, unpinned, shrunk = {}, {}, [], [], []
    for f in findings:
        if f["tier"] == BASELINE:
            counts[f["class"]] = counts.get(f["class"], 0) + 1
        elif f["tier"] == REVIEW:
            review[f["class"]] = review.get(f["class"], 0) + 1
    for cls, n in sorted(counts.items()):
        limit = pinned.get(cls)
        if limit is None:
            unpinned.append({"class": cls, "cells": n})
        elif n > int(limit):
            over.append({"class": cls, "cells": n, "baseline": int(limit)})
    for cls, limit in sorted(pinned.items()):
        got = counts.get(cls, 0)
        if got < int(limit):
            shrunk.append({"class": cls, "cells": got, "baseline": int(limit)})
    ok = not block and not over and not unpinned
    by_class = {}
    for f in block:
        by_class.setdefault(f["class"], []).append(f["key"])
    return ok, {
        "lang": lang, "kind": kind, "cells_examined": None,
        "block_tier_findings": len(block),
        "block_tier_by_class": {c: len(v) for c, v in sorted(by_class.items())},
        "block_tier_examples": {c: v[:examples]
                                for c, v in sorted(by_class.items())},
        "baseline_tier_counts": counts,
        "baseline_pinned": {k: int(v) for k, v in sorted(pinned.items())},
        "baseline_over": over,
        "baseline_unpinned": unpinned,
        "baseline_shrunk_reported_not_failed": shrunk,
        "review_tier_counts": review,
    }


def script_gate_rows(cfg, cells_by_kind: dict, *, lang: str, pack=None,
                     policy=None, stage: str = "42") -> list:
    """The Gate objects for one language: one row per kind, so a failing
    definitions file cannot be hidden by a clean expressions file.

    extra={"lang":..., "kind":...} keys the report row, which is what stops the
    second language's PASS from overwriting the first language's FAIL.
    """
    baselines = _policy(policy, "script_baseline", {}) or {}
    rows = []
    for kind in sorted(cells_by_kind):
        cells = cells_by_kind[kind] or {}
        pinned = ((baselines.get(lang) or {}).get(kind) or {})

        def make(kind=kind, cells=cells, pinned=pinned):
            findings = script_findings(cells, lang=lang, kind=kind, pack=pack)
            ok, detail = script_contract(findings, pinned, lang=lang, kind=kind)
            detail["cells_examined"] = len(cells)
            return ok, detail

        rows.append(Gate(
            G_SCRIPT,
            "every lemma and gloss stays inside the target language's script "
            "(three tiers: BLOCK on cells this run wrote, BASELINE on 2025 "
            "cells, REVIEW for the rest)",
            make, stage=stage, extra={"lang": lang, "kind": kind}))
    return rows


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


# --------------------------------------------------------------------------
# The money gates (patch plan 2.4, 2.6, 2.7). Predicates first, then the two
# builders a stage calls: pre_spend_gates() before the first paid call and
# post_wave_gates() after a wave is in.
#
# Every predicate is a plain function over data so it can be tested without a
# workspace, a stage or a transport -- and so the criterion that was measured
# WRONG (cached/prompt >= 0.90) can be pinned as wrong by a test forever.
# --------------------------------------------------------------------------

# Defaults for the policy numbers. registry/gates.json carries them so they are
# reviewable next to the deck's other baselines; these are the fallbacks if a
# caller passes no policy dict at all.
BILL_TOLERANCE_FACTOR = 1.10        # patch plan 2.6: actual <= quoted x 1.10
CACHE_HIT_MIN_SHARE = 0.95          # patch plan 2.6, second form of G-CACHE
# G-THINK's three. They are POLICY, not measurements: the measurement is the
# per-kind mean and non-zero share that thinking_bounds_by_kind reads off
# stats.json, and these are the margin a human signs around it. Sized so the
# MEDIUM band (mean 578.7 tokens/request) stays two orders of magnitude away
# from the floor, which is the accident the gate exists to catch.
THINK_MEAN_MARGIN = 3.0             # the wave's mean may be 3x the measured mean
THINK_MEAN_FLOOR = 10.0             # ...and never less than this many tokens/req
THINK_NONZERO_SHARE_FLOOR = 0.05    # non-zero share bound, never tighter than 5%

# The refreeze signature. A human writes it once, at the release freeze, after
# the 22 guid_seed reselections and the three alias merges. Never written by
# code: it is a signature, and a program that can sign for the human is not a
# gate. Looked up in <work>/registry/ first so a run host can carry its own.
REFREEZE_STAMP = "refreeze_stamp.json"


def _policy(policy, key: str, default):
    if isinstance(policy, dict) and policy.get(key) is not None:
        return policy[key]
    return default


def read_gates_policy(cfg):
    """registry/gates.json -- packaged defaults with the <work>/registry overlay
    on top, through the same reader every other baseline in that file gets.

    This exists because the two money policy numbers were DEAD DATA. They sat in
    registry/gates.json under a `_note_money_gates` that declares the file a
    human sign-off point, no production caller ever passed a policy dict, and the
    values that actually decided G-BILL and G-CACHE were the module constants
    below. Editing the file a human had signed changed nothing -- a review gate
    that cannot change behaviour is worse than none, because it manufactures the
    belief that those two numbers were reviewed.

    The precedent is in the tree already: s40_migrate reads
    `gates_cfg.get("tie_break_byte_order_max", 0)` off the same file.
    """
    from .registry import Registry              # noqa: PLC0415 - cycle-free
    return dict(Registry(cfg).gates or {})


def _packaged_registry(name: str):
    """A registry file as it ships INSIDE the package, or None.

    Addressed through the top-level package because this module's sibling is
    itself called `registry` (registry.py), so files("ankidkdeck.registry")
    resolves to the module and not to the data directory.
    """
    try:
        ref = resources.files("ankidkdeck").joinpath("registry", name)
        with ref.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, ValueError, ModuleNotFoundError):
        return None


def read_refreeze_stamp(cfg):
    """The refreeze stamp, local copy winning, or None with the paths tried."""
    local = Path(cfg.registry_local) / REFREEZE_STAMP
    if local.exists():
        return read_json(local, default=None), str(local)
    packaged = _packaged_registry(REFREEZE_STAMP)
    if packaged is not None:
        return packaged, "package registry/%s" % REFREEZE_STAMP
    return None, "%s or package registry/%s" % (local, REFREEZE_STAMP)


def scope_is_frozen(stamp, families: int, card_keys=None, where: str = ""):
    """G-SCOPE-FROZEN. No refreeze signature, no spending.

    Two things have to be true, and the second is the one that makes the first
    worth anything:

      1. A stamp exists and its family count equals len(words.json). The
         refreeze is the LAST scope change before release, so a stamp that
         disagrees with the current scope is a stamp for a different deck.
      2. card_keys.json ships inside the package and still has the row count the
         stamp signed for. card_keys is the users' study progress -- a
         guid_seed reselection after the signature is exactly the event this
         gate exists to catch, and comparing row counts is what makes the stamp
         a freeze rather than a date.

    A missing stamp is a FAILURE, not a skip: "the refreeze has not happened
    yet" is the state this gate is for, and it is the state the pipeline is in
    until the release freeze.
    """
    rows = len(card_keys) if isinstance(card_keys, dict) else None
    detail = {"stamp": where, "families_now": families,
              "card_keys_rows_now": rows}
    if not isinstance(stamp, dict):
        detail["why"] = ("no refreeze stamp. The refreeze (22 guid_seed "
                         "reselections + three alias merges) is a once-only "
                         "pre-release step, and paying to translate a scope "
                         "that is about to change is paying twice. Write "
                         "registry/%s with {refrozen_at, families, "
                         "card_keys_rows, by} after the freeze." % REFREEZE_STAMP)
        return False, detail
    detail.update({k: stamp.get(k) for k in
                   ("refrozen_at", "families", "card_keys_rows", "by")})
    problems = []
    if not stamp.get("refrozen_at"):
        problems.append("the stamp has no refrozen_at date")
    if stamp.get("families") != families:
        problems.append("the stamp signed for %r families, words.json has %d"
                        % (stamp.get("families"), families))
    if rows in (None, 0):
        problems.append("card_keys.json is missing or empty inside the package")
    elif stamp.get("card_keys_rows") is not None \
            and stamp["card_keys_rows"] != rows:
        problems.append("the stamp signed for %r card_keys rows, the package "
                        "has %d -- the registry moved after the signature"
                        % (stamp.get("card_keys_rows"), rows))
    detail["violations"] = problems
    return not problems, detail


def budget_has_room(spent_usd, forecast_usd, cap_usd, period: str = "month",
                    period_key: str = "", ledger_anomalies=None):
    """G-BUDGET. period-to-date + this run's forecast must fit under the cap.

    The forecast is not optional. A run that cannot be forecast cannot be
    authorised: `None` here means the bill could not be priced, and the honest
    answer to "will this fit in $10" is then "unknown", which is a refusal.

    `ledger_anomalies` are the events that make period-to-date itself doubtful:
    a stage usage file truncated, deleted or replaced since the ledger absorbed
    it. The ledger now re-reads such a file from the top and the per-call uids
    reconcile it, but a rotation may not be a silent event -- under-counting is
    how money gets spent that nobody approved. So it refuses ONCE, on the run
    that discovers it; the next run sees the new generation on file and passes.
    """
    detail = {"period": period, "period_key": period_key,
              "spent_usd": spent_usd, "forecast_usd": forecast_usd,
              "cap_usd": cap_usd,
              "ledger_anomalies": list(ledger_anomalies or [])}
    if detail["ledger_anomalies"]:
        detail["why"] = (
            "the spend ledger absorbed %d usage-file rotation(s) or "
            "truncation(s) on this run, so the period-to-date total this cap is "
            "checked against changed shape underneath it: %s. The rows were "
            "re-read from the top and reconciled per call, but a spend is not "
            "authorised against a total nobody has looked at. See "
            "reports/spend_ledger.json:ingest_anomalies, then re-run."
            % (len(detail["ledger_anomalies"]),
               "; ".join(str(a.get("why"))
                         for a in detail["ledger_anomalies"])))
        return False, detail
    if cap_usd is None:
        detail["why"] = "no spend_cap_usd configured; the cap is the gate"
        return False, detail
    if forecast_usd is None:
        detail["why"] = ("the bill has no dollar figure for the scenario this "
                         "run would take, so the ceiling cannot be checked. "
                         "An unpriced run is not a cheap run.")
        return False, detail
    total = round(float(spent_usd or 0.0) + float(forecast_usd), 6)
    detail["would_total_usd"] = total
    detail["headroom_usd"] = round(float(cap_usd) - total, 6)
    if total > float(cap_usd):
        detail["why"] = ("%.4f already spent this %s + %.4f forecast = %.4f > "
                         "the %.2f cap. Google's project-level cap does not "
                         "stop an already-submitted batch wave, so this "
                         "arithmetic is the only thing that does."
                         % (float(spent_usd or 0.0), period, float(forecast_usd),
                            total, float(cap_usd)))
        return False, detail
    return True, detail


def bill_within_ceiling(quoted_usd, actual_usd,
                        factor: float = BILL_TOLERANCE_FACTOR):
    """G-BILL. What was actually spent, against what the human accepted.

    The tolerance is on the QUOTE, not on the ceiling: the number a human read
    and approved was the scenario figure, and 10% is the band inside which a
    forecast built from measured constants may miss.
    """
    detail = {"quoted_usd": quoted_usd, "actual_usd": actual_usd,
              "tolerance_factor": factor}
    if quoted_usd is None:
        detail["why"] = "the run was never quoted, so nothing can be compared"
        return False, detail
    allowed = round(float(quoted_usd) * float(factor), 6)
    detail["allowed_usd"] = allowed
    detail["overrun_usd"] = round(float(actual_usd or 0.0) - allowed, 6)
    ok = float(actual_usd or 0.0) <= allowed
    if not ok:
        detail["why"] = ("the wave cost more than the quote plus %.0f%%. Either "
                         "the token model is wrong or the wave did something "
                         "the bill did not describe."
                         % ((float(factor) - 1) * 100))
    return ok, detail


def _thinking_by_kind(rows) -> dict:
    """Per-kind aggregate of one wave's derived thinking. No criterion here."""
    out: dict = {}
    for row in rows:
        kind = str(row.get("kind") or "")
        value = float(row.get("thinking_tokens") or 0)
        node = out.setdefault(kind, {"requests": 0, "thinking_tokens_total": 0.0,
                                     "nonzero_rows": 0, "max_tokens": 0.0,
                                     "max_row": None})
        node["requests"] += 1
        node["thinking_tokens_total"] += value
        if value > 0:
            node["nonzero_rows"] += 1
        if value > node["max_tokens"]:
            node["max_tokens"] = value
            node["max_row"] = {"label": row.get("label"),
                               "thinking_tokens": row.get("thinking_tokens"),
                               "finish_reason": row.get("finish_reason")}
    for node in out.values():
        n = node["requests"]
        node["mean"] = round(node["thinking_tokens_total"] / n, 4) if n else 0.0
        node["nonzero_share"] = round(node["nonzero_rows"] / n, 6) if n else 0.0
        node["thinking_tokens_total"] = int(node["thinking_tokens_total"])
        node["max_tokens"] = int(node["max_tokens"])
    return out


def thinking_is_at_the_measured_level(rows, level: str = "LOW", bounds=None,
                                      alarm_at=None,
                                      margin: float = THINK_MEAN_MARGIN,
                                      mean_floor: float = THINK_MEAN_FLOOR,
                                      share_floor: float =
                                      THINK_NONZERO_SHARE_FLOOR):
    """G-THINK. A wave's thinking DISTRIBUTION against the measured one, per kind.

    WHAT THIS GATE IS FOR, unchanged: thinkingLevel defaults to MEDIUM, MEDIUM
    was measured at mean 578.7 (p95 1,042) thought tokens per request, and the
    derived output cap has no thinking term. "We accidentally ran at MEDIUM", or
    a prompt regression that provokes thinking, has to be visible here.

    WHAT IT USED TO DO, AND WHY THAT WAS WRONG. A kind in MEASURED_OUTPUT_KINDS
    was held to EXACTLY zero on EVERY row, because the measured constant came
    from 62 probe observations that all happened to be zero and was written down
    as an absolute. Then a real definition wave arrived: 3,644 paid requests,
    mean 1.941 thought tokens/request, p95 0, max 797, 44 rows (1.21%) non-zero,
    every one of them finishReason=STOP and healthy. All 44 were counted as
    violations, so a CORRECT wave could not pass -- the same disease as the
    cached/prompt criterion G-CACHE had to abandon and the GB2312 test G-SCRIPT
    had to abandon. A constant measured at n=62 is a statement about a
    distribution's centre, never a per-row ceiling.

    THE CRITERION IS NOW STATISTICAL AND PER KIND:

      mean            the wave's MEAN thoughts/request for the kind, against
                      max(measured_mean x margin, mean_floor, measured_max / n).
                      The third term is what keeps a SMALL wave honest: one row
                      at the highest value ever measured is the tail that was
                      measured, not evidence of a regression, so it can never
                      fail a wave by itself.
      nonzero_share   the share of rows that thought at all, against
                      max(measured_share x margin, share_floor, 1/n) -- the same
                      one-row headroom, for the same reason. This is the term
                      that catches "the prompt changed and now everything
                      thinks a little", which a mean can absorb.
      max             REPORT ONLY. Never a failure. The measured maximum is 797
                      on a healthy wave; any per-row ceiling near that number is
                      the defect this rewrite removed.
      alarm_at        the MEDIUM band (mean tokens/request), read off the
                      artifact. A kind whose MEAN reaches it FAILS whatever its
                      own bound says -- that is the accident, stated in its own
                      words rather than inferred from a margin.

    A kind with no measured bound is WARNED about, not failed, unless it reaches
    the MEDIUM band: it is real, it is on file, and a constant nobody measured
    for that kind is not grounds to call a wave broken.

    KNOWN BLIND SPOT, stated here rather than papered over: on a wave of one or
    two requests the one-row headroom is the whole bound, so a single MEDIUM-
    sized request inside a retry pass is indistinguishable from the measured LOW
    tail (797 tokens were observed at LOW). It is reported as
    `single_row_headroom` per kind. Closing it would need a per-row ceiling, and
    a per-row ceiling on a heavy-tailed distribution is what was just removed.
    """
    bounds = dict(bounds or {})
    by_kind = _thinking_by_kind(rows)
    violations, warnings = [], []
    if not bounds and alarm_at is None:
        # Refuse rather than pass on an empty check: with no measured bound and
        # no MEDIUM band there is no criterion, and "nothing to compare against"
        # is not a clean wave.
        return False, {
            "requests": len(rows), "thinking_level": level,
            "by_kind": by_kind, "violations": [], "warnings": [],
            "why": ("no measured thinking bound for any kind and no MEDIUM band "
                    "on the artifact, so this gate has nothing to adjudicate "
                    "against. Rebase the thinking constants before reading a "
                    "verdict out of it.")}
    for kind, node in sorted(by_kind.items()):
        n = node["requests"]
        bound = bounds.get(kind)
        node["measured"] = bound
        node["single_row_headroom"] = True
        failed = []
        if alarm_at is not None and node["mean"] >= float(alarm_at):
            failed.append({
                "kind": kind, "test": "medium_band",
                "observed_mean": node["mean"], "allowed": float(alarm_at),
                "requests": n,
                "why": ("mean thoughts/request is at or above the measured "
                        "MEDIUM band (%s): this wave did not run at %s"
                        % (alarm_at, level))})
        if bound:
            m_mean = float(bound.get("mean") or 0.0)
            m_max = float(bound.get("max") or 0.0)
            m_share = float(bound.get("nonzero_share") or 0.0)
            mean_allowed = round(max(m_mean * margin, float(mean_floor),
                                     m_max / n if n else 0.0), 4)
            share_allowed = round(max(m_share * margin, float(share_floor),
                                      1.0 / n if n else 0.0), 6)
            node["mean_allowed"] = mean_allowed
            node["nonzero_share_allowed"] = share_allowed
            node["single_row_headroom"] = bool(
                n and (m_max / n >= max(m_mean * margin, float(mean_floor))))
            if node["mean"] > mean_allowed:
                failed.append({
                    "kind": kind, "test": "mean_thoughts_per_request",
                    "observed_mean": node["mean"], "allowed": mean_allowed,
                    "requests": n, "measured_from": bound.get("source"),
                    "why": ("mean thoughts/request is above "
                            "max(measured mean %s x %s, floor %s, measured max "
                            "%s / %d requests)"
                            % (m_mean, margin, mean_floor, m_max, n))})
            if node["nonzero_share"] > share_allowed:
                failed.append({
                    "kind": kind, "test": "nonzero_share",
                    "observed_share": node["nonzero_share"],
                    "allowed": share_allowed, "requests": n,
                    "nonzero_rows": node["nonzero_rows"],
                    "measured_from": bound.get("source"),
                    "why": ("more rows thought than the measured distribution "
                            "supports: max(measured share %s x %s, floor %s, "
                            "1/%d)" % (m_share, margin, share_floor, n))})
        elif node["nonzero_rows"]:
            warnings.append({
                "kind": kind, "requests": n,
                "nonzero_rows": node["nonzero_rows"],
                "mean": node["mean"], "max_tokens": node["max_tokens"],
                "why": "no measured thinking bound for this kind"})
        node["verdict"] = ("FAIL" if failed else
                           ("PASS" if bound else "WARN (unmeasured kind)"))
        violations += failed
    values = sorted({int(r.get("thinking_tokens") or 0) for r in rows})
    detail = {
        "requests": len(rows), "thinking_level": level,
        "criterion": ("per-kind MEAN thoughts/request and NON-ZERO SHARE "
                      "against the measured distribution; the per-row maximum "
                      "is reported, never failed"),
        "mean_margin": margin, "mean_floor_tokens": mean_floor,
        "nonzero_share_floor": share_floor,
        "medium_band_alarm_at": alarm_at,
        "kinds_with_a_measured_bound": sorted(bounds),
        "by_kind": by_kind,
        "distinct_thinking_values": values[:20],
        "violations": violations[:10],
        "warnings": warnings[:10],
        "warning_note": ("non-zero thinking on a kind nobody measured: "
                         "recorded, not failed. At LOW the definition prompt "
                         "was measured at mean 1.941 tokens/request and the "
                         "ranking prompt at 236-275.") if warnings else None}
    if violations:
        detail["why"] = ("this wave's thinking distribution is outside the "
                         "measured one. Thinking is billed at the OUTPUT rate "
                         "and shares maxOutputTokens with the answer, so the "
                         "level or the prompt has changed under the bill.")
    return not violations, detail


def one_prompt_per_wave(rows, bill_shas=None, bill_prompt_id: str = "",
                        cache_prompt_shas=None):
    """G-PROMPT. Every row written by this run carries the bill's prompt.

    Two independent readings, on purpose: the bill's sha comes from the prompt
    builder at bill time, the row's sha comes from the request that was actually
    sent. Comparing a copy of one reading against itself is how a prompt change
    can pass a prompt gate.

    THE CACHED PATH IS THE EXPECTED PATH, and on it the row has no sha of its
    own. systemInstruction and cachedContent are mutually exclusive (hard 400),
    so a cached request carries prompt_sha256 = None -- and this gate used to
    `continue` past every one of them and report ok=True with the word "checked"
    nowhere in its detail. On the one wave this program is actually going to run,
    the sha half of G-PROMPT verified nothing and said nothing.

    So a cached row's prompt identity is taken from THE CACHE instead, in order:

      1. row["cache_prompt_sha256"]  the sha of the prompt that was put in the
                                     cache object, stamped on the row by the
                                     transport at cache-creation time. Preferred:
                                     it is per row, so a wave that used two cache
                                     objects is still checkable row by row.
      2. cache_prompt_shas[name]     {cache resource name: prompt sha} handed to
                                     the builder by the transport.

    And `rows_checked` is ALWAYS reported, because two failures print as ok=True
    without it. Both are refusals now:

      * a row whose prompt sits in a CACHE whose sha nobody recorded. The prompt
        exists, it went on the wire, and it is unverified.
      * zero rows checked while the bill DID quote a sha for a kind in this wave.
        "I verified nothing" and "everything verified" may not print the same
        verdict -- the same anti-pattern G-CACHE was already forbidden from
        (report n/a explicitly, never a quiet pass).

    A call that legitimately has no system prompt and whose kind the bill quotes
    no sha for (the manual review call) is still not a violation: there is
    nothing to compare, and it says so in `kinds_the_bill_does_not_cover`.
    """
    shas = dict(bill_shas or {})
    cache_shas = dict(cache_prompt_shas or {})
    ids = sorted({r.get("prompt_id") or "" for r in rows})
    mismatched, unknown_kind, unverifiable, blind = [], [], [], []
    checked = 0
    via = {"row_sha": 0, "cache_row_sha": 0, "cache_name_map": 0}
    for row in rows:
        got = row.get("prompt_sha256")
        source = "row_sha"
        if got is None:
            got = row.get("cache_prompt_sha256")
            source = "cache_row_sha"
        if got is None and row.get("cache_name"):
            got = cache_shas.get(row.get("cache_name"))
            source = "cache_name_map"
        quoted = shas.get(row.get("kind") or "")
        if got is None:
            item = {"label": row.get("label"), "kind": row.get("kind"),
                    "cache_name": row.get("cache_name"),
                    "the_bill_quotes_a_sha_for_this_kind": quoted is not None}
            if row.get("cache_name"):
                item["why"] = ("the prompt is inside cache %r and no sha was "
                               "recorded for it, so it went on the wire "
                               "unverified" % row.get("cache_name"))
                blind.append(item)
            else:
                item["why"] = "no system prompt on this call"
                unverifiable.append(item)
            continue
        if quoted is None:
            unknown_kind.append(row.get("kind"))
            continue
        checked += 1
        via[source] += 1
        if got != quoted:
            mismatched.append({"label": row.get("label"),
                               "kind": row.get("kind"), "checked_via": source,
                               "row_sha256": got, "bill_sha256": quoted})
    id_ok = (not rows) or ids == [bill_prompt_id]
    # Zero checked is only a pass when the bill quoted no sha for anything in
    # this wave. If it quoted one and nothing was compared against it, the
    # comparison did not happen.
    quotable = any(shas.get(r.get("kind") or "") is not None for r in rows)
    nothing_checked = bool(rows) and checked == 0 and quotable
    detail = {"requests": len(rows), "bill_prompt_id": bill_prompt_id,
              "row_prompt_ids": ids, "bill_shas": shas,
              # The disclosure that was missing: a verdict without this number
              # cannot be read.
              "rows_checked": checked,
              "rows_checked_via": via,
              "rows_whose_cached_prompt_is_unverified": blind[:10],
              "rows_whose_cached_prompt_is_unverified_count": len(blind),
              "rows_with_no_system_prompt": len(unverifiable),
              "rows_with_a_different_sha": mismatched[:10],
              "kinds_the_bill_does_not_cover": sorted(set(unknown_kind)),
              "prompt_id_consistent": id_ok}
    if not id_ok:
        detail["why"] = ("the rows and the bill disagree about which prompt "
                         "pack this run used, and prompt_id is what the "
                         "measured constants are indexed by")
    elif blind:
        detail["why"] = ("%d row(s) carried their system prompt inside an "
                         "explicit cache whose prompt sha nobody recorded. The "
                         "transport has to stamp cache_prompt_sha256 on the row "
                         "at cache-creation time, or hand this gate a "
                         "{cache_name: sha} map; otherwise the wave this program "
                         "is actually going to run is the one wave G-PROMPT "
                         "cannot check." % len(blind))
    elif nothing_checked:
        detail["why"] = ("the bill quotes a prompt sha for a kind in this wave "
                         "and not one of %d row(s) was compared against it. A "
                         "gate that checked nothing does not pass." % len(rows))
    return (id_ok and not mismatched and not blind and not nothing_checked), \
        detail


def cache_hit_is_complete(rows, declared_cache_tokens=None,
                          min_share: float = CACHE_HIT_MIN_SHARE,
                          cache_expected: bool = False,
                          cached_kinds=("definition",)):
    """G-CACHE. cached == declared, per row -- NOT cached/prompt.

    The audit's criterion was `cached/prompt >= 0.90` per row. It is measured
    WRONG: on a fully cached wave the cached count is constant (1,135) while the
    prompt grows with the payload, so the ratio slides from 0.935 at n=1 to
    0.632 at n=20 and the criterion would condemn a perfectly healthy wave. The
    quantity that means "the cache was used" is cached vs DECLARED, which was
    1.00 on 30 of 30 batch rows and on every interactive arm.

    THE DENOMINATOR IS THE WAVE, NOT THE HITS. The patch plan's formula is
    sum(cached) / (declared x REQUESTS); this used to divide by the number of
    rows that happened to carry a cache name, which makes the share a ratio of
    the hits to themselves. Measured consequence: a 100-request definition wave
    in which ONE row used the cache and 99 quietly fell back to an inlined
    system prompt scored share = 1.00 and PASSED, while 99% of the requests paid
    full price. That is the most likely real failure -- an expired cache cannot
    be updated, only recreated, and a recreate changes the resource name, so
    some jobs carry the new name and some fall back.

    `cached_kinds` is what keeps the correct denominator from condemning a
    healthy MIXED wave: the expression prompt is 336 tokens against a measured
    1,024-token floor, so expression requests are uncached BY DESIGN and dividing
    by every row of the wave would fail every real run. The denominator is
    therefore every request of the wave THAT WAS SUPPOSED TO BE CACHED.

    Reports n/a rather than passing when no cache was declared -- unless a cache
    was CONFIGURED, in which case a wave with no cached tokens is the failure
    the gate is for.
    """
    kinds = tuple(cached_kinds or ())
    # CROSS-OWNER EDIT (made for the batch transport; this file is the money gates): a row that
    # produced NO PROMPT TOKENS never executed, so it neither hit nor missed the
    # cache. On the batch surface that is a per-row gRPC error -- measured at
    # prompt=0 and billed $0 (probe W3-4: a cache deleted after submit made all
    # 21 rows fail that way) -- and the wave then retries it. Counting the failed
    # attempt in the denominator failed a healthy wave: one error plus one
    # successful retry scored 1135/(1135 x 2) = 0.5 against a 0.95 threshold.
    # The failure this gate exists for is untouched: a row that silently fell
    # back to an inlined system prompt has prompt tokens and no cache, so it is
    # still counted, and a wave whose cache is dead for EVERY row leaves the
    # scope empty, which with cache_expected is still a refusal.
    #
    # `prompt_tokens > 0` is the WHOLE test. An earlier version also excluded any
    # row carrying an `error` field, which is wider than the reason above needs
    # and unsafe in the one direction that matters: a row that was BILLED (real
    # prompt tokens), used no cache, and also reported an error -- a truncation,
    # an unparseable body -- did hit or miss the cache, and dropping it from the
    # denominator hides exactly the full-price row this gate is looking for.
    executed = [r for r in rows if int(r.get("prompt_tokens") or 0) > 0]
    not_executed = len(rows) - len(executed)
    # Every row of a cacheable kind, executed or not. This is what decides
    # whether the wave DECLARED anything to cache, which is a different question
    # from whether the cache worked: an incremental wave with only expression
    # cells left, or the run after an unrecovered definition request, has nothing
    # to cache and must not be failed for it.
    declared_rows = [r for r in rows if (r.get("kind") or "") in kinds]
    kindless = bool(rows) and not any(r.get("kind") for r in rows)
    scope = [r for r in executed if (r.get("kind") or "") in kinds]
    scope_basis = ("kind in %s, among the %d row(s) that produced prompt tokens"
                   % (list(kinds), len(executed)))
    if kinds and not scope and executed \
            and not any(r.get("kind") for r in executed):
        # Rows with no kind at all (a synthetic or legacy wave): hold the whole
        # wave to the criterion rather than silently checking nothing.
        scope, scope_basis = list(executed), "every row (no kind on any row)"
    with_cache = [r for r in scope if r.get("cache_name")]
    cached = [int(r.get("cached_tokens") or 0) for r in with_cache]
    prompts = [int(r.get("prompt_tokens") or 0) for r in with_cache]
    ratios = [round(c / p, 4) for c, p in zip(cached, prompts) if p]
    detail = {"requests": len(rows),
              # The disclosure fields: what the denominator was and how much of
              # it was actually looked at. A share without these cannot be read.
              "cached_kinds": list(kinds),
              "rows_that_never_executed": not_executed,
              "rows_that_never_executed_note": ("a per-row error is prompt=0 and "
                                                "$0 billed, so it neither hit "
                                                "nor missed the cache"),
              "rows_of_a_cacheable_kind": len(declared_rows),
              "rows_checked": len(scope),
              "rows_checked_basis": scope_basis,
              "requests_with_a_cache": len(with_cache),
              "requests_without_a_cache": len(scope) - len(with_cache),
              "declared_cache_tokens": declared_cache_tokens,
              "min_share": min_share,
              # Recorded as EVIDENCE, never as the criterion: this is the
              # metric the audit proposed and the probes disproved.
              "cached_over_prompt_range": ([min(ratios), max(ratios)]
                                           if ratios else None),
              "cached_over_prompt_note": ("not the criterion: 0.632 at n=20 on "
                                          "a wave whose cache hit 1.00")}
    if not with_cache or not declared_cache_tokens:
        detail["checked"] = False
        detail["verdict"] = "n/a"
        if cache_expected and (declared_rows or kindless or not kinds):
            detail["why"] = ("cache_enabled is set and this wave placed %d "
                             "request(s) of a cacheable kind, but no row carries "
                             "a cache name or the wave declared no cached "
                             "tokens, so it paid the full uncached rate while "
                             "the configuration said otherwise"
                             % len(declared_rows or rows))
            return False, detail
        if cache_expected:
            # cache_enabled with NOTHING CACHEABLE IN THE WAVE. Not a failure:
            # the expression prompt is under the measured 1,024-token floor by
            # design, so an incremental wave with only expression cells left has
            # nothing to cache. Failing it made the documented recovery path
            # ("the cells stay missing and the next run picks them up")
            # unreachable -- the next run was the one with no definition rows.
            detail["verdict"] = "n/a: no request of a cacheable kind in this wave"
            detail["why"] = ("cache_enabled is set, but not one request of this "
                             "wave was of a cacheable kind (%s), so there was "
                             "nothing to cache and nothing to check"
                             % list(kinds))
        return True, detail
    declared = int(declared_cache_tokens)
    exact = [c == declared for c in cached]
    # len(scope), not len(with_cache): see the docstring.
    share = round(sum(cached) / float(declared * len(scope)), 4)
    detail.update({"checked": True, "rows_exactly_declared": sum(exact),
                   "sum_cached_over_declared_x_requests": share,
                   "misses": [{"label": r.get("label"),
                               "cached_tokens": r.get("cached_tokens")}
                              for r, ok in zip(with_cache, exact) if not ok][:10]})
    ok = (len(with_cache) == len(scope) and all(exact)) \
        or share >= float(min_share)
    if not ok:
        detail["why"] = ("cached != declared on %d row(s), %d of %d request(s) "
                         "that should have been cached carried no cache at all, "
                         "and the aggregate share %.4f is under %.2f. An "
                         "expired, misplaced or unreferenced cache is a "
                         "full-price wave."
                         % (len(exact) - sum(exact),
                            len(scope) - len(with_cache), len(scope),
                            share, float(min_share)))
    return ok, detail


# ---- the two builders a stage calls -------------------------------------

def pre_spend_gates(cfg, bill: dict, *, families: int, ledger=None,
                    stage: str = "42") -> list:
    """The gates that must pass BEFORE the first paid call of a run.

    Call site (stage 42, immediately after probe_stats/validate and before the
    first request):

        from ..gates import pre_spend_gates, run_gates
        run_gates(pre_spend_gates(cfg, bill, families=len(families)), cfg,
                  stage="42")

    `bill` is report["bill"] -- the same dict print_bill() just printed, so the
    gate adjudicates the number the human read and not a second computation.

    There is deliberately no `policy` parameter here (there used to be one, and
    the function body never read it, which is how a dead parameter advertises a
    capability that does not exist). Neither pre-spend gate has a registry policy
    number: the cap is configuration (cfg.spend_cap_usd) and the refreeze stamp
    is a human signature. The two numbers that ARE policy are read from
    registry/gates.json by post_wave_gates.
    """
    from .billing import SpendLedger, forecast          # noqa: PLC0415
    fc = forecast(bill, cfg)
    led = ledger if ledger is not None else SpendLedger(cfg)
    # Absorb anything the stage's own fsync'd usage files hold and the ledger
    # has not seen: a wave that CRASHED still spent money, and month-to-date
    # has to know before it authorises the next one. Safe to call even when the
    # per-call sink was wired up -- the rows carry a per-call uid, so whichever
    # door ran first owns the row.
    absorbed = led.ingest()
    spent = led.period_to_date()
    stamp, where = read_refreeze_stamp(cfg)
    card_keys = _packaged_registry("card_keys.json")
    return [
        Gate(G_SCOPE_FROZEN,
             "the scope was refrozen and card_keys.json has not moved since",
             lambda: scope_is_frozen(stamp, families, card_keys, where),
             stage=stage),
        Gate(G_BUDGET,
             "period-to-date spend plus this run's forecast fits under the cap",
             lambda: budget_has_room(
                 spent["usd"], fc["usd"], cfg.spend_cap_usd,
                 period=spent["period"], period_key=spent["period_key"],
                 ledger_anomalies=absorbed.get("anomalies")),
             stage=stage, extra={"scenario": fc["scenario"]}),
    ]


def billed_row(cfg, bill: dict, lang: str) -> dict:
    """One language's bill, from the report dict, completed from the FILE.

    report["bill"][lang] does not carry prompt_id or prompt_sha256 --
    reports/translate_bill_<lang>.json does, and that file is the artifact the
    human read before pressing --confirm-spend. Filling the gaps from disk is
    what lets G-PROMPT compare the wire against the QUOTE rather than against
    the configuration (which would agree with itself by construction).
    """
    row = dict((bill or {}).get(lang) or {})
    if row.get("prompt_sha256") is None or row.get("prompt_id") is None:
        path = Path(cfg.report_dir) / ("translate_bill_%s.json" % lang)
        if path.exists():
            disk = read_json(path, default={}) or {}
            for key in ("prompt_sha256", "prompt_id", "dollars", "mode"):
                if row.get(key) is None:
                    row[key] = disk.get(key)
    return row


def post_wave_gates(cfg, bill: dict, rows, *, lang: str = "",
                    declared_cache_tokens=None, cache_prompt_shas=None,
                    stats=None, policy=None, stage: str = "42") -> list:
    """The gates that adjudicate a wave that has already been paid for.

    `rows` are the usage rows of THIS wave (stage 42's UsageLog.rows, or the
    ledger lines it wrote). `bill` is the same report["bill"] the run quoted.

    `policy` defaults to registry/gates.json (packaged + <work>/registry
    overlay), so the two numbers a human signs in that file are the two numbers
    the gates use. Pass a dict to override -- that is for tests, not for
    production.

    `cache_prompt_shas` is {cache resource name: sha of the prompt inside it},
    from whoever created the cache. Without it, a wave whose rows carry no
    cache_prompt_sha256 has no verifiable prompt identity and G-PROMPT refuses
    instead of passing on an empty check.
    """
    from .billing import expected_scenario, rows_usd_priced  # noqa: PLC0415
    if policy is None:
        policy = read_gates_policy(cfg)
    scenario = expected_scenario(cfg)
    langs = [lang] if lang else sorted(bill)
    billed = {lg: billed_row(cfg, bill, lg) for lg in langs}
    quoted = 0.0
    quotable = True
    for lg in langs:
        value = (billed[lg].get("dollars") or {}).get(scenario)
        if value is None:
            quotable = False
        else:
            quoted += float(value)
    actual = rows_usd_priced(rows, default_model=cfg.gemini_model,
                             default_mode=cfg.mode)
    level = getattr(cfg, "thinking_level", "LOW")
    alarm_at = None
    think_bounds: dict = {}
    # G-THINK's bounds are PER KIND and come off the artifact. MEASURED_OUTPUT_KINDS
    # is deliberately NOT consulted any more: it is the set whose OUTPUT fit was
    # measured, and reusing it as "the kinds held to a thinking constant" is how
    # the definition wave came to be held to an absolute zero. Which kinds have a
    # thinking bound is a property of what has been measured, so it is answered by
    # the file, not by a tuple in the code.
    from .stages.s42_translate import (CACHEABLE_KINDS,  # noqa: PLC0415
                                       thinking_bounds_by_kind,
                                       thinking_per_request)
    if stats:
        think_bounds = thinking_bounds_by_kind(stats, level)
        # The band that means "this ran at MEDIUM". Read off the artifact, never
        # hard-coded: without the measurement there is no band and unmeasured
        # kinds are only warned about.
        alarm_at = thinking_per_request(stats, "MEDIUM", "mean")
    shas = {}
    for lg in langs:
        for kind, sha in (billed[lg].get("prompt_sha256") or {}).items():
            shas[kind] = sha
    quoted_prompt_id = (billed[langs[0]].get("prompt_id") if langs
                        else None) or cfg.prompt_id
    tol = _policy(policy, "bill_tolerance_factor", BILL_TOLERANCE_FACTOR)
    share = _policy(policy, "cache_hit_min_share", CACHE_HIT_MIN_SHARE)
    think_margin = _policy(policy, "think_mean_margin", THINK_MEAN_MARGIN)
    think_floor = _policy(policy, "think_mean_floor_tokens", THINK_MEAN_FLOOR)
    think_share = _policy(policy, "think_nonzero_share_floor",
                          THINK_NONZERO_SHARE_FLOOR)
    extra = {"lang": lang} if lang else {}
    return [
        Gate(G_BILL, "the wave cost no more than the quote plus the tolerance",
             lambda: bill_within_ceiling(quoted if quotable else None,
                                         actual["usd"], tol),
             stage=stage, extra=extra),
        Gate(G_THINK, "the wave's thinking distribution per kind is the "
                      "measured one, within the signed margin",
             lambda: thinking_is_at_the_measured_level(
                 rows, level, think_bounds, alarm_at,
                 margin=think_margin, mean_floor=think_floor,
                 share_floor=think_share),
             stage=stage, extra=extra),
        Gate(G_PROMPT, "every row carries the prompt id and sha the bill quoted",
             lambda: one_prompt_per_wave(rows, shas, quoted_prompt_id,
                                        cache_prompt_shas),
             stage=stage, extra=extra),
        Gate(G_CACHE, "cached tokens equal the declared cache on every row",
             lambda: cache_hit_is_complete(
                 rows, declared_cache_tokens, share,
                 cache_expected=bool(getattr(cfg, "cache_enabled", False)),
                 cached_kinds=tuple(CACHEABLE_KINDS)),
             stage=stage, extra=extra),
    ]
