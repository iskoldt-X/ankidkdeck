"""Stage 42: incremental LLM top-up, POS translation, and GC.

Three properties matter more than the prompts:

  1. THE BILL COMES FIRST. run(confirm=False) computes exactly what would be
     paid for, prints it, writes reports/translate_bill_<lang>.json and
     returns. Nothing is imported from google.genai on that path, so a dry run
     cannot place a call even by accident, and the package installs and runs
     with no LLM dependency at all.
  2. THE UNIT IS A CELL, NOT A FILE. A cell is one (entry_id, dannetid)
     definition or one dannetid expression. The measured post-migration gap is
     ~5,218 cells (English expressions 4,647 + German definitions 571), not the
     14,182 the old per-file counting suggested.
  3. CHECKPOINT AFTER EVERY ENTRY. Any interruption costs one entry.

Ported verbatim from the v2.1 pipeline, because their behaviour is load-bearing:
definition PROMPT V4 and its exactly-n {lemma, gloss} schema lock
(05_translate_definitions.py); the generate -> review -> correct loop with the
"DO NOT USE RUSSIAN" critical rules and batch=20 (06_translate_expressions.py);
the POS prompt (04_translate_pos.py).

Two deliberate departures from those scripts:

  * The key pool is an object. All four v2.1 LLM scripts contain the same latent
    bug: `_request_count = MAX_PER_API` inside an `except` block that never
    declared `global _request_count`, so the assignment created a local and the
    "force a key rotation when the API says slow down" path NEVER FIRED. An
    instance attribute cannot have that bug.
  * Throttling is detected from the exception, not from
    google.api_core.exceptions. The new google-genai SDK does not promise the
    api_core exception classes, and importing them would make an install-time
    dependency out of an error path.

The 2026-08-26 patch changed five more things about the call itself, all of them
measured rather than reasoned:

  * THE SYSTEM PROMPT NO LONGER DEPENDS ON n. The count was interpolated into
    the last line of the prompt, which produced 7 distinct prompt hashes across
    30 payloads -- one per value of n, differing by 1-2 characters. One prompt
    per language is the precondition for one explicit cache per language, which
    halves the definition wave. The count lives in the user message and in the
    schema, which is where it is enforced anyway.
  * thinkingLevel IS SENT, AS A LITERAL. Unset means MEDIUM, measured at mean
    578.7 thought tokens per request and billed at output rates. At LOW the
    derived thinking was 0 across 38 observations, including on a 4.5k prompt.
    thinking_budget=0 is not a substitute: it is accepted and then clamped.
  * temperature IS NOT SENT AT ALL. Deprecated on this model generation, and the
    A/B (24 requests at 0.1 vs 38 with the field absent) showed zero difference
    in count-lock violations, MAX_TOKENS finishes and thinking.
  * finishReason IS READ BEFORE THE JSON IS PARSED. A MAX_TOKENS truncation used
    to surface as a JSONDecodeError inside a five-attempt retry ladder wrapped
    in a five-attempt count-lock ladder: 25 paid calls for one configuration
    error.
  * EVERY RESPONSE'S usageMetadata IS RECORDED. Not one byte of it was read
    before, so there was no way to know what a run had cost or whether the
    cache had been hit.
"""

import dataclasses
import datetime
import json
import math
import os
import re
import time
from pathlib import Path

from ..config import Config
from ..extract import ARTICLE_SHA_SCHEMA
from ..gates import G_ORPH, Gate, run_gates
from ..util import NFC, FatalError, read_json, sha256_str, write_json

MAX_DEFS_PER_BATCH = 20        # 05: one call per entry, capped
MAX_EXPR_PER_BATCH = 20        # 06: MAX_EXPR_PER_BATCH
MAX_RETRIES = 5
# A 503 gets ONE backoff retry, not five. "The model is currently experiencing
# high demand" was 46.4% of requests on the paid tier and 58-68% on the free
# one, so it is not a transient to sit out: the probe wave burned 14 of its 20
# daily free-tier requests on zero-information retries, and on the paid tier
# each one is a real charge for nothing. It is also NOT free-tier specific,
# which is why the ladder is per-class rather than per-tier.
MAX_503_RETRIES = 1
# The ladder for a violated COUNT LOCK. A short array is a dropped sense, never
# something to zip anyway -- but it is also the classic transient, and FATALing
# on the first one throws away every call already paid for in the run.
MAX_COUNT_LOCK_ATTEMPTS = 5
BASE_RETRY_DELAY = 5
CHARS_PER_TOKEN_ESTIMATE = 4   # for the bill only; never used to size a call

# The measured output fit, ceil(a*n + b), times this. There is NO thinking term:
# at thinkingLevel=LOW the derived thinking is 0, so the cap only has to cover
# the JSON.
MAX_OUTPUT_SAFETY_FACTOR = 1.5
# gemini-3.7-flash's output ceiling (config.VERIFIED_MODELS carries it per
# model; this is the guard for the derived cap).
MODEL_OUTPUT_CEILING = 65536
# The request kinds the output fit was MEASURED on. Every point behind
# EXPECTED_OUTPUT came from a DEFINITION request (payload set PS30, 62 points);
# expression, POS and ranking outputs were never measured at all. Sizing them
# from a definition fit is a guess wearing a measurement's clothes -- and the
# guess is not conservative: an expression gloss is "a full, explanatory
# sentence", where a definition gloss is a phrase. They get a flat,
# config-visible cap (cfg.max_output_unmeasured) until someone measures them.
MEASURED_OUTPUT_KINDS = ("definition",)
# The one-shot budget raise on a MAX_TOKENS finish (spec 5.6). A truncation is a
# cap error, so the answer is a BIGGER cap rather than the same request again --
# but exactly once, and never past this, or a runaway request would double its
# way to the model ceiling on our money.
MAX_OUTPUT_RETRY_CEILING = 8192
# Archive reasons that mean RETIRED BY US, not "lost and recoverable". A row
# archived by the clean redo has been deliberately superseded; restoring it is
# undoing a decision that was already paid for, which is the opposite of the
# no-double-billing property _restore_from_archive exists for.
RETIRED_ARCHIVE_REASONS = ("clean_redo",)
# Provenance is a closed vocabulary and pure ASCII, because it is filtered on.
PROVENANCE_RE = re.compile(r"^gemini:[A-Za-z0-9.\-]+\+[A-Za-z0-9.\-]+"
                           r"\+[A-Z]+@\d{4}-\d{2}-\d{2}$")

THROTTLE_MARKERS = ("429", "resource_exhausted", "too many requests",
                    "quota", "rate limit", "ratelimit")
# 403 PERMISSION_DENIED: CachedContent not found (or permission denied). It is a
# 403, not a 404, and it is also what a cache belonging to another key/project
# returns. Recoverable: rebuild the cache and carry on -- never fatal, and never
# a plain retry, because retrying the same dead cache name cannot work.
CACHE_MISSING_MARKERS = ("cachedcontent not found",
                         "cached content not found")
# Verdicts of classify_api_error(). One name per handling path.
ERR_FATAL = "fatal"                  # 400 and friends: raise on the first one
ERR_THROTTLE = "throttle"            # 429: rotate the key, then retry
ERR_UNAVAILABLE = "unavailable"      # 503: one retry, then give up on this call
ERR_RETRYABLE = "retryable"          # other 5xx / transport
ERR_CACHE_MISSING = "cache_missing"  # 403 on the cache: rebuild, then continue


# --------------------------------------------------------------------------
# what to translate
# --------------------------------------------------------------------------

def definition_key(entry_id: str, sense: dict) -> str:
    """Composite key (guide 1.11e): 10 dannetid values are shared across two
    entry_ids, all of them expression senses, so definitions cannot use the
    dannetid alone."""
    return "%s:%s" % (entry_id, sense["dannetid"])


def expression_key(expr: dict) -> str:
    """A shared idiom deliberately collapses to ONE translation."""
    return expr.get("dannetid")


def expression_src_sha(expr: dict) -> str:
    """sha256(NFC(expression text)) -- the string the translator is actually
    given (`expr`; the definition is only an optional `hint`).

    It used to prefer the first SENSE's src_sha, i.e. the sha of the idiom's
    DEFINITION, which is also what stage 41 was writing. That disabled the one
    retranslation trigger the design has: editing an idiom never retranslated it
    and editing its definition retranslated it for nothing. Stage 41 now stores
    this same formula, so a migrated cell is never mistaken for a changed one.
    """
    return sha256_str(NFC(expr.get("expression") or ""))


def expression_hint(expr: dict) -> str:
    """06 passed the first definition as a disambiguating hint."""
    for s in (expr.get("senses") or []):
        if s.get("definition"):
            return s["definition"]
    return ""


def renderable_scope(cfg: Config, entries: dict, families: dict,
                     include_unused: bool = False) -> tuple[set, dict]:
    """The entry_ids a card will actually show.

    Deviation from the guide's pseudocode, deliberate: it iterates every parsed
    entry, which would pay for articles the classifier rejected and the export
    never renders. G-COV is defined over renderable senses, so the scope that
    the gate checks is the scope the bill should quote.

    Falling back to "every parsed entry" when words.json is absent is now behind
    an explicit flag. Measured on the fixture corpus the fallback quotes 57 cells
    instead of 17 -- 3.4x, all of it on articles the classifier rejected and the
    exporter will never render -- and the docstring saying so is not a refusal to
    spend. Running the merge stage first is the answer; --include-unused is for
    the deliberate case.
    """
    if not families:
        if not include_unused:
            raise FatalError(
                "no %s: the translation scope is the set of entries a card will "
                "actually render, and without words.json every rejected article "
                "would be billed too (measured 3.4x on the fixture corpus). Run "
                "the merge stage first, or pass --include-unused to bill every "
                "parsed entry on purpose." % (cfg.json_dir / "words.json"))
        return set(entries), {"basis": "all parsed entries",
                              "why": "words.json not found and --include-unused "
                                     "was given: rejected articles ARE billed"}
    scope = {eid for fam in families.values() for eid in fam.get("entry_ids", [])
             if eid in entries}
    return scope, {"basis": "renderable families (words.json)",
                   "entries_in_scope": len(scope), "entries_parsed": len(entries)}


def pos_keys_in_scope(entries: dict, scope) -> list:
    return sorted({entries[eid].get("pos_key") for eid in scope
                   if entries[eid].get("pos_key")})


def _restore_from_archive(kind: str, key: str, sha, have: dict, archive: dict,
                          restored: list,
                          retired_reasons=RETIRED_ARCHIVE_REASONS) -> bool:
    """A sense that comes back after a DDO edit must not be paid for twice.

    gc() moves a dead row into archive.json and its docstring has always claimed
    this property (guide D7) -- but nothing anywhere read archive.json back, so
    the property was not implemented: remove a sense, run translate (archived),
    put it back, run translate again, and the cell was billed a second time.

    Restoring is conditional on the SOURCE TEXT being unchanged: the archived
    gloss was produced from a specific Danish string, and src_sha is that string.
    A row whose sha no longer matches stays archived and is billed as changed.

    A row RETIRED BY US is never restored, whatever its sha says. The clean redo
    (N-01) archives every live row under reason="clean_redo"; those rows are the
    OLD translator's output, kept for audit, and their shas still match by
    definition. Restoring them was measured to undo a whole redo: crash a
    confirmed redo, then run a plain `translate` -- the natural "carry on"
    action -- and every gemini-2.0-flash row came home, the bill said 0 cells,
    and definitions.json ended up a silent mix of two translators.
    """
    if archive is None:
        return False
    row = (archive.get(kind) or {}).get(key)
    if row is None or row.get("src_sha") != sha:
        return False
    if row.get("reason") in retired_reasons:
        return False
    # `reason` is the archiving reason (why the row was moved out), not a
    # property of the translation. A row coming back into the live table must
    # not carry it: definitions.json rows are compared and exported, and a
    # stray field there is a diff nobody can explain.
    have[key] = {k: v for k, v in row.items() if k != "reason"}
    restored.append({"kind": kind, "key": key,
                     "provenance": row.get("provenance")})
    return True


def _already_redone(row, sha, done_prefix: str) -> bool:
    """Is this live row this redo's own output?

    The resume test for N-01. A row counts as done when its Danish source is
    unchanged AND its provenance carries the identity of the run in progress:
    model + prompt_id + thinking level. The DATE is deliberately not part of the
    comparison -- a redo that crashes at 23:50 and is resumed at 00:10 is the
    same redo, and the alternative is billing it twice for owning a calendar.
    """
    if not done_prefix or row is None:
        return False
    return (row.get("src_sha") == sha
            and str(row.get("provenance") or "").startswith(done_prefix))


def compute_todo(cfg: Config, entries: dict, translations: dict, lang: str,
                 scope=None, archive: dict | None = None,
                 restored: list | None = None, retranslate_all: bool = False,
                 retranslate_reason: str = "clean_redo",
                 done_provenance: str = "", resume: dict | None = None) -> list:
    """Cells that are missing, or whose Danish source text has changed.

    One row per cell: {key, kind, entry_id, lemma, pos_text, dannetid, text,
    grammar, hint, src_sha, reason}. Empty definition texts are not cells --
    there is nothing to translate and the exporter does not render them.

    A key that is missing from the live table but present in archive.json with
    the SAME src_sha is restored into the live table (in place, so the caller
    persists it) and is not billed.

    `retranslate_all` is the clean-retranslation path (owner decision D-01,
    2026-08-26): EVERY cell in scope is a todo row with reason=clean_redo, and
    the archive is not read back. Both halves matter. Without the second one,
    the archive would immediately restore the rows the caller is about to
    archive -- which is also why `rm definitions.json` only ever worked once:
    the next run silently restored the lot from archive.json.

    `done_provenance` makes the redo RESUMABLE, which a 5,565-request multi-hour
    wave has to be. A live row that already carries this run's provenance and an
    unchanged source is this redo's own finished work: it is neither a todo row
    nor a line on the bill. Without it the two options after a crash were
    "silently roll the redo back" (a plain re-run) and "pay for all of it again"
    (a flagged re-run). `resume` receives the arithmetic so the dry path can
    print it.

    `grammar` is DDO's grammar field for the sense (valency frames like
    NOGET/NOGEN, and number/register labels such as "uden pluralis"). 38.8% of
    the senses on file have one and not a byte of it used to reach the model, because only
    s["definition"] went into the payload. It travels as its own field in the
    USER message, never in the system prompt: it must not enter the cached
    prefix, and it is per-sense anyway.
    """
    scope = set(entries) if scope is None else set(scope)
    # `or {}` here was a live bug: an EMPTY live table is falsy, so it was
    # replaced by a fresh dict and the "restore in place, so the caller
    # persists it" contract silently stopped holding -- for exactly the case
    # it exists for, an emptied definitions.json whose rows are all still in
    # archive.json with matching shas. Every one of them was billed again.
    have_defs = translations.get("definitions")
    have_exprs = translations.get("expressions")
    have_defs = {} if have_defs is None else have_defs
    have_exprs = {} if have_exprs is None else have_exprs
    restored = [] if restored is None else restored
    resume = {} if resume is None else resume
    resume.setdefault("definitions_already_redone", 0)
    resume.setdefault("expressions_already_redone", 0)
    resume.setdefault("done_provenance", done_provenance or None)
    # "clean_redo" plus whatever THIS configuration calls it: a row retired by a
    # redo must not come back even if the reason string was renamed. The check
    # matters most on the PLAIN path, which is the one an operator reaches for
    # after a crash.
    retired = tuple(dict.fromkeys(RETIRED_ARCHIVE_REASONS
                                  + (retranslate_reason,)))
    todo = []
    for eid in sorted(scope):
        e = entries.get(eid)
        if e is None:
            continue
        for s in e.get("senses", []):
            text = (s.get("definition") or "").strip()
            if not text:
                continue
            key = definition_key(eid, s)
            row = have_defs.get(key)
            if row is None and not retranslate_all and _restore_from_archive(
                    "definitions", key, s.get("src_sha"), have_defs, archive,
                    restored, retired):
                row = have_defs.get(key)
            reason = None
            if retranslate_all:
                if _already_redone(row, s.get("src_sha"), done_provenance):
                    resume["definitions_already_redone"] += 1
                    continue
                reason = retranslate_reason
            elif row is None:
                reason = "missing"
            elif row.get("src_sha") != s.get("src_sha"):
                reason = "src_sha_changed"
            if reason:
                todo.append({"key": key, "kind": "definition", "entry_id": eid,
                             "lemma": e.get("display_headword") or e.get("lemma"),
                             "pos_text": e.get("pos_text") or "",
                             "dannetid": s.get("dannetid"), "text": text,
                             "grammar": (s.get("grammar") or "").strip(),
                             "hint": "", "src_sha": s.get("src_sha"),
                             "reason": reason})
        for x in e.get("expressions", []):
            text = (x.get("expression") or "").strip()
            key = expression_key(x)
            if not text or not key:
                continue
            sha = expression_src_sha(x)
            row = have_exprs.get(key)
            if row is None and not retranslate_all and _restore_from_archive(
                    "expressions", key, sha, have_exprs, archive, restored,
                    retired):
                row = have_exprs.get(key)
            reason = None
            if retranslate_all:
                if _already_redone(row, sha, done_provenance):
                    resume["expressions_already_redone"] += 1
                    continue
                reason = retranslate_reason
            elif row is None:
                reason = "missing"
            elif row.get("src_sha") != sha:
                reason = "src_sha_changed"
            if reason:
                todo.append({"key": key, "kind": "expression", "entry_id": eid,
                             "lemma": e.get("display_headword") or e.get("lemma"),
                             "pos_text": e.get("pos_text") or "",
                             "dannetid": key, "text": text, "grammar": "",
                             "hint": expression_hint(x), "src_sha": sha,
                             "reason": reason})
    # An expression shared by two entries is ONE cell; keep the first sighting.
    seen, out = set(), []
    for row in todo:
        ident = (row["kind"], row["key"])
        if ident in seen:
            continue
        seen.add(ident)
        out.append(row)
    return out


def _group_by_entry(todo: list, kind: str, max_batch: int) -> list:
    """[(entry_id, [rows])] with at most max_batch rows per element, in a
    deterministic order."""
    by_entry: dict[str, list] = {}
    for row in todo:
        if row["kind"] == kind:
            by_entry.setdefault(row["entry_id"], []).append(row)
    batches = []
    for eid in sorted(by_entry):
        rows = by_entry[eid]
        for i in range(0, len(rows), max_batch):
            batches.append((eid, rows[i:i + max_batch]))
    return batches


# --------------------------------------------------------------------------
# the bill
# --------------------------------------------------------------------------

# The multiplier on the request count, recomputed from the ladders that
# actually exist after the 2026-08-26 patch. Two layers are left:
#
#   count-lock ladder (5)  x  transport ladder inside _generate (5)  =  25
#
# The two that used to be in here are gone: the automatic review pass (the "2 x
# expr_batches" below) and MAX_CORRECTION_ATTEMPTS (5), which together were the
# only 10x amplifier in the design. Review is a hand-run subcommand now.
#
# The label matters as much as the number: this ceiling INCLUDES transport
# retries, i.e. attempts that returned an error and produced no output. The
# audit's 25*def + 50*expr + 25*pos assumed the automatic review loop; it no
# longer applies.
REQUEST_CEILING_FACTOR = MAX_COUNT_LOCK_ATTEMPTS * MAX_RETRIES


def bill_row(todo: list, pos_todo: list) -> dict:
    defs = [r for r in todo if r["kind"] == "definition"]
    exprs = [r for r in todo if r["kind"] == "expression"]
    chars = sum(len(r["text"]) + len(r.get("grammar") or "") + len(r["hint"])
                for r in todo)
    def_calls = len(_group_by_entry(todo, "definition", MAX_DEFS_PER_BATCH))
    expr_batches = len(_group_by_entry(todo, "expression", MAX_EXPR_PER_BATCH))
    pos_calls = 1 if pos_todo else 0
    requests_min = def_calls + expr_batches + pos_calls
    return {
        "definitions": len(defs),
        "definitions_new": sum(1 for r in defs if r["reason"] == "missing"),
        "definitions_changed": sum(1 for r in defs if r["reason"] == "src_sha_changed"),
        "definitions_clean_redo": sum(1 for r in defs
                                      if r["reason"] == "clean_redo"),
        "expressions": len(exprs),
        "expressions_new": sum(1 for r in exprs if r["reason"] == "missing"),
        "expressions_changed": sum(1 for r in exprs if r["reason"] == "src_sha_changed"),
        "expressions_clean_redo": sum(1 for r in exprs
                                      if r["reason"] == "clean_redo"),
        "pos_keys": len(pos_todo),
        "pos_keys_list": pos_todo,
        "cells_total": len(todo),
        "entries_touched": len({r["entry_id"] for r in todo}),
        # Requests, not money. One request per definition batch, one per
        # expression batch (there is no automatic review pass any more), one for
        # the whole POS table.
        "definition_requests": def_calls,
        "expression_requests": expr_batches,
        "pos_requests": pos_calls,
        "requests_min": requests_min,
        "requests_max": requests_min * REQUEST_CEILING_FACTOR,
        "requests_max_basis": ("%d x (count-lock ladder %d x transport ladder %d); "
                              "INCLUDES transport retries, i.e. attempts that "
                              "returned an error and produced no output"
                              % (requests_min, MAX_COUNT_LOCK_ATTEMPTS,
                                 MAX_RETRIES)),
        "source_chars": chars,
        "source_tokens_estimate": math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE),
        # The Danish payload is a FRACTION of the input: the system prompt, the
        # schema, the output and (at MEDIUM) the thinking are all missing here.
        # Counting only this understated a clean redo by roughly 17x, which is
        # why the money stack computes the bill from the probe constants and
        # this field is explicitly labelled as one term of it.
        "source_tokens_estimate_note": ("Danish payload only: excludes the system "
                                        "prompt, the schema, the output tokens "
                                        "and thinking. NOT a bill."),
    }


# The three prices spec 4.2(1) requires the bill to print, and the one that is out
# of bounds:
#
#   cache_works    the expected path: one explicit cache per language, so the
#                  system prompt is charged at the cached input rate once per
#                  request instead of at the full rate.
#   lean_uncached  the fallback that must not be a surprise: the cache fails to
#                  attach and the CURRENT (1,135-token) prompt is inlined on
#                  every request.
#   rich_uncached  the forbidden zone: an enriched (~4.5k-token) prompt inlined
#                  on every request. No branch of this pipeline may run here;
#                  the figure exists so the number is on the page instead of in
#                  someone's head.
BILL_SCENARIOS = ("cache_works", "lean_uncached", "rich_uncached")
FORBIDDEN_SCENARIO = "rich_uncached"


def bill_requests(todo: list, pos_todo: list, lang: str) -> list:
    """One row per REQUEST that would be placed: its size, not its contents.

    This is what the bill file carries instead of the Danish source text. The
    old `cells` block quoted every DDO definition in scope -- 22,282 of them,
    7.6 MB, 28x the incremental bill -- to say something the counts already
    said. A bill has to be auditable, not a second copy of the corpus: per
    request, how many cells, how many characters of input, and the sha of the
    prompt it will be sent with.

    It also answers the money stack's actual question, which is per-request and
    per-n (spec 2.2) rather than per-language.
    """
    rows = []
    for kind, cap in (("definition", MAX_DEFS_PER_BATCH),
                      ("expression", MAX_EXPR_PER_BATCH)):
        sha = prompt_sha256(system_prompt(kind, lang))
        for eid, batch in _group_by_entry(todo, kind, cap):
            chars = sum(len(r["text"]) + len(r.get("grammar") or "")
                        + len(r["hint"]) for r in batch)
            rows.append({"kind": kind, "entry_id": eid, "n": len(batch),
                         "source_chars": chars,
                         "source_tokens_estimate":
                             math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE),
                         "prompt_sha256": sha})
    if pos_todo:
        chars = sum(len(k) for k in pos_todo)
        rows.append({"kind": "pos", "entry_id": None, "n": len(pos_todo),
                     "source_chars": chars,
                     "source_tokens_estimate":
                         math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE),
                     "prompt_sha256": prompt_sha256(system_prompt("pos", lang))})
    return rows


def bill_tokens(todo: list, pos_todo: list, lang: str,
                stats: dict | None) -> dict:
    """Tokens per pricing scenario, computed from MEASURED constants only.

    Tokens first, dollars second (spec 2.2). The token model, term by term:

      output           ceil(a*n + b) per request. MEASURED: 62 points, R2 0.985.
      thinking         requests x THINKING_PER_REQUEST_LOW (p95). MEASURED 0 --
                       and it has to be a measured 0, not an assumed one.
      definition input the measured total-prompt fit (pa*n + pb) minus the
                       measured system-prompt size: the payload+schema half
                       that stays uncached when the cache works.
      expression input ESTIMATED. The expression prompt's size has never been
                       probed, so its payload is source_chars / 4 and its system
                       half is priced at the definition prompt's measured size.
                       Said out loud in `basis` rather than blended into a
                       number that would look measured.

    Returns {"available": False, "why": ...} rather than a plausible number when
    a constant is missing.
    """
    if not stats:
        return {"available": False,
                "why": "no measured constants on disk; the bill cannot be "
                       "priced from anything else"}
    pfit = prompt_token_fit(stats)
    try:
        fit = output_fit(stats=stats)
    except FatalError as exc:
        return {"available": False, "why": str(exc)}
    lean = system_prompt_tokens(stats, lang)
    rich = rich_prompt_tokens(stats)
    think = thinking_per_request(stats, "LOW", "p95")
    missing = [name for name, value in
               (("PROMPT_TOKENS_fit", pfit),
                ("PROMPT_TOKENS_system_only.%s" % lang, lean),
                ("wave2.W2_2_rich.cached", rich),
                ("thinking.THINKING_PER_REQUEST_LOW.p95", think))
               if value is None]
    if missing:
        return {"available": False,
                "why": "missing measured constant(s): %s" % ", ".join(missing)}
    batches = (("definition", _group_by_entry(todo, "definition",
                                             MAX_DEFS_PER_BATCH)),
               ("expression", _group_by_entry(todo, "expression",
                                              MAX_EXPR_PER_BATCH)))
    out: dict = {"available": True, "thinking_per_request_p95": think,
                 "system_tokens_lean": lean, "system_tokens_rich": rich,
                 "basis": ("output and definition input are MEASURED "
                           "(EXPECTED_OUTPUT, PROMPT_TOKENS_fit, "
                           "PROMPT_TOKENS_system_only); expression and pos "
                           "input are ESTIMATED at %d chars/token, with the "
                           "definition prompt's measured size standing in for "
                           "an expression prompt nobody has probed"
                           % CHARS_PER_TOKEN_ESTIMATE)}
    for scenario in BILL_SCENARIOS:
        system = rich if scenario == "rich_uncached" else lean
        cached = uncached = output = requests = 0
        for kind, groups in batches:
            for _eid, batch in groups:
                n = len(batch)
                requests += 1
                output += expected_output_tokens(n, fit)
                if kind == "definition":
                    payload = max(0, math.ceil(pfit[0] * n + pfit[1]) - lean)
                else:
                    chars = sum(len(r["text"]) + len(r.get("grammar") or "")
                                + len(r["hint"]) for r in batch)
                    payload = math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE)
                if scenario == "cache_works":
                    cached += system
                    uncached += payload
                else:
                    uncached += system + payload
        if pos_todo:
            requests += 1
            payload = math.ceil(sum(len(k) for k in pos_todo)
                                / CHARS_PER_TOKEN_ESTIMATE)
            output += expected_output_tokens(len(pos_todo), fit)
            if scenario == "cache_works":
                cached += system
                uncached += payload
            else:
                uncached += system + payload
        out[scenario] = {"requests": requests,
                         "cached_input_tokens": cached,
                         "uncached_input_tokens": uncached,
                         "output_tokens": output,
                         "thinking_tokens": int(math.ceil(requests * think))}
    return out


def dollar_figures(tokens: dict, rates=None, ceiling_usd=None) -> dict:
    """The three dollar figures spec 4.2(1) wants on the bill, plus the ceiling.

    `rates` is the money stack's rate card for (model, mode), as a mapping:

        {"input_usd_per_mtok": float,
         "cached_input_usd_per_mtok": float,
         "output_usd_per_mtok": float}

    Without it every figure is None and `why` says so. A made-up price on a bill
    is worse than a missing one, because the missing one gets read.
    """
    out: dict = {"ceiling_usd": ceiling_usd,
                 "forbidden": FORBIDDEN_SCENARIO,
                 "forbidden_note": ("an enriched prompt inlined on every "
                                    "request. No branch may run here -- the "
                                    "cache is the whole point of enriching."),
                 "rates": dict(rates) if isinstance(rates, dict) else None}
    needed = ("input_usd_per_mtok", "cached_input_usd_per_mtok",
              "output_usd_per_mtok")
    if not tokens.get("available"):
        out["why"] = tokens.get("why", "no token model")
    elif not isinstance(rates, dict) or any(rates.get(k) is None
                                            for k in needed):
        out["why"] = ("no rate card: prices.rate_card(model, mode) must return "
                      "%s. Tokens are counted; dollars are not asserted."
                      % ", ".join(needed))
    else:
        for scenario in BILL_SCENARIOS:
            t = tokens[scenario]
            out[scenario] = round(
                (t["uncached_input_tokens"] * rates["input_usd_per_mtok"]
                 + t["cached_input_tokens"] * rates["cached_input_usd_per_mtok"]
                 + (t["output_tokens"] + t["thinking_tokens"])
                 * rates["output_usd_per_mtok"]) / 1e6, 4)
        if ceiling_usd is not None:
            out["over_ceiling"] = sorted(s for s in BILL_SCENARIOS
                                         if out[s] > ceiling_usd)
        return out
    for scenario in BILL_SCENARIOS:
        out[scenario] = None
    return out


def rate_card_for(cfg: Config) -> tuple:
    """(rates, note) from the money stack's optional prices module.

    This stage does not own prices and must not invent them, so the import is
    soft: `from ..prices import rate_card` when it exists, a stated absence when
    it does not.
    """
    try:                                     # the money stack, when it lands
        from ..prices import rate_card
    except Exception as exc:                 # noqa: BLE001 - optional module
        return None, "no rate card module yet (%s: %s)" % (type(exc).__name__,
                                                           exc)
    try:
        rates = rate_card(cfg.gemini_model, cfg.mode)
    except Exception as exc:                 # noqa: BLE001 - their code
        return None, "prices.rate_card raised %s: %s" % (type(exc).__name__, exc)
    if isinstance(rates, dict):
        return rates, "prices.rate_card(%s, %s)" % (cfg.gemini_model, cfg.mode)
    return None, ("prices.rate_card returned %s, not a mapping of "
                  "input_usd_per_mtok / cached_input_usd_per_mtok / "
                  "output_usd_per_mtok" % type(rates).__name__)


def print_bill(bill: dict, model: str, expr_model: str | None = None,
               mode: str = "standard", thinking_level: str = "LOW",
               prompt_id: str = "", confirmed: bool = False) -> None:
    """ALWAYS runs. No price is INVENTED here: the three dollar figures come
    from the measured token model plus the money stack's rate card, and when
    there is no rate card the line says so instead of guessing."""
    if expr_model and expr_model != model:
        print("--- translation bill (definitions: %s | expressions + pos: %s) ---"
              % (model, expr_model))
    else:
        print("--- translation bill (model: %s) ---" % model)
    print("  mode %s | thinking %s | prompt %s"
          % (mode, thinking_level, prompt_id or "?"))
    total = 0
    for lang in sorted(bill):
        r = bill[lang]
        total += r["cells_total"]
        print("  %-8s %5d cells  (definitions %d: %d new / %d changed / %d redo "
              "| expressions %d: %d new / %d changed / %d redo | pos keys %d)"
              % (lang, r["cells_total"], r["definitions"], r["definitions_new"],
                 r["definitions_changed"], r.get("definitions_clean_redo", 0),
                 r["expressions"], r["expressions_new"],
                 r["expressions_changed"], r.get("expressions_clean_redo", 0),
                 r["pos_keys"]))
        print("           %d entries, %d-%d API requests, ~%d source tokens"
              % (r["entries_touched"], r["requests_min"], r["requests_max"],
                 r["source_tokens_estimate"]))
        resume = r.get("resume") or {}
        if resume.get("definitions_already_redone") \
                or resume.get("expressions_already_redone"):
            print("           RESUME: %d definition + %d expression cell(s) "
                  "already carry this run's provenance and are NOT billed "
                  "again; %d cell(s) remain"
                  % (resume["definitions_already_redone"],
                     resume["expressions_already_redone"], r["cells_total"]))
        money = r.get("dollars") or {}
        if money.get("why"):
            print("           dollars: %s" % money["why"])
        else:
            print("           dollars: cache-works $%s | LEAN uncached $%s | "
                  "RICH uncached $%s (FORBIDDEN) | ceiling $%s"
                  % (money.get("cache_works"), money.get("lean_uncached"),
                     money.get("rich_uncached"), money.get("ceiling_usd")))
            if money.get("over_ceiling"):
                print("           OVER THE CEILING: %s"
                      % ", ".join(money["over_ceiling"]))
    print("  TOTAL %d cells across %d language(s)" % (total, len(bill)))
    print("  request ceiling includes transport retries; source tokens are the "
          "Danish payload only, not a bill")
    if confirmed:
        print("  --confirm-spend IS SET: the calls below are real.")
    else:
        print("  nothing has been sent. Re-run with --confirm-spend to place "
              "calls.")


# --------------------------------------------------------------------------
# GC (offline, free, and never deletes)
# --------------------------------------------------------------------------

def live_keys(entries: dict, scope=None) -> tuple[set, set]:
    scope = set(entries) if scope is None else set(scope)
    defs, exprs = set(), set()
    for eid in scope:
        e = entries.get(eid)
        if e is None:
            continue
        for s in e.get("senses", []):
            defs.add(definition_key(eid, s))
        for x in e.get("expressions", []):
            if x.get("dannetid"):
                exprs.add(x["dannetid"])
    return defs, exprs


def gc(cfg: Config, lang: str, entries: dict | None = None, scope=None) -> dict:
    """Move rows whose key has no live sense into archive.json.

    Archived, never deleted (guide D7): a sense that comes back after a DDO
    edit must not be paid for twice.
    """
    if entries is None:
        entries = read_json(cfg.json_dir / "entries.json")
    tdir = cfg.json_dir / "translations" / lang
    defs = read_json(tdir / "definitions.json", default={})
    exprs = read_json(tdir / "expressions.json", default={})
    archive = read_json(tdir / "archive.json",
                        default={"definitions": {}, "expressions": {}})
    live_defs, live_exprs = live_keys(entries, scope)
    moved = {"definitions": 0, "expressions": 0}
    for k in [k for k in defs if k not in live_defs]:
        archive.setdefault("definitions", {})[k] = defs.pop(k)
        moved["definitions"] += 1
    for k in [k for k in exprs if k not in live_exprs]:
        archive.setdefault("expressions", {})[k] = exprs.pop(k)
        moved["expressions"] += 1
    if moved["definitions"] or moved["expressions"]:
        write_json(tdir / "definitions.json", defs)
        write_json(tdir / "expressions.json", exprs)
        write_json(tdir / "archive.json", archive)
    orphans = sorted(set(defs) - live_defs) + sorted(set(exprs) - live_exprs)
    return {"archived": moved,
            "archive_total": {"definitions": len(archive.get("definitions", {})),
                              "expressions": len(archive.get("expressions", {}))},
            "rows_live": {"definitions": len(defs), "expressions": len(exprs)},
            "orphans_remaining": orphans,
            "definitions": defs, "expressions": exprs, "archive": archive}


def orphans_gate(per_lang: dict):
    """G-ORPH. Coverage of 100% passes happily on dead rows, so the two gates
    only mean something together."""
    bad = {lg: s["orphans_remaining"][:20] for lg, s in per_lang.items()
           if s.get("orphans_remaining")}
    return not bad, {"per_language": {lg: {"archived": s["archived"],
                                           "live": s["rows_live"]}
                                     for lg, s in per_lang.items()},
                     "violations": bad}


# --------------------------------------------------------------------------
# prompts, ported verbatim from the v2.1 scripts
# --------------------------------------------------------------------------

def definition_schema(n_defs: int) -> dict:
    """05: the count lock. minItems == maxItems == n is what made a silent
    truncation impossible, and it stays exactly as it is.

    A short array is not "a wrong answer": zip(rows, got) shifts every gloss
    onto the wrong definition from the missing one onwards, which is
    catastrophic and invisible. The only addition (2026-08-26) is minLength on
    the two strings, about 8 tokens, against an empty cell.

    Deliberately NO `description` fields on this schema or the expression one:
    they are pure output tokens on every request and the prompt already says it.
    """
    return {
        "type": "object",
        "properties": {
            "headword": {"type": "string"},
            "definitions": {
                "type": "array",
                "minItems": n_defs,
                "maxItems": n_defs,
                "items": {
                    "type": "object",
                    "properties": {"lemma": {"type": "string", "minLength": 1},
                                   "gloss": {"type": "string", "minLength": 1}},
                    "required": ["lemma", "gloss"],
                },
            },
        },
        "required": ["headword", "definitions"],
    }


def definition_prompt(lang: str) -> str:
    """PROMPT V4, ported from 05_translate_definitions.py. The 22,734 existing
    cells per language were produced by this text; changing a word makes the new
    cells stylistically foreign to the old ones.

    ONE EDIT, and it is the whole point of this function's signature (2026-08-26,
    +50 characters): the last line used to interpolate n_defs, so the "constant"
    system prompt was a different string for every batch size -- 30 measured
    payloads produced 7 distinct sha256 values, one per value of n, differing by
    one or two characters. An explicit cache is keyed on exact content, so that
    made one-cache-per-language impossible while looking like a formatting
    detail. The count is stated in the user message and enforced by the schema's
    minItems == maxItems, which is where it was always actually enforced.

    Any prompt function reachable from a cached request must therefore depend on
    the LANGUAGE ONLY. test_prompt_is_constant is the thing that keeps it true.
    """
    critical_rule = ""
    if lang.lower() != "english":
        critical_rule = f"""
    **CRITICAL RULE: Both "lemma" and "gloss" MUST be in {lang}. Do NOT use English in your output, unless there is absolutely no way to express a concept in {lang}.**
    """

    system_prompt_base = f"""
    You are a senior lexicographer creating a new Danish-{lang} dictionary specifically for **language learners**. Your translations must be clear, practical, and intuitive.

    ### CORE TASK
    For each Danish definition provided, you must generate a corresponding `lemma` and `gloss` in **{lang}**.

    ### RULES FOR "lemma" (The {lang} headword):
    1.  **Be a real word/phrase:** It must be a concise, common {lang} word or a widely used, natural-sounding fixed phrase.
    2.  **Be practical, not overly technical:** AVOID purely grammatical descriptions. A conceptual hint is better than a niche technical term. If a grammatical term is widely understood by learners in the context of {lang} (e.g., a {lang} term for "noun suffix" when dealing with a suffix that forms nouns), it can be acceptable if truly the best and most concise fit.
    3.  **Balance specificity, naturalness, and informativeness. Focus on DEFINITION'S CORE SEMANTICS:**
        a.  **Capture Key Nuances:** The `lemma` should accurately reflect the specific and defining characteristics of THAT particular Danish definition.
        b.  **Prioritize Natural & Common Usage:** Choose {lang} words/phrases that are natural and commonly used by native speakers.
        c.  **Handling Multiple Meanings of Same Headword:** If multiple Danish definitions of the *same Danish headword* naturally map to the *same core {lang} lemma* (because it genuinely covers those nuances), you MAY reuse that lemma. Rely on distinct `glosses` for precise differentiation.
        d.  **When to Be More Specific (Informativeness):** However, if a Danish definition highlights a *distinct, crucial aspect* (e.g., a specific skill, a combined action, or a specialized context) that a very general {lang} lemma would obscure, prefer a *slightly more specific (yet still natural and common)* {lang} lemma that better conveys this key information. For example, if a definition describes "kicking a ball with light force AND high control", a lemma reflecting both aspects is better than one reflecting only "light kicking" if a natural {lang} term for the combined concept exists.
        e.  **Avoid Over-Simplification & Literal Traps:** Do NOT over-simplify the lemma to the point of losing essential semantic information from the Danish definition. Also, be cautious of being overly influenced by the literal translation of the *Danish headword itself* if the *definition* points to a more nuanced or specific meaning in {lang}. The definition is paramount.
        f.   **Clarity for Learners:** Ultimately, the lemma (in conjunction with the gloss) should provide maximum clarity for a language learner.
    4.  **Conciseness preferred:** Aim for brevity. Single words or short, common phrases in {lang} are ideal, but not at the cost of Rule 3's requirements.

    ### RULES FOR "gloss" (The explanatory translation):
    1.  **Be a clear, complete thought:** It must be a grammatically correct and natural-sounding sentence or a very clear, self-contained explanatory phrase in {lang}. Full sentences are preferred, but clarity and conciseness for the learner are paramount.
    2.  **Be explanatory, focused, AND faithful:** It should clearly explain the meaning and usage context relevant to THAT specific definition, being faithful to the scope and nuances of the Danish original. AVOID over-explaining with information not directly implied by the Danish definition, but ensure all key semantic components of the Danish definition are represented in the {lang} gloss. Focus on direct translation and essential clarification in {lang}.
    3.  **Match formality:** The tone of the gloss in {lang} should generally match the formality of the Danish definition.

    ### OUTPUT FORMAT
    Return pure JSON that **exactly matches the response schema**. The `definitions` array MUST have exactly as many objects as the user message states.
    """

    unified_example = f"""
    ### EXAMPLE OF STRUCTURE (target language is {lang}):
    Input Headword: "spille" (example headword, not necessarily from your data)
    Input Definitions: {{
        "0": "udføre musik på et instrument",
        "1": "deltage i et spil for fornøjelses skyld",
        "2": "i fodbold, aflevere bolden til en medspiller med præcis kontrol og ofte for at skabe en scoringsmulighed"
    }}

    Expected JSON Output Structure:
    {{
    "headword":"spille",
    "definitions":[
    {{
        "lemma":"[A {lang} lemma for 'to play music']",
        "gloss":"[A {lang} gloss explaining playing music on an instrument]."
    }},
    {{
        "lemma":"[A {lang} lemma for 'to play a game']",
        "gloss":"[A {lang} gloss explaining participating in a game for fun]."
    }},
    {{
        "lemma":"[A {lang} lemma for 'to pass with control (football)', informative and natural phrase]",
        "gloss":"[A {lang} gloss explaining the football-specific action of passing with control to create an opportunity, ensuring all key elements are covered]."
    }}
    ]}}
    """
    return f"{critical_rule}\n{system_prompt_base.strip()}\n\n{unified_example.strip()}"


def expression_schema(n_items: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "fixed_expressions": {
                "type": "array",
                "minItems": n_items,
                "maxItems": n_items,
                "items": {
                    "type": "object",
                    "properties": {"lemma": {"type": "string", "minLength": 1},
                                   "gloss": {"type": "string", "minLength": 1}},
                    "required": ["lemma", "gloss"],
                },
            }
        },
        "required": ["fixed_expressions"],
    }


def review_schema() -> dict:
    """The 2025 LLM inspector's schema. NOTHING CALLS THIS ANY MORE.

    The inspector was one extra paid call per expression batch (and up to five
    more when it said no) to answer "does this text contain characters outside
    the target language" -- a character-class question, which the offline script
    gate answers for free on the whole corpus. Kept, unused, because the
    sampled-semantic-review idea would reuse this shape; deleted the day it is
    clear nobody wants it.
    """
    return {
        "type": "object",
        "properties": {
            "contains_other_languages": {"type": "boolean"},
            "detected_words": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["contains_other_languages", "detected_words"],
    }


def expression_prompt(lang: str) -> str:
    """06's generator prompt. The Russian clause exists because of a real
    contamination incident; do not remove it.

    Two edits (2026-08-26), for the same reason as definition_prompt:

      * `n_items` is gone from the last line -- the count is in the user message
        and in the schema, and interpolating it here made the "constant" prompt
        one string per batch size.
      * the correction instruction is gone from the signature. It used to be
        PREPENDED to the system prompt, which changes the cached prefix and so
        forfeits the cache for exactly the requests that are being redone. It is
        appended to the USER message now (see expression_user_payload), and the
        automatic review-and-correct loop that generated it no longer exists:
        review is a hand-run subcommand (owner decision D-08).
    """
    system_prompt_base = f"""
You are a senior lexicographer translating Danish fixed expressions and idioms into {lang} for a dictionary aimed at **language learners**.
### CORE TASK
For each Danish expression in the input, provide a `lemma` and a `gloss` in {lang}.
### CONTEXT PROVIDED
- `expr`: The Danish expression to translate.
- `hint`: An optional Danish definition to clarify the expression's meaning. Use this hint to find the best translation, but do not translate the hint itself.
### RULES FOR "lemma":
- It must be the most natural and common equivalent idiom or phrase in {lang}.
- If a direct idiom exists, prefer it. If not, use a concise, descriptive phrase.
### RULES FOR "gloss":
- It must be a full, explanatory sentence in {lang} that clarifies the meaning and usage of the Danish expression for a learner.
### OUTPUT FORMAT
Return pure JSON matching the schema. The `fixed_expressions` array MUST have exactly as many objects as the user message states.
"""
    critical_rules = ""
    if lang.lower() != "english":
        critical_rules = f"""
**CRITICAL RULES: These are non-negotiable.**
1.  The primary language for both "lemma" and "gloss" MUST be **{lang}**.
2.  You must avoid all other languages. **DO NOT USE RUSSIAN under any circumstances.**
3.  **AS A LAST RESORT, ONLY IF** a concept is truly untranslatable into {lang}, you are permitted to use a concise English word or phrase in the `gloss`. This should be extremely rare. Never use English in the `lemma`.
"""
    return f"{critical_rules.strip()}\n\n{system_prompt_base.strip()}"


# --------------------------------------------------------------------------
# user payloads -- everything that varies per request lives HERE
# --------------------------------------------------------------------------

def definition_user_payload(entry_label: str, rows: list,
                            correction: str = "") -> str:
    """The Danish side of one definition request.

    Three things travel here and nowhere else: the headword label, the count
    (the schema enforces it, the prompt refers to it), and DDO's per-sense
    `grammar` note. The grammar block is emitted only when at least one row has
    one, so an entry without grammar produces byte-identical bytes to before.
    """
    payload = {str(i): r["text"] for i, r in enumerate(rows)}
    grammar = {str(i): (r.get("grammar") or "").strip()
               for i, r in enumerate(rows) if (r.get("grammar") or "").strip()}
    out = ('Headword: "%s"\n'
           "Expecting exactly %d definition objects.\n"
           "Input Definitions JSON:\n%s"
           % (entry_label, len(payload),
              json.dumps(payload, ensure_ascii=False, indent=2)))
    if grammar:
        out += ("\nGrammar notes from the source dictionary, keyed by the same "
                "index (valency frames and number/register labels; translate "
                "the definition, not the note):\n%s"
                % json.dumps(grammar, ensure_ascii=False, indent=2))
    if correction:
        # Appended, never prepended to the system prompt: the system prompt is
        # the cached prefix, and a per-batch instruction in front of it forfeits
        # the discount on exactly the requests being redone.
        out += "\n\nIMPORTANT CORRECTION: %s" % correction
    return out


def expression_user_payload(lang: str, entry_label: str, rows: list,
                            correction: str = "") -> str:
    """The Danish side of one expression request.

    `correction` is APPENDED, never prepended to the system prompt: the system
    prompt is the cached prefix, and putting a per-batch instruction in front of
    it forfeits the cache on precisely the requests being redone.
    """
    payload = {str(i): {"expr": r["text"], "hint": r["hint"]}
               for i, r in enumerate(rows)}
    out = ('Headword: "%s"\n'
           "Please translate the following %d fixed expressions into %s:\n%s"
           % (entry_label, len(payload), lang,
              json.dumps(payload, ensure_ascii=False, indent=2)))
    if correction:
        out += "\n\nIMPORTANT CORRECTION: %s" % correction
    return out


def prompt_sha256(text: str) -> str:
    """The identity of a prompt. An explicit cache is keyed on exact content, so
    this is also the identity of the cache -- and of the measured constants,
    which are only valid for the prompt they were measured on."""
    return sha256_str(text)


# The system prompt for one kind of request. THE ONLY PLACE that mapping is
# written down.
#
# It exists because the bill and the wire had two independent copies of it. The
# bill file computed prompt_sha256(definition_prompt(lang)) at its own call site
# while the request built its system instruction at another, so replacing the
# prompt builder at one of them (which is exactly what the prompt-pack work
# does) would have left G-PROMPT comparing a stale sha to itself and reporting
# agreement: a gate that certifies the thing it is not looking at.
_SYSTEM_PROMPTS = {"definition": lambda lang: definition_prompt(lang),
                   "expression": lambda lang: expression_prompt(lang),
                   "pos": lambda lang: pos_prompt(lang)}


def system_prompt(kind: str, lang: str) -> str:
    """The system prompt that a `kind` request for `lang` is sent with."""
    builder = _SYSTEM_PROMPTS.get(kind)
    if builder is None:
        raise FatalError(
            "no system prompt is registered for request kind %r (known: %s). A "
            "kind whose prompt is built somewhere else cannot be checked "
            "against the bill." % (kind, ", ".join(sorted(_SYSTEM_PROMPTS))))
    return builder(lang)


def prompt_shas(lang: str) -> dict:
    """{"definition": sha, "expression": sha} for one language.

    The bill's prompt_sha256 block and doctor's printout both come from here, so
    they cannot drift from what CallContext.request() puts on the wire.
    """
    return {kind: prompt_sha256(system_prompt(kind, lang))
            for kind in ("definition", "expression")}


def review_prompt(lang: str, json_string: str) -> str:
    """06's reviewer. English gets its own wording, as in the original.

    NOTHING CALLS THIS ANY MORE -- see review_schema. The hand-run correction
    pass uses the ordinary generator prompt plus a correction in the user
    message, because a redo has to produce a translation, not a verdict.
    """
    if lang.lower() == "english":
        allowed = "the English language"
    else:
        allowed = f"the {lang} or English languages"
    return f"""
You are a meticulous language quality inspector. Your only task is to detect if the lemma part of the following text contains any **CHARACTERS** that are NOT part of {allowed}.

- You MUST ignore JSON structure characters like {{, }}, " and keys like "lemma", "gloss".
- If no other languages are found, "contains_other_languages" must be false and "detected_words" must be an empty list.

Text to inspect:
---
{json_string}
---
"""


def pos_schema(tags: list, lang: str) -> dict:
    """04: the schema pins the exact key set, which is the only check that
    matters -- the exporter groups the card front by pos_key and a missing key
    silently un-labels a whole group."""
    properties = {tag: {"type": "string",
                        "description": f"The {lang} translation for the Danish POS tag '{tag}'."}
                  for tag in tags}
    return {"type": "object", "properties": properties,
            "required": list(properties.keys())}


def pos_prompt(lang: str) -> str:
    return f"""
You are a linguistic assistant specializing in lexicography. Your task is to translate a list of Danish part-of-speech (POS) abbreviations into their full, clear equivalents in {lang}.

### INSTRUCTIONS
- For each key in the input JSON, provide its translation as the value.
- The translation should be the common term for that part-of-speech in {lang}.
- Adhere strictly to the JSON output format. The output MUST be a single JSON object with the exact same keys as the input, and all values must be strings.
- If the target language is Chinese, DO NOT include any pinyin or other phonetic transcriptions.
"""


# --------------------------------------------------------------------------
# the API layer (only ever reached with confirm=True)
# --------------------------------------------------------------------------

# Every measured constant the money math or the request sizing CONSUMES. The
# rule "nothing measured is hard-coded" only bites if the reader refuses to run
# without the values: before this list existed, probe_stats() checked the file,
# the model and EXPECTED_OUTPUT.a/b and nothing else, so a stats.json with no
# cache floor and no thinking constant sailed straight through
# `translate --confirm-spend` and only `doctor` (which translate does not call)
# complained.
REQUIRED_STATS_KEYS = (
    ("EXPECTED_OUTPUT.a", "the output fit's slope; sizes every request"),
    ("EXPECTED_OUTPUT.b", "the output fit's intercept"),
    ("thinking.THINKING_PER_REQUEST_LOW",
     "thinking tokens per request at LOW -- measured 0, and it has to be a "
     "measured 0 rather than an assumed one"),
    ("wave2.EXPLICIT_CACHE_FLOOR",
     "the minimum cacheable prompt; the discount path depends on it"),
    ("budget.MAX_OUTPUT_FORMULA",
     "the formula the cap is derived from, in the probe's own words"),
)


def _stats_get(stats: dict, dotted: str):
    node = stats
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def missing_stats_keys(stats: dict) -> list:
    """Which of REQUIRED_STATS_KEYS this artifact does not carry.

    One implementation, used by both the spend gate (probe_stats) and doctor, so
    the command that says "fit to spend" and the stage that spends cannot
    disagree about what fit means.
    """
    return [key for key, _why in REQUIRED_STATS_KEYS
            if _stats_get(stats, key) in (None, {}, [])]


def probe_stats(cfg: Config) -> dict:
    """The measured LLM constants, from disk.

    NOTHING measured is hard-coded in this package. A missing file, a model that
    does not match the configured one, or a missing key is a REFUSAL, not a
    warning and not a default: every one of those states means the number that
    would be used was measured on something else.
    """
    path = cfg.probe_stats_path
    if not path.exists():
        raise FatalError(
            "no measured LLM constants at %s. The expected-output fit, the "
            "thinking constant and the cache floor are all measured values; "
            "there is no default for them. Run the probe suite (or copy its "
            "stats.json into place) before spending." % path)
    stats = read_json(path)
    model = stats.get("model")
    if model != cfg.gemini_model:
        raise FatalError(
            "%s was measured on model %r but the configured model is %r. Every "
            "constant in that file is a property of the model that produced it."
            % (path, model, cfg.gemini_model))
    missing = missing_stats_keys(stats)
    if missing:
        why = dict(REQUIRED_STATS_KEYS)
        raise FatalError(
            "%s is missing measured constant(s) the spend consumes: %s. A "
            "missing key is not a zero and not a default -- it means nobody "
            "measured it on %s."
            % (path, "; ".join("%s (%s)" % (k, why[k]) for k in missing),
               cfg.gemini_model))
    return stats


def thinking_per_request(stats: dict, level: str = "LOW",
                        stat: str = "p95") -> float | None:
    """The measured thinking tokens per request at one level, or None.

    p95 by default, not mean: the bill is a ceiling a human accepts in advance,
    and half the requests being above the number they accepted is not a
    ceiling. At LOW every one of the 38 observations was 0, so p95 and mean
    agree there -- the distinction exists for the levels where they do not.
    """
    node = _stats_get(stats, "thinking.THINKING_PER_REQUEST_%s" % level)
    if isinstance(node, dict):
        value = node.get(stat)
        return None if value is None else float(value)
    return None if node is None else float(node)


def prompt_token_fit(stats: dict) -> tuple | None:
    """(a, b) of the measured TOTAL prompt-token fit, a*n + b.

    Measured on the definition wave: system prompt + schema + n Danish payload
    rows. Subtracting the system half gives the per-request payload cost, which
    is what stays uncached when the cache works.
    """
    fit = stats.get("PROMPT_TOKENS_fit") or {}
    a, b = fit.get("a"), fit.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return float(a), float(b)


def system_prompt_tokens(stats: dict, lang: str) -> int | None:
    """The measured token count of the LEAN system prompt for one language."""
    value = (stats.get("PROMPT_TOKENS_system_only") or {}).get(lang)
    return int(value) if isinstance(value, (int, float)) else None


def rich_prompt_tokens(stats: dict) -> int | None:
    """The measured token count of the RICH prompt (the 4.5k enrichment probe).

    Only ever used to price the FORBIDDEN scenario: a rich prompt inlined on
    every request with no cache. Nothing is allowed to run there.
    """
    value = _stats_get(stats, "wave2.W2_2_rich.cached")
    return int(value) if isinstance(value, (int, float)) else None


def output_fit(cfg: Config | None = None, stats: dict | None = None) -> tuple:
    """(a, b) of the measured candidates ~= a*n + b fit.

    Measured: a = 35.964, b = 23.07, R^2 = 0.985 over 62 points -- but the
    numbers come off the disk, because that is the only way a re-measurement can
    reach the code that sizes requests.
    """
    if stats is None:
        if cfg is None:
            raise FatalError("output_fit needs either a Config or a stats dict")
        stats = probe_stats(cfg)
    fit = stats.get("EXPECTED_OUTPUT") or {}
    a, b = fit.get("a"), fit.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise FatalError(
            "EXPECTED_OUTPUT.a / EXPECTED_OUTPUT.b missing from the measured "
            "constants; the output cap cannot be derived without them.")
    return float(a), float(b)


def expected_output_tokens(n: int, fit: tuple) -> int:
    """ceil(a*n + b). The output side of the bill, and the basis of the cap."""
    a, b = fit
    return math.ceil(a * n + b)


def max_output_tokens(n: int, fit: tuple, floor: int = 1024,
                      ceiling: int = MODEL_OUTPUT_CEILING) -> int:
    """The truncation guard for a DEFINITION request with n cells in it.

    ceil(a*n + b) * 1.5, WITH NO THINKING TERM -- at thinkingLevel=LOW the
    derived thinking was 0 in 38 observations, including on a 4.5k prompt, so
    there is nothing to reserve for it.

    `floor` is not padding and it does not cost anything: billing is on tokens
    actually produced, and 1024 is the lowest cap that was measured end to end
    (n=8 produced 250-307 output tokens under it). Below it is untested. The old
    flat 8192 was not wrong so much as uninformative -- it could not distinguish
    "the model stopped early" from "the cap stopped it".

    THE HEADROOM IS THIN AT THE TOP OF THE RANGE, and only definitions were
    measured. Observed maximum candidate counts against the cap this formula
    derives:

        n=8    408 observed / 1024 cap = 2.51x
        n=12   498 observed / 1024 cap = 2.06x
        n=20   783 observed / 1115 cap = 1.42x

    1.42x is the real safety margin on a full batch, not the 1.5 factor. That is
    why MEASURED_OUTPUT_KINDS exists (an expression batch's output distribution
    was never measured and its glosses are whole sentences) and why _generate
    raises the budget once on a MAX_TOKENS finish instead of aborting a paid
    wave over a cap.
    """
    derived = math.ceil(expected_output_tokens(n, fit) * MAX_OUTPUT_SAFETY_FACTOR)
    return min(ceiling, max(floor, derived))


def resolve_max_output(cfg: Config, n: int, fit: tuple,
                       kind: str = "definition") -> int:
    """The cap for one request. Measured kinds get the fit; the rest get a flat one.

    cfg.max_output_tokens pins one cap for every request (an investigation
    setting). Otherwise: a DEFINITION request is sized from the measured fit,
    and every other kind gets cfg.max_output_unmeasured, because there is no
    measurement to derive anything from. A flat generous cap costs nothing
    (billing is on tokens produced) and a wrong derived cap costs a truncated
    paid call, so the asymmetry decides it.
    """
    if cfg.max_output_tokens:
        return min(MODEL_OUTPUT_CEILING, int(cfg.max_output_tokens))
    if kind not in MEASURED_OUTPUT_KINDS:
        return min(MODEL_OUTPUT_CEILING, max(int(cfg.max_output_floor),
                                             int(cfg.max_output_unmeasured)))
    return max_output_tokens(n, fit, floor=int(cfg.max_output_floor))


def derived_thinking(usage) -> int:
    """total - prompt - candidates - toolUse.

    NEVER read thoughtsTokenCount: protobuf omits zero-valued fields, so the key
    is absent exactly when thinking is 0 and present when it is not -- which
    means "field missing" and "field zero" are indistinguishable at the one place
    where the difference is the whole question. The identity was verified to
    hold on both arms of a LOW/MEDIUM pair.
    """
    total = _usage_int(usage, "total_token_count", "totalTokenCount")
    prompt = _usage_int(usage, "prompt_token_count", "promptTokenCount")
    cand = _usage_int(usage, "candidates_token_count", "candidatesTokenCount")
    tools = _usage_int(usage, "tool_use_prompt_token_count",
                       "toolUsePromptTokenCount")
    return max(0, total - prompt - cand - tools)


def _usage_int(usage, snake: str, camel: str) -> int:
    """usageMetadata reaches us as an SDK object on the interactive path and as
    plain JSON on the batch path. One reader for both."""
    if usage is None:
        return 0
    for name in (snake, camel):
        if isinstance(usage, dict):
            if usage.get(name) is not None:
                return int(usage[name])
        elif getattr(usage, name, None) is not None:
            return int(getattr(usage, name))
    return 0


def normalize_usage(usage, *, model: str, label: str, kind: str = "",
                    mode: str = "standard", cache_name: str | None = None,
                    prompt_id: str = "", finish_reason: str = "",
                    n_expected: int | None = None) -> dict:
    """One usage row, in the shape the ledger and the gates read.

    Used by BOTH transports on purpose: the derived-thinking rule and the
    "cached is a SUBSET of prompt, never an addend" rule have to be implemented
    once or they will disagree.
    """
    prompt = _usage_int(usage, "prompt_token_count", "promptTokenCount")
    cached = _usage_int(usage, "cached_content_token_count",
                        "cachedContentTokenCount")
    cand = _usage_int(usage, "candidates_token_count", "candidatesTokenCount")
    total = _usage_int(usage, "total_token_count", "totalTokenCount")
    tools = _usage_int(usage, "tool_use_prompt_token_count",
                       "toolUsePromptTokenCount")
    return {
        "label": label, "kind": kind, "model": model, "mode": mode,
        "prompt_id": prompt_id, "cache_name": cache_name,
        "finish_reason": finish_reason, "n_expected": n_expected,
        "prompt_tokens": prompt,
        # A SUBSET of prompt_tokens. Adding the two is the single easiest way to
        # get the bill wrong, and three reviewers warned about it separately.
        "cached_tokens": cached,
        "uncached_prompt_tokens": max(0, prompt - cached),
        "candidates_tokens": cand,
        "tool_use_tokens": tools,
        "thinking_tokens": derived_thinking(usage),
        "total_tokens": total,
    }


def append_jsonl(path, obj) -> None:
    """Append one JSON object as one line, flushed and fsync'd before returning.

    Spend records are the one artifact that has to survive the process, not just
    the run: a crash is exactly when a human needs to know what was already
    paid for. An append of a single line is atomic enough for that (nothing
    rewrites earlier lines), and the fsync is what makes "the row is on disk"
    true rather than "the row is in a buffer the crash took with it".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_synced(path, obj) -> None:
    """write_json, then fsync the directory entry.

    For the small evidence files that are written from a raise path (count-lock
    violations, the usage roll-up): atomic_write_text renames a temp file into
    place, which is durable for content but not for the directory entry.
    """
    write_json(path, obj)
    d = Path(path).parent
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class UsageLog:
    """Every response's usageMetadata, in call order.

    `sink` is the hook the money stack owns (reports/spend_ledger.json is
    append-only and written per interaction). Without a sink the rows still
    accumulate here and the stage writes them out, so no run is unaccounted for.

    `path` is the crash-safe half. Keeping the rows in memory and writing them
    at the end of a successful run was measured to lose everything that mattered:
    five paid calls, then a count-lock FatalError, and
    reports/translate_usage.json did not exist -- while _generate's own docstring
    promised "a crash mid-run cannot leave a paid call unaccounted for". With
    `path` set, every row is appended and fsync'd as it is recorded, so five
    paid calls leave five rows whatever happens next. Disk FIRST, then the sink:
    if the money stack's hook raises, our own record is already durable.
    """

    def __init__(self, sink=None, path=None):
        self.rows: list = []
        self.sink = sink
        self.path = path

    def record(self, row: dict) -> dict:
        self.rows.append(row)
        if self.path is not None:
            append_jsonl(self.path, row)
        if self.sink is not None:
            self.sink(row)
        return row

    def totals(self) -> dict:
        keys = ("prompt_tokens", "cached_tokens", "uncached_prompt_tokens",
                "candidates_tokens", "thinking_tokens", "total_tokens")
        out = {k: sum(r.get(k) or 0 for r in self.rows) for k in keys}
        out["requests"] = len(self.rows)
        out["finish_reasons"] = sorted({r.get("finish_reason") or "?"
                                        for r in self.rows})
        return out


@dataclasses.dataclass(frozen=True)
class LlmRequest:
    """One request, independent of the transport that will carry it.

    The standard path hands this to _generate; the batch path serializes it into
    a JSONL row. Both read the same fields, which is what makes "the bill, the
    JSONL and the interactive call agree" checkable rather than hoped for.

    `system` and `cache_name` are MUTUALLY EXCLUSIVE and the exclusion is
    checked at construction: sending both is a hard 400 ("CachedContent can not
    be used with GenerateContent request setting system_instruction, tools or
    tool_config"). Neither is also legal -- a hand-run review call deliberately
    carries no system instruction, and passing an empty one is not the same
    request.
    """
    kind: str                       # definition | expression | review | pos | rank
    label: str
    user: str
    schema: dict | None
    n_expected: int | None
    max_output_tokens: int
    thinking_level: str = "LOW"
    system: str | None = None
    cache_name: str | None = None
    service_tier: str | None = None

    def __post_init__(self):
        if self.system and self.cache_name:
            raise FatalError(
                "%s: systemInstruction XOR cachedContent. Sending both is a "
                "hard 400 -- the system prompt has to move INTO the cache, not "
                "travel beside it." % self.label)

    def generation_config(self) -> dict:
        """The wire shape of generationConfig, camelCase, no temperature.

        temperature is absent by construction, not by omission: it is
        deprecated on this model generation and the A/B measured no difference.
        """
        cfg = {"responseMimeType": "application/json",
               "thinkingConfig": {"thinkingLevel": self.thinking_level},
               "maxOutputTokens": int(self.max_output_tokens)}
        if self.schema is not None:
            cfg["responseSchema"] = self.schema
        return cfg


def classify_api_error(exc: Exception) -> str:
    """Which of the five handling paths an exception belongs to.

    Before this existed there was one `except Exception` with a five-attempt
    ladder, wrapped in a five-attempt count-lock ladder: a 400 caused by our own
    request cost 25 paid attempts to discover, and a dead cache reference looked
    exactly like a transient.
    """
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    if any(m in text for m in CACHE_MISSING_MARKERS):
        return ERR_CACHE_MISSING
    if "400" in text or "invalid_argument" in text or "invalid argument" in text:
        return ERR_FATAL
    if any(m in text for m in THROTTLE_MARKERS):
        return ERR_THROTTLE
    if "503" in text or "unavailable" in text or "high demand" in text:
        return ERR_UNAVAILABLE
    if any(code in text for code in ("500", "502", "504", "internal")):
        return ERR_RETRYABLE
    if "403" in text or "permission" in text:
        # A 403 that is not the cache message is a credentials problem: retrying
        # cannot fix it.
        return ERR_FATAL
    return ERR_RETRYABLE


class CacheUnavailable(RuntimeError):
    """The referenced explicit cache is gone (403). Recoverable: rebuild it and
    continue -- the wave keeps its place, nothing is re-billed."""


def _is_throttle(exc: Exception) -> bool:
    """Kept as a name because "is this the API saying slow down" is a question
    other code asks; the answer now comes from the one classifier."""
    return classify_api_error(exc) == ERR_THROTTLE


class KeyPool:
    """Rotating pool of Gemini keys, with the v2.1 rotation bug designed out.

    v2.1 kept the counter in a module global and reset it from inside an except
    block that had no `global` statement; the rebind was local and the throttle
    rotation never happened. Here the counter is instance state, so
    force_rotate() cannot silently fail.
    """

    def __init__(self, keys: list, max_per_key: int):
        if not keys:
            raise FatalError(
                "GEMINI_API_KEYS is empty. Export it as a comma-separated list "
                "of keys before confirming any spend.")
        self.keys = keys
        self.max_per_key = max(1, int(max_per_key))
        self.idx = 0
        self.used_on_key = 0
        self.total_requests = 0
        self.rotations = 0
        self._client = None

    def _build(self):
        from google import genai  # lazy: never imported on the bill-only path
        return genai.Client(api_key=self.keys[self.idx])

    def client(self):
        if self._client is None:
            self._client = self._build()
        elif self.used_on_key >= self.max_per_key:
            self.idx = (self.idx + 1) % len(self.keys)
            self.used_on_key = 0
            self.rotations += 1
            self._client = self._build()
        return self._client

    def count(self) -> None:
        self.used_on_key += 1
        self.total_requests += 1

    def force_rotate(self) -> None:
        """Called when the API says slow down. THIS is the line v2.1 could not
        execute."""
        self.used_on_key = self.max_per_key


def _pool_from_env(cfg: Config | None = None) -> KeyPool:
    """The keys, and how many requests each one takes before rotating.

    MAX_PER_API used to be handed to int() with no guard, so one mistyped
    environment variable raised a bare ValueError straight past main()'s
    FatalError handler -- a traceback instead of a sentence, on the money path.
    """
    keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    default = str(cfg.max_per_api_key if cfg else 5)
    raw = os.getenv("MAX_PER_API", default)
    try:
        max_per_key = int(raw)
    except (TypeError, ValueError):
        raise FatalError(
            "MAX_PER_API=%r is not an integer. It is the number of requests one "
            "key takes before the pool rotates; unset it or give it a number."
            % (raw,)) from None
    if cfg is not None and cfg.cache_enabled:
        # An explicit cache is bound to the key/project that created it --
        # another key referencing it gets 403 PERMISSION_DENIED -- so a cached
        # run pins ONE key. Rotation and caching are not compatible.
        idx = int(cfg.cache_key_index)
        if idx >= len(keys):
            raise FatalError(
                "cache_key_index = %d but GEMINI_API_KEYS has %d key(s). A "
                "cache belongs to the key that created it." % (idx, len(keys)))
        pool = KeyPool([keys[idx]], max_per_key=10 ** 9)
        pool.pinned_reason = ("cache_enabled: a cache is bound to its creating "
                              "key/project")
        return pool
    return KeyPool(keys, max_per_key)


@dataclasses.dataclass
class Completion:
    """What one successful call produced. `usage` is the normalized row."""
    parsed: dict
    finish_reason: str
    usage: dict


def _finish_reason(resp) -> str:
    """The finish reason as a plain string, from an SDK enum or a raw dict."""
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return "NO_CANDIDATE"
    fr = getattr(cands[0], "finish_reason", None)
    if fr is None and isinstance(cands[0], dict):
        fr = cands[0].get("finishReason") or cands[0].get("finish_reason")
    if fr is None:
        return "UNKNOWN"
    return str(getattr(fr, "name", fr))


def _generate(pool: KeyPool, model: str, req: LlmRequest, *,
              usage: UsageLog | None = None, prompt_id: str = "",
              mode: str = "standard") -> Completion:
    """One schema-locked call. Returns a Completion, never a bare dict.

    Four properties this function now has and did not have:

      1. systemInstruction XOR cachedContent, asserted BEFORE anything is
         imported or sent. The guard is here rather than in the batch builder
         because stage 50 imports this function directly.
      2. thinkingLevel and maxOutputTokens are always sent, temperature never.
      3. finishReason is checked BEFORE json.loads. A MAX_TOKENS truncation used
         to surface as a JSONDecodeError and be retried five times inside a
         five-attempt count-lock ladder: 25 paid calls to discover a cap.
      4. Errors are classified. 400 raises on the first one, 503 gets exactly
         one retry, a dead cache reference raises CacheUnavailable so the caller
         can rebuild it, and only genuine transients use the full ladder.
      5. A MAX_TOKENS finish RAISES THE BUDGET ONCE and retries (spec 5.6),
         doubling the cap up to MAX_OUTPUT_RETRY_CEILING, before it aborts.
         Truncation is a cap error, so resending the identical request is
         useless -- but aborting a multi-hour paid wave over one under-sized cap
         is worse, and the only measured MAX_TOKENS finishes in the probe set
         came from a level (MEDIUM) and a kind (expression-shaped output) whose
         budget nobody has measured. Both attempts appear in the ledger, with
         the cap each one used.

    The usage row is recorded BEFORE the response is parsed, and an error row is
    recorded for a failed attempt too, so a crash mid-run cannot leave a paid
    call unaccounted for. `usage.path` is what makes that true on DISK and not
    just in memory.
    """
    if getattr(req, "system", None) and getattr(req, "cache_name", None):
        raise FatalError(
            "%s: systemInstruction XOR cachedContent. Sending both is a hard "
            "400; the system prompt has to move INTO the cache." % req.label)

    from google.genai import types

    cap = int(req.max_output_tokens)
    # The sha of the prompt ACTUALLY SENT, on every ledger row. The bill's
    # prompt_sha256 and this are two independent readings of the same object, so
    # G-PROMPT can be checked against the wire instead of against another copy
    # of the same assumption.
    system_sha = prompt_sha256(req.system) if getattr(req, "system", None) else None
    kwargs = {
        "max_output_tokens": cap,
        "response_mime_type": "application/json",
        "thinking_config": types.ThinkingConfig(
            thinking_level=req.thinking_level),
    }
    if req.schema is not None:
        kwargs["response_schema"] = req.schema
    if req.system:
        # The reviewer call carries no system instruction; passing an empty one
        # is not the same request, so the field is set only when there is text.
        kwargs["system_instruction"] = req.system
    if req.cache_name:
        kwargs["cached_content"] = req.cache_name
    if req.service_tier:
        kwargs["service_tier"] = req.service_tier

    def _row(**extra) -> dict:
        row = normalize_usage(None, model=model, label=req.label, kind=req.kind,
                              mode=mode, cache_name=req.cache_name,
                              prompt_id=prompt_id, n_expected=req.n_expected)
        row["max_output_tokens"] = cap
        row["prompt_sha256"] = system_sha
        row.update(extra)
        return row

    last = None
    attempt = 0
    unavailable_seen = 0
    budget_raised = False
    while attempt < MAX_RETRIES:
        attempt += 1
        cli = pool.client()
        try:
            resp = cli.models.generate_content(
                model=model,
                contents=[req.user],
                config=types.GenerateContentConfig(**kwargs),
            )
            pool.count()
        except Exception as exc:  # noqa: BLE001 - the classifier is the handler
            # A request that failed was still PLACED: on the free tier a 503
            # consumes one of the 20 daily requests and returns nothing, and on
            # any tier the rotation counter is about requests put on a key, not
            # about successes.
            pool.count()
            kind = classify_api_error(exc)
            last = "%s: %s [%s]" % (type(exc).__name__, exc, kind)
            if usage is not None:
                usage.record(_row(error=kind, error_text=str(exc)[:400],
                                  attempt=attempt))
            if kind == ERR_CACHE_MISSING:
                raise CacheUnavailable(
                    "%s: the explicit cache is gone (%s). Rebuild it and "
                    "continue; do not retry the same cache name."
                    % (req.label, exc)) from exc
            if kind == ERR_FATAL:
                raise FatalError(
                    "%s: the request itself was refused, so retrying it "
                    "unchanged cannot help: %s" % (req.label, exc)) from exc
            if kind == ERR_THROTTLE:
                pool.force_rotate()
            elif kind == ERR_UNAVAILABLE:
                unavailable_seen += 1
                if unavailable_seen > MAX_503_RETRIES:
                    raise FatalError(
                        "%s: %d consecutive 503s. This is not a transient to "
                        "sit out -- it was 46.4%% of requests on the paid tier "
                        "in the probe wave, and every retry is a charge (or a "
                        "free-tier daily request) for nothing: %s"
                        % (req.label, unavailable_seen, last))
            time.sleep(BASE_RETRY_DELAY * attempt)
            continue

        finish = _finish_reason(resp)
        row = normalize_usage(getattr(resp, "usage_metadata", None), model=model,
                              label=req.label, kind=req.kind, mode=mode,
                              cache_name=req.cache_name, prompt_id=prompt_id,
                              finish_reason=finish, n_expected=req.n_expected)
        row["attempt"] = attempt
        row["max_output_tokens"] = cap
        row["prompt_sha256"] = system_sha
        if usage is not None:
            usage.record(row)
        # finishReason FIRST. These are configuration errors, not transients:
        # retrying the identical request reproduces them. MAX_TOKENS gets one
        # DIFFERENT request (a bigger budget) rather than the same one again.
        if finish == "MAX_TOKENS" and not budget_raised \
                and cap < MAX_OUTPUT_RETRY_CEILING:
            budget_raised = True
            last = "finishReason=MAX_TOKENS at max_output_tokens=%d" % cap
            cap = min(MAX_OUTPUT_RETRY_CEILING, cap * 2)
            kwargs["max_output_tokens"] = cap
            print("  budget raise: %s was truncated at max_output_tokens=%d; "
                  "retrying ONCE at %d (thinking_level=%s)"
                  % (req.label, row["max_output_tokens"], cap,
                     req.thinking_level))
            continue
        if finish == "MAX_TOKENS":
            raise FatalError(
                "%s: the response was TRUNCATED (finishReason=MAX_TOKENS) at "
                "max_output_tokens=%d for n=%s%s. The derived cap is "
                "ceil(a*n + b) * %s with NO thinking term, which only holds at "
                "thinking_level=LOW -- this run is at %s, and maxOutputTokens "
                "is a budget that thoughts and candidates SHARE. Raise "
                "max_output_unmeasured (or max_output_tokens) for this kind, or "
                "drop the thinking level, before re-running."
                % (req.label, cap, req.n_expected,
                   ", and it was already retried once at the raised budget"
                   if budget_raised else
                   " (already at the %d retry ceiling, so the budget was not "
                   "raised)" % MAX_OUTPUT_RETRY_CEILING,
                   MAX_OUTPUT_SAFETY_FACTOR, req.thinking_level))
        if finish in ("NO_CANDIDATE", "UNKNOWN") or not getattr(resp, "text", ""):
            raise FatalError(
                "%s: the response carried no usable candidate "
                "(finishReason=%s). Retrying it unchanged reproduces it."
                % (req.label, finish))
        if finish != "STOP":
            raise FatalError(
                "%s: finishReason=%s, which is not STOP. Not retried: the same "
                "request produces the same verdict." % (req.label, finish))
        try:
            return Completion(parsed=json.loads(resp.text), finish_reason=finish,
                              usage=row)
        except (ValueError, TypeError) as exc:
            last = "%s: %s [unparseable body under finishReason=STOP]" % (
                type(exc).__name__, exc)
            time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError("%s failed after %d attempts: %s"
                     % (req.label, MAX_RETRIES, last))


@dataclasses.dataclass
class CallContext:
    """Everything a request needs that is not the request.

    One object rather than eight parameters, because the batch transport, the
    hand-run review subcommand and the interactive path all need the same set
    and any of them silently dropping (say) the cache name or the prompt id
    would be invisible.
    """
    cfg: Config
    pool: KeyPool
    fit: tuple
    lang: str
    usage: UsageLog = dataclasses.field(default_factory=UsageLog)
    prompt_id: str = ""
    mode: str = "standard"
    cache_name: str | None = None
    # Count-lock violations, WITH the finishReason of the offending response.
    # Without it "the model dropped a sense" and "the cap truncated the JSON"
    # are the same log line, and they need opposite fixes.
    violations: list = dataclasses.field(default_factory=list)
    # Where those violations are written, as they happen. The file used to be
    # written after both translation loops, i.e. never on the one path that
    # needs it: a count-lock ladder that runs out RAISES, and the evidence for
    # why went with it.
    violations_path: Path | None = None

    def request(self, kind: str, label: str, user: str, schema: dict | None,
                n_expected: int | None, system: str | None) -> LlmRequest:
        """Build one request: the ONE place the cap, the thinking level, the
        cache/system exclusion and the service tier are decided.

        `system` is checked against system_prompt(kind, lang) for the kinds this
        module owns. Passing a different object for a definition or expression
        request is the failure mode F5 exists for -- the bill would describe one
        prompt and the wire would carry another -- so it is refused here rather
        than reconciled later.
        """
        n = n_expected or 1
        if kind in _SYSTEM_PROMPTS and system is not None \
                and system != system_prompt(kind, self.lang):
            raise FatalError(
                "%s: the system instruction handed to CallContext.request() is "
                "not system_prompt(%r, %r). The bill, the cache key and "
                "G-PROMPT all read the prompt through that one function, so a "
                "second source for it would make them agree about the wrong "
                "text." % (label, kind, self.lang))
        return LlmRequest(
            kind=kind, label=label, user=user, schema=schema, n_expected=n_expected,
            max_output_tokens=resolve_max_output(self.cfg, n, self.fit, kind),
            thinking_level=self.cfg.thinking_level,
            # With a cache attached the system prompt IS the cache: sending
            # both is a hard 400.
            system=None if self.cache_name else system,
            cache_name=self.cache_name,
            service_tier=self.cfg.effective_service_tier)

    def call(self, model: str, req: LlmRequest) -> Completion:
        return _generate(self.pool, model, req, usage=self.usage,
                         prompt_id=self.prompt_id, mode=self.mode)

    def note_violation(self, row: dict) -> None:
        """Record a violation AND put it on disk now (stage 50's pattern).

        Every violation is a paid call that produced nothing usable, and the
        ladder it belongs to can end in a raise. Writing the whole (short) list
        each time is cheaper than the alternative: a fatal run whose evidence
        file does not exist.
        """
        self.violations.append(row)
        if self.violations_path is not None:
            write_json_synced(self.violations_path, self.violations)


def _count_lock_row(kind: str, label: str, expected: int, got,
                    completion: Completion | None, attempt: int,
                    cap: int | None = None) -> dict:
    """A count-lock violation, WITH the finishReason and the cap that produced
    it. "The model dropped a sense" and "the cap truncated the JSON" used to be
    the same log line, and they need opposite fixes."""
    return {"kind": kind, "label": label, "expected": expected, "got": got,
            "attempt": attempt, "max_output_tokens": cap,
            "finish_reason": completion.finish_reason if completion else None,
            "candidates_tokens": (completion.usage.get("candidates_tokens")
                                  if completion else None)}


def _translate_definition_batch(ctx: CallContext, model: str, entry_label: str,
                                rows: list, correction: str = "") -> list:
    n = len(rows)
    user = definition_user_payload(entry_label, rows, correction)
    label = "definition batch %s" % entry_label
    last = None
    # The 2025 count lock, kept verbatim: a short array means the model dropped a
    # sense, and zipping it would shift every gloss onto the wrong definition. It
    # is RETRIED (5x, with backoff, as v2.1 did) before it is fatal: this is the
    # classic transient, and aborting on the first one throws away every call
    # already paid for in a multi-hour run.
    for attempt in range(1, MAX_COUNT_LOCK_ATTEMPTS + 1):
        time.sleep(ctx.cfg.def_request_interval)
        req = ctx.request("definition", label, user, definition_schema(n), n,
                          system_prompt("definition", ctx.lang))
        comp = ctx.call(model, req)
        out = comp.parsed.get("definitions")
        if isinstance(out, list) and len(out) == n:
            return out
        last = len(out) if isinstance(out, list) else out
        ctx.note_violation(_count_lock_row("definition", entry_label, n, last,
                                           comp, attempt,
                                           req.max_output_tokens))
        print("  count lock: definition batch %s returned %s objects, expected "
              "%d, finishReason=%s (attempt %d/%d)"
              % (entry_label, last, n, comp.finish_reason, attempt,
                 MAX_COUNT_LOCK_ATTEMPTS))
        time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError(
        "definition batch for %s returned %s objects, expected %d, after %d "
        "attempts" % (entry_label, last, n, MAX_COUNT_LOCK_ATTEMPTS))


def _translate_expression_batch(ctx: CallContext, model: str, entry_label: str,
                                rows: list, correction: str = "") -> list:
    """One generation request per batch. No review call, no correction loop.

    The loop that used to be here (generate -> review -> correct, up to
    MAX_CORRECTION_ATTEMPTS=5) was the only 10x amplifier in the design, and it
    ran on every batch whether or not anything was wrong: two requests per batch
    minimum, ten in the bad case, and a contamination FATAL could discard a
    multi-hour paid run. Review is a hand-run subcommand now (owner decision
    D-08) and the offline script gate does the detection the reviewer call was
    doing. `correction` is for that subcommand, and it travels in the USER
    message so the cached prefix is untouched.
    """
    n = len(rows)
    user = expression_user_payload(ctx.lang, entry_label, rows, correction)
    label = "expression batch %s" % entry_label
    last = None
    for attempt in range(1, MAX_COUNT_LOCK_ATTEMPTS + 1):
        time.sleep(ctx.cfg.expr_request_interval)
        req = ctx.request("expression", label, user, expression_schema(n), n,
                          system_prompt("expression", ctx.lang))
        comp = ctx.call(model, req)
        items = comp.parsed.get("fixed_expressions")
        if isinstance(items, list) and len(items) == n:
            return items
        last = len(items) if isinstance(items, list) else items
        ctx.note_violation(_count_lock_row("expression", entry_label, n, last,
                                           comp, attempt,
                                           req.max_output_tokens))
        print("  count lock: expression batch %s returned %s objects, expected "
              "%d, finishReason=%s (attempt %d/%d)"
              % (entry_label, last, n, comp.finish_reason, attempt,
                 MAX_COUNT_LOCK_ATTEMPTS))
        time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError(
        "expression batch for %s returned %s objects, expected %d, after %d "
        "attempts" % (entry_label, last, n, MAX_COUNT_LOCK_ATTEMPTS))


def _translate_pos(ctx: CallContext, model: str, tags: list) -> dict:
    """The POS fallback for a language the checked-in registry does not cover.

    Once registry/pos_translations.json covers every pos_key in scope for every
    configured language, this path is unreachable -- which is the intended end
    state: 12 hand-written strings retire an LLM call that would otherwise have
    to be ported to every transport.
    """
    payload = {tag: "" for tag in tags}
    user = ("Please translate the following Danish POS tags into %s:\n%s"
            % (ctx.lang, json.dumps(payload, indent=2, ensure_ascii=False)))
    time.sleep(ctx.cfg.pos_request_interval)
    comp = ctx.call(model, ctx.request(
        "pos", "pos translation", user, pos_schema(tags, ctx.lang), len(tags),
        system_prompt("pos", ctx.lang)))
    parsed = comp.parsed
    if set(parsed) != set(tags):
        raise FatalError(
            "pos translation key mismatch: missing %s extra %s"
            % (sorted(set(tags) - set(parsed)), sorted(set(parsed) - set(tags))))
    return parsed


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------

def _provenance(model: str, prompt_id: str = "", thinking_level: str = "LOW",
                date: str | None = None) -> str:
    """gemini:<model>+<prompt_id>+<THINKING>@<date>, ASCII, closed vocabulary.

    The model alone was not enough to identify a cell: the same model with a
    different prompt pack or a different thinking level is a different
    translator, and the measured thinking constant is only valid for the prompt
    it was measured on. A later audit filters on this string, so its shape is
    asserted rather than hoped for.
    """
    prov = "gemini:%s+%s+%s@%s" % (model, prompt_id or "unset",
                                   (thinking_level or "unset").upper(),
                                   date or datetime.date.today().isoformat())
    if not PROVENANCE_RE.match(prov):
        raise FatalError(
            "provenance %r is not in the closed form "
            "gemini:<model>+<prompt_id>+<THINKING>@<date>; model and prompt_id "
            "must be ASCII letters, digits, dot or dash." % prov)
    return prov


def provenance_prefix(model: str, prompt_id: str = "",
                      thinking_level: str = "LOW") -> str:
    """The date-less head of a provenance string: who translated, not when.

    `gemini:<model>+<prompt_id>+<THINKING>@`. This is the identity a resumed
    clean redo compares against (see _already_redone): the same translator on a
    second calendar day is the same translator.
    """
    return "gemini:%s+%s+%s@" % (model, prompt_id or "unset",
                                 (thinking_level or "unset").upper())


def archive_everything(defs: dict, exprs: dict, archive: dict, reason: str,
                       keep_prefix: str = "") -> dict:
    """Move every live row into archive.json under an explicit reason.

    The clean-retranslation path (D-01). Destructive, so it runs only behind
    --confirm-spend, and it stamps WHY: an archive whose rows all look alike
    cannot answer "was this row retired by DDO, or by our own rebuild?".

    `keep_prefix` is the resume half: a row whose provenance already starts with
    the running redo's identity is that redo's own output and STAYS LIVE.
    Archiving it would retire work this run just paid for, and (because the
    archive is not read back on a redo) the next resume would buy it again.
    """
    moved = {"definitions": 0, "expressions": 0}
    kept = {"definitions": 0, "expressions": 0}
    for kind, table in (("definitions", defs), ("expressions", exprs)):
        dest = archive.setdefault(kind, {})
        for key in sorted(table):
            row = dict(table[key])
            if keep_prefix and str(row.get("provenance") or "").startswith(
                    keep_prefix):
                kept[kind] += 1
                continue
            row["reason"] = reason
            dest[key] = row
            moved[kind] += 1
            del table[key]
    return {**moved, "kept": kept}


def _drift_report(cfg: Config, entries: dict, write: bool = False) -> dict:
    """Guide 4.10 stage42_retranslate_changed: the article_sha ledger, so a DDO
    edit is detected by content and never by DDO's own lastmod (hus kept
    lastmod=1995-03-01 across a real content change).

    Two round-2 corrections:

      * The ledger is written only on a CONFIRMED run. It used to be rewritten
        unconditionally, so a bill-only run consumed
        "entries_changed_since_last_run" and the second dry run reported 0 --
        and `translate --lang German` then `--lang English` gave the second
        language a "nothing changed" report. Retranslation is driven by per-sense
        src_sha, so this only ever affected the human artifact; the human
        artifact is the whole point of it.
      * The file carries a SCHEMA number. article_sha's input set changed once
        (content fields only), and without a schema stamp every entry in the
        corpus reads as "changed since last run" -- which a human reads as "DDO
        moved". On a schema bump the report says so and the ledger is re-seeded.
    """
    path = cfg.json_dir / "ledger" / "content_hashes.json"
    raw = read_json(path, default={})
    if isinstance(raw, dict) and "hashes" in raw:
        known, schema = raw.get("hashes") or {}, raw.get("schema")
    else:
        # Schema 1 wrote a bare {entry_id: sha} map with no version stamp.
        known, schema = (raw if isinstance(raw, dict) else {}), 1
    payload = {"schema": ARTICLE_SHA_SCHEMA,
               "hashes": {eid: e.get("article_sha") for eid, e in entries.items()}}
    if known and schema != ARTICLE_SHA_SCHEMA:
        out = {"schema_changed": True, "ledger_schema": schema,
               "parser_schema": ARTICLE_SHA_SCHEMA,
               "entries_changed_since_last_run": None,
               "changed_sample": [], "entries_new": len(entries),
               "note": "parser schema changed; no drift information for this "
                       "run. The ledger is re-seeded on the next confirmed run."}
    else:
        changed = sorted(eid for eid, e in entries.items()
                         if eid in known and known[eid] != e.get("article_sha"))
        fresh = sorted(eid for eid in entries if eid not in known)
        out = {"schema_changed": False, "parser_schema": ARTICLE_SHA_SCHEMA,
               "entries_changed_since_last_run": len(changed),
               "changed_sample": changed[:20], "entries_new": len(fresh)}
    if write:
        write_json(path, payload)
    out["ledger_written"] = bool(write)
    return out


# Every artifact that has to exist after money has been spent, whether the run
# finished or fell over. The .jsonl is the crash-proof one (appended and fsync'd
# per call); the rest are written on the way out of the spending section.
SPEND_RECORD_FILES = ("reports/<stage>_usage.jsonl (one line per call, fsync'd)",
                      "reports/<stage>_usage.json (the same rows, as an array)",
                      "review/count_lock_violations_<lang>.json (as they happen)",
                      "reports/<stage>_report.json (with crashed=... on a "
                      "fatal path)")


def _persist_spend_records(cfg: Config, usage: UsageLog, violations: dict,
                           report: dict | None = None, error: BaseException | None = None,
                           usage_file: str = "translate_usage.json",
                           report_file: str = "translate_report.json") -> None:
    """Put the spend records on disk on the way OUT of the spending section.

    Called on both exits, success and exception. The measured failure it exists
    for: five paid calls, a count-lock FatalError, and
    reports/translate_usage.json, review/count_lock_violations_German.json and
    reports/translate_report.json all absent -- while _generate's own docstring
    promised "a crash mid-run cannot leave a paid call unaccounted for". True in
    memory, false on disk, and a crash is the only time it matters.

    NEVER RAISES. It runs on the exception path, and an error raised here would
    replace the error a human actually needs to read.
    """
    try:
        write_json_synced(cfg.report_dir / usage_file, usage.rows)
        for lang, rows in (violations or {}).items():
            if rows:
                write_json_synced(
                    cfg.review_dir / ("count_lock_violations_%s.json" % lang),
                    rows)
        if error is not None and report is not None:
            report["crashed"] = {"error": type(error).__name__,
                                 "message": str(error)[:2000],
                                 "records": list(SPEND_RECORD_FILES)}
            report["usage"] = usage.totals()
            report["note"] = ("THIS RUN FAILED AFTER PLACING PAID CALLS. The "
                              "usage rows above are what was already spent; "
                              "the drift ledger was NOT consumed.")
            write_json_synced(cfg.report_dir / report_file, report)
    except Exception as exc:                 # noqa: BLE001 - must not mask
        print("  WARNING: could not write the spend records (%s: %s). The "
              "per-call ledger at reports/*.jsonl is still authoritative."
              % (type(exc).__name__, exc))


# Where the hand-run review reads its work list. Written by the offline script
# gate, not by an LLM: the detection that the 2025 reviewer call was doing is a
# character-class check, and a character-class check does not need a model.
REVIEW_FLAG_FILES = ("script_violations_%s.json", "translate_flags_%s.json")


def review_flags(cfg: Config, lang: str) -> list:
    """The flagged cells for one language, from work/review/.

    Row shape: {"kind": "definition"|"expression", "key": str,
                "reason": str, "detected": [str]}.
    """
    for pattern in REVIEW_FLAG_FILES:
        path = cfg.review_dir / (pattern % lang)
        if path.exists():
            data = read_json(path, default=[])
            rows = data.get("violations") if isinstance(data, dict) else data
            return [r for r in (rows or []) if isinstance(r, dict) and r.get("key")]
    return []


def _correction_instruction(flags: list, lang: str) -> str:
    """One instruction for one batch, built from every flag in that batch."""
    detected, reasons = [], []
    for flag in flags:
        for d in (flag.get("detected") or []):
            if d not in detected:
                detected.append(str(d))
        reason = flag.get("reason")
        if reason and reason not in reasons:
            reasons.append(str(reason))
    text = ("Your previous answer for this batch was rejected: %s. Regenerate "
            "the whole batch in pure %s."
            % ("; ".join(reasons) or "it failed the output check", lang))
    if detected:
        text += (" These forbidden strings must not appear anywhere in the "
                 "output: %s." % ", ".join(detected[:20]))
    return text


def review(cfg: Config, registry=None, lang: str | None = None,
           keys: list | None = None, confirm: bool = False,
           include_unused: bool = False, usage_sink=None) -> dict:
    """`ankidkdeck review --lang X --fix <keys>` -- the hand-run correction pass.

    This is what is left of the 2025 generate -> review -> correct loop after
    owner decision D-08, and the differences are the point:

      * it NEVER runs by itself. No generation path calls it, so the default
        expression wave is one request per batch instead of two-to-ten;
      * its work list is a FILE a human (or an offline gate) put there, not a
        model's opinion;
      * the correction instruction goes into the USER message, so the cached
        system prefix -- and therefore the discount -- survives the redo;
      * one pass per invocation. There is no MAX_CORRECTION_ATTEMPTS: if a cell
        is still wrong, a human decides to run it again.
    """
    cfg.validate()
    if not lang:
        raise FatalError("review needs --lang: the flag files are per language")
    entries = read_json(cfg.json_dir / "entries.json")
    families = read_json(cfg.json_dir / "words.json", default={})
    scope, scope_note = renderable_scope(cfg, entries, families, include_unused)
    tdir = cfg.json_dir / "translations" / lang
    defs = read_json(tdir / "definitions.json", default={})
    exprs = read_json(tdir / "expressions.json", default={})
    flags = {r["key"]: r for r in review_flags(cfg, lang)}
    wanted = list(keys) if keys else sorted(flags)
    # Every cell in scope, so the Danish side of a flagged key can be rebuilt.
    cells = {r["key"]: r for r in compute_todo(
        cfg, entries, {"definitions": {}, "expressions": {}}, lang, scope,
        retranslate_all=True, retranslate_reason="review_fix")}
    missing = [k for k in wanted if k not in cells]
    rows = [cells[k] for k in wanted if k in cells]
    report = {"language": lang, "scope": scope_note, "confirmed": bool(confirm),
              "flags_on_file": len(flags), "requested": len(wanted),
              "keys_not_in_scope": missing[:20],
              "definitions": sum(1 for r in rows if r["kind"] == "definition"),
              "expressions": sum(1 for r in rows if r["kind"] == "expression")}
    print("--- review pass (%s) ---" % lang)
    print("  %d flagged cell(s) on file, %d requested, %d resolvable"
          % (len(flags), len(wanted), len(rows)))
    if not confirm:
        report["note"] = ("dry run: no LLM module was imported and no request "
                          "was made. Re-run with --confirm-spend to redo these "
                          "cells.")
        print("  nothing has been sent. Re-run with --confirm-spend.")
        write_json(cfg.report_dir / ("review_report_%s.json" % lang), report)
        return report
    if not rows:
        report["note"] = "nothing to redo"
        write_json(cfg.report_dir / ("review_report_%s.json" % lang), report)
        return report

    transport_guard(cfg)
    stats = probe_stats(cfg)
    cfg.validate(spending=True, stats=stats)
    fit = output_fit(stats=stats)
    pool = _pool_from_env(cfg)
    usage = UsageLog(sink=usage_sink,
                     path=cfg.report_dir / ("review_usage_%s.jsonl" % lang))
    ctx = CallContext(cfg=cfg, pool=pool, fit=fit, lang=lang, usage=usage,
                      prompt_id=cfg.prompt_id, mode=cfg.mode,
                      violations_path=cfg.review_dir
                      / ("count_lock_violations_%s.json" % lang))
    prov = _provenance(cfg.gemini_model, cfg.prompt_id, cfg.thinking_level)
    prov_expr = _provenance(cfg.expressions_model, cfg.prompt_id,
                            cfg.thinking_level)
    redone = {"definitions": 0, "expressions": 0}
    usage_file = "review_usage_%s.json" % lang
    report_file = "review_report_%s.json" % lang
    try:
        for kind, table, model, provenance, cap in (
                ("definition", defs, cfg.gemini_model, prov, MAX_DEFS_PER_BATCH),
                ("expression", exprs, cfg.expressions_model, prov_expr,
                 MAX_EXPR_PER_BATCH)):
            for eid, batch in _group_by_entry(rows, kind, cap):
                label = ("%s %s" % (batch[0]["lemma"],
                                    batch[0]["pos_text"])).strip()
                correction = _correction_instruction(
                    [flags.get(r["key"], {}) for r in batch], lang)
                if kind == "definition":
                    got = _translate_definition_batch(ctx, model, label, batch,
                                                      correction=correction)
                else:
                    got = _translate_expression_batch(ctx, model, label, batch,
                                                      correction=correction)
                for row, obj in zip(batch, got):
                    table[row["key"]] = {"lemma": obj.get("lemma"),
                                         "gloss": obj.get("gloss"),
                                         "src_sha": row["src_sha"],
                                         "provenance": provenance}
                    redone[kind + "s"] += 1
                write_json(tdir / ("%ss.json" % kind), table)
    except BaseException as exc:             # noqa: BLE001 - re-raised below
        # Money was placed before this point. The records go down first.
        report["redone"] = redone
        _persist_spend_records(cfg, usage, {lang: ctx.violations}, report=report,
                               error=exc, usage_file=usage_file,
                               report_file=report_file)
        raise
    report.update({"redone": redone, "usage": usage.totals(),
                   "count_lock_violations": len(ctx.violations),
                   "provenance": prov})
    _persist_spend_records(cfg, usage, {lang: ctx.violations},
                           usage_file=usage_file, report_file=report_file)
    write_json(cfg.report_dir / report_file, report)
    return report


def transport_guard(cfg: Config) -> None:
    """Refuse to spend on a transport that is not wired up yet.

    Without this, `--mode batch --confirm-spend` would place ordinary
    interactive calls and label every ledger row "batch" -- i.e. pay standard
    rates and record half of them. Same class of problem as cache_enabled with
    no cache: believing a discount applies is more expensive than knowing it
    does not.

    The batch module owns the removal of these two checks. Delete a branch when
    (and only when) the thing it is standing in for actually exists.
    """
    if cfg.mode == "batch":
        raise FatalError(
            "mode = batch: the batch transport (JSONL writer, job registry, "
            "wave splitter, cache lifecycle) is not part of this stage yet. "
            "Running it would place standard-rate calls and file them as batch. "
            "Use --mode standard (or flex) for now.")
    if cfg.cache_enabled:
        raise FatalError(
            "cache_enabled = true: nothing creates or attaches an explicit "
            "cache yet, so this run would pay the full uncached rate while the "
            "configuration says otherwise. Turn it off until the cache "
            "lifecycle is wired up.")


# Spec 5.9, first half: the bind audit is frozen before the first paid call.
BIND_AUDIT_SNAPSHOT = "bind_report_pre_translate.json"


def _snapshot_bind_audit(cfg: Config) -> None:
    """Freeze bind_report.json before this workspace's first paid translate.

    bind's numbers -- n_legacy / n_bound / n_dropped / bind_rate -- are THE
    audit of the recovered 2025 asset, and they are computed against the LIVE
    translation tables. Once a paid translate has written gemini rows into those
    tables, a later `bind` (or a `build`, which runs bind) is answering a
    different question and filing the answer under the same name: keys that a
    legacy row would have taken are now occupied, so the row is counted as a
    shared_dannetid_conflict drop and the bind rate falls for a reason that has
    nothing to do with 2025.

    Written ONCE and never overwritten. The first paid run is the boundary, and
    a second snapshot would move the boundary to wherever the pollution already
    reached. Stage 41 carries the other half (it labels a polluted report as
    polluted instead of pretending).
    """
    src = cfg.report_dir / "bind_report.json"
    dst = cfg.report_dir / BIND_AUDIT_SNAPSHOT
    if dst.exists() or not src.exists():
        return
    data = read_json(src, default=None)
    if data is None:
        return
    write_json(dst, {
        "snapshot_of": "reports/bind_report.json",
        "taken_at": datetime.date.today().isoformat(),
        "taken_before": "the first paid translate run on this workspace",
        "why": ("bind counts against the live translation tables, so a paid "
                "translate changes its answer. This copy is the pre-LLM "
                "audit of the 2025 asset and is never rewritten."),
        "report": data})
    print("  bind audit snapshot written: reports/%s" % BIND_AUDIT_SNAPSHOT)


DRY_PATH_WRITES = ("translations/<lang>/definitions.json (restored rows only)",
                   "translations/<lang>/expressions.json (restored rows only)",
                   "translations/<lang>/archive.json (gc)",
                   "reports/translate_bill_<lang>.json",
                   "reports/translate_report.json",
                   "reports/gates_report.json (G-ORPH)")


def run(cfg: Config, registry=None, lang: str | None = None,
        confirm: bool = False, do_gc: bool = True,
        include_unused: bool = False, retranslate_all: bool = False,
        phase: str = "all", usage_sink=None) -> dict:
    """Bill, then (with confirm) spend.

    `run(confirm=False)` IS NOT READ-ONLY, and the help text now says so: it
    runs gc(), writes the files listed in DRY_PATH_WRITES and runs G-ORPH, which
    can FatalError before a human ever sees the bill. "Nothing was sent" is
    true; "nothing changed" is not. Snapshot work/ before the first dry run.

    `phase` exists for the batch transport, where one wave is two invocations
    (submit, then ingest hours later) and both of them are --confirm-spend runs.
    The drift ledger must be consumed once per wave, at the END of the ingest --
    see below.

    `retranslate_all` is the clean retranslation (D-01, N-01). It is
    RESUMABLE: rows that already carry this run's provenance are neither
    re-billed nor re-archived, so a crash halfway through a 5,565-request wave
    costs the remainder and nothing else. The plain path, meanwhile, refuses to
    restore rows the redo retired -- without that, the natural "just run it
    again" after a crash silently rolled the whole redo back to the old model.
    """
    cfg.validate()
    retranslate_all = bool(retranslate_all or cfg.retranslate_all)
    if phase not in ("all", "submit", "ingest"):
        raise FatalError("phase = %r is not one of all, submit, ingest" % phase)
    if phase != "all" and cfg.mode != "batch":
        # Not fatal -- the phases are a property of the wave, not of the wire,
        # and the ledger accounting is testable without a transport. But on the
        # interactive surface a "submit" places every call there is, so say so.
        print("  NOTE: phase=%s on the %s transport: there is no separate submit "
              "and ingest here, so this run does all of the work and only the "
              "drift ledger behaves as if it were split." % (phase, cfg.mode))
    entries = read_json(cfg.json_dir / "entries.json")
    families = read_json(cfg.json_dir / "words.json", default={})
    langs = [lang] if lang else list(cfg.langs)
    # `german` is not `German`: none of the 22,734 migrated cells are visible
    # under the lowercase key, so --confirm-spend would pay the full
    # from-scratch price and write to translations/german/. The CLI checks this
    # too; the stage checks it because the stage is also called directly.
    unknown = [lg for lg in langs if lg not in cfg.langs]
    if unknown:
        raise FatalError(
            "unknown language(s) %s. Configured languages are: %s. Add the "
            "language to `langs` in ankidkdeck.toml before spending on it."
            % (", ".join(repr(u) for u in unknown), ", ".join(cfg.langs)))
    scope, scope_note = renderable_scope(cfg, entries, families, include_unused)
    pos_wanted = pos_keys_in_scope(entries, scope)
    reg_pos = getattr(registry, "pos_translations", None) or {}

    report: dict = {"languages": {}, "scope": scope_note, "confirmed": bool(confirm),
                    "model": cfg.gemini_model,
                    "model_expressions": cfg.expressions_model,
                    "mode": cfg.mode, "thinking_level": cfg.thinking_level,
                    "prompt_id": cfg.prompt_id, "phase": phase,
                    "retranslate_all": retranslate_all,
                    "model_verified": cfg.model_is_verified(),
                    # The drift ledger is READ here and WRITTEN at the end of a
                    # successful confirmed run (see below). It used to be
                    # written at this line -- before a single call had been
                    # placed -- so a crash on the first request consumed
                    # entries_changed_since_last_run, and under the batch
                    # transport (submit and ingest are both --confirm-spend) it
                    # was consumed twice per wave.
                    "drift": _drift_report(cfg, entries, write=False)}
    bill: dict = {}
    gc_stats: dict = {}
    state: dict = {}
    archived_for_redo: dict = {}
    # The identity a resumed redo compares live rows against: same model, same
    # prompt pack, same thinking level, ANY date. Only meaningful on the redo
    # path -- on the incremental path src_sha already decides.
    done_prefix = provenance_prefix(cfg.gemini_model, cfg.prompt_id,
                                   cfg.thinking_level) if retranslate_all else ""
    # The measured constants, for the bill. Read on the DRY path too, because
    # spec 4.2(1) is a dry-run requirement: three dollar figures, offline. Missing
    # or unusable is not fatal HERE (the spend gate below is where it refuses)
    # -- the bill says which constant it could not find instead of inventing it.
    try:
        bill_stats = probe_stats(cfg)
    except FatalError as exc:
        bill_stats, bill_stats_why = None, str(exc)
    else:
        bill_stats_why = None
    rates, rates_note = rate_card_for(cfg)

    for lg in langs:
        tdir = cfg.json_dir / "translations" / lg
        if do_gc:
            g = gc(cfg, lg, entries, scope)
            defs, exprs = g.pop("definitions"), g.pop("expressions")
            archive = g.pop("archive")
            gc_stats[lg] = g
        else:
            defs = read_json(tdir / "definitions.json", default={})
            exprs = read_json(tdir / "expressions.json", default={})
            archive = read_json(tdir / "archive.json",
                               default={"definitions": {}, "expressions": {}})
        pos = read_json(tdir / "pos.json", default={})
        restored: list = []
        resume: dict = {}
        todo = compute_todo(cfg, entries, {"definitions": defs, "expressions": exprs},
                            lg, scope, archive=archive, restored=restored,
                            retranslate_all=retranslate_all,
                            retranslate_reason=cfg.retranslate_reason,
                            done_provenance=done_prefix, resume=resume)
        if restored:
            # A returning sense is un-archived for free, on the DRY path too:
            # the row already exists and paying for it again is the defect.
            write_json(tdir / "definitions.json", defs)
            write_json(tdir / "expressions.json", exprs)
        # A pos_key the checked-in registry already covers is NOT billed. Stage
        # 42's POS call still exists for a language the registry does not know,
        # which is what makes adding a language a no-registry-edit operation.
        have_pos = set(pos) | set(reg_pos.get(lg) or {})
        pos_todo = [k for k in pos_wanted if k not in have_pos]
        state[lg] = {"dir": tdir, "defs": defs, "exprs": exprs, "pos": pos,
                     "todo": todo, "pos_todo": pos_todo, "archive": archive}
        bill[lg] = bill_row(todo, pos_todo)
        bill[lg]["restored_from_archive"] = len(restored)
        bill[lg]["resume"] = dict(resume)
        tokens = bill_tokens(todo, pos_todo, lg, bill_stats)
        if bill_stats_why and not tokens.get("available"):
            tokens["why"] = bill_stats_why
        bill[lg]["tokens"] = tokens
        bill[lg]["dollars"] = dollar_figures(tokens, rates, cfg.spend_cap_usd)
        bill[lg]["dollars"]["rate_card_source"] = rates_note
        write_json(cfg.report_dir / ("translate_bill_%s.json" % lg),
                   {"language": lg, "model": cfg.gemini_model,
                    "model_expressions": cfg.expressions_model,
                    "mode": cfg.mode, "thinking_level": cfg.thinking_level,
                    "prompt_id": cfg.prompt_id,
                    # From system_prompt(), the same function CallContext builds
                    # the request with -- not a second reading of the prompt.
                    "prompt_sha256": prompt_shas(lg),
                    "retranslate_all": retranslate_all,
                    "scope": scope_note,
                    "pos_keys_from_registry": sorted(
                        set(pos_wanted) & set(reg_pos.get(lg) or {})),
                    "restored_from_archive_rows": restored[:50],
                    **bill[lg],
                    # PER REQUEST, not per cell, and NO DDO TEXT. The cells
                    # block used to carry every Danish definition in scope --
                    # 22,282 of them on a clean redo, 7.6 MB, 28x the
                    # incremental bill -- to restate what the counts already
                    # said. What a bill needs is sizes and shas.
                    "requests": bill_requests(todo, pos_todo, lg),
                    # WHICH cells and WHY, one line each. Renamed from `cells`
                    # (a list of full rows, DDO text included) because the shape
                    # changed as well as the contents: a silent type change is
                    # worse than a name a reader has to notice.
                    "cells_by_key": {r["key"]: r["reason"] for r in todo},
                    "cells_note": ("keys and reasons only; sizes are per "
                                   "request in `requests`. The Danish source "
                                   "text lives in json/entries.json and does "
                                   "not belong in a report artifact -- it was "
                                   "7.6 MB of this file on a clean redo.")})

    print_bill(bill, cfg.gemini_model, cfg.expressions_model, mode=cfg.mode,
               thinking_level=cfg.thinking_level, prompt_id=cfg.prompt_id,
               confirmed=bool(confirm))
    if retranslate_all:
        print("  RETRANSLATE-ALL: every cell above is being paid for again. On "
              "--confirm-spend the existing rows are moved into archive.json "
              "with reason=%r and the archive is NOT read back. Rows that "
              "already carry this run's provenance (%s) are kept and NOT "
              "re-billed." % (cfg.retranslate_reason, done_prefix))
    if not cfg.model_is_verified():
        print("  WARNING: model %s is not on the verified list; no measured "
              "constant and no price applies to it." % cfg.gemini_model)
    report["bill"] = bill
    if gc_stats:
        report["gc"] = {lg: {k: v for k, v in g.items() if k != "orphans_remaining"}
                        for lg, g in gc_stats.items()}
        run_gates([Gate(G_ORPH, "no translation row survives without a live sense "
                                "(archived, never deleted)",
                        lambda: orphans_gate(gc_stats), stage="42")], cfg, stage="42")

    if not confirm:
        report["note"] = ("bill only: no LLM module was imported and no request was "
                          "made. Re-run with --confirm-spend to place the calls.")
        report["dry_path_wrote"] = list(DRY_PATH_WRITES)
        write_json(cfg.report_dir / "translate_report.json", report)
        return report

    # ---------------- past this line, money is spent ----------------
    # Measured constants first: no file, wrong model or a missing key is a
    # refusal, because the alternative is sizing paid requests from a number
    # that was measured on something else.
    transport_guard(cfg)
    stats = probe_stats(cfg)
    # thinking_level is a SPEND-time gate, not a load-time one: a dry run may
    # quote any level, but placing paid calls at a level whose thinking cost
    # nobody measured means the cap formula (no thinking term) does not apply.
    cfg.validate(spending=True, stats=stats)
    fit = output_fit(stats=stats)
    pool = _pool_from_env(cfg)
    model = cfg.gemini_model
    # Expressions, the POS table (and stage 50's ranking) may run on their own
    # model: short outputs whose failure mode is contamination, not truncation.
    expr_model = cfg.expressions_model
    prov = _provenance(model, cfg.prompt_id, cfg.thinking_level)
    prov_expr = _provenance(expr_model, cfg.prompt_id, cfg.thinking_level)
    # Per-call, fsync'd, append-only. This is the file that answers "what had
    # already been paid for?" after a crash; everything else is a roll-up of it.
    usage = UsageLog(sink=usage_sink,
                     path=cfg.report_dir / "translate_usage.jsonl")
    violations_by_lang: dict = {}
    _snapshot_bind_audit(cfg)
    try:
        for lg in langs:
            st = state[lg]
            tdir, defs, exprs = st["dir"], st["defs"], st["exprs"]
            if retranslate_all:
                # Destructive, and only here: the bill-only path quotes the redo
                # without touching a row. The archive is not read back for this
                # language (compute_todo above), which is what makes the redo
                # actually happen -- deleting definitions.json by hand only ever
                # worked once, because the next run silently restored it all.
                # keep_prefix is the resume half: rows this redo already
                # produced stay live instead of being retired and bought again.
                moved = archive_everything(defs, exprs, st["archive"],
                                           cfg.retranslate_reason,
                                           keep_prefix=done_prefix)
                write_json(tdir / "definitions.json", defs)
                write_json(tdir / "expressions.json", exprs)
                write_json(tdir / "archive.json", st["archive"])
                archived_for_redo[lg] = moved
                print("  %s: archived %d definition and %d expression row(s) "
                      "with reason=%r; kept %d + %d already redone in this "
                      "wave" % (lg, moved["definitions"], moved["expressions"],
                                cfg.retranslate_reason,
                                moved["kept"]["definitions"],
                                moved["kept"]["expressions"]))
            ctx = CallContext(cfg=cfg, pool=pool, fit=fit, lang=lg, usage=usage,
                              prompt_id=cfg.prompt_id, mode=cfg.mode,
                              violations_path=cfg.review_dir
                              / ("count_lock_violations_%s.json" % lg))
            violations_by_lang[lg] = ctx.violations
            written = {"definitions": 0, "expressions": 0}
            for eid, rows in _group_by_entry(st["todo"], "definition",
                                             MAX_DEFS_PER_BATCH):
                label = "%s %s" % (rows[0]["lemma"], rows[0]["pos_text"])
                got = _translate_definition_batch(ctx, model, label.strip(), rows)
                for row, obj in zip(rows, got):
                    defs[row["key"]] = {"lemma": obj.get("lemma"),
                                        "gloss": obj.get("gloss"),
                                        "src_sha": row["src_sha"],
                                        "provenance": prov}
                    written["definitions"] += 1
                write_json(tdir / "definitions.json", defs)  # after EVERY entry
            for eid, rows in _group_by_entry(st["todo"], "expression",
                                             MAX_EXPR_PER_BATCH):
                label = "%s %s" % (rows[0]["lemma"], rows[0]["pos_text"])
                got = _translate_expression_batch(ctx, expr_model, label.strip(),
                                                  rows)
                for row, obj in zip(rows, got):
                    # row["src_sha"] is sha256(NFC(expression text)) -- the same
                    # formula stage 41 stored, so a migrated row and a fresh one
                    # are indistinguishable to the next drift check.
                    exprs[row["key"]] = {"lemma": obj.get("lemma"),
                                         "gloss": obj.get("gloss"),
                                         "src_sha": row["src_sha"],
                                         "provenance": prov_expr}
                    written["expressions"] += 1
                write_json(tdir / "expressions.json", exprs)
            if st["pos_todo"]:
                # Translate the ~14 data-pos-key values, never the 41 mangled
                # 2025 display strings (guide 1.11f). The call asks for the full
                # key set (the model needs the whole paradigm to be consistent)
                # but only the MISSING keys are written: the card front is
                # grouped by these strings, so re-wording a key that already
                # shipped would reshuffle every card front for no reason.
                mapping = _translate_pos(ctx, expr_model, list(pos_wanted))
                merged = dict(st["pos"])
                for key in st["pos_todo"]:
                    merged[key] = mapping[key]
                write_json(tdir / "pos.json", merged)
                st["pos"] = merged
            report["languages"][lg] = {
                "written": written,
                "pos_keys_written": len(st["pos_todo"]),
                "definition_rows": len(defs), "expression_rows": len(exprs),
                "pos_rows": len(st["pos"]), "provenance": prov,
                "provenance_expressions": prov_expr,
                "count_lock_violations": len(ctx.violations),
                "archived_for_redo": archived_for_redo.get(lg),
            }
    except BaseException as exc:             # noqa: BLE001 - re-raised below
        # Money has been spent by now. Every record goes down BEFORE the
        # exception continues on its way: five paid calls then a FatalError used
        # to leave no usage file, no violations file and no report at all.
        _persist_spend_records(cfg, usage, violations_by_lang, report=report,
                               error=exc)
        raise
    # One record per wave, next to the tokens it cost. The dollar figure is the
    # money stack's to fill in from these counts and the rate card.
    report["waves"] = [{"mode": cfg.mode, "phase": phase,
                        "cache_name": None, "prompt_id": cfg.prompt_id,
                        "thinking_level": cfg.thinking_level,
                        "languages": list(langs), **usage.totals(),
                        "cached_share": (usage.totals()["cached_tokens"]
                                         / usage.totals()["prompt_tokens"]
                                         if usage.totals()["prompt_tokens"]
                                         else None),
                        "spend_usd": None}]
    report["usage"] = usage.totals()
    report["spend_records"] = list(SPEND_RECORD_FILES)
    _persist_spend_records(cfg, usage, violations_by_lang)
    report["api"] = {"requests": pool.total_requests, "key_rotations": pool.rotations,
                     "keys_in_pool": len(pool.keys)}
    # The drift ledger is consumed HERE: after the last call of a successful
    # confirmed run, and under the batch transport only on the ingest phase --
    # submit and ingest are both --confirm-spend runs, and writing it on both
    # would consume entries_changed_since_last_run twice per wave.
    if phase in ("all", "ingest"):
        report["drift"] = _drift_report(cfg, entries, write=True)
        report["drift"]["ledger_written_at_phase"] = phase
    else:
        report["drift"]["ledger_written_at_phase"] = None
    write_json(cfg.report_dir / "translate_report.json", report)
    return report
