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
"""

import datetime
import json
import math
import os
import time

from ..config import Config
from ..extract import ARTICLE_SHA_SCHEMA
from ..gates import G_ORPH, Gate, run_gates
from ..util import NFC, FatalError, read_json, sha256_str, write_json

MAX_DEFS_PER_BATCH = 20        # 05: one call per entry, capped
MAX_EXPR_PER_BATCH = 20        # 06: MAX_EXPR_PER_BATCH
MAX_RETRIES = 5
# 06's review -> correct rounds. v2.1 retried up to 5 times before giving up on
# an entry; 3 rounds with no backoff meant one flaky response aborted a paid
# multi-hour run (measured: 64 calls spent before the contamination FATAL).
MAX_CORRECTION_ATTEMPTS = 5
# The same ladder for a violated COUNT LOCK. A short array is a dropped sense,
# never something to zip anyway -- but it is also the classic transient, and
# FATALing on the first one throws away every call already paid for in the run.
MAX_COUNT_LOCK_ATTEMPTS = 5
BASE_RETRY_DELAY = 5
DEF_REQUEST_INTERVAL = 2.1     # 05: 10 RPM free tier
EXPR_REQUEST_INTERVAL = 5.0    # 06: 30 RPM free tier, two calls per batch
POS_REQUEST_INTERVAL = 1.1     # 04: 60 RPM free tier
CHARS_PER_TOKEN_ESTIMATE = 4   # for the bill only; never used to size a call

THROTTLE_MARKERS = ("429", "resource_exhausted", "too many requests",
                    "quota", "rate limit", "ratelimit")


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
                          restored: list) -> bool:
    """A sense that comes back after a DDO edit must not be paid for twice.

    gc() moves a dead row into archive.json and its docstring has always claimed
    this property (guide D7) -- but nothing anywhere read archive.json back, so
    the property was not implemented: remove a sense, run translate (archived),
    put it back, run translate again, and the cell was billed a second time.

    Restoring is conditional on the SOURCE TEXT being unchanged: the archived
    gloss was produced from a specific Danish string, and src_sha is that string.
    A row whose sha no longer matches stays archived and is billed as changed.
    """
    if archive is None:
        return False
    row = (archive.get(kind) or {}).get(key)
    if row is None or row.get("src_sha") != sha:
        return False
    have[key] = dict(row)
    restored.append({"kind": kind, "key": key,
                     "provenance": row.get("provenance")})
    return True


def compute_todo(cfg: Config, entries: dict, translations: dict, lang: str,
                 scope=None, archive: dict | None = None,
                 restored: list | None = None) -> list:
    """Cells that are missing, or whose Danish source text has changed.

    One row per cell: {key, kind, entry_id, lemma, pos_text, dannetid, text,
    hint, src_sha, reason}. Empty definition texts are not cells -- there is
    nothing to translate and the exporter does not render them.

    A key that is missing from the live table but present in archive.json with
    the SAME src_sha is restored into the live table (in place, so the caller
    persists it) and is not billed.
    """
    scope = set(entries) if scope is None else set(scope)
    have_defs = translations.get("definitions") or {}
    have_exprs = translations.get("expressions") or {}
    restored = [] if restored is None else restored
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
            if row is None and _restore_from_archive(
                    "definitions", key, s.get("src_sha"), have_defs, archive,
                    restored):
                row = have_defs.get(key)
            reason = None
            if row is None:
                reason = "missing"
            elif row.get("src_sha") != s.get("src_sha"):
                reason = "src_sha_changed"
            if reason:
                todo.append({"key": key, "kind": "definition", "entry_id": eid,
                             "lemma": e.get("display_headword") or e.get("lemma"),
                             "pos_text": e.get("pos_text") or "",
                             "dannetid": s.get("dannetid"), "text": text,
                             "hint": "", "src_sha": s.get("src_sha"),
                             "reason": reason})
        for x in e.get("expressions", []):
            text = (x.get("expression") or "").strip()
            key = expression_key(x)
            if not text or not key:
                continue
            sha = expression_src_sha(x)
            row = have_exprs.get(key)
            if row is None and _restore_from_archive(
                    "expressions", key, sha, have_exprs, archive, restored):
                row = have_exprs.get(key)
            reason = None
            if row is None:
                reason = "missing"
            elif row.get("src_sha") != sha:
                reason = "src_sha_changed"
            if reason:
                todo.append({"key": key, "kind": "expression", "entry_id": eid,
                             "lemma": e.get("display_headword") or e.get("lemma"),
                             "pos_text": e.get("pos_text") or "",
                             "dannetid": key, "text": text,
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

def bill_row(todo: list, pos_todo: list) -> dict:
    defs = [r for r in todo if r["kind"] == "definition"]
    exprs = [r for r in todo if r["kind"] == "expression"]
    chars = sum(len(r["text"]) + len(r["hint"]) for r in todo)
    def_calls = len(_group_by_entry(todo, "definition", MAX_DEFS_PER_BATCH))
    expr_batches = len(_group_by_entry(todo, "expression", MAX_EXPR_PER_BATCH))
    return {
        "definitions": len(defs),
        "definitions_new": sum(1 for r in defs if r["reason"] == "missing"),
        "definitions_changed": sum(1 for r in defs if r["reason"] == "src_sha_changed"),
        "expressions": len(exprs),
        "expressions_new": sum(1 for r in exprs if r["reason"] == "missing"),
        "expressions_changed": sum(1 for r in exprs if r["reason"] == "src_sha_changed"),
        "pos_keys": len(pos_todo),
        "pos_keys_list": pos_todo,
        "cells_total": len(todo),
        "entries_touched": len({r["entry_id"] for r in todo}),
        # Requests, not money: expressions cost one generate + one review per
        # batch, and a failed review costs another round (up to 3).
        "requests_min": def_calls + 2 * expr_batches + (1 if pos_todo else 0),
        "requests_max": def_calls + 2 * expr_batches * MAX_CORRECTION_ATTEMPTS
                        + (1 if pos_todo else 0),
        "source_chars": chars,
        "source_tokens_estimate": math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE),
    }


def print_bill(bill: dict, model: str, expr_model: str | None = None) -> None:
    """ALWAYS runs. No price is asserted: the per-token rate is a decision for
    the human at the moment of spending, and a made-up number in a log is worse
    than none."""
    if expr_model and expr_model != model:
        print("--- translation bill (definitions: %s | expressions + pos: %s) ---"
              % (model, expr_model))
    else:
        print("--- translation bill (model: %s) ---" % model)
    total = 0
    for lang in sorted(bill):
        r = bill[lang]
        total += r["cells_total"]
        print("  %-8s %5d cells  (definitions %d: %d new / %d changed | "
              "expressions %d: %d new / %d changed | pos keys %d)"
              % (lang, r["cells_total"], r["definitions"], r["definitions_new"],
                 r["definitions_changed"], r["expressions"],
                 r["expressions_new"], r["expressions_changed"], r["pos_keys"]))
        print("           %d entries, %d-%d API requests, ~%d source tokens"
              % (r["entries_touched"], r["requests_min"], r["requests_max"],
                 r["source_tokens_estimate"]))
    print("  TOTAL %d cells across %d language(s)" % (total, len(bill)))
    print("  price per token is not asserted here; check the current rate card")
    print("  nothing has been sent. Re-run with --confirm-spend to place calls.")


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
    truncation impossible."""
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
                    "properties": {"lemma": {"type": "string"},
                                   "gloss": {"type": "string"}},
                    "required": ["lemma", "gloss"],
                },
            },
        },
        "required": ["headword", "definitions"],
    }


def definition_prompt(lang: str, n_defs: int) -> str:
    """PROMPT V4, ported from 05_translate_definitions.py without edits. The
    22,734 existing cells per language were produced by this text; changing a
    word makes the new cells stylistically foreign to the old ones."""
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
    Return pure JSON that **exactly matches the response schema**. The `definitions` array MUST have a length of {n_defs}.
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
                    "properties": {"lemma": {"type": "string"},
                                   "gloss": {"type": "string"}},
                    "required": ["lemma", "gloss"],
                },
            }
        },
        "required": ["fixed_expressions"],
    }


def review_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "contains_other_languages": {"type": "boolean"},
            "detected_words": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["contains_other_languages", "detected_words"],
    }


def expression_prompt(lang: str, n_items: int, correction_instruction: str = "") -> str:
    """06, ported without edits. The review-and-correct loop and the Russian
    clause exist because of a real contamination incident; do not remove them."""
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
Return pure JSON matching the schema. The `fixed_expressions` array MUST have a length of {n_items}.
"""
    critical_rules = ""
    if lang.lower() != "english":
        critical_rules = f"""
**CRITICAL RULES: These are non-negotiable.**
1.  The primary language for both "lemma" and "gloss" MUST be **{lang}**.
2.  You must avoid all other languages. **DO NOT USE RUSSIAN under any circumstances.**
3.  **AS A LAST RESORT, ONLY IF** a concept is truly untranslatable into {lang}, you are permitted to use a concise English word or phrase in the `gloss`. This should be extremely rare. Never use English in the `lemma`.
"""
    if correction_instruction:
        return (f"**IMPORTANT CORRECTION:** {correction_instruction}\n\n"
                f"{critical_rules.strip()}\n\n{system_prompt_base.strip()}")
    return f"{critical_rules.strip()}\n\n{system_prompt_base.strip()}"


def review_prompt(lang: str, json_string: str) -> str:
    """06's reviewer. English gets its own wording, as in the original."""
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

def _is_throttle(exc: Exception) -> bool:
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(m in text for m in THROTTLE_MARKERS)


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


def _pool_from_env() -> KeyPool:
    keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    return KeyPool(keys, os.getenv("MAX_PER_API", "5"))


def _generate(pool: KeyPool, model: str, system: str, user: str, schema: dict,
              temperature: float, label: str) -> dict:
    """One schema-locked call with the v2.1 retry ladder. Returns parsed JSON."""
    from google.genai import types

    kwargs = {"temperature": temperature, "max_output_tokens": 8192,
              "response_mime_type": "application/json", "response_schema": schema}
    if system:
        # 06's reviewer call carries no system instruction; passing an empty
        # one is not the same request.
        kwargs["system_instruction"] = system
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        cli = pool.client()
        try:
            resp = cli.models.generate_content(
                model=model,
                contents=[user],
                config=types.GenerateContentConfig(**kwargs),
            )
            pool.count()
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001 - the ladder is the handler
            if _is_throttle(exc):
                pool.force_rotate()
            last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError("%s failed after %d attempts: %s" % (label, MAX_RETRIES, last))


def _translate_definition_batch(pool, model, lang, entry_label, rows) -> list:
    payload = {str(i): r["text"] for i, r in enumerate(rows)}
    n = len(payload)
    user = ('Headword: "%s"\n'
            "Expecting exactly %d definition objects.\n"
            "Input Definitions JSON:\n%s"
            % (entry_label, n, json.dumps(payload, ensure_ascii=False, indent=2)))
    last = None
    # The 2025 count lock, kept verbatim: a short array means the model dropped a
    # sense, and zipping it would shift every gloss onto the wrong definition. It
    # is now RETRIED (5x, with backoff, as v2.1 did) before it is fatal: this is
    # the classic transient, and aborting on the first one throws away every call
    # already paid for in a multi-hour run.
    for attempt in range(1, MAX_COUNT_LOCK_ATTEMPTS + 1):
        time.sleep(DEF_REQUEST_INTERVAL)
        parsed = _generate(pool, model, definition_prompt(lang, n), user,
                           definition_schema(n), 0.1,
                           "definition batch %s" % entry_label)
        out = parsed.get("definitions")
        if isinstance(out, list) and len(out) == n:
            return out
        last = len(out) if isinstance(out, list) else out
        print("  count lock: definition batch %s returned %s objects, expected "
              "%d (attempt %d/%d)"
              % (entry_label, last, n, attempt, MAX_COUNT_LOCK_ATTEMPTS))
        time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError(
        "definition batch for %s returned %s objects, expected %d, after %d "
        "attempts" % (entry_label, last, n, MAX_COUNT_LOCK_ATTEMPTS))


def _translate_expression_batch(pool, model, lang, entry_label, rows) -> list:
    """06's generate -> review -> correct loop, with v2.1's retry budget."""
    payload = {str(i): {"expr": r["text"], "hint": r["hint"]}
               for i, r in enumerate(rows)}
    n = len(payload)
    user = ('Headword: "%s"\n'
            "Please translate the following %d fixed expressions into %s:\n%s"
            % (entry_label, n, lang, json.dumps(payload, ensure_ascii=False, indent=2)))
    detected: list = []
    short = None
    for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
        correction = ""
        if detected:
            correction = (
                "Your previous attempt for this batch included forbidden words: %s. "
                "Please regenerate the entire batch, ensuring these words are "
                "completely removed and replaced with pure %s equivalents."
                % (detected, lang))
        time.sleep(EXPR_REQUEST_INTERVAL)
        parsed = _generate(pool, model, expression_prompt(lang, n, correction), user,
                           expression_schema(n), 0.1,
                           "expression batch %s" % entry_label)
        items = parsed.get("fixed_expressions")
        if not isinstance(items, list) or len(items) != n:
            # Same ladder as the definition count lock: retried, not fatal.
            short = len(items) if isinstance(items, list) else items
            print("  count lock: expression batch %s returned %s objects, "
                  "expected %d (attempt %d/%d)"
                  % (entry_label, short, n, attempt, MAX_CORRECTION_ATTEMPTS))
            time.sleep(BASE_RETRY_DELAY * attempt)
            continue
        time.sleep(EXPR_REQUEST_INTERVAL)
        verdict = _generate(pool, model, "", review_prompt(
            lang, json.dumps(items, ensure_ascii=False)), review_schema(), 0.0,
            "expression review %s" % entry_label)
        if not verdict.get("contains_other_languages"):
            return items
        detected = verdict.get("detected_words") or ["unknown"]
        time.sleep(BASE_RETRY_DELAY * attempt)
    raise FatalError(
        "expression batch for %s failed after %d rounds (contamination: %s, "
        "last short array: %s)"
        % (entry_label, MAX_CORRECTION_ATTEMPTS, detected, short))


def _translate_pos(pool, model, lang, tags: list) -> dict:
    payload = {tag: "" for tag in tags}
    user = ("Please translate the following Danish POS tags into %s:\n%s"
            % (lang, json.dumps(payload, indent=2, ensure_ascii=False)))
    time.sleep(POS_REQUEST_INTERVAL)
    parsed = _generate(pool, model, pos_prompt(lang), user, pos_schema(tags, lang),
                       0.0, "pos translation")
    if set(parsed) != set(tags):
        raise FatalError(
            "pos translation key mismatch: missing %s extra %s"
            % (sorted(set(tags) - set(parsed)), sorted(set(parsed) - set(tags))))
    return parsed


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------

def _provenance(model: str) -> str:
    return "gemini:%s@%s" % (model, datetime.date.today().isoformat())


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


def run(cfg: Config, registry=None, lang: str | None = None,
        confirm: bool = False, do_gc: bool = True,
        include_unused: bool = False) -> dict:
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
                    "drift": _drift_report(cfg, entries, write=confirm)}
    bill: dict = {}
    gc_stats: dict = {}
    state: dict = {}

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
        todo = compute_todo(cfg, entries, {"definitions": defs, "expressions": exprs},
                            lg, scope, archive=archive, restored=restored)
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
                     "todo": todo, "pos_todo": pos_todo}
        bill[lg] = bill_row(todo, pos_todo)
        bill[lg]["restored_from_archive"] = len(restored)
        write_json(cfg.report_dir / ("translate_bill_%s.json" % lg),
                   {"language": lg, "model": cfg.gemini_model,
                    "model_expressions": cfg.expressions_model,
                    "scope": scope_note,
                    "pos_keys_from_registry": sorted(
                        set(pos_wanted) & set(reg_pos.get(lg) or {})),
                    "restored_from_archive_rows": restored[:50],
                    **bill[lg],
                    "cells": [{k: v for k, v in r.items() if k != "hint"}
                              for r in todo]})

    print_bill(bill, cfg.gemini_model, cfg.expressions_model)
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
        write_json(cfg.report_dir / "translate_report.json", report)
        return report

    # ---------------- past this line, money is spent ----------------
    pool = _pool_from_env()
    model = cfg.gemini_model
    # Expressions, the POS table (and stage 50's ranking) may run on their own
    # model: short outputs whose failure mode is contamination, not truncation.
    expr_model = cfg.expressions_model
    prov = _provenance(model)
    prov_expr = _provenance(expr_model)
    for lg in langs:
        st = state[lg]
        tdir, defs, exprs = st["dir"], st["defs"], st["exprs"]
        written = {"definitions": 0, "expressions": 0}
        for eid, rows in _group_by_entry(st["todo"], "definition", MAX_DEFS_PER_BATCH):
            label = "%s %s" % (rows[0]["lemma"], rows[0]["pos_text"])
            got = _translate_definition_batch(pool, model, lg, label.strip(), rows)
            for row, obj in zip(rows, got):
                defs[row["key"]] = {"lemma": obj.get("lemma"), "gloss": obj.get("gloss"),
                                    "src_sha": row["src_sha"], "provenance": prov}
                written["definitions"] += 1
            write_json(tdir / "definitions.json", defs)  # checkpoint after EVERY entry
        for eid, rows in _group_by_entry(st["todo"], "expression", MAX_EXPR_PER_BATCH):
            label = "%s %s" % (rows[0]["lemma"], rows[0]["pos_text"])
            got = _translate_expression_batch(pool, expr_model, lg, label.strip(), rows)
            for row, obj in zip(rows, got):
                # row["src_sha"] is sha256(NFC(expression text)) -- the same
                # formula stage 41 stored, so a migrated row and a fresh one are
                # indistinguishable to the next drift check.
                exprs[row["key"]] = {"lemma": obj.get("lemma"), "gloss": obj.get("gloss"),
                                     "src_sha": row["src_sha"], "provenance": prov_expr}
                written["expressions"] += 1
            write_json(tdir / "expressions.json", exprs)
        if st["pos_todo"]:
            # Translate the ~14 data-pos-key values, never the 41 mangled 2025
            # display strings (guide 1.11f). The call asks for the full key set
            # (the model needs the whole paradigm to be consistent) but only the
            # MISSING keys are written: the card front is grouped by these
            # strings, so re-wording a key that already shipped would reshuffle
            # every card front for no reason.
            mapping = _translate_pos(pool, expr_model, lg, list(pos_wanted))
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
        }
    report["api"] = {"requests": pool.total_requests, "key_rotations": pool.rotations,
                     "keys_in_pool": len(pool.keys)}
    write_json(cfg.report_dir / "translate_report.json", report)
    return report
