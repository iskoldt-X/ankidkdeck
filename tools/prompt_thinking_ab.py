#!/usr/bin/env python3
"""The thinking A/B gate for a prompt-pack upgrade. Patch plan item 4.4.

WHY THIS EXISTS
---------------
Enriching the definition prompt from the frozen 1,135 tokens to the rich ~2,800
costs about $0.50 per language in cached input -- a rounding error. The thing
that could make it expensive is THINKING: every 100 extra thought tokens per
request is +$0.68 per language, and thinkingLevel is the one knob whose default
(MEDIUM) was measured at mean 578.7 thought tokens per request.

The measurement on file says the risk is small: a 4,564-token rich prompt at
thinkingLevel=LOW produced derived thinking = 0. But that is ONE observation,
not a median of twenty, and the constant on disk was measured on the FROZEN
prompt. Consumption rule 6 therefore invalidates it the moment prompt_id
changes, and the money gates refuse to spend until someone re-measures. This
tool is that re-measurement.

WHAT IT DOES
------------
Places the same requests twice -- once with the frozen prompt, once with the
rich one -- through the PRODUCTION request constructors, and never through a
copy of them. That rule is not a style preference: the whole point of measuring
is to measure what the pipeline will send, and a probe that builds its own
request measures the probe.

It then evaluates the five criteria of patch plan 4.4, of which four are
mechanical and one is a human's job:

  (a) median(thoughts | RICH@LOW) <= 1.25 * median(thoughts | LEAN@LOW)
  (b) G-SCRIPT violations in the RICH arm == 0
  (c) part-of-speech shape conformance >= 95%, attributed per block by
      running the RAMP_STAGES block sets as extra arms
  (d) 40 blind pairs judged by a human; if they cannot tell, ship LEAN
  (e) the measured thinking constant is invalidated by the prompt change, and
      the artifact has to be rebased before anything may be spent

(d) is emitted as a file and reported PENDING. A tool cannot pass it, and a
tool that pretended to would be the most expensive line in this repository:
criterion (d) is the one that says "if the rich prompt buys nothing, keep the
money".

COST AND SAFETY
---------------
Nothing is placed without --confirm-spend. Without it the tool prints the plan,
the entry list and the estimated cost, and exits 0 -- which is also how you get
the exact backfill command to run afterwards.

Every response's usage is appended to <probes>/calls.jsonl, fsync'd, BEFORE it
is interpreted. A crash after five paid calls has to leave five rows: an
accounting artifact written at the end of a successful run is an accounting
artifact that does not exist on the runs that need it.

WHICH SURFACE
-------------
`--mode interactive` (the default) is the original behaviour: one synchronous
call per entry per arm. `--mode batch` places each arm as ONE BATCH JOB through
the shipped transport, which is what the owner mandated after the Chinese month:
the interactive surface 503-storms (83.9% of requests during the measured
storm), and an A/B whose two arms meet a storm at different rates measures the
weather rather than the prompt.

The batch path reuses the shipped modules whole -- the JSONL writer, the job
state machine, the submit, the poll, the crash recovery, the key-join
reconciliation with its bijection guard, and the explicit-cache lifecycle -- and
keeps its OWN job registry file. That separation is load-bearing rather than
cosmetic: transport._resume_in_flight adopts every in-flight job whose `lang`
field matches and reads no label at all, so an A/B job in the production
registry would be adopted by the next translate and its answers written into the
shipped translation tables. See AB_REGISTRY_FILE.

Both surfaces write the SAME downstream artifacts -- <probes>/calls.jsonl rows
in the schema backfill_probe_stats reads, reports/prompt_ab_usage.jsonl rows
labelled with the surface they really ran on, criteria (a)(b)(c), and the blind
pairs plus their separate answer key.

USAGE
-----
  # 1. see the plan, spend nothing
  python3 tools/prompt_thinking_ab.py --lang German --entries 20

  # 2. place the calls (the owner presses this)
  python3 tools/prompt_thinking_ab.py --lang German --entries 20 --confirm-spend

  # 2b. ...or place them on the batch surface, one job per arm
  python3 tools/prompt_thinking_ab.py --lang German --entries 20 \\
      --mode batch --confirm-spend

  # 3. hand the ledger to the artifact, which re-derives the constants AND
  #    records which pack version each language was measured on
  python3 tools/backfill_probe_stats.py --probes <probes> \\
      --declare-prompt-id rich-core-1 --rebase-measurement --write
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# arm -> (prompt_id, ramp stage or None). The two ends are the decision the 4.4
# A/B exists to make; the ramp stages in between are the PER-BLOCK ATTRIBUTION
# half of criterion (c) -- "which block earned its tokens". They were recorded
# as prompts.RAMP_STAGES and nothing could run them, so the question could not
# be answered at all. They are measurement arms only: only stage3 is a byte
# prefix of RICH, so an intermediate stage is never a shippable variant.
ARMS = {"lean": ("v4-frozen", None),
        "rich": ("rich-core-1", None),
        "stage1": ("rich-core-1", "stage1"),
        "stage2": ("rich-core-1", "stage2"),
        "stage3": ("rich-core-1", "stage3")}
DEFAULT_ARMS = "lean,rich"
# The arms that criterion (a) and the blind test compare. An attribution run
# adds arms; it does not change what the decision is made on.
DECISION_ARMS = ("lean", "rich")

# Infinitive particles that must never open a verb lemma. The frozen prompt's
# own POS block names all three.
PARTICLES = ("to ", "zu ", "at ")
ARTICLES = ("the ", "a ", "an ", "der ", "die ", "das ", "ein ", "eine ",
            "el ", "la ", "los ", "las ", "un ", "una ")


def pos_shape_violations(lang, produced, pack):
    """Criterion (c): does the lemma have the SHAPE its part of speech needs?

    Mechanical only, and deliberately so -- these are the four rules whose
    conformance was measured on the shipped corpus, so a rate here is
    comparable to a rate there rather than to an opinion:

      verbs   no infinitive particle (94.0% conformance in English 2025)
      nouns   no article (99.4% in German, 99.5% in Spanish)
      lemma   inside the pack's lemma charset (Han-only for Chinese)
      lemma   non-empty
    """
    from ankidkdeck import gates                       # noqa: PLC0415

    profile = gates.script_profile(pack)
    bad = []
    for row in produced:
        lemma = (row.get("lemma") or "").strip()
        pos = row.get("pos_key") or ""
        low = lemma.lower()
        if not lemma:
            bad.append({**row, "why": "empty lemma"})
            continue
        if pos == "vb." and low.startswith(PARTICLES):
            bad.append({**row, "why": "verb lemma opens with an infinitive "
                                      "particle"})
            continue
        if pos.startswith("sb.") and low.startswith(ARTICLES):
            bad.append({**row, "why": "noun lemma opens with an article"})
            continue
        if profile["han_allowed"] and not profile["latin_in_lemma_allowed"]:
            if any(ch.isascii() and ch.isalpha() for ch in lemma):
                bad.append({**row, "why": "Latin letters in a Han-only lemma"})
    return bad


def pick_entries(cfg, lang, n_entries):
    """The same entries for both arms, chosen deterministically.

    Sorted by entry_id and filtered to 2..8 senses: a single-sense entry
    measures nothing about a batch and a 20-sense entry is a tail case, while
    the corpus mean is 3.75 senses per definition request.
    """
    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415
    from ankidkdeck.util import read_json                # noqa: PLC0415

    entries = read_json(cfg.json_dir / "entries.json")
    families = read_json(cfg.json_dir / "words.json", default={})
    scope, _ = s42.renderable_scope(cfg, entries, families, False)
    out = []
    for eid in sorted(scope):
        entry = entries.get(eid) or {}
        senses = [s for s in (entry.get("senses") or [])
                  if (s.get("definition") or "").strip()]
        if not 2 <= len(senses) <= 8:
            continue
        rows = [{"key": s42.definition_key(eid, s),
                 "text": s.get("definition"),
                 "grammar": s.get("grammar") or "",
                 "pos_key": entry.get("pos_key") or ""} for s in senses]
        label = "%s %s" % (entry.get("display_headword") or entry.get("lemma"),
                           entry.get("pos_text") or "")
        out.append({"entry_id": eid, "label": label.strip(), "rows": rows})
        if len(out) >= n_entries:
            break
    return out


def probe_row(arm, prompt_id, prompt_chars, level, label, usage_dict, n):
    """One line of the probe ledger, in the shape backfill_probe_stats reads.

    `request_fingerprint.prompt_chars` is what puts this call in a prompt
    FAMILY, and the family is what makes the thinking number mean anything: the
    same thinkingLevel=LOW produced 0 thought tokens on the definition prompt
    and 236-275 on the ranking prompt, so a pooled number describes neither.

    `n` AND `n_expected`, and the duplication is deliberate. The batch size is
    the x-axis of both fits backfill_probe_stats re-derives, and it reads that
    axis as `fp.get("n")` -- the key the wave-1 probe harness wrote. This tool
    wrote only `n_expected`, so EVERY A/B row, on both surfaces, was silently
    skipped for EXPECTED_OUTPUT, PROMPT_TOKENS_fit and the prompt_sha256_per_n
    cross-check: `--declare-prompt-id rich-core-1 --rebase-measurement` could
    never have re-derived the rich arm's own fits from the ledger the A/B
    writes, which is the whole of criterion (e). `n` is what the reader needs;
    `n_expected` stays because it is what the previous rows carry.
    """
    return {"record": "call", "probe_id": "prompt_ab_%s" % arm,
            "arm": arm, "prompt_id": prompt_id, "label": label,
            "config": {"thinking_config": {"thinking_level": level}},
            "request_fingerprint": {"prompt_chars": prompt_chars,
                                    "n": n, "n_expected": n},
            "usageMetadata": usage_dict}


def usage_to_camel(row):
    """The production usage row, in the camelCase shape the probe ledger uses.

    Both shapes exist for a reason (the wire speaks camelCase, the ledger and
    the gates speak snake_case) and the conversion is here rather than in
    either of them, because this tool is the only thing that needs both.
    """
    return {"promptTokenCount": row.get("prompt_tokens") or 0,
            "cachedContentTokenCount": row.get("cached_tokens") or 0,
            "candidatesTokenCount": row.get("candidates_tokens") or 0,
            "toolUsePromptTokenCount": row.get("tool_use_tokens") or 0,
            "totalTokenCount": row.get("total_tokens") or 0}


# --------------------------------------------------------------------------
# the batch surface
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS. The interactive surface 503-storms -- 83.9% of requests
# during the storm measured in the Chinese month -- and an A/B whose two arms
# meet a storm at different rates is not an A/B, it is a measurement of the
# weather. The batch transport is the surface the production definition wave
# already runs on, so measuring there measures what the pipeline will send.
#
# HOW THIS IS KEPT OUT OF THE PRODUCTION WAVE. Two layers, and the order below
# is the order of load-bearingness -- corrected after both reviewers checked it
# by experiment, because an earlier version of this comment had it backwards and
# a reader who believed it could have "simplified" the mechanism away.
#
#   THE MECHANISM is ab_wave_tag: the A/B's jobs carry a WAVE TAG in their
#   `lang` field ("AB-German-lean-<sel>"), never a language.
#   transport._resume_in_flight and _ingest_ready both select on EXACT equality
#   (`if job.get("lang") != lang: continue`), so a production translate for
#   "German" cannot match an A/B record even if the two shared one registry
#   file. Verified by both reviewers with the file deliberately shared: the
#   production resume adopted nothing. DO NOT REMOVE OR SHORTEN THE TAG.
#
#   DEFENCE IN DEPTH is the separate registry file. It matters because the
#   registry has operations that do NOT filter by `lang` and so do couple the
#   two: find_by_fingerprint, next_job_id's slot allocation, summary() and
#   cache_prompt_shas() (the last is read by transport at two call sites). It
#   also keeps a human reading work/batch/jobs.json from seeing A/B rows, and it
#   keeps the isolation standing if the tag scheme is ever changed.
AB_REGISTRY_FILE = "ab_jobs.json"

# keys.language_tag DROPS every non-letter, which would collapse the three
# attribution arms into ONE row-key namespace: AB-German-stage1/2/3 all become
# "ABGermanstage". Digits are translated to letters instead of dropped, so the
# stage arms and the selection digest stay distinct in `ledger["row_key"]`.
_DIGITS_TO_LETTERS = str.maketrans("0123456789", "abcdefghij")


def ab_selection_id(todo):
    """A short digest of the CELL SET this run measures.

    It goes into the wave tag, which makes the tag an identity for the
    measurement rather than for the arm alone. Without it, a second run with a
    different --entries adopts the first run's stored outcome (the tag matches)
    while report["cells"] is computed from the NEW selection -- so
    prompt_ab_verdict.json states an n it did not have, and the LEAN-vs-RICH
    decision is read off the wrong sample. The fingerprint one layer down does
    include the row keys, but the adoption short-circuits before it is reached.
    """
    from ankidkdeck.util import sha256_str                # noqa: PLC0415

    cell_keys = sorted(str(r["key"]) for r in todo)
    return sha256_str("%d|%s" % (len(cell_keys),
                                 ",".join(cell_keys)))[:8]


def ab_wave_tag(lang, arm, selection):
    """The `lang` FIELD of this arm's batch jobs. It is NOT a language.

    Four shipped functions read that field, and all four want separation per
    ARM (the arms carry different system prompts and therefore different caches)
    AND per CELL SET (see ab_selection_id):

      transport._resume_in_flight   which jobs this invocation adopts
      transport._ingest_ready       which jobs a production wave would ingest
      registry.wave_fingerprint     the wave identity
      keys.make_key                 the row key segment

    The REAL language never leaves CallContext, so the system prompt and the
    Danish payload are the real language's throughout.
    """
    return "AB-%s-%s-%s" % (lang, arm, selection)


def ab_key_tag(tag):
    """`tag` as the letters-only segment keys.make_key needs. See
    _DIGITS_TO_LETTERS: translated, not stripped, so nothing collides."""
    from ankidkdeck.batch import keys as batch_keys       # noqa: PLC0415

    return batch_keys.language_tag(tag.translate(_DIGITS_TO_LETTERS))


def todo_rows_for(cfg, lang, batches):
    """The A/B's chosen cells as PRODUCTION todo rows.

    pick_entries() carries only what the interactive arm needs (key, text,
    grammar, pos_key). The batch path needs the full row -- src_sha for the
    plan's cells, lemma and pos_text for the request label, hint for the token
    estimate -- so the rows are recomputed by compute_todo, the same function
    the translate wave uses, and then narrowed to the keys already chosen.
    Deriving them here rather than widening pick_entries keeps the interactive
    arm byte-for-byte what it was.
    """
    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415
    from ankidkdeck.util import read_json                # noqa: PLC0415

    entries = read_json(cfg.json_dir / "entries.json")
    scope = {b["entry_id"] for b in batches}
    wanted = {r["key"] for b in batches for r in b["rows"]}
    todo = s42.compute_todo(cfg, entries,
                            {"definitions": {}, "expressions": {}}, lang,
                            scope, retranslate_all=True,
                            retranslate_reason="prompt_ab")
    return [r for r in todo
            if r["kind"] == "definition" and r["key"] in wanted]


# The surface the INTERACTIVE arms place their calls on, and the ONE place it
# is named. Stage 50's RANK_MODE and stage 42's REVIEW_MODE are the template:
# one explicit constant driving the rate, the request ceiling and the ledger
# label, so the three cannot disagree.
#
# WHY IT IS NOT cfg.mode. run_arm has always stamped its ledger rows
# mode="standard" (it calls _generate, which IS the synchronous surface), but
# cfg.mode is whatever the operator's toml says -- very plausibly "batch" during
# a batch month. Quoting the interactive arms off cfg.mode would price them at
# batch rates against a ledger booked at standard, and would quote the BATCH
# request ceiling of 4 for a path that actually takes the interactive count-lock
# x transport ladder of 25. That is exactly the review() defect this project
# fixed in the same round; it is not being reintroduced one file over.
AB_INTERACTIVE_MODE = "standard"


def ab_apply_surface(cfg, surface, *, cache_enabled):
    """Put cfg on the surface this run will really use, and SAY SO.

    Both fields are overrides of the operator's ankidkdeck.toml, and
    `cache_enabled` in particular commits storage spend nobody configured, so
    every change is printed and returned for the report rather than applied
    quietly.

    The values are not cosmetic -- four shipped functions read exactly these two
    fields to decide what is legal and what a call costs: transport_guard,
    _pool_from_env, Config.effective_service_tier and billing.expected_scenario.
    Leaving cfg.mode at whatever the toml says would price the interactive arms
    at batch rates and quote them the batch request ceiling; see
    AB_INTERACTIVE_MODE.
    """
    changed = {}
    if cfg.mode != surface:
        changed["mode"] = {"was": cfg.mode, "now": surface}
        cfg.mode = surface
    if bool(cfg.cache_enabled) != bool(cache_enabled):
        changed["cache_enabled"] = {"was": bool(cfg.cache_enabled),
                                    "now": bool(cache_enabled)}
        cfg.cache_enabled = bool(cache_enabled)
    for field, move in sorted(changed.items()):
        print("  ab: OVERRIDING your config -- %s %r -> %r (this run is a %s "
              "A/B; the label, the rate and the request ceiling all have to "
              "describe the surface the calls are placed on)"
              % (field, move["was"], move["now"], surface))
    return changed


def ab_bill(cfg, lang, todo, arms, stats, *, surface):
    """A forecast-shaped quote, ONE ENTRY PER ARM, on a NAMED surface.

    billing.forecast sums over the keys of this mapping, and an A/B places the
    same cells once per arm -- so filing all the arms under one key would quote
    a two-arm run at half its size and G-BUDGET would adjudicate the wrong
    number.

    `surface` is "batch" or AB_INTERACTIVE_MODE, never cfg.mode: it decides both
    the rate card and the request ceiling, and both must describe the surface
    the calls are really placed on. See AB_INTERACTIVE_MODE.

    KNOWN UNDER-STATEMENT, stated rather than hidden: bill_tokens sizes the
    system half from PROMPT_TOKENS_system_only, which was measured on the LEAN
    prompt (~1,142 tokens). The rich arm's prompt is ~2,822, so on the cached
    batch surface its cached-input line is quoted at about 40% of the truth. At
    the A/B's size (tens of requests) the whole understatement is under a cent.
    It is named here because a quote that is wrong by a knowable amount should
    say so on the artifact, not in someone's memory.
    """
    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415

    rates, rates_note = s42.rate_card_for(cfg, mode=surface)
    tokens = s42.bill_tokens(todo, [], lang, stats)
    bill = {}
    for arm in arms:
        bill[ab_wave_tag(lang, arm, ab_selection_id(todo))] = dict(
            s42.bill_row(todo, [], surface), tokens=tokens,
            arm=arm, prompt_id=ARMS[arm][0], surface=surface,
            dollars=dict(s42.dollar_figures(tokens, rates, cfg.spend_cap_usd),
                         rate_card_source=rates_note),
            basis_note=("the system half is the MEASURED lean size; the rich "
                        "arm's cached prefix is about 2.5x that, so on a "
                        "cached surface this arm's cached-input line "
                        "under-states it"))
    return bill


# The N-09 rule the A/B must be exempt from, and the exemption's own name. R6
# refuses a spend whose live system prompt has drifted from the one the measured
# constants were taken on -- "measured on LEAN, spent on RICH". THAT IS THE
# TOOL'S ENTIRE PURPOSE: the A/B exists to produce the measurement that rebases
# the constant, so it cannot be gated on already having it. Measured drift for
# the rich arm is +158.7% against a 10% tolerance, so R6 refuses every rich arm,
# for ever, by construction.
#
# BEFORE THIS EXISTED THE EXEMPTION WAS AN ACCIDENT. The block was evaluated
# once, with whatever prompt happened to be active -- always LEAN, because
# main() activates it before anything -- and the rich arm then spent under a
# green verdict the gate had never formed an opinion on. That is this
# codebase's own named failure mode ("a gate that never ran is not a gate that
# passed"), and the tool's artifact asserted the opposite in
# criteria.e_constant_invalidated.
#
# THE REPLACEMENT IS NOT A HOLE. R6 asks "is the prompt on the wire the prompt
# the constants describe?", which the A/B cannot answer by design. It is
# replaced by the question the A/B CAN answer -- "is the prompt on the wire the
# prompt this arm DECLARED?" -- as a sha256 identity against
# prompts.build_definition_prompt for the arm's own id and block set. A
# mismatch still refuses, and every other rule (including R2's cache floor,
# which is the remaining size check) is evaluated unchanged.
AB_R6_EXEMPTION = "R6-prompt-size-ab-identity"


def ab_consumption_rules(cfg, lang, arm, stats):
    """The N-09 rules for ONE arm, with R6 swapped for the identity check.

    The arm's prompt must already be active: this reads the LIVE
    system_prompt(), which is the same object CallContext.request will refuse to
    deviate from, and compares it with what the arm declares.
    """
    from ankidkdeck import billing, prompts                # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42     # noqa: PLC0415
    from ankidkdeck.util import FatalError, sha256_str     # noqa: PLC0415

    pid, stage = ARMS[arm]
    blocks = None if stage is None else prompts.ramp_stage_blocks(stage)
    declared = prompts.build_definition_prompt(lang, prompt_id=pid,
                                               block_names=blocks)
    live = s42.system_prompt("definition", lang)
    texts = {kind: s42.system_prompt(kind, lang)
             for kind in ("definition", "expression")}
    rows = []
    for row in billing.consumption_rules(cfg, stats, prompts=texts):
        if row["rule"] != "R6-prompt-size":
            rows.append(row)
            continue
        rows.append({
            "rule": AB_R6_EXEMPTION, "spec_rule": "6",
            "ok": live == declared, "blocking": True,
            "detail": {
                "arm": arm, "prompt_id": pid, "ramp_stage": stage,
                "blocks": None if blocks is None else list(blocks),
                "declared_sha256": sha256_str(declared),
                "live_sha256": sha256_str(live),
                "declared_chars": len(declared), "live_chars": len(live),
                "replaces": "R6-prompt-size",
                "why": ("R6 compares the live prompt against the size the "
                        "constants were measured on, which the A/B is here to "
                        "re-measure -- the rich arm drifts about 159% and R6 "
                        "would refuse it for ever. The A/B is EXEMPT from that "
                        "comparison, deliberately and by name, and is held to "
                        "an identity check instead: the prompt on the wire must "
                        "be exactly the prompt this arm declares. Criterion (e) "
                        "still requires backfill_probe_stats "
                        "--declare-prompt-id --rebase-measurement before any "
                        "PRODUCTION spend on the rich prompt."),
            }})
    bad = [r for r in rows if r["blocking"] and not r["ok"]]
    if bad:
        raise FatalError(
            "%d consumption rule(s) refuse this arm (%s):\n%s"
            % (len(bad), arm, "\n".join(
                "  %s (patch plan N-09 rule %s): %s"
                % (r["rule"], r["spec_rule"],
                   json.dumps(r["detail"], ensure_ascii=False, sort_keys=True))
                for r in bad)))
    return rows


def ab_gates(cfg, bill, families, out_path=None):
    """G-SCOPE-FROZEN + G-BUDGET, WITHOUT writing the production gates report.

    s42._pre_spend goes through gates.run_gates, which persists into
    cfg.report_dir/gates_report.json and merges rows by (id, stage, extra) with
    "a later run of the SAME scope wins". pre_spend_gates emits G-SCOPE-FROZEN
    and G-BUDGET at stage 42 -- the same merge keys a production stage-42 run
    emits -- so an A/B run turned a red production gates report GREEN. Measured:
    a report carrying failed:[G-BUDGET, G-SCOPE-FROZEN] read failed:[] after one
    A/B run.

    A measurement tool must not participate in the release artifact. The gates
    are evaluated exactly as run_gates evaluates them, the same failure message
    is raised, and the verdict is written to the A/B's OWN file so it stays
    auditable.
    """
    from ankidkdeck.gates import failure_message, pre_spend_gates  # noqa: PLC0415
    from ankidkdeck.util import FatalError, write_json             # noqa: PLC0415

    count = families if isinstance(families, int) else len(families or {})
    rows = []
    for gate in pre_spend_gates(cfg, bill, families=count):
        ok, detail = gate.fn()
        # The stage is OVERRIDDEN rather than inherited. These rows never merge
        # with production's -- they are in the A/B's own file -- and (id, stage,
        # extra) is exactly the key gates._write_report merges on, so a row
        # labelled "42" sitting in an A/B artifact is the kind of thing that
        # gets copied back into the wrong file one day.
        rows.append({"id": gate.id, "description": gate.description,
                     "stage": "42-prompt-ab", "production_stage": gate.stage,
                     "extra": dict(gate.extra), "ok": bool(ok),
                     "detail": detail})
    if out_path is not None:
        write_json(out_path, {"results": rows})
    message = failure_message(rows)
    if message:
        raise FatalError(message)
    return rows


def ab_pre_spend(cfg, lang, todo, arms, stats, *, surface):
    """The gate block every production paid path evaluates, PER ARM.

    Same block and the same order stage 42's review() runs: the transport guard,
    the spending half of cfg.validate, the N-09 consumption rules, then
    G-SCOPE-FROZEN and G-BUDGET over a bill. It runs BEFORE any cache and before
    the first job is planned, which is the only place it can refuse something
    that has not been paid for yet.

    PER ARM, because the arms differ in the one thing the rules are about: the
    system prompt. Evaluating once, with whatever pack happened to be active,
    adjudicated LEAN and let RICH spend behind it. Each arm's prompt is
    activated here before its own rules are asked, and the process is left on
    the LAST arm's pack -- run_arm / run_arm_batch re-activate per arm anyway,
    so no caller inherits a stale one.

    G-SCOPE-FROZEN and G-BUDGET see the WHOLE run's bill on every pass: budget
    is a property of the invocation, not of one arm, and per-arm budgeting would
    let a five-arm run through on five individually-affordable quotes.
    """
    from ankidkdeck import prompts                         # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42     # noqa: PLC0415
    from ankidkdeck.util import read_json                  # noqa: PLC0415

    s42.transport_guard(cfg)
    cfg.validate(spending=True, stats=stats)
    bill = ab_bill(cfg, lang, todo, arms, stats, surface=surface)
    families = read_json(cfg.json_dir / "words.json", default={})
    out = {"surface": surface, "bill": bill,
           "consumption_rules": {}, "pre_spend_gates": {}}
    for arm in arms:
        pid, stage = ARMS[arm]
        prompts.reset()
        prompts.activate(cfg, prompt_id=pid, ramp_stage=stage)
        out["consumption_rules"][arm] = ab_consumption_rules(cfg, lang, arm,
                                                             stats)
        out["pre_spend_gates"][arm] = ab_gates(
            cfg, bill, families,
            out_path=cfg.report_dir / "prompt_ab_gates.json")
    return out


def ab_cache(cfg, lang, tag, reg, client, stats, *, requests, system, report):
    """Create, or verify and extend, THIS ARM's explicit cache.

    One cache per arm, because the arms differ precisely in the system prompt
    and the system prompt is what the cache holds. transport._cache_for cannot
    be reused as-is: it derives the registry key from the same `lang` it builds
    the prompt from, and here those are two different strings (the wave tag and
    the real language). Everything it would have called is called here --
    caches.cache_ttl_plan, caches.create / remaining_seconds / extend -- so the
    floor check, the TTL arithmetic and the free-tier refusal are the shipped
    ones.
    """
    from ankidkdeck.batch import caches as batch_caches   # noqa: PLC0415
    from ankidkdeck.batch import transport as batch_transport  # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42    # noqa: PLC0415

    if not cfg.cache_enabled:
        return None
    key = "%s/definition" % tag
    plan_ttl = batch_caches.cache_ttl_plan(
        requests, poll_deadline_s=batch_transport.JOB_WAIT_SECONDS,
        factor=cfg.cache_ttl_factor)
    ttl = plan_ttl["ttl_seconds"]
    record = reg.cache_record(key)
    if record is not None:
        handle = batch_caches.CacheHandle.from_record(record)
        try:
            # caches.get before every submit: the rows resolve the cache when
            # they EXECUTE, hours later, and the failure is per row.
            left = batch_caches.remaining_seconds(client, handle)
            if left is None or left < ttl:
                batch_caches.extend(client, handle, ttl)
            reg.remember_cache(key, handle.as_record())
            reg.remember_declared(tag, handle.declared_tokens)
            return handle
        except s42.CacheUnavailable as exc:
            # An expired cache cannot be updated, only recreated, and the
            # recreate changes the resource name.
            print("  ab-batch: %s" % exc)
            reg.forget_cache(key)
    handle = batch_caches.create(
        client, model=cfg.gemini_model, system_text=system, lang=lang,
        kind="definition", ttl=ttl, stats=stats,
        display_name="ankidkdeck-%s-def" % tag)
    reg.remember_cache(key, handle.as_record())
    reg.remember_declared(tag, handle.declared_tokens)
    report.setdefault("cache", {}).setdefault("created", []).append(
        {"tag": tag, "name": handle.name,
         "declared_tokens": handle.declared_tokens,
         "prompt_sha256": handle.prompt_sha256, "ttl": plan_ttl})
    print("  ab-batch: cache %s for %s, %d declared tokens, ttl %ds (%s)"
          % (handle.name, tag, handle.declared_tokens, ttl,
             plan_ttl["decided_by"]))
    return handle


def ab_cache_delete(cfg, tag, reg, client, report):
    """Delete this arm's cache once its job is terminal and on disk.

    NOT while a job of this arm is still in flight: batch resolves the cache
    when each ROW executes, and a cache deleted after a successful submit made
    all 21 rows of the probe job fail with gRPC code 7. Same rule, same reason,
    as transport._end_of_wave_cache.
    """
    from ankidkdeck.batch import caches as batch_caches   # noqa: PLC0415

    record = reg.cache_record("%s/definition" % tag)
    if not record:
        return
    in_flight = [j["job_id"] for j in reg.in_flight() if j.get("lang") == tag]
    if in_flight:
        report.setdefault("cache", {}).setdefault("kept", []).append(
            {"tag": tag, "name": record.get("name"),
             "jobs_in_flight": sorted(in_flight)})
        return
    handle = batch_caches.CacheHandle.from_record(record)
    hours = max(0.0, float(handle.ttl_seconds) / 3600.0)
    cost = batch_caches.storage_cost(handle, hours, cfg.mode)
    result = batch_caches.delete(client, handle)
    reg.forget_cache("%s/definition" % tag)
    report.setdefault("cache", {}).setdefault("deleted", []).append(
        dict(result, tag=tag, storage_cost=cost))


# How advanced a job record is, for ab_job_of. RECOVERED beats DOWNLOADED beats
# anything still in flight beats a terminal failure; ties go to the LAST record
# written. A state this map does not name (FAILED) ranks 0, which is what makes
# a stale failure lose to the success that followed it.
_AB_JOB_RANK = {"RECOVERED": 4, "DOWNLOADED": 3, "SUBMITTED": 2, "PLANNED": 2}


def ab_job_of(reg, tag):
    """This arm's MOST ADVANCED job record, or None.

    NOT the first match. `reg.jobs()` is a JSON-backed dict, so plain iteration
    is INSERTION order, and a first-match lookup returns the OLDEST record for
    the tag. That is a duplicate-spend loop, and the state it needs is an
    ordinary event rather than an exotic one:

      1. a job hits the documented 48h expiry. EXPIRED is in
         registry.TERMINAL_JOB_STATES and _drain_one records it FAILED with
         resubmittable=True -- correctly, since nothing was billed.
      2. the operator re-runs. find_by_fingerprint deliberately skips a
         resubmittable FAILED record, next_job_id hands out `-a2`, the arm runs,
         is ingested, is marked RECOVERED, and its cache is deleted and
         forgotten at end of wave. Records for the tag are now
         [FAILED, RECOVERED].
      3. EVERY later invocation of the same command used to match the FAILED
         record, see neither RECOVERED nor DOWNLOADED, and fall into the submit
         branch. ab_cache finds no cache record (step 2 deleted it) so it MINTS
         A NEW ONE; wave_fingerprint folds cache_name into its material, so the
         fingerprint is new; so find_by_fingerprint does not stop the resubmit;
         so next_job_id walks on to `-a3` and the arm is PAID FOR AGAIN, in
         full, on every subsequent run -- with `resumed` reporting False, so
         nothing in the output says it had already been measured.

    Every shipped anti-double-bill layer was working. The tool was asking the
    registry the wrong question.
    """
    best = None
    best_rank = -1
    for job in reg.jobs().values():
        if job.get("lang") != tag:
            continue
        rank = _AB_JOB_RANK.get(job.get("state"), 0)
        if rank >= best_rank:          # >= : within a rank, the LAST one wins
            best, best_rank = job, rank
    return best


def ab_job_cell_keys(job):
    """The cell keys a stored job actually asked for, sorted.

    The second half of the adoption guard. ab_selection_id already puts the cell
    set in the tag, so a mismatch here should be unreachable -- this reads the
    keys back out of the PLAN the registry stored at submit time and compares
    them, so that if the tag scheme is ever weakened the run refuses instead of
    serving a measurement of a different sample under this one's name.
    """
    return sorted(cell["key"]
                  for row in (job.get("plan") or [])
                  for cell in (row.get("cells") or []))


def run_arm_batch(cfg, lang, arm, batches, todo, ledger_path, usage_log, *,
                  client, stats, reg, report, sleep=None, now=None):
    """Place this arm's calls as ONE BATCH JOB, through the shipped transport.

    Returns the same dict run_arm() returns, so verdict() cannot tell the two
    surfaces apart -- which is the point: criteria (a), (b) and (c) and the
    blind pairs are computed from this dict and must not change shape with the
    transport.

    RESTARTABLE at every step, because a batch drain is a foreground wait of up
    to 50 hours and the process dying inside it is a normal event:

      SUBMITTED       transport._resume_in_flight polls and downloads it. Never
                      a second create -- batches.create is not idempotent.
      PLANNED         adopted by matching the uploaded input file, or released
                      if nothing was ever uploaded. Same function, same rules.
      DOWNLOADED      the results are on disk; this run ingests them.
      RECOVERED       already ingested. The arm's result is read back out of the
                      registry record rather than recomputed, so a re-run neither
                      re-appends ledger rows nor resubmits.
    """
    from ankidkdeck import prompts                        # noqa: PLC0415
    from ankidkdeck.batch import registry as batch_registry    # noqa: PLC0415
    from ankidkdeck.batch import transport as batch_transport  # noqa: PLC0415
    from ankidkdeck.batch import waves as batch_waves     # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42    # noqa: PLC0415
    from ankidkdeck.util import FatalError                # noqa: PLC0415

    pid, stage = ARMS[arm]
    # Switch the WHOLE process to this arm's prompt family, exactly as the
    # interactive arm does: CallContext.request refuses a system instruction
    # that is not system_prompt(kind, lang), so the arm is a property of the run
    # and never an argument smuggled past the check.
    prompts.reset()
    prompts.activate(cfg, prompt_id=pid, ramp_stage=stage)
    system = s42.system_prompt("definition", lang)
    prompt_chars = len(system)
    effective_id = prompts.effective_prompt_id(lang)
    tag = ab_wave_tag(lang, arm, ab_selection_id(todo))
    want_keys = sorted(str(r["key"]) for r in todo)
    pool = s42._pool_from_env(cfg)
    summary = {"jobs": []}

    # FIRST, before anything is created: finish what a previous invocation left
    # in flight. This is the shipped recovery, not a second implementation of
    # it, and it is what makes a re-run with the same arguments safe.
    batch_transport._resume_in_flight(cfg, reg, client, tag, summary=summary,
                                      sleep=sleep, now=now)
    job = ab_job_of(reg, tag)
    if job is not None:
        # Belt to ab_selection_id's braces. Unreachable while the cell set is in
        # the tag; if that ever weakens, refuse rather than serve a measurement
        # of one sample under another sample's name.
        got_keys = ab_job_cell_keys(job)
        if got_keys and got_keys != want_keys:
            raise FatalError(
                "arm %s: job %r was submitted for %d cell(s) and this run "
                "selected %d. Adopting it would report one sample's criteria "
                "under another sample's n. Nothing was resubmitted."
                % (arm, job.get("job_id"), len(got_keys), len(want_keys)))
    if job is not None and job.get("state") == batch_registry.RECOVERED:
        print("  ab-batch: arm %s was already ingested as %s (%d cells); "
              "nothing resubmitted"
              % (arm, job["job_id"], len(want_keys)))
        return dict(job["outcome"], arm=arm, prompt_id=pid, ramp_stage=stage,
                    blocks=(None if stage is None
                            else list(prompts.ramp_stage_blocks(stage))),
                    prompt_chars=prompt_chars, resumed=True)

    if job is None or job.get("state") != batch_registry.DOWNLOADED:
        # EVERY REFUSAL THAT CAN STILL FIRE, FIRES BEFORE caches.create. A
        # CachedContent is a billable object with a multi-hour TTL, and the
        # group-count refusal below is a NEW one production does not have --
        # reachable by an ordinary `--entries` mistake. Planning needs nothing
        # from the cache: `cacheable` is a property of the CONFIGURATION (a
        # handle is returned iff cfg.cache_enabled), and the context's
        # cache_name is attached afterwards, before the JSONL is written.
        ctx = s42.CallContext(cfg=cfg, pool=pool, fit=s42.output_fit(cfg),
                              lang=lang, usage=usage_log, prompt_id=pid,
                              mode="batch", cache_name=None)
        system_tokens = s42.system_prompt_tokens(stats, lang)
        prompt_fit = s42.prompt_token_fit(stats)
        if system_tokens is None or prompt_fit is None:
            raise FatalError(
                "the wave splitter needs PROMPT_TOKENS_system_only.%s and "
                "PROMPT_TOKENS_fit from the measured constants, and one of "
                "them is not on disk. The enqueued limit is a hard refusal at "
                "submit; guessing a request's size is not a way to meet it."
                % lang)
        planned = batch_transport.plan_requests(
            s42, ctx, todo, "definition", ab_key_tag(tag),
            system_tokens=system_tokens, prompt_fit=prompt_fit,
            cacheable=bool(cfg.cache_enabled))
        if not planned:
            raise FatalError("arm %s planned no request at all" % arm)
        # ONE JOB PER ARM is what makes the arms comparable: two jobs would be
        # two queue positions and two drain windows for one measurement. The
        # shipped splitter decides whether that holds, rather than an assumption
        # about how big 20 entries are.
        counts_cached, _ = batch_waves.enqueued_counts_cached(stats)
        groups = batch_waves.split_into_jobs(
            [{"entry_id": p.entry_id, "cached_tokens": p.cached_tokens,
              "uncached_tokens": p.uncached_tokens, "planned": p}
             for p in planned],
            target_tokens=batch_waves.job_token_target(),
            counts_cached=counts_cached)
        if len(groups) != 1:
            raise FatalError(
                "arm %s would need %d batch jobs at --entries %d, and the A/B "
                "is one job per arm so that both arms take one queue position "
                "and one drain window. Re-run with fewer entries."
                % (arm, len(groups), len(batches)))
        # Only now is anything billable created.
        handle = ab_cache(cfg, lang, tag, reg, client, stats,
                          requests=len(planned), system=system, report=report)
        ctx.cache_name = handle.name if handle else None
        job = batch_transport._submit_and_drain(
            s42, cfg, ctx, reg, client, tag, "definition", planned,
            handle=handle, wave=0, index=0, effective_id=effective_id,
            summary=summary, sleep=sleep, now=now)

    if job.get("state") != batch_registry.DOWNLOADED:
        raise FatalError(
            "arm %s ended in state %s (%s) with no result file on disk. A "
            "submitted job that produced nothing is money committed for "
            "nothing and it does not resolve itself: look the job up in the "
            "console before re-running."
            % (arm, job.get("state"), job.get("failure") or "no reason "
               "recorded"))

    result = ab_ingest(cfg, lang, arm, job, todo, ledger_path, usage_log,
                       prompt_chars=prompt_chars, report=report)
    # RECOVERED only after the rows are on disk. This is what stops a re-run
    # from ingesting the same paid job twice: the ledger dedupes on a per-call
    # (ts, seq) a second process cannot reproduce, so the guard has to be the
    # job's state.
    reg.mark_recovered(job["job_id"], result)
    # The job is terminal and its results are on disk, so the cache is safe to
    # delete. Storage is billed by the token-hour.
    ab_cache_delete(cfg, tag, reg, client, report)
    return dict(result, arm=arm, prompt_id=pid, ramp_stage=stage,
                blocks=(None if stage is None
                        else list(prompts.ramp_stage_blocks(stage))),
                prompt_chars=prompt_chars, resumed=False)


def ab_row_uid(job_id, row_key):
    """The DETERMINISTIC identity of one ingested row, for S2's dedupe.

    Not (ts, seq): billing.usage_row_uid dedupes on those, and a second process
    cannot reproduce them, which is precisely why a re-ingest used to
    double-count. (job, row key) is reproducible by any process, for ever.
    """
    return "%s|%s" % (job_id, row_key)


def ab_rows_already_written(path):
    """The ab_row_uid values an append-only file already carries.

    A crash INSIDE ab_ingest -- after row 10 of 20 -- leaves the job DOWNLOADED,
    and the next run re-ingests the whole result file. The job-state guard
    (DOWNLOADED -> RECOVERED) closes the whole-file case and cannot close this
    one: the state only advances after the last row. So the appends themselves
    have to be idempotent. Duplicates here are not merely untidy -- they inflate
    month-to-date in the spend ledger and over-weight one arm in
    <probes>/calls.jsonl, which is the artifact backfill_probe_stats re-derives
    the consumption constants from.
    """
    seen = set()
    if path is None or not Path(path).exists():
        return seen
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue        # a torn last line from a crash; not a uid
            uid = row.get("ab_row_uid")
            if uid:
                seen.add(uid)
    return seen


def ab_ingest(cfg, lang, arm, job, todo, ledger_path, usage_log, *,
              prompt_chars, report):
    """Reconcile one downloaded A/B job and write everything downstream needs.

    The join is reconcile()'s, key to key, with its bijection hard guard intact:
    60 rows will not shuffle, but the guard is what makes "joined by key" a fact
    rather than a hope, and the first real wave measured the output NOT to be in
    input order past row 1000.

    IDEMPOTENT PER ROW (see ab_rows_already_written). Re-ingesting a job is a
    normal consequence of a crash mid-ingest, and it must reproduce the same
    measurement without appending a single duplicate line. The outcome itself is
    always recomputed -- only the disk append is skipped -- so a partially
    ingested job completes rather than being half-counted.
    """
    from ankidkdeck.batch import reconcile as batch_reconcile  # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42    # noqa: PLC0415

    pid = ARMS[arm][0]
    lines = []
    with open(job["results_path"], encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                lines.append(json.loads(raw))
    order: dict = {}
    outcomes = batch_reconcile.reconcile(job["plan"], lines, report=order)
    print("  ab-batch: %s" % batch_reconcile.order_note(order))
    pos_key_of = {r["key"]: (r.get("pos_key") or "") for r in todo}
    declared = job.get("declared_cache_tokens")
    produced, thoughts = [], []
    cache_rows = {"rows": 0, "cached_equals_declared": 0, "declared": declared}
    written_probe = ab_rows_already_written(ledger_path)
    written_usage = ab_rows_already_written(getattr(usage_log, "path", None))
    reused = 0
    for row, outcome in zip(job["plan"], outcomes):
        # The ledger row, built by the SAME normalize_usage the production batch
        # path uses, and labelled mode="batch" because that is the surface it
        # really ran on.
        ledger = s42.normalize_usage(
            outcome.usage, model=job["model"],
            label="%s batch %s" % (outcome.kind,
                                   row.get("label") or row["key"]),
            kind=outcome.kind, mode="batch", cache_name=job.get("cache_name"),
            cache_prompt_sha256=job.get("cache_prompt_sha256"),
            prompt_id=pid, finish_reason=outcome.finish_reason,
            n_expected=outcome.n_expected)
        ledger["max_output_tokens"] = row.get("cap")
        # None by construction on the cached path (systemInstruction XOR
        # cachedContent); cache_prompt_sha256 above is what carries the prompt.
        ledger["prompt_sha256"] = (None if job.get("cache_name")
                                   else s42.prompt_sha256(
                                       s42.system_prompt("definition", lang)))
        ledger["job_name"] = job.get("job_name")
        ledger["row_key"] = outcome.key
        ledger["arm"] = arm
        uid = ab_row_uid(job["job_id"], outcome.key)
        ledger["ab_row_uid"] = uid
        if outcome.error:
            ledger["error"] = "batch_row_error"
            ledger["error_text"] = json.dumps(outcome.error)[:400]
        if uid in written_usage:
            reused += 1
        else:
            usage_log.record(ledger)
        # Disk first, then interpretation. Always. Same probe_row, same
        # usage_to_camel, same file as the interactive arm -- so
        # backfill_probe_stats reads one schema and not two.
        if uid not in written_probe:
            s42.append_jsonl(
                ledger_path,
                dict(probe_row(arm, pid, prompt_chars, cfg.thinking_level,
                               row.get("label") or outcome.key,
                               usage_to_camel(ledger), outcome.n_expected),
                     ab_row_uid=uid))
        thoughts.append(ledger.get("thinking_tokens") or 0)
        if declared:
            cache_rows["rows"] += 1
            if int(ledger.get("cached_tokens") or 0) == int(declared):
                cache_rows["cached_equals_declared"] += 1
        if not outcome.ok:
            continue
        for cell, obj in zip(row["cells"], outcome.items):
            produced.append({"key": cell["key"],
                             "pos_key": pos_key_of.get(cell["key"], ""),
                             "lemma": (obj or {}).get("lemma") or "",
                             "gloss": (obj or {}).get("gloss") or "",
                             "provenance": "prompt_ab:%s" % pid})
    if declared and cache_rows["rows"] != cache_rows["cached_equals_declared"]:
        # Reported loudly, not raised: the money is already spent and the
        # measurement is still the measurement. `cached == declared` per row is
        # the criterion (the cached/prompt ratio the audit wanted slides from
        # 0.935 at n=1 to 0.632 at n=20 on a wave that is hitting 1.00).
        print("  ab-batch: WARNING arm %s: %d/%d row(s) reported cached == "
              "declared (%s). The cached-input line of this arm's bill is not "
              "what it was quoted at."
              % (arm, cache_rows["cached_equals_declared"], cache_rows["rows"],
                 declared))
    if reused:
        print("  ab-batch: arm %s: %d row(s) were already on disk from an "
              "interrupted ingest and were NOT appended again" % (arm, reused))
    # Keyed by the job's own `lang` field, which IS this arm's wave tag -- read
    # off the record rather than rebuilt, so the report cannot disagree with the
    # registry about which wave it is describing.
    report.setdefault("cache_check", {})[job["lang"]] = cache_rows
    report.setdefault("order_cross_check", {})[job["lang"]] = order
    return {"calls": len(thoughts), "thoughts": thoughts, "produced": produced,
            "job_id": job["job_id"], "job_name": job.get("job_name"),
            "rows_already_written": reused,
            "cache_check": cache_rows, "order_cross_check": order}


def run_batch_ab(cfg, lang, arms, batches, ledger_path, *, client=None,
                 sleep=None, now=None):
    """Every arm as its own batch job, with the pre-spend block in front.

    Returns ({arm: result}, report). The result dicts are the ones verdict()
    reads, in the shape run_arm() produces, so the criteria and the blind pairs
    are computed identically on both surfaces.
    """
    from ankidkdeck.batch import registry as batch_registry   # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42        # noqa: PLC0415

    # WHAT THIS RUN IS, stated to the three shipped functions that decide what
    # is legal from exactly these two fields: transport_guard (cache_enabled is
    # only allowed on mode=batch, because nothing else drives the cache
    # lifecycle), _pool_from_env (a cache belongs to the key that created it, so
    # a cached run pins ONE key instead of rotating) and
    # Config.effective_service_tier (a batch JSONL row must not carry
    # serviceTier -- jsonl.build_row refuses one that does).
    #
    # THE CACHE IS ON, deliberately, and it is not a cost decision. The
    # production definition wave is cached, and the whole question the A/B
    # answers is what the pipeline will send: an uncached A/B would measure a
    # request shape production never uses. Both arms clear the measured
    # 1,024-token explicit-cache floor (lean ~1,142, rich ~2,822), so neither
    # arm is disadvantaged by the choice.
    #
    # BOTH ARE OVERRIDES OF THE OPERATOR'S ankidkdeck.toml, and cache_enabled
    # commits storage spend nobody configured, so they are ANNOUNCED rather than
    # applied quietly: "--mode batch" reads as "which surface", not "and caching
    # is now on regardless of your config".
    overrides = ab_apply_surface(cfg, "batch", cache_enabled=True)
    stats = s42.probe_stats(cfg)
    todo = todo_rows_for(cfg, lang, batches)
    report = {"mode": "batch", "arms": list(arms),
              "registry_file": AB_REGISTRY_FILE, "config_overrides": overrides,
              "cells": len(todo), "requests_per_arm": len(batches)}
    # BEFORE any cache and before any job: the only place a gate can still
    # refuse something nobody has paid for. PER ARM -- see ab_pre_spend.
    report.update(ab_pre_spend(cfg, lang, todo, arms, stats, surface="batch"))
    reg = batch_registry.JobRegistry(cfg, file=AB_REGISTRY_FILE)
    pool = s42._pool_from_env(cfg)
    client = client if client is not None else pool.client()
    results = {}
    for arm in arms:
        usage = s42.UsageLog(path=cfg.report_dir / "prompt_ab_usage.jsonl")
        print("--- arm %s (%s%s) on the BATCH surface ---"
              % (arm, ARMS[arm][0],
                 "" if ARMS[arm][1] is None else " " + ARMS[arm][1]))
        results[arm] = run_arm_batch(cfg, lang, arm, batches, todo,
                                     ledger_path, usage, client=client,
                                     stats=stats, reg=reg, report=report,
                                     sleep=sleep, now=now)
        print("  %d calls, thinking tokens: %s"
              % (results[arm]["calls"], results[arm]["thoughts"]))
    report["registry"] = reg.summary()
    return results, report


def plan(cfg, lang, batches, arms, mode="interactive"):
    from ankidkdeck import prompts                      # noqa: PLC0415

    print("--- prompt thinking A/B plan (nothing has been sent) ---")
    detail = ("   (one job per arm, explicit cache per arm, registry %s)"
              % AB_REGISTRY_FILE) if mode == "batch" else ""
    print("  surface             %s%s" % (mode, detail))
    print("  language            %s   pack %s"
          % (lang, prompts.pack_version(lang)))
    print("  entries             %d  (%d senses total)"
          % (len(batches), sum(len(b["rows"]) for b in batches)))
    print("  thinking level      %s" % cfg.thinking_level)
    print("  model               %s" % cfg.gemini_model)
    total = 0
    for arm in arms:
        pid, stage = ARMS[arm]
        blocks = None if stage is None else prompts.ramp_stage_blocks(stage)
        text = prompts.build_definition_prompt(lang, prompt_id=pid,
                                               block_names=blocks)
        est = prompts.estimate_tokens(text)
        calls = len(batches)
        total += calls
        print("  arm %-6s %-12s %6d chars / ~%5d tok x %d calls%s"
              % (arm, pid, len(text), est, calls,
                 "" if stage is None else "   blocks: " + ",".join(blocks)))
    print("  requests to place   %d" % total)
    print()
    print("  criterion (a) needs the lean and rich arms; (b) and (c) are")
    print("  computed for EVERY arm, which is (c)'s per-block attribution;")
    print("  (d) writes 40 blind pairs for a human, answers in a separate")
    print("  file; (e) is the artifact rebase, a separate command:")
    print()
    print("    python3 tools/backfill_probe_stats.py --probes <probes> \\")
    print("        --declare-prompt-id %s --rebase-measurement --write"
          % ARMS["rich"][0])
    print()
    print("  Re-run with --confirm-spend to place the calls.")


def run_arm(cfg, lang, arm, batches, ledger_path, usage_log):
    """Place this arm's calls through the production constructors."""
    from ankidkdeck import prompts                      # noqa: PLC0415
    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415

    pid, stage = ARMS[arm]
    # Switch the WHOLE process to this arm's prompt family and block set.
    # CallContext.request refuses a system instruction that is not
    # system_prompt(kind, lang), which is exactly the guard that makes this the
    # only correct way to do it: the arm is a property of the run, not an
    # argument smuggled past the check.
    prompts.reset()
    prompts.activate(cfg, prompt_id=pid, ramp_stage=stage)
    system = s42.system_prompt("definition", lang)
    prompt_chars = len(system)
    pool = s42._pool_from_env(cfg)
    ctx = s42.CallContext(cfg=cfg, pool=pool, fit=s42.output_fit(cfg),
                          lang=lang, usage=usage_log, prompt_id=pid,
                          mode="standard")
    produced, thoughts = [], []
    for batch in batches:
        n = len(batch["rows"])
        before = len(usage_log.rows)
        got = s42._translate_definition_batch(ctx, cfg.gemini_model,
                                              batch["label"], batch["rows"])
        for row in usage_log.rows[before:]:
            # Disk first, then interpretation. Always.
            s42.append_jsonl(ledger_path,
                             probe_row(arm, pid, prompt_chars,
                                       cfg.thinking_level, batch["label"],
                                       usage_to_camel(row), n))
            thoughts.append(row.get("thinking_tokens") or 0)
        for src, out in zip(batch["rows"], got):
            produced.append({"key": src["key"], "pos_key": src["pos_key"],
                             "lemma": (out or {}).get("lemma") or "",
                             "gloss": (out or {}).get("gloss") or "",
                             "provenance": "prompt_ab:%s" % pid})
    return {"arm": arm, "prompt_id": pid, "ramp_stage": stage,
            "blocks": (None if stage is None
                       else list(prompts.ramp_stage_blocks(stage))),
            "prompt_chars": prompt_chars,
            "calls": len(thoughts), "thoughts": thoughts,
            "produced": produced}


def run_interactive_ab(cfg, lang, arms, batches, ledger_path):
    """Every arm on the SYNCHRONOUS surface, with the same pre-spend block.

    THIS PATH USED TO HAVE NO PRE-SPEND GATE AT ALL. `main()` ran only
    cfg.validate() and the thinking-level check, so
    `--arms lean,rich,stage1,stage2,stage3 --entries 20 --confirm-spend` placed
    up to 100 paid requests behind no G-BUDGET, no G-SCOPE-FROZEN and no N-09
    rule -- the same hole this project closed on stage 50 and on review().

    Three things are DECIDED here rather than inherited, because pasting the
    batch block in unchanged would have been worse than leaving the hole:

      the surface       AB_INTERACTIVE_MODE, never cfg.mode. run_arm stamps its
                        ledger rows mode="standard"; quoting them off a toml
                        that says "batch" during a batch month would price them
                        at half and quote the batch request ceiling of 4 for a
                        path that takes the interactive 25x ladder.
      the cache         OFF, because the interactive arm genuinely cannot cache:
                        run_arm builds its CallContext with no cache_name, so
                        every request carries the system prompt inline. Saying
                        so to cfg is what makes expected_scenario quote
                        lean_uncached -- the column this path is really billed
                        on -- rather than cache_works, and it is what lets
                        transport_guard pass instead of hard-refusing a command
                        that worked yesterday on a cache_enabled toml.
      the arms          the rules are asked PER ARM with that arm's prompt
                        active. See ab_pre_spend and AB_R6_EXEMPTION: a block
                        evaluated once reports green on the very arm it most
                        needs to look at.
    """
    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415

    overrides = ab_apply_surface(cfg, AB_INTERACTIVE_MODE, cache_enabled=False)
    stats = s42.probe_stats(cfg)
    todo = todo_rows_for(cfg, lang, batches)
    report = {"mode": "interactive", "arms": list(arms),
              "config_overrides": overrides,
              "cells": len(todo), "requests_per_arm": len(batches)}
    report.update(ab_pre_spend(cfg, lang, todo, arms, stats,
                               surface=AB_INTERACTIVE_MODE))
    results = {}
    for arm in arms:
        usage = s42.UsageLog(path=cfg.report_dir / "prompt_ab_usage.jsonl")
        print("--- arm %s (%s%s) ---"
              % (arm, ARMS[arm][0],
                 "" if ARMS[arm][1] is None else " " + ARMS[arm][1]))
        results[arm] = run_arm(cfg, lang, arm, batches, ledger_path, usage)
        print("  %d calls, thinking tokens: %s"
              % (results[arm]["calls"], results[arm]["thoughts"]))
    return results, report


def verdict(results, lang, pack, out_dir, ratio=1.25, pos_min=0.95):
    from ankidkdeck import gates                        # noqa: PLC0415

    lean = results.get("lean")
    rich = results.get("rich")
    out = {"criteria": {}, "decision": "LEAN"}
    if lean and rich and lean["thoughts"] and rich["thoughts"]:
        m_lean = statistics.median(lean["thoughts"])
        m_rich = statistics.median(rich["thoughts"])
        # A median of zero makes the ratio undefined, and zero-to-zero is the
        # expected result: state the comparison rather than dividing by it.
        ok_a = m_rich <= max(ratio * m_lean, m_lean)
        out["criteria"]["a_thinking_median"] = {
            "ok": bool(ok_a), "median_lean": m_lean, "median_rich": m_rich,
            "allowed": ratio * m_lean, "n_lean": len(lean["thoughts"]),
            "n_rich": len(rich["thoughts"]),
            "note": ("both medians are 0, which is what the single prior "
                     "observation predicted" if m_lean == m_rich == 0 else "")}
    else:
        out["criteria"]["a_thinking_median"] = {
            "ok": False, "note": "both arms are required for criterion (a)"}
    # (b) and (c) for EVERY arm that ran. Per-arm is the point: patch plan 4.4's
    # criterion (c) asks "which block earned its token", and a single number for
    # the whole rich prompt cannot answer it. The DECISION is still read off the
    # rich arm -- the stages are attribution, not candidates.
    by_arm = {}
    for name in sorted(results):
        got = results[name]
        cells = {r["key"]: r for r in got["produced"]}
        findings = gates.script_findings(cells, lang=lang, kind="definitions",
                                         pack=pack)
        blocking = [f for f in findings if f["tier"] != gates.REVIEW]
        bad = pos_shape_violations(lang, got["produced"], pack)
        rate = 1.0 - (len(bad) / max(1, len(got["produced"])))
        by_arm[name] = {
            "prompt_id": got.get("prompt_id"),
            "ramp_stage": got.get("ramp_stage"),
            "blocks": got.get("blocks"),
            "cells": len(cells),
            "script_blocking": len(blocking),
            "script_by_class": {c: sum(1 for f in blocking
                                       if f["class"] == c)
                                for c in sorted({f["class"]
                                                 for f in blocking})},
            "pos_conformance": round(rate, 4),
            "pos_violations": len(bad),
            "thinking_median": (statistics.median(got["thoughts"])
                                if got.get("thoughts") else None),
            "examples": [{"key": b["key"], "lemma": b["lemma"],
                          "why": b["why"]} for b in bad[:8]],
        }
    out["by_arm"] = by_arm
    if rich:
        r = by_arm["rich"]
        out["criteria"]["b_script_violations"] = {
            "ok": r["script_blocking"] == 0, "cells": r["cells"],
            "blocking": r["script_blocking"], "by_class": r["script_by_class"],
            "review_only": None}
        out["criteria"]["c_pos_shape"] = {
            "ok": r["pos_conformance"] >= pos_min,
            "conformance": r["pos_conformance"], "minimum": pos_min,
            "violations": r["pos_violations"], "examples": r["examples"],
            "per_block_attribution": {
                name: {"blocks": v["blocks"],
                       "pos_conformance": v["pos_conformance"],
                       "thinking_median": v["thinking_median"]}
                for name, v in sorted(by_arm.items())},
            "attribution_note": (
                "run --arms lean,stage1,stage2,rich to fill this in; with only "
                "lean and rich it reports the two ends and attributes nothing."
                if not any(v["ramp_stage"] for v in by_arm.values()) else "")}
    pairs_path = key_path = None
    if lean and rich:
        by_key = {r["key"]: r for r in lean["produced"]}
        rng = random.Random(20260826)
        pairs = []
        for r in rich["produced"]:
            other = by_key.get(r["key"])
            if not other:
                continue
            flip = rng.random() < 0.5
            pairs.append({"key": r["key"],
                          "A": (other if flip else r)["lemma"] + " -- "
                               + (other if flip else r)["gloss"],
                          "B": (r if flip else other)["lemma"] + " -- "
                               + (r if flip else other)["gloss"],
                          "answer_A_is": ("lean" if flip else "rich")})
        rng.shuffle(pairs)
        pairs = pairs[:40]
        # TWO FILES. The answer key used to sit on every pair, one field along
        # from the text the judge was reading -- and criterion (d) is the stop
        # loss on the whole enrichment decision ("if they cannot tell, ship LEAN
        # and keep the money"), so a file with the answers in it is not a blind
        # test and its PENDING verdict means nothing.
        pairs_path = Path(out_dir) / "prompt_ab_blind_pairs.json"
        key_path = Path(out_dir) / "prompt_ab_blind_key.json"
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        pairs_path.write_text(json.dumps(
            [{"key": q["key"], "A": q["A"], "B": q["B"]} for q in pairs],
            ensure_ascii=False, indent=2), encoding="utf-8")
        key_path.write_text(json.dumps(
            {"note": "Answers for prompt_ab_blind_pairs.json. Do not open "
                     "this file until the pairs have been judged.",
             "seed": 20260826,
             "answers": {q["key"]: q["answer_A_is"] for q in pairs}},
            ensure_ascii=False, indent=2), encoding="utf-8")
    out["criteria"]["d_blind_test"] = {
        "ok": None, "pairs": len(pairs) if pairs_path else 0,
        "file": str(pairs_path) if pairs_path else None,
        "answer_key": str(key_path) if key_path else None,
        "note": "A HUMAN decides this one, from the pairs file only. The "
                "answers are in a SEPARATE file. If the pairs are "
                "indistinguishable, ship LEAN and keep the money."}
    out["criteria"]["e_constant_invalidated"] = {
        "ok": None,
        "note": "The measured thinking constant belongs to the prompt it was "
                "measured on. Run backfill_probe_stats with "
                "--declare-prompt-id and --rebase-measurement before any "
                "--confirm-spend on the rich prompt; without it the money "
                "gates refuse, which is correct."}
    mechanical = [v.get("ok") for k, v in out["criteria"].items()
                  if v.get("ok") is not None]
    out["mechanical_criteria_pass"] = bool(mechanical) and all(mechanical)
    out["decision"] = ("RICH pending the human blind test"
                       if out["mechanical_criteria_pass"] else "LEAN")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--entries", type=int, default=20,
                    help="entries per arm (default 20, the median the single "
                         "prior rich observation is being replaced with)")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="lean,rich decides; add stage1,stage2 for criterion "
                         "(c)'s per-block attribution (known: %s)"
                         % ",".join(sorted(ARMS)))
    ap.add_argument("--probes", help="directory holding calls.jsonl "
                                     "(default <work>/probes)")
    ap.add_argument("--config", help="ankidkdeck.toml (default: the usual "
                                     "search path)")
    ap.add_argument("--mode", choices=("interactive", "batch"),
                    default="interactive",
                    help="which SURFACE places the calls. interactive is the "
                         "original behaviour, unchanged. batch runs one job "
                         "per arm through the shipped batch transport, with an "
                         "explicit cache per arm and its own job registry "
                         "(%s) -- the interactive surface 503-storms, and an "
                         "A/B whose arms meet a storm at different rates is a "
                         "measurement of the weather." % AB_REGISTRY_FILE)
    ap.add_argument("--confirm-spend", action="store_true",
                    help="place the calls. Without this nothing is sent.")
    args = ap.parse_args(argv)

    from ankidkdeck import prompts                      # noqa: PLC0415
    from ankidkdeck.config import load_config           # noqa: PLC0415

    cfg = load_config(Path(args.config)) if args.config else load_config()
    cfg.validate()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print("unknown arm(s): %s (known: %s)"
              % (", ".join(unknown), ", ".join(sorted(ARMS))), file=sys.stderr)
        return 2
    if cfg.thinking_level != "LOW":
        print("thinking_level is %s, not LOW. The A/B measures the LOW arm; "
              "measuring anything else would produce a constant the pipeline "
              "does not use." % cfg.thinking_level, file=sys.stderr)
        return 2

    prompts.activate(cfg, prompt_id=ARMS["lean"][0])
    batches = pick_entries(cfg, args.lang, args.entries)
    if not batches:
        print("no entries in scope with 2-8 senses: is this work dir built?",
              file=sys.stderr)
        return 2
    if not args.confirm_spend:
        plan(cfg, args.lang, batches, arms, mode=args.mode)
        return 0

    probes = Path(args.probes) if args.probes else (cfg.work_dir / "probes")
    probes.mkdir(parents=True, exist_ok=True)
    ledger = probes / "calls.jsonl"
    pack = prompts.packs.load(args.lang, cfg)
    # BOTH surfaces go through a run_*_ab that evaluates the pre-spend block per
    # arm before it places anything. The interactive branch used to inline its
    # own loop here and place calls behind no gate at all.
    if args.mode == "batch":
        results, surface_report = run_batch_ab(cfg, args.lang, arms, batches,
                                               ledger)
    else:
        results, surface_report = run_interactive_ab(cfg, args.lang, arms,
                                                     batches, ledger)
    out = verdict(results, args.lang, pack, cfg.work_dir / "review")
    out["lang"] = args.lang
    out["mode"] = args.mode
    out["model"] = cfg.gemini_model
    out["ledger"] = str(ledger)
    out["surface_report"] = surface_report
    path = cfg.report_dir / "prompt_ab_verdict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(out["criteria"], ensure_ascii=False, indent=2))
    print("verdict -> %s   (%s)" % (out["decision"], path))
    print()
    print("next, and required before any rich --confirm-spend:")
    print("  python3 tools/backfill_probe_stats.py --probes %s \\" % probes)
    print("      --declare-prompt-id %s --rebase-measurement --write"
          % ARMS["rich"][0])
    return 0 if out["mechanical_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
