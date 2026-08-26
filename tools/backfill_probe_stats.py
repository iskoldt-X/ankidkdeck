#!/usr/bin/env python3
"""N-09: re-derive the measured LLM constants from the RAW probe ledger and
reconcile work/probes/stats.json against them.

    python3 tools/backfill_probe_stats.py --probes ~/v3run/work/probes
    python3 tools/backfill_probe_stats.py --probes ~/v3run/work/probes \\
        --declare-prompt-id v4-frozen --write

Why a tool and not a hand edit. stats.json is the file that authorises spending:
every request's output cap, the cache floor, the thinking constant and the whole
bill are read from it, and nothing in the package has a default for any of them.
A number that got into it by hand has no provenance, and "the constant that
sizes paid requests was typed in" is exactly the state the consumption rules
exist to forbid. So this recomputes each consumption-critical value from
calls.jsonl -- the append-only, one-line-per-call probe ledger, written before
any of it was parsed -- and then says, per key: CONFIRMED (the artifact agrees
with the raw ledger), DISAGREES (it does not, with both numbers), or MISSING.

What it will WRITE (only with --write, and only what is missing or provably
stale):

  prompt_id / prompt_lineage    which prompt PACK the constants were measured
                                on. Nothing in the artifact said, and consumption
                                rule 6 ("measured on LEAN, spent on RICH") cannot
                                be enforced against an artifact that does not
                                say. Requires --declare-prompt-id: the tool will
                                not invent a lineage, it will only record the one
                                a human states, together with the evidence for
                                and against it. RELABELLING an existing basis
                                onto a different pack additionally requires
                                --rebase-measurement (see below).
  wave2.floor_error_verbatim    the cache-floor error text. The artifact's copy
                                is an EARLIER 429 ("prepayment credits are
                                depleted") from the same probe arm; the 400 that
                                actually states the floor arrived 16 minutes
                                later (01:04:40 vs 01:20:27) and was never
                                written back, taking floor_classifier to
                                UNCLASSIFIED with it. The deriver takes the FIRST
                                match, so this is "the early wrong value was
                                kept", not "a later value overwrote the right
                                one". The 400 is still in calls.jsonl.
  CONSUMPTION_GUARD             set when a required constant is missing,
                                disagrees, or has been relabelled onto a pack
                                nobody re-measured; REMOVED when none of that
                                holds. Two readers, both of which refuse while it
                                is present: `ankidkdeck doctor` and consumption
                                rule R1-guard in billing.consumption_rules (which
                                assert_ready_to_spend raises on, before the first
                                paid call). It had ONE reader for a while --
                                doctor -- while this docstring claimed two, i.e.
                                the switch designed to stop a spend could not.

What it will NOT write: any constant the raw ledger cannot support, and any key
nothing consumes. SCHEMA_TOKENS is the example of the second kind -- the patch
plan lists it as missing, but the schema is already inside PROMPT_TOKENS_fit's
intercept (the fit was measured on system prompt + schema + payload), so the
money math never asks for it. An unused key is not a gap.

Zero API calls. Reads calls.jsonl and stats.json; writes stats.json (with a
dated pre-image next to it) and a provenance report.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

# The keys the production code actually consumes, and who consumes them. Kept in
# step with s42_translate.REQUIRED_STATS_KEYS plus the money stack's readers; the
# test asserts the required half is a subset of this table.
CONSUMED = {
    "EXPECTED_OUTPUT.a": "sizes every request (max_output_tokens) and the bill's output side",
    "EXPECTED_OUTPUT.b": "same fit's intercept",
    "PROMPT_TOKENS_fit.a": "the bill's per-request input tokens",
    "PROMPT_TOKENS_fit.b": "same fit's intercept (system prompt + schema)",
    "PROMPT_TOKENS_system_only": "the cached half of the input, per language",
    "CHARS_PER_TOKEN": "offline prompt sizing (consumption rule 6)",
    "thinking.THINKING_PER_REQUEST_LOW": "the bill's thinking term and G-THINK",
    "wave2.EXPLICIT_CACHE_FLOOR": "the cache constructor's floor",
    "wave2.W2_2_rich.cached": "the forbidden RICH-uncached figure on the bill",
    "budget.MAX_OUTPUT_FORMULA": "the formula the cap is derived from, in words",
    "temperature.VERDICT": "the decision not to send temperature",
}

# Tolerances for "the artifact agrees with the raw ledger". A regression fit
# depends on which observations the deriver included, and this tool cannot know
# that, so a small relative difference is agreement and a large one is a finding.
# The FIT comparison uses its own, wider band (10%) because it is asserted on the
# PREDICTION rather than on the coefficients -- stated at the comparison.
FIT_TOLERANCE = 0.05
FIT_PREDICTION_TOLERANCE = 0.10

# What actually happened to wave2.floor_error_verbatim, stated once so the
# artifact and this tool cannot drift apart. The original note said a "later"
# 429 had overwritten the 400; the timestamps say the opposite.
FLOOR_NOTE = (
    "restored from calls.jsonl by tools/backfill_probe_stats.py. The field held "
    "the EARLIER of two errors from the same probe arm (W2-1 a_tiny): a 429 "
    "'prepayment credits are depleted' at 2026-08-26T01:04:40+0200. The 400 that "
    "actually states the floor arrived 16 minutes LATER, at 01:20:27, and was "
    "never written back -- derive_stats takes the FIRST match, so nothing "
    "overwrote anything: the wrong value was simply the one on file first, and "
    "floor_classifier went to UNCLASSIFIED with it.")


def p95(values):
    """Linear-interpolated 95th percentile -- the same convention the probe's
    deriver used (verified: 13 MEDIUM observations -> 1042.0)."""
    s = sorted(values)
    if not s:
        return None
    idx = 0.95 * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (idx - lo) * (s[hi] - s[lo]), 4)


def linfit(points):
    """Least squares (a, b, r2, n) for y = a*x + b."""
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n < 2:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    mean = sy / n
    sst = sum((y - mean) ** 2 for _, y in pts)
    sse = sum((y - (a * x + b)) ** 2 for x, y in pts)
    r2 = 1 - sse / sst if sst else None
    return {"a": round(a, 3), "b": round(b, 1),
            "r2": round(r2, 4) if r2 is not None else None, "points": n}


def usage_int(usage, camel):
    value = (usage or {}).get(camel)
    return int(value) if isinstance(value, (int, float)) else 0


def derived_thinking(usage):
    """total - prompt - candidates - toolUse. NEVER thoughtsTokenCount: protobuf
    omits zero-valued fields, so the field is absent exactly when the answer is
    zero -- which is the one case the question is being asked in."""
    return max(0, usage_int(usage, "totalTokenCount")
               - usage_int(usage, "promptTokenCount")
               - usage_int(usage, "candidatesTokenCount")
               - usage_int(usage, "toolUsePromptTokenCount"))


def thinking_level(row):
    cfg = row.get("config") or {}
    tc = cfg.get("thinking_config") or cfg.get("thinkingConfig") or {}
    level = tc.get("thinking_level") or tc.get("thinkingLevel")
    return str(level).upper() if level else None


def load_calls(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rederive(rows: list) -> dict:
    """Every consumption-critical constant, recomputed from the raw ledger.

    ONE PROMPT FAMILY AT A TIME. Both fits and the thinking constant are
    properties of a prompt, not of a model: the ledger holds three different
    system prompts (the definition prompt at 5,123-5,124 characters, the
    homograph-ranking prompt at 1,503, and an enriched 4,564-token arm), and
    pooling them produces a prompt-token slope 38% off the artifact's because a
    handful of large-prompt points drag it. So the fits are computed on the
    DOMINANT family -- the definition wave, which is what the bill is about --
    and the thinking numbers are reported per family as well as per level. That
    split is what surfaced the finding below.
    """
    calls = [r for r in rows if r.get("record") == "call"
             and r.get("usageMetadata")]
    out: dict = {"calls_with_usage": len(calls), "rows_in_ledger": len(rows)}

    # ---- which prompt was each call carrying ----
    sizes = [(r.get("request_fingerprint") or {}).get("prompt_chars")
             for r in calls]
    sizes = [s for s in sizes if isinstance(s, int)]
    modal = max(set(sizes), key=sizes.count) if sizes else None
    # +/- 8 characters is the same prompt: the constantisation defect made the
    # definition prompt vary by 1-2 characters with the batch size, and the next
    # nearest prompt in the ledger is 3,600 characters away.
    same = lambda s: isinstance(s, int) and modal is not None \
        and abs(s - modal) <= 8                             # noqa: E731
    out["prompt_family_chars"] = modal
    out["prompt_families"] = {str(s): sizes.count(s) for s in sorted(set(sizes))}

    def family(row) -> str:
        chars = (row.get("request_fingerprint") or {}).get("prompt_chars")
        if same(chars):
            return "definition(%s)" % modal
        return "other(%s)" % chars

    # ---- thinking, per level, and per (level, family) ----
    by_level: dict = {}
    by_pair: dict = {}
    by_probe: dict = {}
    for row in calls:
        level = thinking_level(row)
        if not level:
            continue
        value = derived_thinking(row["usageMetadata"])
        by_level.setdefault(level, []).append(value)
        by_pair.setdefault((level, family(row)), []).append(value)
        if level == "LOW":
            by_probe.setdefault(row.get("probe_id") or "?", []).append(value)

    def stat(values) -> dict:
        return {"mean": round(sum(values) / len(values), 1), "p95": p95(values),
                "max": max(values), "n_observations": len(values),
                "distinct_values": sorted(set(values))[:10]}

    out["thinking"] = {level: stat(v) for level, v in sorted(by_level.items())}
    out["thinking_by_family"] = {"%s|%s" % pair: stat(v)
                                 for pair, v in sorted(by_pair.items())}
    out["thinking_at_low_by_probe"] = {k: stat(v)
                                       for k, v in sorted(by_probe.items())}

    # ---- the two fits, on the dominant prompt family only ----
    out_points, prompt_points = [], []
    for row in calls:
        fp = row.get("request_fingerprint") or {}
        n = fp.get("n")
        if not isinstance(n, int) or not same(fp.get("prompt_chars")):
            continue
        usage = row["usageMetadata"]
        cand = usage_int(usage, "candidatesTokenCount")
        prompt = usage_int(usage, "promptTokenCount")
        if cand:
            out_points.append((n, cand))
        if prompt:
            prompt_points.append((n, prompt))
    out["EXPECTED_OUTPUT"] = linfit(out_points)
    out["PROMPT_TOKENS_fit"] = linfit(prompt_points)

    # ---- temperature arms ----
    arms = {"t_set": 0, "no_temperature": 0}
    for row in calls:
        cfg = row.get("config") or {}
        arms["t_set" if cfg.get("temperature") is not None
             else "no_temperature"] += 1
    out["temperature_arms"] = arms

    # ---- the cache floor, from the server's own words ----
    floor = None
    verbatim = None
    for row in rows:
        if row.get("min_total_token_count"):
            floor = int(row["min_total_token_count"])
            verbatim = verbatim or row.get("error_verbatim") \
                or row.get("error_text") or row.get("error_text_trunc")
        text = str(row.get("error_text") or row.get("error_text_trunc") or "")
        if "min_total_token_count" in text and verbatim is None:
            verbatim = text
    out["EXPLICIT_CACHE_FLOOR"] = floor
    out["floor_error_verbatim"] = verbatim

    # ---- prompt identity, per n ----
    per_n: dict = {}
    for row in calls:
        fp = row.get("request_fingerprint") or {}
        if fp.get("prompt_sha256") and isinstance(fp.get("n"), int):
            per_n.setdefault(str(fp["n"]), set()).add(fp["prompt_sha256"])
    out["prompt_sha256_per_n"] = {k: sorted(v) for k, v in sorted(
        per_n.items(), key=lambda kv: int(kv[0]))}
    out["prompt_sha256_distinct"] = sorted({s for v in per_n.values()
                                            for s in v})
    out["prompt_chars_seen"] = sorted({
        (row.get("request_fingerprint") or {}).get("prompt_chars")
        for row in calls
        if (row.get("request_fingerprint") or {}).get("prompt_chars")})

    # PER FAMILY, because a flat list inverts the size gate it feeds. Consumption
    # rule 6 sizes the definition prompt against max(measured_prompt_chars), and
    # the patch plan's 4.4 A/B measures LEAN and RICH on the same model: after
    # that run the flat maximum is the RICH size, and the LEAN prompt -- which
    # 4.4 calls a FREE rollback, LEAN being a pure prefix of RICH -- drifts 57%
    # from its own basis and gets refused. The rule has to compare a prompt with
    # the family it belongs to.
    fam_chars: dict = {}
    for row in calls:
        chars = (row.get("request_fingerprint") or {}).get("prompt_chars")
        # A zero-character prompt is the no-system-instruction arm, not a prompt
        # family: it would put a 0 in the band a size is compared against.
        if not isinstance(chars, int) or chars <= 0:
            continue
        fam_chars.setdefault("definition" if same(chars) else "other",
                             set()).add(chars)
    out["prompt_chars_by_family"] = {k: sorted(v)
                                     for k, v in sorted(fam_chars.items())}

    # What the ledger MEASURED, as one value. If a human declares a new prompt
    # pack and this has not changed, no new probe wave happened and the constants
    # are still the old pack's however they are labelled.
    out["ledger_fingerprint"] = hashlib.sha256(json.dumps(
        {"shas": out["prompt_sha256_distinct"],
         "chars": out["prompt_chars_seen"],
         "calls": out["calls_with_usage"]},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]

    # ---- what it cost ----
    out["spend_usd_est"] = round(sum(float(r.get("cost_usd_est") or 0.0)
                                     for r in rows), 6)
    out["usage_totals"] = {
        key: sum(usage_int(r["usageMetadata"], key) for r in calls)
        for key in ("promptTokenCount", "cachedContentTokenCount",
                    "candidatesTokenCount", "totalTokenCount")}
    return out


def dotted(stats: dict, path: str):
    node = stats
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def close(a, b, tol=FIT_TOLERANCE) -> bool:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if b == 0:
        return abs(a) <= tol
    return abs(a - b) / abs(b) <= tol


def reconcile(stats: dict, raw: dict) -> list:
    """One row per consumption-critical key: CONFIRMED / DISAGREES / MISSING."""
    rows = []

    def add(key, verdict, artifact, ledger, note=""):
        rows.append({"key": key, "verdict": verdict, "in_artifact": artifact,
                     "from_calls_jsonl": ledger,
                     "consumed_by": CONSUMED.get(key, ""), "note": note})

    # The fits are compared on their PREDICTIONS, not on their coefficients. A
    # regression's coefficients depend on which observations the deriver
    # included, and this tool cannot know that (the ledger supports several
    # defensible inclusion sets whose slopes span 34.7-36.2); what the code
    # actually consumes is ceil(a*n + b) at n = 1..20, and agreement THERE is
    # the property worth asserting. Disagreement at the coefficient level
    # between two fits over the same data is noise; disagreement at n=20 is a
    # re-measurement.
    for top in ("EXPECTED_OUTPUT", "PROMPT_TOKENS_fit"):
        have = dotted(stats, top) or {}
        mine = raw.get(top) or {}
        if not isinstance(have.get("a"), (int, float)):
            add(top, "MISSING", None, mine or None)
            continue
        deltas = {}
        worst = 0.0
        for n in (1, 8, 20):
            hp = have["a"] * n + have["b"]
            mp = mine["a"] * n + mine["b"]
            rel = abs(hp - mp) / abs(hp) if hp else 0.0
            worst = max(worst, rel)
            deltas["n=%d" % n] = {"artifact": round(hp, 1),
                                  "ledger": round(mp, 1),
                                  "delta_pct": round(rel * 100, 2)}
        add(top, "CONFIRMED" if worst <= FIT_PREDICTION_TOLERANCE
            else "DISAGREES",
            {"a": have.get("a"), "b": have.get("b"),
             "points": have.get("points")},
            {"a": mine.get("a"), "b": mine.get("b"),
             "points": mine.get("points")},
            "predictions %s (agreement is asserted on the prediction, worst "
            "%.2f%%)" % (json.dumps(deltas), worst * 100))

    fam = "LOW|definition(%s)" % raw.get("prompt_family_chars")
    for level in ("LOW", "MEDIUM"):
        key = "thinking.THINKING_PER_REQUEST_%s" % level
        have = dotted(stats, key) or {}
        # LOW is compared against the DEFINITION FAMILY, not the pooled ledger:
        # the constant is what the definition wave will cost, and that is the
        # wave the bill is about.
        mine = ((raw.get("thinking_by_family") or {}).get(fam) if level == "LOW"
                else (raw.get("thinking") or {}).get(level)) or {}
        if not have:
            add(key, "MISSING" if level == "LOW" else "ABSENT (not required)",
                None, mine or None)
            continue
        # n_observations may legitimately differ (the deriver used a narrower
        # arm set); the VALUES may not.
        same = all(close(have.get(f), mine.get(f), 0.01)
                   for f in ("mean", "p95", "max"))
        add(key, "CONFIRMED" if same else "DISAGREES",
            {f: have.get(f) for f in ("mean", "p95", "max", "n_observations")},
            {f: mine.get(f) for f in ("mean", "p95", "max", "n_observations")},
            "compared on the %s arms" % (fam if level == "LOW" else "MEDIUM"))

    # The finding this per-family split exists to surface: thinkingLevel=LOW is
    # a measured 0 on the DEFINITION prompt and a measured 236-275 on the
    # homograph-ranking prompt. Both are LOW. A gate that fails on any non-zero
    # row would fail every healthy ranking wave -- the same shape of mistake as
    # the cached/prompt criterion -- so the scope has to be ON FILE.
    other_low = {k: v for k, v in (raw.get("thinking_by_family") or {}).items()
                 if k.startswith("LOW|") and k != fam and v.get("max")}
    if other_low:
        add("thinking.THINKING_PER_REQUEST_LOW_scope",
            "PRESENT" if dotted(stats,
                                "thinking.THINKING_PER_REQUEST_LOW_scope")
            else "MISSING",
            dotted(stats, "thinking.THINKING_PER_REQUEST_LOW_scope"),
            other_low,
            "LOW is not zero on every prompt: %s. The artifact states a bare "
            "zero, so anything reading it as a property of the MODEL is "
            "reading it wrong." % ", ".join(sorted(other_low)))

    have = dotted(stats, "wave2.EXPLICIT_CACHE_FLOOR")
    add("wave2.EXPLICIT_CACHE_FLOOR",
        "CONFIRMED" if have and have == raw.get("EXPLICIT_CACHE_FLOOR")
        else ("MISSING" if have is None else "DISAGREES"),
        have, raw.get("EXPLICIT_CACHE_FLOOR"))

    have = str(dotted(stats, "wave2.floor_error_verbatim") or "")
    is_floor = "min_total_token_count" in have
    add("wave2.floor_error_verbatim",
        "CONFIRMED" if is_floor else "DISAGREES",
        have[:120], (raw.get("floor_error_verbatim") or "")[:120],
        "" if is_floor else ("the artifact's copy is a different error; the "
                            "floor's own 400 is still in the ledger"))

    for key in ("PROMPT_TOKENS_system_only", "CHARS_PER_TOKEN",
                "budget.MAX_OUTPUT_FORMULA", "temperature.VERDICT",
                "wave2.W2_2_rich.cached"):
        have = dotted(stats, key)
        add(key, "PRESENT" if have not in (None, {}, []) else "MISSING",
            have, None,
            "not recomputable from the ledger alone" if key !=
            "temperature.VERDICT" else
            "arms in the ledger: %s" % raw.get("temperature_arms"))

    add("prompt_id", "PRESENT" if stats.get("prompt_id") else "MISSING",
        stats.get("prompt_id"), None,
        "consumption rule 6 has nothing to compare without it")
    return rows


def live_pack_versions() -> dict:
    """{language: {version, sha256}} for the packs installed right now.

    Declared next to the prompt_id because the pack IS the prompt: `prompt_id`
    names which blocks are assembled, the pack supplies most of their text, and
    editing `fixed_renderings_table` changes what the model is told. Recording
    only the family let a 60-character pack edit change the shipped prompt with
    both halves of consumption rule 6 passing. Rule R6-pack-version compares
    this declaration against the live packs, so a pack bump now needs the same
    explicit --declare-prompt-id --rebase-measurement a family change needs.
    """
    try:
        from ankidkdeck import prompts                # noqa: PLC0415
        return prompts.pack_identity(prompts.available())
    except Exception as exc:                          # noqa: BLE001
        return {"_unavailable": str(exc)}


def prompt_lineage(stats: dict, raw: dict, prompt_id: str,
                   rebased_from=None) -> dict:
    """The declaration consumption rule 6 needs, with its own counter-evidence.

    The honest shape of this record: the probes measured a prompt whose text
    THIS PROGRAM NO LONGER SENDS (item 1.7 made the prompt constant across batch
    sizes, which is why the ledger has one sha per n instead of one sha), so a
    sha equality check can never pass and pretending otherwise would put a
    permanent false green in the artifact. What CAN be checked is the pack
    version and the size, so the record carries both, and it carries the shas it
    was measured on so a reader can see the discontinuity for themselves.

    `size_band_basis` is the part rule 6 actually reads, and it is scoped BOTH
    ways: to a prompt_id (whose pack these sizes belong to) and to a prompt
    family (which prompt inside the pack). Neither scope is decoration --
    without the first, --declare-prompt-id relabels LEAN measurements onto a RICH
    pack in one argument; without the second, the 4.4 A/B makes the documented
    free LEAN rollback fail its own gate.
    """
    shas = raw.get("prompt_sha256_distinct") or []
    system = stats.get("PROMPT_TOKENS_system_only") or {}
    families = raw.get("prompt_chars_by_family") or {}
    out = {
        "prompt_id": prompt_id,
        "declared_by": "human, via tools/backfill_probe_stats.py "
                       "--declare-prompt-id",
        "declared_at": datetime.date.today().isoformat(),
        "measured_prompt_sha256": shas,
        "measured_prompt_chars": raw.get("prompt_chars_seen"),
        "measured_system_tokens": system,
        # WHICH PACKS. Read by consumption rule R6-pack-version.
        "pack_versions": live_pack_versions(),
        "size_band_basis": {
            "prompt_id": prompt_id,
            "by_family": families,
            "ledger_fingerprint": raw.get("ledger_fingerprint"),
            "note": ("consumption rule 6 sizes each prompt against its OWN "
                     "family here, nearest measured value, not against the "
                     "largest prompt in the ledger. `other` is the "
                     "homograph-ranking prompt, which is not a batch size of "
                     "the definition prompt and is 3,600 characters away from "
                     "it -- pooling them is what would break the size band "
                     "after the 4.4 LEAN/RICH A/B."),
        },
        "measured_prompt_chars_note": (
            "flat, across every prompt in the ledger, kept for readers. It is "
            "NOT the gate's basis: %s is." % "size_band_basis.by_family"),
        "sha_equality_is_not_checkable": (
            "the probe set sent %d distinct system prompt shas: %d for the "
            "definition prompt (one per batch size, 1-2 characters apart) and "
            "the rest for other prompts entirely (the homograph-ranking prompt "
            "at 1,503 characters is not a batch size of anything). Patch plan "
            "item 1.7 replaced the per-n definition prompt with one constant "
            "prompt per language, so the sha that WILL be sent is not among "
            "these and never can be. Consumption rule 6 is therefore enforced "
            "on the pack version (prompt_id) plus a 10%% size band against the "
            "prompt's own family, which accepts the constantisation edit (the "
            "German definition prompt is 5,134 characters against a measured "
            "5,124: +10 characters, 0.2%%) and refuses an enrichment (5,124 -> "
            "~11,970 characters, +134%%)."
            % (len(shas), len(raw.get("prompt_sha256_per_n") or {}))),
        "invalidated_by": ("any prompt_id change AND any pack_version change. "
                           "The thinking constant, the prompt-token fit and the "
                           "system prompt size are properties of one prompt "
                           "TEXT, and the pack is most of that text."),
    }
    if rebased_from:
        out["rebased_from"] = rebased_from
        out["rebased_at"] = datetime.date.today().isoformat()
        out["rebase_note"] = (
            "the size-band basis was moved from pack %r to pack %r by a human "
            "passing --rebase-measurement. If the ledger fingerprint did not "
            "change with it, no new probe wave stands behind the new label and "
            "CONSUMPTION_GUARD is set." % (rebased_from, prompt_id))
    return out


def relabel_refusal(stats: dict, raw: dict, declared: str, rebase: bool):
    """Why --declare-prompt-id may not simply relabel this artifact, or None.

    The hole this closes, reproduced by reviewer B: the flag rewrote prompt_id to
    any string with exit 0 and no warning, while measured_prompt_chars and the
    thinking constants stayed exactly the LEAN ones. Rule 6's whole reason to
    exist is "measured on LEAN, spent on RICH", and its stronger half could be
    switched off by one argument. The tool already computes everything needed to
    notice: if the ledger fingerprint behind the basis is unchanged, no new probe
    wave happened and the constants belong to the old pack whatever the label
    says.
    """
    lineage = stats.get("prompt_lineage") or {}
    band = lineage.get("size_band_basis") or {}
    basis_id = band.get("prompt_id") or lineage.get("prompt_id") \
        or stats.get("prompt_id")
    if not basis_id or basis_id == declared:
        return None
    recorded = band.get("ledger_fingerprint")
    if recorded is None:
        evidence = ("The basis records no ledger fingerprint at all, so nothing "
                    "in the artifact can show that a new probe wave stands "
                    "behind the new label.")
        state = "unknown (no fingerprint recorded on the basis)"
    elif recorded == raw.get("ledger_fingerprint"):
        evidence = ("The raw ledger has not changed since that basis was "
                    "written, so no new measurement stands behind the new "
                    "label at all.")
        state = "UNCHANGED since the basis was written"
    else:
        evidence = ("The raw ledger HAS changed, which is necessary but not "
                    "sufficient: the move is still a human decision.")
        state = "changed"
    if rebase:
        return None
    return {
        "basis_prompt_id": basis_id, "declared_prompt_id": declared,
        "ledger_state": state,
        "why": ("this artifact's size-band basis was measured on prompt pack "
                "%r and you are declaring %r. %s Relabelling would let the "
                "thinking constant, the prompt-token fit and the system prompt "
                "size -- all properties of ONE pack -- authorise a spend on "
                "another one, which is the exact failure consumption rule 6 "
                "exists to prevent. Re-run the 4.4 A/B on the new pack, then "
                "pass --rebase-measurement to move the basis deliberately (it "
                "records who moved it, and sets CONSUMPTION_GUARD if the raw "
                "ledger did not change)." % (basis_id, declared, evidence)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", required=True,
                    help="the probe artifact directory (holds calls.jsonl and "
                         "stats.json)")
    ap.add_argument("--stats", help="stats.json path (default: <probes>/stats.json)")
    ap.add_argument("--calls", help="calls.jsonl path (default: <probes>/calls.jsonl)")
    ap.add_argument("--declare-prompt-id", metavar="ID",
                    help="record which prompt PACK the constants were measured "
                         "on. Required to clear the consumption guard; the tool "
                         "will not invent it.")
    ap.add_argument("--rebase-measurement", action="store_true",
                    help="move the size-band basis onto the pack named by "
                         "--declare-prompt-id. Required whenever that pack is "
                         "not the one the current basis was measured on: "
                         "relabelling silently is how LEAN constants come to "
                         "authorise a RICH spend. Sets CONSUMPTION_GUARD when "
                         "the raw ledger has not changed, i.e. when no new "
                         "probe wave stands behind the new label.")
    ap.add_argument("--write", action="store_true",
                    help="patch stats.json (a dated pre-image is written next "
                         "to it first) and write the provenance report")
    args = ap.parse_args(argv)

    probes = Path(args.probes).expanduser()
    stats_path = Path(args.stats).expanduser() if args.stats \
        else probes / "stats.json"
    calls_path = Path(args.calls).expanduser() if args.calls \
        else probes / "calls.jsonl"
    for path in (stats_path, calls_path):
        if not path.exists():
            print("missing: %s" % path, file=sys.stderr)
            return 2

    raw = rederive(load_calls(calls_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    rows = reconcile(stats, raw)

    # Before anything is printed as a plan: may this declaration be made at all?
    # Refusing here rather than after the reconciliation table keeps a refusal
    # from reading like a footnote to a success.
    if args.declare_prompt_id:
        refusal = relabel_refusal(stats, raw, args.declare_prompt_id,
                                  args.rebase_measurement)
        if refusal:
            print("REFUSED: %s" % refusal["why"], file=sys.stderr)
            print("  basis pack   %s" % refusal["basis_prompt_id"],
                  file=sys.stderr)
            print("  declared     %s" % refusal["declared_prompt_id"],
                  file=sys.stderr)
            print("  raw ledger   %s" % refusal["ledger_state"],
                  file=sys.stderr)
            return 3

    print("--- raw ledger: %s ---" % calls_path)
    print("  %d rows, %d of them calls with usage; est spend $%.6f"
          % (raw["rows_in_ledger"], raw["calls_with_usage"],
             raw["spend_usd_est"]))
    print("  thinking by level: %s"
          % {k: {"mean": v["mean"], "p95": v["p95"], "n": v["n_observations"]}
             for k, v in raw["thinking"].items()})
    print("--- reconciliation against %s ---" % stats_path)
    width = max(len(r["key"]) for r in rows)
    for row in rows:
        print("  %-9s %-*s  artifact=%s  ledger=%s"
              % (row["verdict"], width, row["key"],
                 json.dumps(row["in_artifact"], ensure_ascii=False)[:70],
                 json.dumps(row["from_calls_jsonl"], ensure_ascii=False)[:60]))
        if row["note"]:
            print("            %s" % row["note"])

    # The floor's error text and the LOW scope note are BACKFILLABLE from the
    # ledger, so they are not blockers -- this run fixes them. Everything else
    # missing or disagreeing is.
    fixable = ("wave2.floor_error_verbatim",
               "thinking.THINKING_PER_REQUEST_LOW_scope")
    blocking = [r for r in rows
                if r["verdict"] in ("MISSING", "DISAGREES")
                and r["key"] not in fixable
                and not (r["key"] == "prompt_id" and args.declare_prompt_id)]
    patch = copy.deepcopy(stats)
    changes = []

    # 0. the scope of the LOW thinking constant, from the ledger
    scope_row = next((r for r in rows
                      if r["key"] == "thinking.THINKING_PER_REQUEST_LOW_scope"),
                     None)
    if scope_row and scope_row["verdict"] == "MISSING":
        patch.setdefault("thinking", {})
        patch["thinking"]["THINKING_PER_REQUEST_LOW_scope"] = (
            "DEFINITION-PROMPT ARMS ONLY (%s characters of system prompt). "
            "Measured 0 there. NOT a property of the model or of the level: at "
            "the same thinkingLevel=LOW the homograph-ranking prompt produced "
            "236 and 275 thought tokens with the field PRESENT and "
            "finishReason=STOP. A gate that fails on any non-zero derived "
            "thinking would fail every healthy ranking wave."
            % raw.get("prompt_family_chars"))
        patch["thinking"]["THINKING_PER_REQUEST_LOW_by_probe"] = \
            raw.get("thinking_at_low_by_probe")
        patch["thinking"]["THINKING_AT_LOW_BY_PROMPT_FAMILY"] = {
            k: v for k, v in (raw.get("thinking_by_family") or {}).items()
            if k.startswith("LOW|")}
        changes.append("thinking.THINKING_PER_REQUEST_LOW_scope + by_probe + "
                       "by_prompt_family")

    # 1. the floor's own error text, back from the ledger
    floor_row = next(r for r in rows
                     if r["key"] == "wave2.floor_error_verbatim")
    if floor_row["verdict"] == "DISAGREES" and raw.get("floor_error_verbatim"):
        patch.setdefault("wave2", {})
        patch["wave2"]["floor_error_verbatim"] = raw["floor_error_verbatim"]
        patch["wave2"]["floor_error_overwritten_note"] = FLOOR_NOTE
        patch["wave2"]["floor_classifier"] = "FLOOR(min_total_token_count)"
        changes.append("wave2.floor_error_verbatim + floor_classifier")
    elif "overwritten by a later" in str(
            (stats.get("wave2") or {}).get("floor_error_overwritten_note") or ""):
        # The restore was right and the EXPLANATION of it was backwards, and the
        # backwards version is already in the shipped artifact: the 429 is 16
        # minutes EARLIER than the 400 (01:04:40 vs 01:20:27) and the deriver
        # takes the first match, so nothing was overwritten. A wrong causal
        # story in the file that authorises spending is a defect in the file.
        patch.setdefault("wave2", {})
        patch["wave2"]["floor_error_overwritten_note"] = FLOOR_NOTE
        changes.append("wave2.floor_error_overwritten_note (causality was "
                       "stated backwards)")

    # 2. the prompt lineage, if a human declared one. Written only when it
    #    actually differs: re-running with the same --declare-prompt-id used to
    #    rewrite the file and truncate `backfilled.changes` from three entries to
    #    one, i.e. erase the evidence of what this backfill had done -- on the
    #    file that authorises spending. `declared_at` is carried forward for the
    #    same reason, so the artifact is byte-stable across days too.
    rebased_from = None
    if args.declare_prompt_id:
        old_lineage = stats.get("prompt_lineage") or {}
        old_basis = ((old_lineage.get("size_band_basis") or {}).get("prompt_id")
                     or old_lineage.get("prompt_id"))
        if args.rebase_measurement and old_basis \
                and old_basis != args.declare_prompt_id:
            rebased_from = old_basis
        want = prompt_lineage(stats, raw, args.declare_prompt_id,
                              rebased_from=rebased_from)
        if old_lineage.get("prompt_id") == args.declare_prompt_id \
                and old_lineage.get("declared_at"):
            want["declared_at"] = old_lineage["declared_at"]
        if old_lineage.get("rebased_at") and rebased_from is None:
            for key in ("rebased_from", "rebased_at", "rebase_note"):
                if old_lineage.get(key) is not None:
                    want[key] = old_lineage[key]
        if stats.get("prompt_id") != args.declare_prompt_id \
                or old_lineage != want:
            patch["prompt_id"] = args.declare_prompt_id
            patch["prompt_lineage"] = want
            changes.append("prompt_id + prompt_lineage")

    # 3. the guard itself: set it when something is missing, remove it when
    #    nothing is. This is the switch doctor and consumption rule R1-guard
    #    read, and it is also where a rebase with no new measurement lands.
    guard = None
    if rebased_from and (stats.get("prompt_lineage") or {}) \
            .get("size_band_basis", {}).get("ledger_fingerprint") \
            == raw.get("ledger_fingerprint"):
        guard = ("the size-band basis was rebased from pack %r to %r by hand "
                 "and the raw probe ledger did not change, so no new "
                 "measurement stands behind the new label. Run the patch plan's "
                 "4.4 A/B on %r and re-derive before spending."
                 % (rebased_from, args.declare_prompt_id,
                    args.declare_prompt_id))
        patch["CONSUMPTION_GUARD"] = guard
        changes.append("CONSUMPTION_GUARD set (rebase with no new measurement)")
    elif blocking:
        guard = ("Wave-1/2/3 artifact is not fit to authorise a spend: %s. "
                 "Re-derive or declare these before --confirm-spend."
                 % "; ".join("%s (%s)" % (r["key"], r["verdict"])
                             for r in blocking))
        patch["CONSUMPTION_GUARD"] = guard
        changes.append("CONSUMPTION_GUARD set")
    elif stats.get("CONSUMPTION_GUARD"):
        patch.pop("CONSUMPTION_GUARD", None)
        changes.append("CONSUMPTION_GUARD removed")

    print("--- verdict ---")
    if blocking:
        print("  NOT FIT TO SPEND: %s"
              % "; ".join("%s %s" % (r["verdict"], r["key"])
                          for r in blocking))
    else:
        print("  every consumption-critical constant is present and agrees "
              "with the raw ledger")
    print("  changes %s: %s"
          % ("to write" if args.write else "that --write would make",
             ", ".join(changes) or "none"))

    if args.write and changes:
        # ONE pre-image, whatever day the second run happens on. Keying it to
        # today's date meant a re-run on another day wrote a file called
        # `pre-backfill-<later date>` whose CONTENT was the already-backfilled
        # artifact -- a fake pre-image, on the file that authorises spending.
        existing = sorted(stats_path.parent.glob("%s.pre-backfill-*.json"
                                                 % stats_path.stem))
        if existing:
            print("  pre-image already on file: %s" % existing[0])
        else:
            pre = stats_path.with_name(
                "%s.pre-backfill-%s.json" % (stats_path.stem,
                                             datetime.date.today().isoformat()))
            pre.write_text(json.dumps(stats, ensure_ascii=False, indent=1,
                                      sort_keys=True), encoding="utf-8")
            print("  pre-image: %s" % pre)
        # CUMULATIVE. This used to be the current run's list, so a second run
        # replaced "three things were backfilled" with "one thing was" and the
        # provenance of the other two was gone from the artifact.
        prior = list((stats.get("backfilled") or {}).get("changes") or [])
        cumulative = prior + [c for c in changes if c not in prior]
        patch["backfilled"] = {
            "at": datetime.date.today().isoformat(),
            "by": "tools/backfill_probe_stats.py",
            "from": str(calls_path),
            "changes": cumulative,
            "changes_this_run": changes,
            "why": ("the artifact authorises spending, so every value in it has "
                    "to come from the probe ledger rather than from a hand"),
        }
        stats_path.write_text(json.dumps(patch, ensure_ascii=False, indent=1,
                                         sort_keys=True), encoding="utf-8")
        report = stats_path.with_name("stats_backfill_report.json")
        report.write_text(json.dumps(
            {"at": datetime.date.today().isoformat(),
             "calls": str(calls_path), "stats": str(stats_path),
             "rederived": raw, "reconciliation": rows,
             "changes": cumulative, "changes_this_run": changes,
             "consumption_guard": guard},
            ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        print("  wrote %s and %s" % (stats_path, report))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
