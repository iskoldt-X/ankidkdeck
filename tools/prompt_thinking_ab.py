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

USAGE
-----
  # 1. see the plan, spend nothing
  python3 tools/prompt_thinking_ab.py --lang German --entries 20

  # 2. place the calls (the owner presses this)
  python3 tools/prompt_thinking_ab.py --lang German --entries 20 --confirm-spend

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
    """
    return {"record": "call", "probe_id": "prompt_ab_%s" % arm,
            "arm": arm, "prompt_id": prompt_id, "label": label,
            "config": {"thinking_config": {"thinking_level": level}},
            "request_fingerprint": {"prompt_chars": prompt_chars,
                                    "n_expected": n},
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


def plan(cfg, lang, batches, arms):
    from ankidkdeck import prompts                      # noqa: PLC0415

    print("--- prompt thinking A/B plan (nothing has been sent) ---")
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
        plan(cfg, args.lang, batches, arms)
        return 0

    from ankidkdeck.stages import s42_translate as s42   # noqa: PLC0415

    probes = Path(args.probes) if args.probes else (cfg.work_dir / "probes")
    probes.mkdir(parents=True, exist_ok=True)
    ledger = probes / "calls.jsonl"
    pack = prompts.packs.load(args.lang, cfg)
    results = {}
    for arm in arms:
        usage = s42.UsageLog(path=cfg.report_dir / "prompt_ab_usage.jsonl")
        print("--- arm %s (%s%s) ---"
              % (arm, ARMS[arm][0],
                 "" if ARMS[arm][1] is None else " " + ARMS[arm][1]))
        results[arm] = run_arm(cfg, args.lang, arm, batches, ledger, usage)
        print("  %d calls, thinking tokens: %s"
              % (results[arm]["calls"], results[arm]["thoughts"]))
    out = verdict(results, args.lang, pack, cfg.work_dir / "review")
    out["lang"] = args.lang
    out["model"] = cfg.gemini_model
    out["ledger"] = str(ledger)
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
