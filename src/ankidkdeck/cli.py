"""The single entry point: `ankidkdeck <subcommand>`.

Every subcommand is one stage, every stage is a pure read-files-write-files
function, and nothing is configured by editing source any more -- that was the
v1/v2 workflow this package retires.

Two rules are enforced here rather than inside the stages:

  * Net is constructed ONLY for the subcommands that are allowed to touch the
    network (sitemap, crawl, audio). A parse or an export cannot make a request
    even if a future edit tried to.
  * Spending money needs a flag. `translate` and `priority` print a bill and
    stop; --confirm-spend is the only way past it.

"PRINTS A BILL AND STOPS" IS NOT "READ-ONLY", and the confusion has cost real
work: without --confirm-spend, `translate` still runs gc() and rewrites
definitions.json / expressions.json / archive.json and two reports, and runs
G-ORPH (which can FatalError before a human sees the bill); `priority` still
rewrites words.json, priority_orders.json and ranking_queue.json. Nothing is
SENT. Something is CHANGED. Snapshot work/ before the first dry run of either:

    cp -a work work.before-translate        # or: tar -C work -cf ../work.tar .

Usage sketch:

    ankidkdeck --work work crawl --pilot
    ankidkdeck build
    ankidkdeck priority                           # writes the queue, no calls
    ankidkdeck doctor                             # what a spend would use
    ankidkdeck translate --lang German            # prints the bill, exits
    ankidkdeck translate --lang German --confirm-spend
    ankidkdeck export --lang German --check-determinism

The runbook order is build -> priority -> translate -> audio -> export, and the
`priority` step is not optional bookkeeping: s30_merge and s50_priority write the
same words.json field, so a build after a priority run silently restores every
homograph display order.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import MODES, load_config
from .util import FatalError, read_json


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _flatten(prefix: str, value, out: list, depth: int = 0) -> None:
    if isinstance(value, dict) and depth < 2:
        for k in value:
            _flatten(f"{prefix}.{k}" if prefix else str(k), value[k], out, depth + 1)
        return
    if isinstance(value, list):
        if not value or not isinstance(value[0], (dict, list)):
            shown = ", ".join(str(v) for v in value[:6])
            more = "" if len(value) <= 6 else f" (+{len(value) - 6} more)"
            out.append((prefix, f"[{shown}]{more}" if value else "[]"))
        else:
            out.append((prefix, f"{len(value)} rows"))
        return
    out.append((prefix, value))


def print_report(name: str, report) -> None:
    print("[%s]" % name)
    if not isinstance(report, dict):
        print("  %s" % report)
        return
    rows: list = []
    for key in report:
        _flatten(key, report[key], rows)
    width = max((len(k) for k, _ in rows), default=0)
    for k, v in rows:
        print("  %-*s  %s" % (width, k, v))


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------

def add_global_options(p: argparse.ArgumentParser) -> None:
    """The global options are accepted BEFORE and AFTER the subcommand.

    `ankidkdeck export --lang German --work work` is what a human types, and an
    argparse that only accepts them before the subcommand rejects it. The
    defaults are SUPPRESS so that an option given before the subcommand is not
    overwritten by the subparser's own (absent) default.
    """
    p.add_argument("--work", metavar="PATH", type=Path, default=argparse.SUPPRESS,
                   help="workspace directory (default: ./work). Holds raw/ json/ "
                        "audio/ reports/ review/ and is gitignored.")
    p.add_argument("--config", metavar="PATH", type=Path,
                   default=argparse.SUPPRESS,
                   help="TOML config file (default: ./ankidkdeck.toml)")
    p.add_argument("--legacy-workspace", metavar="PATH", type=Path,
                   default=argparse.SUPPRESS,
                   help="read-only path to the recovered 2025 workspace "
                        "(download_map.json, ddo_entries.json, translations, audio)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ankidkdeck",
        description="Build Danish frequency Anki decks from Den Danske Ordbog.")
    p.add_argument("--version", action="version", version="ankidkdeck " + __version__)
    add_global_options(p)
    common = argparse.ArgumentParser(add_help=False)
    add_global_options(common)
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True,
                           parser_class=lambda **kw: argparse.ArgumentParser(
                               parents=[common], **kw))

    s = sub.add_parser("wordlist", help="pin the wordlist artifact (stage 00)")
    s.add_argument("--accept-new-wordlist", action="store_true",
                   help="confirm a wordlist change; without it a changed sha256 "
                        "is fatal, because the whole deck would re-rank")

    sub.add_parser("sitemap", help="download the sitemap inventory, 9 requests "
                                   "(stage 10)")

    s = sub.add_parser("crawl", help="fetch DDO pages (stage 12)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true",
                   help="300 words spread across the rank range; must be clean "
                        "before the full run is unlocked")
    g.add_argument("--full", action="store_true", help="phase A: one GET per word")
    g.add_argument("--phase-b", action="store_true",
                   help="lemma completion: parse+classify what we have, then "
                        "fetch each kept lemma that was never queried directly")
    g.add_argument("--phase-c", action="store_true",
                   help="registry overrides (form_to_lemma)")

    sub.add_parser("parse", help="parse raw pages into entries.json (stage 20)")
    sub.add_parser("classify", help="four-bucket classifier (stage 22)")
    sub.add_parser("resolve", help="reverse index / overrides / unresolved.json "
                                   "(stage 21)")
    sub.add_parser("merge", help="family merge, GUID freeze, dense rank (stage 30)")
    sub.add_parser("bind", help="bind migrated translations to dannetid (stage 41)")
    sub.add_parser("migrate", help="offline re-key of the 2025 assets (stage 40)")
    sub.add_parser("build", help="parse -> classify -> resolve -> merge -> bind")

    s = sub.add_parser("priority", help="homograph display order (stage 50)")
    s.add_argument("--confirm-spend", action="store_true",
                   help="place the Gemini ranking calls for the queued families. "
                        "The ranking always runs on the standard surface, never "
                        "batch. NOTE: even without this flag the stage REWRITES "
                        "words.json, priority_orders.json and ranking_queue.json "
                        "-- it places no call, but it is not read-only.")

    s = sub.add_parser("translate", help="incremental LLM top-up (stage 42)")
    s.add_argument("--lang", help="one target language; default: all configured")
    s.add_argument("--confirm-spend", action="store_true",
                   help="place the Gemini calls the bill just quoted")
    s.add_argument("--mode", choices=MODES,
                   help="transport for this run (default: the configured mode). "
                        "batch is half price and asynchronous; flex is standard "
                        "plus serviceTier=flex. Nothing downgrades by itself.")
    s.add_argument("--phase", choices=("all", "submit", "ingest"), default="all",
                   help="batch only: submit a wave, or ingest a finished one. "
                        "The drift ledger is consumed on the ingest.")
    s.add_argument("--retranslate-all", action="store_true",
                   help="CLEAN RETRANSLATION: bill every cell in scope, not only "
                        "the missing and changed ones. With --confirm-spend the "
                        "existing rows are moved into archive.json with "
                        "reason=clean_redo and the archive is NOT read back, so "
                        "they are really regenerated. Without --confirm-spend it "
                        "only quotes the bill and touches nothing.")
    s.add_argument("--no-gc", action="store_true",
                   help="skip archiving translation rows with no live sense")
    s.add_argument("--include-unused", action="store_true",
                   help="bill every parsed entry when words.json is absent, "
                        "including the articles the classifier rejected "
                        "(measured 3.4x the cells). Normally you run the merge "
                        "stage first instead.")

    s = sub.add_parser("review", help="hand-run correction pass (stage 42). "
                                      "NEVER triggered automatically.")
    s.add_argument("--lang", required=True, help="target language")
    s.add_argument("--fix", metavar="KEYS",
                   help="comma-separated cell keys to redo; default: every cell "
                        "flagged in work/review/script_violations_<lang>.json")
    s.add_argument("--confirm-spend", action="store_true",
                   help="place the correction calls")

    sub.add_parser("doctor", help="print the EFFECTIVE spend configuration and "
                                  "the state of the measured constants")

    s = sub.add_parser("audio", help="audio cache and delta download (stage 60)")
    s.add_argument("--seed-legacy", action="store_true",
                   help="hardlink/copy from the 2025 workspace instead of "
                        "downloading (~4,629 requests saved)")
    s.add_argument("--sweep-orphans", action="store_true",
                   help="move cached mp3s that entries.json no longer references "
                        "into audio/_orphans/. Only safe once the parse is "
                        "complete: against a partial entries.json this "
                        "quarantines the whole cache.")

    s = sub.add_parser("export", help="render cards and write the .apkg (stage 70)")
    s.add_argument("--lang", required=True, help="target language")
    s.add_argument("--check-determinism", action="store_true",
                   help="build twice and compare notes + media (G-DET)")

    sub.add_parser("status", help="print the ledger and report summaries")
    sub.add_parser("gates", help="print the accumulated gate report")
    return p


# --------------------------------------------------------------------------
# command bodies
# --------------------------------------------------------------------------

def _registry(cfg):
    from .registry import Registry
    return Registry(cfg)


def _net(cfg):
    from .net import Net
    return Net(cfg)


def check_lang(cfg, lang) -> None:
    """--lang must name a CONFIGURED language, checked before anything runs.

    `--lang german` is not `--lang German`: none of the 22,734 migrated cells
    are visible under the lowercase key, so `translate --lang german
    --confirm-spend` would quietly pay the full from-scratch price and write to
    translations/german/. Export is protected by G-COV -- the money is not. The
    escape hatch is `langs` in ankidkdeck.toml, not a typo.
    """
    if lang is None:
        return
    if lang not in cfg.langs:
        raise FatalError(
            "unknown --lang %r. Configured languages are: %s. Add it to `langs` "
            "in ankidkdeck.toml first -- a new language means a full "
            "from-scratch translation bill." % (lang, ", ".join(cfg.langs)))


def _lemmas_needed(cfg, registry) -> set:
    """Phase B's input: the lemma pages we have never asked for.

    The article set depends on WHICH form you query (/har -> 1 article,
    /have -> 3), so this is a correctness requirement, not an optimisation. It
    needs a provisional parse + classify, which is why phase B runs them first.
    """
    from .stages import s20_parse, s22_classify
    print_report("parse (provisional)", s20_parse.run(cfg, registry))
    print_report("classify (provisional)", s22_classify.run(cfg, registry))
    entries = read_json(cfg.json_dir / "entries.json")
    classification = read_json(cfg.json_dir / "classification.json")
    ledger = read_json(cfg.json_dir / "fetch_ledger.json", default={})
    wordset = {w["word"] for w in read_json(cfg.json_dir / "wordlist.json")["words"]}
    kept = {m["entry_id"] for c in classification.values()
            for m in (c.get("members") or [])}
    lemmas = {entries[eid]["lemma"] for eid in kept if eid in entries}
    return {lem for lem in lemmas if lem not in wordset and lem not in ledger}


def run_command(args, cfg) -> int:
    cmd = args.command
    # One place, before any stage runs: every subcommand that takes --lang
    # (translate, export, and anything added later) is validated against
    # cfg.langs. Nothing is fetched, parsed or billed for a typo.
    check_lang(cfg, getattr(args, "lang", None))
    if cmd == "wordlist":
        from .stages import s00_wordlist
        print_report("wordlist", s00_wordlist.run(cfg, args.accept_new_wordlist))
        return 0
    if cmd == "sitemap":
        from .stages import s10_sitemap
        reg = _registry(cfg)
        print_report("sitemap", s10_sitemap.run(cfg, _net(cfg), reg.gates))
        return 0
    if cmd == "crawl":
        from .stages import s12_download
        reg = _registry(cfg)
        net = _net(cfg)
        if args.pilot:
            print_report("crawl pilot", s12_download.run_pilot(cfg, net, reg.gates))
        elif args.full:
            print_report("crawl phase A", s12_download.run_phase_a(cfg, net))
        elif args.phase_b:
            print_report("crawl phase B",
                         s12_download.run_phase_b(cfg, net, _lemmas_needed(cfg, reg)))
        else:
            print_report("crawl phase C", s12_download.run_phase_c(cfg, net, reg))
        return 0
    if cmd in ("parse", "classify", "resolve", "merge", "bind", "migrate", "build"):
        from .stages import (s20_parse, s21_resolve, s22_classify, s30_merge,
                             s40_migrate, s41_bind)
        reg = _registry(cfg)
        steps = {"parse": [("parse", lambda: s20_parse.run(cfg, reg))],
                 "classify": [("classify", lambda: s22_classify.run(cfg, reg))],
                 "resolve": [("resolve", lambda: s21_resolve.run(cfg, reg))],
                 "merge": [("merge", lambda: s30_merge.run(cfg, reg))],
                 "bind": [("bind", lambda: s41_bind.run(cfg, reg))],
                 "migrate": [("migrate", lambda: s40_migrate.run(cfg, reg))]}
        steps["build"] = (steps["parse"] + steps["classify"] + steps["resolve"]
                          + steps["merge"] + steps["bind"])
        for name, fn in steps[cmd]:
            print_report(name, fn())
        return 0
    if cmd == "priority":
        from .stages import s50_priority
        print_report("priority", s50_priority.run(cfg, _registry(cfg),
                                                  confirm=args.confirm_spend))
        return 0
    if cmd == "translate":
        from .stages import s42_translate
        check_lang(cfg, args.lang)
        if args.mode:
            cfg.mode = args.mode
            cfg.validate()
        print_report("translate", s42_translate.run(
            cfg, _registry(cfg), lang=args.lang, confirm=args.confirm_spend,
            do_gc=not args.no_gc, include_unused=args.include_unused,
            retranslate_all=args.retranslate_all, phase=args.phase))
        return 0
    if cmd == "review":
        from .stages import s42_translate
        check_lang(cfg, args.lang)
        keys = [k.strip() for k in (args.fix or "").split(",") if k.strip()]
        print_report("review", s42_translate.review(
            cfg, _registry(cfg), lang=args.lang, keys=keys or None,
            confirm=args.confirm_spend))
        return 0
    if cmd == "doctor":
        return doctor(cfg)
    if cmd == "audio":
        from .stages import s60_audio
        print_report("audio", s60_audio.run(cfg, _net(cfg),
                                            seed_legacy=args.seed_legacy,
                                            sweep_orphans=args.sweep_orphans,
                                            registry=_registry(cfg)))
        return 0
    if cmd == "export":
        from .stages import s70_export
        check_lang(cfg, args.lang)
        print_report("export", s70_export.run(cfg, _registry(cfg), args.lang,
                                              check_determinism=args.check_determinism))
        return 0
    if cmd == "status":
        print_report("status", status(cfg))
        return 0
    if cmd == "gates":
        return gates_report(cfg)
    raise FatalError("unknown command: %s" % cmd)


def status(cfg) -> dict:
    ledger = read_json(cfg.json_dir / "fetch_ledger.json", default={})
    by_status: dict = {}
    for row in ledger.values():
        st = row.get("status", "unknown")
        by_status[st] = by_status.get(st, 0) + 1
    entries = read_json(cfg.json_dir / "entries.json", default={})
    families = read_json(cfg.json_dir / "words.json", default={})
    out = {
        "work_dir": str(cfg.work_dir),
        "wordlist": len(read_json(cfg.json_dir / "wordlist.json",
                                  default={"words": []})["words"]),
        "fetch_ledger": {"words_attempted": len(ledger), **by_status},
        "entries": len(entries),
        "entries_empty": sum(1 for e in entries.values() if e.get("empty")),
        "families": len(families),
        "cards": sum(1 for f in families.values() if f.get("freq_rank")),
        "audio_cached": len(read_json(cfg.audio_dir / "manifest.json", default={})),
    }
    for lang in cfg.langs:
        tdir = cfg.json_dir / "translations" / lang
        out["translations." + lang] = {
            "definitions": len(read_json(tdir / "definitions.json", default={})),
            "expressions": len(read_json(tdir / "expressions.json", default={})),
            "pos": len(read_json(tdir / "pos.json", default={})),
        }
    unresolved = read_json(cfg.report_dir / "unresolved.json", default=[])
    out["unresolved_words"] = len(unresolved)
    return out


def doctor(cfg) -> int:
    """Print the configuration that will ACTUALLY be used, and where it came from.

    This command exists because of a specific near miss: the run host's
    ankidkdeck.toml carried four lines, none of them about the model, so the
    effective model there was the source default -- and a --confirm-spend would
    have paid for a model nothing had been measured on and welded its name into
    every cell's provenance. Nothing in the pipeline printed the effective spend
    configuration before money was placed. Now one command does, and it exits
    non-zero when the run is not fit to spend.
    """
    from .config import VERIFIED_MODELS
    from .gates import read_refreeze_stamp
    from .stages.s42_translate import (CACHEABLE_KINDS, REQUIRED_STATS_KEYS,
                                       missing_stats_keys, prompt_shas,
                                       rate_card_for, thinking_per_request,
                                       unmeasured_thinking_prior)
    problems = []
    print("--- effective spend configuration ---")
    print("  work_dir            %s" % cfg.work_dir)
    print("  model               %s%s"
          % (cfg.gemini_model,
             "" if cfg.model_is_verified() else "   <-- NOT VERIFIED"))
    print("  model (expr/pos)    %s" % cfg.expressions_model)
    if not cfg.model_is_verified():
        problems.append("model %s is not on the verified list (%s)"
                        % (cfg.gemini_model, ", ".join(sorted(VERIFIED_MODELS))))
    else:
        meta = VERIFIED_MODELS[cfg.gemini_model]
        print("  constants measured  %s   rate card read %s"
              % (meta["constants_measured_at"], meta["rate_card_read_at"]))
    print("  mode                %s   (service tier: %s)"
          % (cfg.mode, cfg.effective_service_tier or "-"))
    print("  thinking level      %s%s"
          % (cfg.thinking_level,
             "" if cfg.thinking_level == "LOW"
             else "   <-- NOT LOW: the derived output cap has no thinking term"))
    print("  temperature         not sent (deprecated on this model generation)")
    print("  max output tokens   %s"
          % ("derived per request for definitions (floor %d), flat %d for the "
             "kinds nobody measured" % (cfg.max_output_floor,
                                       cfg.max_output_unmeasured)
             if not cfg.max_output_tokens else cfg.max_output_tokens))
    # Print the shas of the prompts THIS config would send, not the shas of
    # the default prompt: cfg.prompt_id selects the variant, and a doctor that
    # printed the frozen shas under a rich prompt_id would certify the wrong
    # text right above the line that names the id.
    from . import prompts as _prompts
    _prompts.activate(cfg)
    print("  prompt_id           %s   (variant %s, packs %s)"
          % (cfg.prompt_id, _prompts.variant_for(cfg.prompt_id),
             ", ".join("%s=%s" % (lang, _prompts.pack_version(lang))
                       for lang in cfg.langs) or "-"))
    for lang in cfg.langs:
        shas = prompt_shas(lang)
        print("    %-9s def %s  expr %s"
              % (lang, shas["definition"][:12], shas["expression"][:12]))
    print("  explicit cache      %s (ttl factor %s, key index %d)"
          % ("on" if cfg.cache_enabled else "off", cfg.cache_ttl_factor,
             cfg.cache_key_index))
    print("  spend cap           $%s per %s" % (cfg.spend_cap_usd,
                                                cfg.spend_cap_period))
    print("  retranslate_all     %s (reason %r)"
          % (cfg.retranslate_all, cfg.retranslate_reason))
    print("  throttle            def %ss / expr %ss / pos %ss / rank %ss, "
          "%d requests per key"
          % (cfg.def_request_interval, cfg.expr_request_interval,
             cfg.pos_request_interval, cfg.rank_request_interval,
             cfg.max_per_api_key))
    print("  measured RPM/RPD    %s / %s (read %s)"
          % (cfg.rpm_limit if cfg.rpm_limit is not None else "UNMEASURED",
             cfg.rpd_limit if cfg.rpd_limit is not None else "UNMEASURED",
             cfg.rate_limits_measured_at or "never"))
    if cfg.rpm_limit is None or cfg.rpd_limit is None:
        print("    (free tier is a hard 20 requests/model/day and a 503 counts; "
              "the paid tier's per-minute limit has never been measured)")

    print("--- measured constants ---")
    path = cfg.probe_stats_path
    if not path.exists():
        print("  %s   MISSING" % path)
        problems.append("no measured constants at %s: the output cap, the "
                        "thinking constant and the cache floor have no defaults"
                        % path)
    else:
        stats = read_json(path, default={})
        print("  file                %s" % path)
        print("  measured_at         %s" % stats.get("measured_at"))
        print("  model               %s%s"
              % (stats.get("model"),
                 "" if stats.get("model") == cfg.gemini_model
                 else "   <-- DOES NOT MATCH THE CONFIGURED MODEL"))
        if stats.get("model") != cfg.gemini_model:
            problems.append("the measured constants were produced on %r, the "
                            "configured model is %r"
                            % (stats.get("model"), cfg.gemini_model))
        guard = stats.get("CONSUMPTION_GUARD")
        if guard:
            print("  CONSUMPTION_GUARD   %s" % guard)
            problems.append("the constants file still declares a "
                            "CONSUMPTION_GUARD: %s" % guard)
        fit = stats.get("EXPECTED_OUTPUT") or {}
        print("  output fit          a=%s b=%s (R2 %s over %s points)"
              % (fit.get("a"), fit.get("b"), fit.get("r2"), fit.get("points")))
        print("  fit measured on     definition requests only; every other kind "
              "uses the flat cap")
        low = thinking_per_request(stats, "LOW", "p95")
        print("  thinking @ LOW      %s (p95)"
              % ("MISSING" if low is None else low))
        # The measured zero is PROMPT-scoped, not level-scoped: the ranking
        # prompt produced 236-275 thought tokens at the same LOW. So the bill
        # books an unmeasured kind at a labelled prior, and the one output a
        # human reads before --confirm-spend has to say which of the two numbers
        # on that line is a measurement.
        prior, prior_source = unmeasured_thinking_prior(stats)
        print("  thinking @ LOW      %s/request on kinds NOBODY MEASURED "
              "(expression, pos) -- a PRIOR, not a measurement" % prior)
        print("    source            %s" % prior_source)
        print("    scope of the 0    %s"
              % ((stats.get("thinking") or {})
                 .get("THINKING_PER_REQUEST_LOW_scope")
                 or "NOT RECORDED -- the artifact states a bare zero"))
        print("  cached input rate   applies to %s only (the measured "
              "explicit-cache minimum is %s tokens; every other kind pays the "
              "uncached rate in every scenario)"
              % (", ".join(CACHEABLE_KINDS),
                 (stats.get("wave2") or {}).get("EXPLICIT_CACHE_FLOOR")))
        # No fallback for the implicit floor: it is a DIFFERENT number from the
        # explicit one and the real artifact does not carry it, so printing a
        # source constant here would put an invented 4096 in the one output a
        # human reads before pressing --confirm-spend.
        print("  explicit cache floor %s   implicit %s"
              % ((stats.get("wave2") or {}).get("EXPLICIT_CACHE_FLOOR"),
                 stats.get("IMPLICIT_CACHE_FLOOR", "n/a (not in this artifact)")))
        # One list of required keys, shared with the spend gate, so doctor's
        # verdict and translate's refusal cannot disagree about what fit means.
        why = dict(REQUIRED_STATS_KEYS)
        for key in missing_stats_keys(stats):
            problems.append("missing measured constant %s (%s)"
                            % (key, why[key]))
        if cfg.thinking_level != "LOW":
            level = thinking_per_request(stats, cfg.thinking_level, "p95")
            print("  thinking @ %-8s %s (p95)   override ack: %s"
                  % (cfg.thinking_level, "MISSING" if level is None else level,
                     cfg.thinking_level_override_ack))
            # Spending above LOW takes the measurement AND the acknowledgement.
            # The message says which half is missing and why the pin exists, in
            # the one output a human reads before pressing --confirm-spend.
            pinned = ("thinkingLevel is pinned to LOW for this program "
                      "(decision record: gemini-docs-verification.md) because "
                      "maxOutputTokens is ONE budget shared by thoughts and "
                      "candidates while the derived cap "
                      "(ceil(a*n + b) * 1.5) has NO thinking term. At MEDIUM "
                      "the measured p95 is 1,042 thought tokens against an "
                      "n=20 batch's entire 1,115-token cap, and both "
                      "MAX_TOKENS finishes in the probe set came from MEDIUM")
            if level is None:
                problems.append(
                    "thinking_level = %s has no measured thinking cost in %s, "
                    "so a paid run at this level would be sized by a constant "
                    "measured on another one. %s"
                    % (cfg.thinking_level, path, pinned))
            elif not cfg.thinking_level_override_ack:
                problems.append(
                    "thinking_level = %s is measured (p95 %s) but "
                    "thinking_level_override_ack is false. A measured thinking "
                    "cost the cap formula never reads is a number nobody is "
                    "using, so the measurement alone does not license the "
                    "spend. %s. Set thinking_level_override_ack = true to "
                    "spend anyway -- it means \"I know the output-cap math "
                    "ignores thinking\"" % (cfg.thinking_level, level, pinned))

    # The one state left where the configuration promises something the code
    # cannot deliver. translate refuses it outright (s42.transport_guard); doctor
    # has to say so, or its verdict would contradict the stage's. The two other
    # entries that used to be here -- mode = batch and cache_enabled -- are gone
    # because the transport and the cache lifecycle exist now.
    if cfg.cache_enabled and cfg.mode != "batch":
        problems.append("cache_enabled = true with mode = %s: the "
                        "explicit-cache lifecycle is driven by the batch wave, "
                        "so on this transport nothing would create the cache "
                        "and the run would pay the full uncached rate while the "
                        "bill quoted the cached one" % cfg.mode)
    if cfg.mode == "batch":
        print("  batch transport       jobs and results under %s; one job in "
              "flight (across invocations too), results downloaded on "
              "completion, a drain that dies is resumed and never resubmitted"
              % (cfg.work_dir / "batch"))

    # The two pre-spend gates. Reported rather than adjudicated: this command
    # places no call, so it must not absorb a refusal that belongs to the run.
    # But G-SCOPE-FROZEN refuses every --confirm-spend until the refreeze has
    # happened, and finding that out by being refused is worse than being told.
    stamp, where = read_refreeze_stamp(cfg)
    if isinstance(stamp, dict):
        print("  refreeze stamp        %s (%s families, %s card_keys rows, by "
              "%s)" % (stamp.get("refrozen_at"), stamp.get("families"),
                       stamp.get("card_keys_rows"), stamp.get("by")))
    else:
        print("  refreeze stamp        NOT PRESENT (%s). G-SCOPE-FROZEN will "
              "refuse every --confirm-spend until the release refreeze is done "
              "and signed: paying to translate a scope that is about to change "
              "is paying twice." % where)
    print("  spend cap             $%s/month, checked by G-BUDGET against "
          "month-to-date PLUS this run's forecast SUMMED over the languages in "
          "the run" % cfg.spend_cap_usd)

    print("--- prices ---")
    rates, rates_note = rate_card_for(cfg)
    print("  %s" % rates_note)
    if rates:
        print("  %s" % rates)

    print("--- verdict ---")
    if problems:
        for p in problems:
            print("  BLOCKED: %s" % p)
        print("  this configuration is NOT fit to spend on.")
        return 1
    print("  fit to spend, as far as this command can see. It REPORTS the "
          "refreeze stamp and the cap above but adjudicates neither: this "
          "command places no call, so G-SCOPE-FROZEN and G-BUDGET stay with the "
          "run that would spend, and a missing stamp above means the next "
          "--confirm-spend is refused.")
    return 0


def gates_report(cfg) -> int:
    """Print the accumulated gate report and AGGREGATE it: exit non-zero if any
    recorded row fails.

    Rows are keyed on (id, stage, extra), so the per-language export gates
    (G-COV / G-RATE / G-MEDIA / G-DET) keep one row per language. Before that,
    a passing German export overwrote a failing Chinese one and this command
    certified the release all-green.
    """
    from .gates import row_label
    path = cfg.report_dir / "gates_report.json"
    data = read_json(path, default={})
    if not data:
        print("no gate report yet: %s" % path)
        return 1
    rows = data.get("results", [])
    width = max((len(row_label(r)) for r in rows), default=12)
    for row in rows:
        # No marker at all on a report written before this accounting existed:
        # "unknown" must not print as "carried".
        carried_marker = ("" if row.get("executed_this_run", True)
                          else "   [CARRIED]")
        print("%-4s %-*s %-10s %s%s"
              % ("PASS" if row.get("ok") else "FAIL", width, row_label(row),
                 "stage " + str(row.get("stage") or "?"), row.get("description"),
                 carried_marker))
    bad = [row_label(r) for r in rows if not r.get("ok")]
    print("%d gate row(s) recorded, %d failing%s"
          % (len(rows), len(bad), (": " + ", ".join(bad)) if bad else ""))
    # The row count is NOT a release verdict. Rows accumulate across stages AND
    # across runs, so an all-PASS list can coexist with declared gates that have
    # never executed on this workspace -- and G-SITEMAP, one of them, is what
    # adjudicates merge_report.sitemap_shortfall_families.
    never = data.get("gate_ids_never_run") or []
    print("stages represented: %s ; %d of %d declared gates have a verdict here"
          % (", ".join(data.get("stages_reported") or ["?"]),
             len(data.get("gate_ids_with_a_verdict") or []),
             data.get("gates_declared") or 0))
    # `gates` READS the report, it does not run gates -- so from its own point of
    # view every row is carried. The distinction that matters is inside the file:
    # which rows the run that produced it actually executed. 12 recorded rows
    # were 10 executed plus 2 left by an earlier run at stages that build never
    # touches.
    carried = data.get("gate_rows_carried_from_an_earlier_run")
    if carried is None:
        print("(this report predates the executed-here accounting: it cannot say "
              "which rows the run that wrote it actually executed)")
    else:
        fresh = data.get("gate_rows_executed_this_run") or []
        print("%d row(s) were executed by the run that wrote this report "
              "(stages %s); %d carried from an earlier run%s"
              % (len(fresh), ", ".join(data.get("stages_executed_this_run")
                                       or ["-"]),
                 len(carried), (": " + ", ".join(carried)) if carried else ""))
    if never:
        print("NOT RUN on this workspace (%d): %s" % (len(never), ", ".join(never)))
    print("manual gates are never recorded here: G-IMPORT (the Anki smoke test, "
          "tools/import_smoke_test.md) and G-REVIEW (a human reading "
          "review/rejected.json) need a signature, not a script.")
    return 1 if bad else 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(getattr(args, "config", None),
                      work_dir=getattr(args, "work", None),
                      legacy_workspace=getattr(args, "legacy_workspace", None))
    try:
        return run_command(args, cfg)
    except FatalError as exc:
        print("FATAL: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; every stage checkpoints, so re-running resumes",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
