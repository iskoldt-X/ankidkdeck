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

Usage sketch:

    ankidkdeck --work work crawl --pilot
    ankidkdeck build
    ankidkdeck translate --lang German            # prints the bill, exits
    ankidkdeck translate --lang German --confirm-spend
    ankidkdeck export --lang German --check-determinism
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .util import FatalError, read_json

NETWORK_COMMANDS = {"sitemap", "crawl", "audio"}


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
                   help="place the Gemini ranking calls for the queued families")

    s = sub.add_parser("translate", help="incremental LLM top-up (stage 42)")
    s.add_argument("--lang", help="one target language; default: all configured")
    s.add_argument("--confirm-spend", action="store_true",
                   help="place the Gemini calls the bill just quoted")
    s.add_argument("--no-gc", action="store_true",
                   help="skip archiving translation rows with no live sense")

    s = sub.add_parser("audio", help="audio cache and delta download (stage 60)")
    s.add_argument("--seed-legacy", action="store_true",
                   help="hardlink/copy from the 2025 workspace instead of "
                        "downloading (~4,629 requests saved)")

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
        print_report("translate", s42_translate.run(
            cfg, _registry(cfg), lang=args.lang, confirm=args.confirm_spend,
            do_gc=not args.no_gc))
        return 0
    if cmd == "audio":
        from .stages import s60_audio
        print_report("audio", s60_audio.run(cfg, _net(cfg),
                                            seed_legacy=args.seed_legacy))
        return 0
    if cmd == "export":
        from .stages import s70_export
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


def gates_report(cfg) -> int:
    """Print the accumulated gate report. Gates run inside their stages; this
    is the standing record of what passed, and what a human still owes."""
    path = cfg.report_dir / "gates_report.json"
    data = read_json(path, default={})
    if not data:
        print("no gate report yet: %s" % path)
        return 1
    for row in data.get("results", []):
        print("%-4s %-12s %-10s %s"
              % ("PASS" if row.get("ok") else "FAIL", row.get("id"),
                 "stage " + str(row.get("stage") or "?"), row.get("description")))
    failed = data.get("failed") or []
    print("%d gate(s) recorded, %d failing%s"
          % (len(data.get("results", [])), len(failed),
             (": " + ", ".join(failed)) if failed else ""))
    print("manual gates are never recorded here: G-IMPORT (the Anki smoke test, "
          "tools/import_smoke_test.md) and G-REVIEW (a human reading "
          "review/rejected.json) need a signature, not a script.")
    return 1 if failed else 0


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
