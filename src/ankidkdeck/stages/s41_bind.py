"""Stage 41: bind the migrated 2025 translations to 2026 dannetid keys.

The second half of "track 1". Stage 40 could only address the 2025 side by
(entry_id, danish_text) because `dannetid` does not exist in the 2025 markup
(0 of 5,267 files). Once stage 20 has parsed the 2026 pages, each legacy row is
matched inside its own entry -- byte-exact text first, the 2025 sense address
second -- and only then is dannetid promoted to the storage key:

    definitions   key = "<entry_id>:<dannetid>"   composite; 10 dannetid values
                                                  are shared across entries
    expressions   key = "<dannetid>"              a shared idiom deliberately
                                                  collapses to ONE translation

THE HONEST GATE, replacing the 2025 "coverage == 2025 coverage, not one row
lost" (which was unsatisfiable and so never enforced):
    n_bound + n_dropped == n_legacy, every drop carries a reason code from a
    closed set, n_unexplained == 0.

The bind is ENTRY-SCOPED BY DESIGN. A row is keyed (entry_id, dannetid) and is
bound whether or not a family renders that entry: stage 42's GC archives the
orphans and G-ORPH enforces that it did, so bind-then-GC is the written order
and a re-instated article's translations come for free. bind_report.json reports
the volume (n_bound_on_unused_entries) so the archive size is predictable
instead of discovered.

Losing a row is only accounted for honestly if the three ways of losing it are
told apart. The entry is absent from entries.json -> DDO deleted the article.
The entry is present but every word that saw it rejected it -> the classifier
dropped it (`rejected_article`, guide 6.3 population 3). Otherwise it binds.
"""

from ..config import Config
from ..gates import DROP_REASONS, G_BIND, Gate, bind_accounting, run_gates
from ..util import NFC, FatalError, read_json, sha256_str, write_json
from .s21_resolve import rejected_everywhere_ids

WARN_BIND_RATE = 0.97


def _legacy(cfg: Config, name: str):
    p = cfg.json_dir / "legacy" / name
    if not p.exists():
        raise FatalError(
            f"missing {p} -- run the migrate stage (40) against the recovered "
            "2025 workspace before binding.")
    return read_json(p)


def _invert_sense_paths(paths: dict) -> dict:
    """{entry_id: {definition_text: sense_path}}. First path wins when one
    entry repeats a definition string, matching stage 40's write order."""
    out: dict[str, dict] = {}
    for eid, by_path in paths.items():
        inv = out.setdefault(eid, {})
        for path, text in by_path.items():
            inv.setdefault(text, path)
    return out


def _place(out: dict, key: str, row: dict, drop, legacy_row: dict, where: str) -> bool:
    """First bind wins. An identical rebind is fine (stage 41 is idempotent);
    a conflicting rebind keeps the incumbent and is logged, which is how the
    deliberate shared-dannetid collapse stays accountable."""
    prev = out.get(key)
    if prev is None:
        out[key] = row
        return True
    if (prev.get("lemma"), prev.get("gloss")) == (row.get("lemma"), row.get("gloss")):
        # Idempotent, and deliberately NOT a mutation: the incumbent's src_sha
        # describes the Danish string ITS gloss was produced from. For a
        # dannetid-keyed idiom the two rows can come from different Danish
        # strings, and overwriting the sha makes the retranslation trigger lie.
        return True
    drop(legacy_row, "shared_dannetid_conflict",
         {"key": key, "kept": {"lemma": prev.get("lemma"), "gloss": prev.get("gloss")},
          "where": where})
    return False


def run(cfg: Config, registry=None) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    inv_paths = _invert_sense_paths(read_json(cfg.json_dir / "legacy" / "sense_paths.json",
                                              default={}))
    # Read after stage 21, so an entry recovered through the reverse index or an
    # override is NOT counted as rejected.
    classification = read_json(cfg.json_dir / "classification.json", default={})
    rejected = rejected_everywhere_ids(classification)
    families = read_json(cfg.json_dir / "words.json", default={})
    rendered_entries = {eid for fam in families.values()
                        for eid in (fam.get("entry_ids") or [])}
    per_lang: dict[str, dict] = {}
    report: dict = {"warnings": [],
                    "entries_rejected_everywhere": len(rejected),
                    "entries_in_a_family": len(rendered_entries),
                    "bind_scope": "entry-scoped by design; rows on entries no "
                                  "family renders are kept and archived by "
                                  "stage 42's GC (G-ORPH), so the bind rate is "
                                  "not a loss"}

    def entry_verdict(eid: str):
        """(entry, drop_reason). None reason means 'bind it'."""
        e = entries.get(eid)
        if e is None:
            return None, "article_gone_from_ddo"
        if eid in rejected:
            return e, "rejected_article"
        return e, None

    for lang in cfg.langs:
        legacy_defs = _legacy(cfg, f"legacy_{lang}_definitions.json")
        legacy_exprs = _legacy(cfg, f"legacy_{lang}_expressions.json")
        tdir = cfg.json_dir / "translations" / lang
        out_defs = read_json(tdir / "definitions.json", default={})
        out_exprs = read_json(tdir / "expressions.json", default={})
        dropped: list = []
        counts = {"definitions": {"bound": 0, "dropped": 0, "legacy": 0,
                                  "via_text": 0, "via_sense_path": 0,
                                  "bound_on_unused_entries": 0},
                  "expressions": {"bound": 0, "dropped": 0, "legacy": 0,
                                  "via_expression": 0, "via_variant": 0,
                                  "bound_on_unused_entries": 0}}

        def drop(row: dict, reason: str, extra: dict | None = None):
            dropped.append({"reason": reason, **row, **(extra or {})})

        # ---- definitions ---------------------------------------------------
        for eid in sorted(legacy_defs):
            e, verdict = entry_verdict(eid)
            unused = eid not in rendered_entries
            for text, tr in legacy_defs[eid].items():
                counts["definitions"]["legacy"] += 1
                base = {"entry_id": eid, "text": text, "lemma": tr.get("lemma"),
                        "provenance": tr.get("provenance")}
                if verdict:
                    drop(base, verdict)
                    counts["definitions"]["dropped"] += 1
                    continue
                cand = [s for s in e["senses"] if s["definition"] == text]
                via = "via_text"
                if not cand:
                    path = inv_paths.get(eid, {}).get(text)
                    if path:
                        cand = [s for s in e["senses"] if s.get("sense_path") == path]
                        via = "via_sense_path"
                if not cand:
                    drop(base, "sense_text_changed")
                    counts["definitions"]["dropped"] += 1
                    continue
                s = cand[0]
                row = {"lemma": tr.get("lemma"), "gloss": tr.get("gloss"),
                       "src_sha": s["src_sha"], "provenance": tr.get("provenance")}
                if _place(out_defs, f"{eid}:{s['dannetid']}", row, drop, base,
                          "definitions"):
                    counts["definitions"]["bound"] += 1
                    counts["definitions"][via] += 1
                    if unused:
                        counts["definitions"]["bound_on_unused_entries"] += 1
                else:
                    counts["definitions"]["dropped"] += 1

        # ---- expressions ---------------------------------------------------
        for eid in sorted(legacy_exprs):
            e, verdict = entry_verdict(eid)
            unused = eid not in rendered_entries
            by_expr, by_variant = {}, {}
            for x in (e or {}).get("expressions", []):
                by_expr.setdefault(x["expression"], x)
                for v in x.get("variants", []):
                    by_variant.setdefault(v, x)
            for text, tr in legacy_exprs[eid].items():
                counts["expressions"]["legacy"] += 1
                base = {"entry_id": eid, "text": text, "lemma": tr.get("lemma"),
                        "provenance": tr.get("provenance")}
                if verdict:
                    drop(base, verdict)
                    counts["expressions"]["dropped"] += 1
                    continue
                # Legacy expression keys are the EXPRESSION TEXT, not its
                # definition; the "" separator makes it byte-stable (689/689).
                x = by_expr.get(text)
                via = "via_expression"
                if x is None:
                    x = by_variant.get(text)
                    via = "via_variant"
                if x is None:
                    drop(base, "expression_text_changed")
                    counts["expressions"]["dropped"] += 1
                    continue
                if not x.get("dannetid"):
                    drop(base, "source_gap", {"why": "2026 expression carries no dannetid"})
                    counts["expressions"]["dropped"] += 1
                    continue
                # src_sha is the sha of the EXPRESSION TEXT, which is what 2025
                # actually sent to the LLM (payload field `expr`; the idiom's
                # definition was only an optional `hint`). Stage 40 already
                # hashed it -- overwriting it with the definition's sha disabled
                # the one retranslation trigger (D7/1.7) for all 12,716
                # expression cells x 4 languages: editing an idiom never
                # retranslated, editing its definition retranslated spuriously.
                # Stage 42's expression_src_sha() computes the same formula.
                sha = tr.get("src_sha") or sha256_str(NFC(text))
                row = {"lemma": tr.get("lemma"), "gloss": tr.get("gloss"),
                       "src_sha": sha, "provenance": tr.get("provenance")}
                if _place(out_exprs, x["dannetid"], row, drop, base, "expressions"):
                    counts["expressions"]["bound"] += 1
                    counts["expressions"][via] += 1
                    if unused:
                        counts["expressions"]["bound_on_unused_entries"] += 1
                else:
                    counts["expressions"]["dropped"] += 1

        write_json(tdir / "definitions.json", out_defs)
        write_json(tdir / "expressions.json", out_exprs)
        write_json(tdir / "dropped.json", dropped)

        n_legacy = counts["definitions"]["legacy"] + counts["expressions"]["legacy"]
        n_bound = counts["definitions"]["bound"] + counts["expressions"]["bound"]
        n_dropped = counts["definitions"]["dropped"] + counts["expressions"]["dropped"]
        reasons: dict[str, int] = {}
        for d in dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        unexplained = (n_legacy - n_bound - n_dropped
                       + sum(v for k, v in reasons.items() if k not in DROP_REASONS))
        per_lang[lang] = {"n_legacy": n_legacy, "n_bound": n_bound,
                          "n_dropped": n_dropped, "n_unexplained": unexplained,
                          "reasons": reasons,
                          "bind_rate": round(n_bound / n_legacy, 4) if n_legacy else None,
                          "detail": counts,
                          # Report only: these rows are the volume stage 42's GC
                          # will archive. Not a defect, but it should be
                          # predictable rather than discovered.
                          "n_bound_on_unused_entries":
                              counts["definitions"]["bound_on_unused_entries"]
                              + counts["expressions"]["bound_on_unused_entries"],
                          "keys_definitions": len(out_defs),
                          "keys_expressions": len(out_exprs)}
        rate = per_lang[lang]["bind_rate"]
        if rate is not None and rate < WARN_BIND_RATE:
            report["warnings"].append(
                f"{lang}: bind rate {rate:.2%} is below {WARN_BIND_RATE:.0%} -- read "
                f"translations/{lang}/dropped.json before trusting the deck")

    run_gates([
        Gate(G_BIND, "n_bound + n_dropped == n_legacy per language, every drop "
                     "carries a reason code from the closed set, n_unexplained == 0",
             lambda: bind_accounting(per_lang), stage="41"),
    ], cfg, stage="41")

    report["per_language"] = per_lang
    report["drop_reason_codes"] = sorted(DROP_REASONS)
    write_json(cfg.report_dir / "bind_report.json", report)
    return report
