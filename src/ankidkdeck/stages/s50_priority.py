"""Stage 50: homograph display order, inherited from the 2025 rankings.

615 Gemini rankings were paid for in 2025 and they are keyed by FILENAME, which
2026 does not have. The bridge (stage 40) turns filenames into entry_ids, and
four of the 615 keys collapse to fewer ids on the way -- the same-word
re-downloads rom/ram/hiv/gloria -- so the mapping is dedupe-keep-first, not
one-to-one.

Two rules make the inheritance honest:

  * A family that inherits orders from more than one source word gets a STABLE
    MERGE with the lowest-rank source setting the tone, and every contradiction
    is logged (measured: 5 -- en/en-acute, tusinde/tusind, sol/so, rose/rosa).
  * A stored order is reused ONLY when SOMEONE ACTUALLY RANKED IT and its entry
    set equals the family's entry set exactly. A family that gained a homograph
    since 2025 has never been ranked with that homograph in it, so it goes back
    in the queue instead of inheriting a stale order.

`ranked` IS THAT FIRST CLAUSE, and it is why this stage is not a no-op. Every
multi-article family gets an order written to priority_orders.json -- determinism
needs one even for a family nobody has ever ranked -- so the entry-set test alone
is satisfied by the stage's OWN deterministic fallback. The documented workflow
is `priority` (see the bill) then `priority --confirm-spend`, and on the second
run every queued family read back its own fallback as if it were an authority:
the queue emptied, zero calls were placed, ranking_queue.json was overwritten
with [], and "a new homograph has no rank" became unreachable forever. So the
reuse test requires prev["ranked"], which is true only for an order that came
from a 2025 ranking covering the WHOLE entry set, from a Gemini call, or from an
earlier such order. A fallback order is written, used, and never mistaken for a
verdict.

Deviation from the guide's pseudocode, on the owner's instruction: the ANCHOR
STAYS FIRST. Guide 4.11 writes f["entry_ids"] = order + tail, which lets a
ranked non-anchor article head the card; but family_id IS the anchor entry_id,
the registry keys the GUID seed on it, G-ANCHOR asserts it is neither demoted
nor an affix, and stage 70 reads Etymology from it. Letting the ranking move it
would split identity from display. The ranking therefore orders everything
AFTER the anchor. Reviewers: this is the one place where the guide and the
launch instruction disagree.

Unranked entries go last -- 09:534-537's fallback, kept because it is safe.
"""

import datetime
import json
import time

from ..config import Config
from ..gates import G_ORDER, Gate, run_gates
from ..util import FatalError, read_json, write_json

# One family per call, as in 03_rank_homographs.py; 1.6s keeps the free tier's
# 30 RPM with margin.
RANK_REQUEST_INTERVAL = 1.6


def dedupe_keep_first(seq) -> list:
    seen, out = set(), []
    for x in seq:
        if x is None or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def stable_merge(base: list, extra: list) -> list:
    """Merge extra into base without ever reordering base.

    base is the order proposed by the lowest-rank source word; it sets the
    tone. An id only in extra is placed after the nearest preceding id they
    share, else before the nearest following one, else at the end -- so the
    result is deterministic and independent of dict iteration order.
    """
    out = list(base)
    for i, e in enumerate(extra):
        if e in out:
            continue
        pos = None
        for prev in reversed(extra[:i]):
            if prev in out:
                pos = out.index(prev) + 1
                break
        if pos is None:
            for nxt in extra[i + 1:]:
                if nxt in out:
                    pos = out.index(nxt)
                    break
        out.insert(len(out) if pos is None else pos, e)
    return out


def anchor_first(anchor: str, order: list) -> list:
    """Identity beats commonness: the anchor heads the card, the ranking orders
    the rest."""
    return [anchor] + [e for e in order if e != anchor]


def _order_gate(rows: list):
    """G-ORDER. The new entry_ids must be a permutation of the old one with the
    anchor still at position 0: a display-order stage that can lose or duplicate
    an article would silently delete meanings from a card."""
    bad = [r for r in rows if not r["ok"]]
    return not bad, {"families": len(rows), "violations": bad[:20]}


def _rank_schema(n_ids: int) -> dict:
    """03: the permutation lock -- exactly the input ids, no extras, no
    omissions."""
    return {"type": "object",
            "properties": {"sorted_ids": {"type": "array", "minItems": n_ids,
                                          "maxItems": n_ids,
                                          "items": {"type": "string"}}},
            "required": ["sorted_ids"]}


def rank_prompt() -> str:
    """Ported verbatim from 03_rank_homographs.py (the Editor-in-Chief prompt
    the 615 stored orders were produced with). The only change is that ids are
    entry_ids now, because filenames no longer exist -- the example keeps its
    original shape."""
    system_prompt = """
You are the Editor-in-Chief for a new Danish-English dictionary aimed at foreign language learners. Your current task is to rank the different meanings of Danish homographs based on their frequency and relevance in contemporary Danish.
### CONTEXT
I will provide a Danish headword and a JSON array of its various entries. Each entry has:
- "id": A unique identifier (the DDO entry_id).
- "pos": The part of speech.
- "definitions": One or more Danish definitions, separated by "; ".
### YOUR TASK
Analyze the entries and sort them in descending order of common usage. The first item in your list should be the most fundamental and frequently used meaning. The last item should be the most specialized, archaic, or least common meaning.
### OUTPUT FORMAT
You MUST return a JSON object with a single key, "sorted_ids". The value must be an array of strings, containing all the provided "id"s sorted by priority. The array must contain the exact same IDs as the input, with no extras or omissions.
"""
    example_prompt = """
**Example:**
Input headword: "en"
Entries:
[
  {"id": "11021940", "pos": "talord", "definitions": "symboliserer tallet 1"},
  {"id": "11021941", "pos": "adverbium", "definitions": "blot, kun"},
  {"id": "11021942", "pos": "artikel", "definitions": "bruges for at angive ubestemt form"},
  {"id": "11021943", "pos": "pronomen", "definitions": "en uspecificeret person"}
]
Your output MUST be a sorted list, for example:
{
  "sorted_ids": [
    "11021942",
    "11021943",
    "11021940",
    "11021941"
  ]
}
"""
    return "%s\n\n%s" % (system_prompt.strip(), example_prompt.strip())


def _rank_payload(entries: dict, entry_ids: list) -> list:
    rows = []
    for eid in entry_ids:
        e = entries[eid]
        defs = "; ".join(d for d in (s.get("definition") for s in e.get("senses", []))
                         if d)
        rows.append({"id": eid, "pos": e.get("pos_text") or e.get("pos_key") or "N/A",
                     "definitions": defs})
    return rows


def run(cfg: Config, registry=None, confirm: bool = False) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    families = read_json(cfg.json_dir / "words.json")
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    bridge = read_json(cfg.json_dir / "legacy" / "filename_to_entry_id.json",
                       default={})
    stored = read_json(cfg.json_dir / "priority_orders.json", default={})

    rank_of = {w["word"]: w["rank"] for w in wordlist}
    legacy_path = (cfg.legacy_workspace / "priority_map.json"
                   if cfg.legacy_workspace else None)
    legacy = read_json(legacy_path, default={}) if legacy_path else {}
    report: dict = {"legacy_priority_keys": len(legacy),
                    "bridge_entries": len(bridge)}
    if legacy and not bridge:
        raise FatalError(
            "priority_map.json is present but work/json/legacy/"
            "filename_to_entry_id.json is not -- run the migrate stage first, "
            "or the 615 paid-for orderings cannot be re-keyed.")

    # ---- 1. filename lists -> entry_id lists, dedupe keep-first -------------
    collapsed, unbridged = [], []
    by_word: dict[str, list] = {}
    for word, filenames in sorted(legacy.items()):
        eids_raw = []
        for fn in filenames:
            eid = bridge.get(fn)
            if eid is None:
                unbridged.append({"word": word, "filename": fn})
                continue
            eids_raw.append(eid)
        eids = dedupe_keep_first(eids_raw)
        if len(eids) != len(eids_raw):
            collapsed.append({"word": word, "filenames": filenames,
                              "entry_ids": eids,
                              "collapsed_by": len(eids_raw) - len(eids)})
        by_word[word] = eids
    report["keys_collapsing_after_bridge"] = len(collapsed)
    report["collapsed_keys"] = collapsed
    report["unbridged_filenames"] = len(unbridged)

    # ---- 2. proposals per family -------------------------------------------
    fams_of_word: dict[str, list] = {}
    for fid, fam in families.items():
        for m in fam.get("members", []):
            fams_of_word.setdefault(m["word"], []).append(fid)
    proposals: dict[str, list] = {}
    orphan_words = []
    for word, eids in by_word.items():
        fids = fams_of_word.get(word)
        if not fids:
            orphan_words.append(word)
            continue
        for fid in fids:
            proposals.setdefault(fid, []).append(
                (rank_of.get(word, 10 ** 9), word, eids))
    report["priority_words_with_no_family"] = len(orphan_words)
    report["priority_words_with_no_family_sample"] = sorted(orphan_words)[:20]

    # ---- 3. apply ------------------------------------------------------------
    conflicts, queue, gate_rows = [], [], []
    sources = {"stored": 0, "legacy": 0, "none": 0, "singleton": 0}
    for fid in sorted(families):
        fam = families[fid]
        eids = list(fam.get("entry_ids") or [])
        anchor = fam["anchor_entry_id"]
        if len(eids) <= 1:
            sources["singleton"] += 1
            continue
        want = set(eids)

        prev = stored.get(fid)
        provenance = None
        if prev and prev.get("ranked") and set(prev.get("order") or []) == want:
            # Reuse needs BOTH an exact set match (guide 1.11d / A's P13) and an
            # order somebody actually ranked.
            order, tail, source = list(prev["order"]), [], "stored"
            # Carry the paid-for provenance forward. Overwriting it with the
            # literal "stored" erased "gemini:<model>@<date>" on the first plain
            # re-run, i.e. the record of what had been paid for.
            provenance = prev.get("provenance")
        else:
            props = sorted(proposals.get(fid, []))
            merged: list = []
            for _, _, p_eids in props:
                merged = stable_merge(merged, p_eids)
            if len({tuple(p[2]) for p in props}) > 1:
                conflicts.append({"family_id": fid, "lemma": fam.get("lemma"),
                                  "proposals": [{"rank": r, "word": w,
                                                 "entry_ids": ids}
                                                for r, w, ids in props]})
            order = [e for e in merged if e in want]
            tail = [e for e in eids if e not in order]
            source = "legacy" if order else "none"
        sources[source] = sources.get(source, 0) + 1

        if tail:
            # A homograph nobody ever ranked: no stored order is valid for this
            # entry set, so the family goes to the queue and the newcomer goes
            # last until it is ranked.
            queue.append({"family_id": fid, "lemma": fam.get("lemma"),
                          "entry_ids": eids, "ranked_prefix": order,
                          "unranked": tail, "status": "pending",
                          "why": "no order covers the current entry set"})

        new_order = anchor_first(anchor, order + tail)
        ok = (set(new_order) == want and len(new_order) == len(set(new_order))
              and new_order[0] == anchor)
        gate_rows.append({"family_id": fid, "ok": ok, "before": eids,
                          "after": new_order})
        fam["entry_ids"] = new_order
        fam["priority_source"] = source
        # `ranked` is the REUSE AUTHORITY and it is deliberately narrower than
        # "an order exists": a `none` order is this stage's own deterministic
        # fallback, and a `legacy` order with a non-empty tail covers only part
        # of the current entry set -- which is exactly the case the exact-set
        # rule exists for. Neither is a verdict, so neither may be reused.
        stored[fid] = {"order": new_order, "source": source,
                       "provenance": provenance,
                       "entry_set": sorted(want),
                       "ranked": source in ("stored", "legacy", "gemini")
                                 and not tail}

    run_gates([Gate(G_ORDER, "each family's entry order is a permutation of its "
                             "own articles with the anchor still first",
                    lambda: _order_gate(gate_rows), stage="50")], cfg, stage="50")

    conflicts_path = cfg.report_dir / "priority_conflicts.json"
    # MERGED, never replaced. `conflicts` is only appended on the non-reuse
    # branch, so on a plain re-run every family takes the reuse branch and an
    # unconditional write emptied the file -- deleting the audit record of the 5
    # real two-source contradictions (en/en-acute, tusinde/tusind, sol/so,
    # rose/rosa) after exactly one run.
    prev_conflicts = read_json(conflicts_path, default=[])
    by_fid = {r.get("family_id"): r for r in prev_conflicts if isinstance(r, dict)}
    for row in conflicts:
        by_fid[row["family_id"]] = row
    merged_conflicts = [by_fid[k] for k in sorted(by_fid, key=str)]

    write_json(cfg.json_dir / "words.json", families)
    write_json(cfg.json_dir / "priority_orders.json", stored)
    write_json(conflicts_path, merged_conflicts)
    write_json(cfg.report_dir / "ranking_queue.json", queue)
    report.update({"multi_entry_families": len(gate_rows),
                   "order_sources": sources,
                   "orders_reusable_next_run": sum(1 for v in stored.values()
                                                   if v.get("ranked")),
                   "conflicts_logged": len(conflicts),
                   "conflicts_on_file": len(merged_conflicts),
                   "queue_for_ranking": len(queue)})

    print("--- homograph ranking queue ---")
    print("  %d multi-article families, %d inherited a 2025 order, %d reused a "
          "stored order" % (len(gate_rows), sources.get("legacy", 0),
                            sources.get("stored", 0)))
    print("  %d families need a fresh Gemini ranking (1 request each)"
          % len(queue))
    print("  %d two-source contradictions logged to reports/priority_conflicts.json"
          % len(conflicts))
    if not confirm:
        print("  nothing has been sent. Re-run with --confirm-spend to place calls.")
        report["note"] = ("queue only: no LLM module was imported and no request "
                          "was made.")
        write_json(cfg.report_dir / "priority_report.json", report)
        return report

    # ---------------- past this line, money is spent ----------------
    from .s42_translate import _generate, _pool_from_env

    pool = _pool_from_env()
    # The ranking is a short permutation, not prose: it shares the expressions
    # model rather than the definition model (config.expressions_model).
    model = cfg.expressions_model
    prov = "gemini:%s@%s" % (model, datetime.date.today().isoformat())
    ranked = 0
    for row in queue:
        fid = row["family_id"]
        fam = families[fid]
        eids = list(fam["entry_ids"])
        payload = _rank_payload(entries, eids)
        user = ('--- YOUR TURN ---\nInput headword: "%s"\nEntries:\n%s'
                % (fam.get("lemma"), json.dumps(payload, ensure_ascii=False,
                                                indent=2)))
        time.sleep(RANK_REQUEST_INTERVAL)
        parsed = _generate(pool, model, rank_prompt(), user,
                           _rank_schema(len(eids)), 0.1, "ranking %s" % fid)
        sorted_ids = parsed.get("sorted_ids")
        if not isinstance(sorted_ids, list) or set(sorted_ids) != set(eids) \
                or len(sorted_ids) != len(eids):
            raise FatalError(
                "ranking for family %s returned %s, which is not a permutation "
                "of %s" % (fid, sorted_ids, eids))
        new_order = anchor_first(fam["anchor_entry_id"], sorted_ids)
        fam["entry_ids"] = new_order
        fam["priority_source"] = "gemini"
        # `source` stays a CLOSED vocabulary (stored/legacy/gemini/none) and the
        # model+date string lives in its own field. One key that is sometimes an
        # enum and sometimes a free-form provenance string cannot be filtered on.
        stored[fid] = {"order": new_order, "source": "gemini",
                       "provenance": prov, "entry_set": sorted(set(eids)),
                       "ranked": True}
        row["status"] = "ranked"
        row["order"] = new_order
        row["provenance"] = prov
        ranked += 1
        # checkpoint after EVERY family: an interrupted run loses one call
        write_json(cfg.json_dir / "words.json", families)
        write_json(cfg.json_dir / "priority_orders.json", stored)
        # The queue keeps its rows WITH their status. Overwriting the file with
        # [] once a confirm run had resolved it destroyed the only record of what
        # had needed ranking -- and a run interrupted halfway left no trace of
        # which families were still pending.
        write_json(cfg.report_dir / "ranking_queue.json", queue)
    report["families_ranked"] = ranked
    report["queue_still_pending"] = sum(1 for r in queue
                                        if r.get("status") != "ranked")
    report["api"] = {"requests": pool.total_requests,
                     "key_rotations": pool.rotations}
    write_json(cfg.report_dir / "priority_report.json", report)
    return report
