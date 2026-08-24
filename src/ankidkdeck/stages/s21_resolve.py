"""Stage 21: resolve wordlist words that classified to zero entries.

Layers, in order: (1) forward page already gave members; (2) curated
registry/form_to_lemma.json overrides; (3) reverse form index over every kept
article's flex table (this is what recovers er -> vaere); (4) known_no_entry.

The override moved AHEAD of the reverse index (round 2). Guide 4.7 lists the
index first, but a curated override was then unreachable whenever the automatic
layer had ANY hit -- and correcting the automatic layer is the entire reason
form_to_lemma.json exists. Human curation wins; an override that names a lemma
page we never crawled still falls through to the index and is still reported.

THE RECOVERY DOOR USES THE FRONT DOOR'S RULE. Both layers route every candidate
through classify_one() and keep the bucket it returns. Before, `attach()`
hardcoded bucket="form": `indexable()` filters only affix / demoted /
rejected-everywhere, so the abbreviation, multiword-neighbour and unrelated
rejections the classifier applies on the forward page were simply not applied
here, and every recovered member was mislabelled -- which then made
s30._relation() call an official alternative spelling an "inflection" and kept it
out of the visible Variants list.

OWNER DECISION (2026-08-24): a word that still resolves to nothing is SKIPPED
and recorded in unresolved.json for later human review -- the pipeline never
stops for it. That decision only works if the file is readable, so every row
carries an explicit `reason` and the counters are derived FROM the rows.
"""

from collections import defaultdict

from ..config import Config
from ..gates import is_affix_entry
from ..util import NFC, nk, read_json, write_json
from .s22_classify import classify_one

# Reason codes for unresolved.json. Closed set, one row per skipped word.
REASON_NOHIT = "nohit"                     # DDO has no such word
REASON_ALL_REJECTED = "all_rejected"       # the page had articles; all rejected
REASON_OVERRIDE_NOT_CRAWLED = "override_lemma_not_crawled"
REASON_NO_SURVIVOR = "no_survivor"         # nothing left to attach
UNRESOLVED_REASONS = (REASON_NOHIT, REASON_ALL_REJECTED,
                      REASON_OVERRIDE_NOT_CRAWLED, REASON_NO_SURVIVOR)


def rejected_everywhere_ids(classification: dict) -> set:
    """Entries that every word which SAW them declined to keep.

    `xrefs` count as seen. Exclusive exactness moves a case-only homograph from
    members to xrefs, and if no word rejected it outright its `seen` count was
    zero -- so it failed the `seen > 0` test and stayed in the reverse index.
    That let layer 3 hand a word the erbium-class symbol article as a member:
    the Er/er hazard re-entering through the recovery door. Guide 4.7 builds the
    index over every KEPT article, and an xref is not kept.
    """
    seen: dict[str, int] = defaultdict(int)
    kept: dict[str, int] = defaultdict(int)
    for c in classification.values():
        for m in (c.get("members") or []):
            kept[m["entry_id"]] += 1
            seen[m["entry_id"]] += 1
        for r in (c.get("rejected") or []):
            seen[r["entry_id"]] += 1
        for eid in (c.get("xrefs") or []):
            seen[eid] += 1
    return {eid for eid, n in seen.items() if n > 0 and kept.get(eid, 0) == 0}


def run(cfg: Config, registry) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    classification = read_json(cfg.json_dir / "classification.json")
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    ledger = read_json(cfg.json_dir / "fetch_ledger.json", default={})
    demoted_pos = registry.demoted_pos_keys
    alias_pairs = registry.alias_pairs

    rejected_everywhere = rejected_everywhere_ids(classification)

    def indexable(eid: str) -> bool:
        """What layer 2 and layer 3 are allowed to hand a word.

        A demoted or affix article is never recovered: the classifier rejects it
        on the forward page for a reason, and re-admitting it here (with a
        fabricated demoted flag, as this stage used to do) is how the symbol
        article gets a chance to head a family.
        """
        e = entries[eid]
        return (eid not in rejected_everywhere
                and e.get("pos_key") not in demoted_pos
                and not is_affix_entry(e))

    rev = defaultdict(set)
    for eid in entries:
        if not indexable(eid):
            continue
        for f in entries[eid]["form_index"]:
            rev[f].add(eid)

    recovery_rejects: list = []

    def judged(word: str, c: dict, eids, evidence: str) -> list:
        """classify_one() on every candidate, exactly as the forward page does.

        Returns [(entry_id, bucket, why)] for the survivors; a refusal is written
        into the word's own `rejected` list with a `recovery_` reason so the
        recovery door leaves the same audit trail the front door does.
        """
        keep = []
        for eid in eids:
            bucket, why = classify_one(word, entries[eid], demoted_pos, alias_pairs)
            if bucket == "reject":
                row = {"entry_id": eid, "headword": entries[eid]["lemma"],
                       "pos_key": entries[eid].get("pos_key"),
                       "reason": "recovery_" + why}
                c["rejected"].append(row)
                recovery_rejects.append({"word": word, "via": evidence, **row})
                continue
            keep.append((eid, bucket, why))
        return keep

    def attach(c: dict, judged_rows, evidence: str) -> None:
        for eid, bucket, why in judged_rows:
            c["members"].append({
                "entry_id": eid,
                # The bucket the CLASSIFIER returned, never the literal "form":
                # s30._relation() reads it, and alt_forms_html only renders
                # members whose relation is variant/alias.
                "bucket": bucket,
                # the REAL flag, read from the entry -- never hardcoded False
                "demoted": entries[eid].get("pos_key") in demoted_pos,
                "evidence": evidence, "why": why})

    unresolved = []
    by_reason: dict[str, int] = {}
    override_problems: list = []
    resolved = {"forward": 0, "reverse_index": 0, "override": 0,
                "known_no_entry": 0}
    for w in wordlist:
        word = w["word"]
        c = classification.setdefault(
            word, {"members": [], "xrefs": [], "rejected": [], "resolved_by": None})
        if c["members"]:
            resolved["forward"] += 1
            continue
        # Layer 2: the curated override, AHEAD of the automatic index.
        override_lemma = registry.form_to_lemma.get(word)
        override_reason = None
        if override_lemma:
            key = nk(NFC(override_lemma))
            of_lemma = [e for e, v in entries.items() if v["lemma_key"] == key]
            usable = judged(word, c, sorted(e for e in of_lemma if indexable(e)),
                            "override")
            if usable:
                attach(c, usable, "override")
                c["resolved_by"] = "override"
                resolved["override"] += 1
                continue
            # An override that names a lemma page we never crawled is a curation
            # bug the human has to see; it must not fall through unmarked, even
            # when the reverse index happens to rescue the word below.
            override_reason = (REASON_NO_SURVIVOR if of_lemma
                               else REASON_OVERRIDE_NOT_CRAWLED)
            override_problems.append({"word": word, "override_lemma": override_lemma,
                                      "reason": override_reason,
                                      "entries_with_that_lemma": len(of_lemma)})
        # Layer 3: the reverse form index over every kept article's flex table.
        hits = judged(word, c, sorted(rev.get(nk(word), set())), "reverse_index")
        if hits:
            attach(c, hits, "reverse_index")
            c["resolved_by"] = "reverse_index"
            resolved["reverse_index"] += 1
            continue
        if word in registry.known_no_entry:
            c["resolved_by"] = "known_no_entry"
            resolved["known_no_entry"] += 1
            continue
        status = (ledger.get(word) or {}).get("status")
        if status == "nohit":
            reason = REASON_NOHIT
        elif override_reason:
            reason = override_reason
        elif c["rejected"]:
            reason = REASON_ALL_REJECTED
        else:
            reason = REASON_NO_SURVIVOR
        by_reason[reason] = by_reason.get(reason, 0) + 1
        unresolved.append({
            "word": word, "rank": w["rank"], "reason": reason,
            "fetch_status": status,
            "articles_on_page": (ledger.get(word) or {}).get("article_count"),
            "override_lemma": override_lemma,
            # WHY it was rejected, not just what: the owner's skip-and-record
            # ruling rests on this being readable (plan section 4.11).
            "rejected": [{"entry_id": r["entry_id"], "headword": r["headword"],
                          "pos_key": r.get("pos_key"), "reason": r.get("reason")}
                         for r in c["rejected"]],
        })

    write_json(cfg.json_dir / "classification.json", classification)
    write_json(cfg.report_dir / "unresolved.json", unresolved)
    write_json(cfg.report_dir / "recovery_rejected.json", recovery_rejects)
    # Derived from the rows, so the counters and the file can never disagree.
    resolved["unresolved"] = len(unresolved)
    resolved["unresolved_by_reason"] = {r: by_reason.get(r, 0)
                                        for r in UNRESOLVED_REASONS}
    resolved["nohit"] = by_reason.get(REASON_NOHIT, 0)
    resolved["entries_rejected_everywhere"] = len(rejected_everywhere)
    resolved["entries_in_reverse_index"] = len(
        {e for eids in rev.values() for e in eids})
    resolved["recovery_candidates_rejected"] = len(recovery_rejects)
    resolved["recovery_rejected_by_reason"] = _count(
        r["reason"] for r in recovery_rejects)
    resolved["recovered_buckets"] = _count(
        m["bucket"] for c in classification.values()
        for m in (c.get("members") or [])
        if m.get("evidence") in ("reverse_index", "override"))
    # An override that could not be used is a CURATION bug: reported even when
    # the reverse index rescued the word, because otherwise it is invisible.
    resolved["override_problems"] = len(override_problems)
    resolved["override_problems_sample"] = override_problems[:20]
    write_json(cfg.report_dir / "resolve_report.json", resolved)
    return resolved


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
