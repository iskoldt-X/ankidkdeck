"""Stage 21: resolve wordlist words that classified to zero entries.

Layers, in order: (1) forward page already gave members; (2) reverse form index
over every kept article's flex table (this is what recovers er -> vaere);
(3) curated registry/form_to_lemma.json overrides; (4) known_no_entry.

OWNER DECISION (2026-08-24): a word that still resolves to nothing is SKIPPED
and recorded in unresolved.json for later human review -- the pipeline never
stops for it. That decision only works if the file is readable, so every row
carries an explicit `reason` and the counters are derived FROM the rows.
"""

from collections import defaultdict

from ..config import Config
from ..gates import is_affix_entry
from ..util import NFC, nk, read_json, write_json

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

    def attach(c: dict, eids, evidence: str) -> None:
        for eid in eids:
            c["members"].append({
                "entry_id": eid, "bucket": "form",
                # the REAL flag, read from the entry -- never hardcoded False
                "demoted": entries[eid].get("pos_key") in demoted_pos,
                "evidence": evidence})

    unresolved = []
    by_reason: dict[str, int] = {}
    resolved = {"forward": 0, "reverse_index": 0, "override": 0,
                "known_no_entry": 0}
    for w in wordlist:
        word = w["word"]
        c = classification.setdefault(
            word, {"members": [], "xrefs": [], "rejected": [], "resolved_by": None})
        if c["members"]:
            resolved["forward"] += 1
            continue
        hits = sorted(rev.get(nk(word), set()))
        if hits:
            attach(c, hits, "reverse_index")
            c["resolved_by"] = "reverse_index"
            resolved["reverse_index"] += 1
            continue
        override_lemma = registry.form_to_lemma.get(word)
        override_reason = None
        if override_lemma:
            key = nk(NFC(override_lemma))
            of_lemma = [e for e, v in entries.items() if v["lemma_key"] == key]
            usable = sorted(e for e in of_lemma if indexable(e))
            if usable:
                attach(c, usable, "override")
                c["resolved_by"] = "override"
                resolved["override"] += 1
                continue
            # An override that names a lemma page we never crawled is a curation
            # bug the human has to see; it must not fall through unmarked.
            override_reason = (REASON_NO_SURVIVOR if of_lemma
                               else REASON_OVERRIDE_NOT_CRAWLED)
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
    # Derived from the rows, so the counters and the file can never disagree.
    resolved["unresolved"] = len(unresolved)
    resolved["unresolved_by_reason"] = {r: by_reason.get(r, 0)
                                        for r in UNRESOLVED_REASONS}
    resolved["nohit"] = by_reason.get(REASON_NOHIT, 0)
    resolved["entries_rejected_everywhere"] = len(rejected_everywhere)
    resolved["entries_in_reverse_index"] = len(
        {e for eids in rev.values() for e in eids})
    write_json(cfg.report_dir / "resolve_report.json", resolved)
    return resolved
