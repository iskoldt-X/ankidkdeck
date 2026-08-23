"""Stage 21: resolve wordlist words that classified to zero entries.

Layers, in order: (1) forward page already gave members; (2) reverse form index
over every kept article's flex table (this is what recovers er -> vaere);
(3) curated registry/form_to_lemma.json overrides; (4) known_no_entry.

OWNER DECISION (2026-08-24): a word that still resolves to nothing is SKIPPED
and recorded in unresolved.json for later human review -- the pipeline never
stops for it.
"""

from collections import defaultdict

from ..config import Config
from ..util import NFC, nk, read_json, write_json


def run(cfg: Config, registry) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    classification = read_json(cfg.json_dir / "classification.json")
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    ledger = read_json(cfg.json_dir / "fetch_ledger.json", default={})

    # Reverse form index over kept (non-rejected-everywhere) entries. An entry
    # is indexable if at least one word accepted it, or nothing saw it yet.
    rejected_everywhere = set()
    seen = defaultdict(int)
    kept = defaultdict(int)
    for c in classification.values():
        for m in c["members"]:
            kept[m["entry_id"]] += 1
            seen[m["entry_id"]] += 1
        for r in c["rejected"]:
            seen[r["entry_id"]] += 1
    for eid in entries:
        if seen.get(eid, 0) > 0 and kept.get(eid, 0) == 0:
            rejected_everywhere.add(eid)

    rev = defaultdict(set)
    for eid, e in entries.items():
        if eid in rejected_everywhere:
            continue
        for f in e["form_index"]:
            rev[f].add(eid)

    unresolved = []
    resolved = {"forward": 0, "reverse_index": 0, "override": 0,
                "known_no_entry": 0, "nohit": 0, "unresolved": 0}
    for w in wordlist:
        word = w["word"]
        c = classification.setdefault(
            word, {"members": [], "xrefs": [], "rejected": [], "resolved_by": None})
        if c["members"]:
            resolved["forward"] += 1
            continue
        hits = sorted(rev.get(nk(word), set()))
        if hits:
            for eid in hits:
                c["members"].append({"entry_id": eid, "bucket": "form",
                                     "demoted": False, "evidence": "reverse_index"})
            c["resolved_by"] = "reverse_index"
            resolved["reverse_index"] += 1
            continue
        lemma = registry.form_to_lemma.get(word)
        if lemma:
            eids = sorted(e for e, v in entries.items()
                          if v["lemma_key"] == nk(NFC(lemma)) and e not in rejected_everywhere)
            if eids:
                for eid in eids:
                    c["members"].append({"entry_id": eid, "bucket": "form",
                                         "demoted": False, "evidence": "override"})
                c["resolved_by"] = "override"
                resolved["override"] += 1
                continue
        if word in registry.known_no_entry:
            c["resolved_by"] = "known_no_entry"
            resolved["known_no_entry"] += 1
            continue
        status = (ledger.get(word) or {}).get("status")
        if status == "nohit":
            resolved["nohit"] += 1
        resolved["unresolved"] += 1 if status != "nohit" else 0
        unresolved.append({
            "word": word, "rank": w["rank"], "fetch_status": status,
            "articles_on_page": (ledger.get(word) or {}).get("article_count"),
            "rejected_headwords": [r["headword"] for r in c["rejected"]],
        })

    write_json(cfg.json_dir / "classification.json", classification)
    write_json(cfg.report_dir / "unresolved.json", unresolved)
    resolved["unresolved_total_recorded"] = len(unresolved)
    write_json(cfg.report_dir / "resolve_report.json", resolved)
    return resolved
