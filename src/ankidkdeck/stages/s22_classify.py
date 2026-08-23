"""Stage 22: the four-bucket classifier.

Buckets, in headline order: exact_cs (NFC codepoint-equal) > form (in the
article's own flex table) > variant (orthographic variant / alias) > exact_ci
(equal only after casefold -- where Er(erbium)/er lives). Decisive bucket-2
signal is flex-table containment; tab labels are refuted (kan->khan). Exactness
is EXCLUSIVE: a word that is its own dictionary headword is never absorbed as a
form of another lemma (godt has 15 senses of its own AND sits in god's flex
table). Affix/junk detection is by headword shape and data-pos-key, never by
sitemap shard.
"""

from ..config import Config
from ..util import NFC, nk, read_json, write_json

BUCKET_ORDER = {"exact_cs": 0, "form": 1, "variant": 2, "exact_ci": 3}


def _squash(s: str) -> str:
    return nk(s).replace(" ", "").replace("-", "")


def _is_variant(word: str, e: dict, alias_pairs: set) -> bool:
    if _squash(word) == _squash(e["lemma"]):
        return True
    if nk(word) in {nk(a["form"]) for a in e["alt_spellings"]}:
        return True
    pair = (nk(word), nk(e["lemma"]))
    return pair in alias_pairs or (pair[1], pair[0]) in alias_pairs


def classify_one(word: str, e: dict, demoted_pos: set, alias_pairs: set):
    q, hw = NFC(word), e["lemma"]
    if hw.startswith("-") or hw.endswith("-"):
        return "reject", "affix"                      # -kvinde (kvinder), -ske (sker)
    if " " in hw and " " not in q:
        return "reject", "multiword_neighbour"        # en bloc, engroshandel
    if hw.rstrip(".") == q and hw != q:
        return "reject", "abbreviation"               # min. (min), o.k. (ok)
    demoted = e.get("pos_key") in demoted_pos
    if NFC(hw) == q:
        return "exact_cs", ("demoted" if demoted else "exact")
    if nk(q) in set(e["form_index"]):
        return "form", "flex_table"
    if _is_variant(q, e, alias_pairs):
        return "variant", "orthographic_variant"
    if nk(hw) == nk(q):
        # Case-only match: Er(erbium)/er, VAR/var, Se(selenium)/se.
        if demoted:
            return "reject", "case_only_demoted_pos"
        return "exact_ci", "case_only"
    return "reject", "unrelated"


def run(cfg: Config, registry) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    wordset = {w["word"] for w in wordlist}
    # candidates per word = the articles parsed from that word's own page(s)
    cands: dict[str, list[str]] = {}
    for eid, e in entries.items():
        for w in e.get("source_words", []):
            cands.setdefault(w, []).append(eid)

    demoted_pos = registry.demoted_pos_keys
    alias_pairs = registry.alias_pairs
    classification: dict[str, dict] = {}
    rejected_report = []
    bucket2_report = []
    for w in sorted(wordset | set(cands)):
        c = {"members": [], "xrefs": [], "rejected": [], "resolved_by": None}
        for eid in sorted(set(cands.get(w, []))):
            e = entries[eid]
            bucket, why = classify_one(w, e, demoted_pos, alias_pairs)
            if bucket == "reject":
                row = {"entry_id": eid, "headword": e["lemma"],
                       "pos_key": e.get("pos_key"), "reason": why}
                c["rejected"].append(row)
                rejected_report.append({"word": w, **row})
                continue
            c["members"].append({"entry_id": eid, "bucket": bucket,
                                 "demoted": e.get("pos_key") in demoted_pos})
            if bucket == "form":
                bucket2_report.append({"word": w, "entry_id": eid, "lemma": e["lemma"]})
        # Exclusive exactness: measured cost on 2025 data is zero; prevents the
        # live godt/god hazard at rank 53.
        if any(m["bucket"] in ("exact_cs", "exact_ci") for m in c["members"]):
            c["xrefs"] = [m["entry_id"] for m in c["members"] if m["bucket"] == "form"]
            c["members"] = [m for m in c["members"] if m["bucket"] != "form"]
        c["members"].sort(key=lambda m: (
            BUCKET_ORDER[m["bucket"]], m["demoted"],
            -len(entries[m["entry_id"]]["senses"]), m["entry_id"]))
        if c["members"]:
            c["resolved_by"] = "forward"
        classification[w] = c

    write_json(cfg.json_dir / "classification.json", classification)
    write_json(cfg.report_dir / "rejected.json", rejected_report)
    write_json(cfg.report_dir / "bucket2_accepted.json", bucket2_report)
    # Gate: no accepted member may be an affix-page entry.
    for w, c in classification.items():
        for m in c["members"]:
            if entries[m["entry_id"]].get("pos_key") in {"førsteled", "sidsteled", "suffiks", "præfiks"}:
                raise AssertionError(f"affix entry accepted for {w!r}: {m['entry_id']}")
    return {"words": len(classification),
            "with_members": sum(1 for c in classification.values() if c["members"]),
            "rejected_edges": len(rejected_report)}
