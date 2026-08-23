"""Stage 30: family merge, GUID freeze, unique assignment, dense rank.

One family == one card. The merge unit is settled (final guide 1.2): connected
components of the bipartite graph (wordlist word, entry_id), computed AFTER the
classifier and after the exclusive-exactness rule. A per-group loop is wrong for
a measured reason -- 37 query words sit in two "duplicate groups" (`have` heads
both have-1 sb. and have-2 vb.), and a group loop emits two cards for them and
then collides. The largest real component has 8 words
(mod moede moeder moedes moedet moedt moedte moedtes).

Identity (final guide 1.4) is deliberately split three ways:
    family_id      = the anchor article's entry_id      -- stable, DDO-owned
    guid_seed      = registry/card_keys.json, append-only, human-reviewed
    display_headword = DDO's h1 including span.super    -- card face only
The seed is FROZEN here once per major version and never recomputed: it is the
users' study progress. guid_for() hashes the bytes, and 55 (headword, wordlist
word) pairs differ only by case -- ('Er','er') at rank 1.
"""

from collections import defaultdict

from .. import __version__
from ..config import Config
from ..gates import (G_ANCHOR, G_ASSIGN, G_RANK, G_SEED, Gate,
                     anchors_not_demoted_or_affix, dense_unique_ranks,
                     registry_seed_bytes, run_gates, unique_assignment)
from ..util import NFC, FatalError, nk, read_json, sha256_str, write_json
from .s22_classify import BUCKET_ORDER


class UnionFind:
    """Nodes are ("W", word) and ("E", entry_id) so the two sides of the
    bipartite graph can never be confused with each other."""

    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> list[tuple[list, list]]:
        groups: dict = defaultdict(lambda: ([], []))
        for node in list(self.parent):
            words, eids = groups[self.find(node)]
            (words if node[0] == "W" else eids).append(node[1])
        out = [(sorted(w), sorted(e)) for w, e in groups.values()]
        out.sort(key=lambda c: (c[0], c[1]))
        return out


def _dedupe_ordered(items) -> list:
    seen, out = set(), []
    for x in items:
        if x is None:
            continue
        x = NFC(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def run(cfg: Config, registry) -> dict:
    entries = read_json(cfg.json_dir / "entries.json")
    classification = read_json(cfg.json_dir / "classification.json")
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    sitemap = read_json(cfg.json_dir / "sitemap.json", default={})

    v2_path = cfg.json_dir / "legacy" / "v2_querywords.json"
    if not v2_path.exists():
        raise FatalError(
            "missing %s -- the GUID seeds cannot be frozen without the list of "
            "QueryWords that actually shipped in v2.1. Run the migrate stage "
            "(40) against the recovered 2025 workspace first "
            "(ankidkdeck migrate --legacy-workspace <path>)." % v2_path
        )
    v2_querywords = read_json(v2_path)

    demoted_pos = registry.demoted_pos_keys
    report: dict = {}

    def demoted(eid: str) -> bool:
        return entries[eid].get("pos_key") in demoted_pos

    # ---- 1. connected components. NEVER a group loop. --------------------
    wrank = {w["word"]: w["rank"] for w in wordlist}
    uf = UnionFind()
    off_wordlist = []
    for word, c in sorted(classification.items()):
        members = c.get("members") or []
        if word not in wrank:
            # classification is keyed on wordlist words; anything else would
            # silently add a card nobody asked for.
            if members:
                off_wordlist.append(word)
            continue
        for m in members:
            eid = m["entry_id"]
            if eid not in entries:
                raise FatalError(
                    f"classification references unknown entry_id {eid} for {word!r}")
            uf.union(("W", word), ("E", eid))
    report["words_off_wordlist_ignored"] = off_wordlist

    comps, refused = [], []
    for words, eids in uf.components():
        heads = sorted({entries[e]["lemma"] for e in eids if not demoted(e)})
        if len(heads) > 1:
            # Refuse, never guess: a component holding two real lemmas would
            # merge two dictionary words onto one card. Post-classifier the
            # 2025 replay yields 0 of these, so this is a net, not a workflow.
            refused.append({"heads": heads, "words": words, "entry_ids": eids})
            continue
        comps.append({"words": words, "entry_ids": eids, "refused": False})
    for r in refused:
        for w in r["words"]:
            own = sorted({m["entry_id"] for m in classification[w]["members"]})
            comps.append({"words": [w], "entry_ids": own, "refused": True})
    write_json(cfg.review_dir / "merge_conflicts.json", refused)
    comps.sort(key=lambda c: (c["words"], c["entry_ids"]))

    # ---- 2. family_id and anchor ----------------------------------------
    def anchor_of(eids: list[str]) -> str:
        ranked = [e for e in eids if not entries[e]["empty"]] or list(eids)
        return max(ranked, key=lambda e: (len(entries[e]["senses"]),
                                          not demoted(e), -int(e)))

    families: dict[str, dict] = {}
    fam_of: dict[tuple[str, str], str] = {}
    disambiguated = []
    for comp in comps:
        if not comp["entry_ids"]:
            continue
        anchor = anchor_of(comp["entry_ids"])
        fid = anchor
        if fid in families:
            # Only reachable through the refused-component fallback, where two
            # words can keep the same article set. Keep the ids distinct rather
            # than silently re-merging what we just refused to merge.
            fid = f"{anchor}#{comp['words'][0]}"
            disambiguated.append({"anchor": anchor, "family_id": fid,
                                  "words": comp["words"]})
            if fid in families:
                raise FatalError(f"family_id collision that cannot be broken: {fid}")
        others = sorted((e for e in comp["entry_ids"] if e != anchor),
                        key=lambda e: (-len(entries[e]["senses"]), e))
        families[fid] = {
            "family_id": fid,
            "anchor_entry_id": anchor,
            "lemma": entries[anchor]["lemma"],
            "display_headword": entries[anchor]["display_headword"],
            "entry_ids": [anchor] + others,
            "members": [],
            "cross_refs": [],
            "merge_state": "refused" if comp["refused"] else None,
        }
        for w in comp["words"]:
            for e in comp["entry_ids"]:
                fam_of[(w, e)] = fid
    report["family_ids_disambiguated"] = disambiguated

    # ---- 3. UNIQUE ASSIGNMENT: one wordlist form -> exactly one family ----
    assignments: dict[str, dict] = {}
    unplaced: list[str] = []
    for w in wordlist:
        word = w["word"]
        members = (classification.get(word) or {}).get("members") or []
        if not members:
            # Owner decision 2026-08-24: a word that resolves to nothing is
            # skipped and recorded (stage 21 wrote it to unresolved.json),
            # never a stop.
            unplaced.append(word)
            continue
        pick = min(
            (BUCKET_ORDER.get(m["bucket"], 99),
             0 if entries[m["entry_id"]]["lemma_key"] == nk(word) else 1,
             int(m["entry_id"]), m["entry_id"], m["bucket"])
            for m in members)
        eid, bucket = pick[3], pick[4]
        fid = fam_of[(word, eid)]
        fam = families[fid]
        assignments[word] = {"family_id": fid, "via": bucket, "entry_id": eid,
                             "evidence": next((m.get("evidence") for m in members
                                               if m["entry_id"] == eid), None)}
        relation = ("anchor" if nk(word) == nk(fam["lemma"])
                    else "inflection" if bucket == "form"
                    else "variant" if bucket == "variant" else "alias")
        fam["members"].append({"word": word, "wiktionary_rank": w["rank"],
                               "relation": relation})
        for x in (classification[word].get("xrefs") or []):
            if x in fam["entry_ids"] or x not in entries:
                continue
            fam["cross_refs"].append({"entry_id": x, "lemma": entries[x]["lemma"],
                                      "pos_key": entries[x].get("pos_key")})
    report["unplaced_words"] = len(unplaced)
    report["unplaced_sample"] = unplaced[:25]

    # ---- 4. FREEZE the GUID seeds -----------------------------------------
    # Runs AFTER assignment, not before: the seed is chosen from the family's
    # members, and members only exist once every word has been assigned. (The
    # guide's pseudocode puts freeze at step 3 and members at step 4; that
    # ordering cannot execute.)
    memberless = [f for f, fam in families.items() if not fam["members"]]
    if memberless:
        raise FatalError(
            "families with no assigned member (a component exists that no "
            "wordlist word claims): %s" % memberless[:10])

    new_rows = {}
    for fid, fam in families.items():
        if fid in registry.card_keys:
            continue
        carried = [m["word"] for m in fam["members"] if m["word"] in v2_querywords]
        seed = (min(carried, key=lambda x: (v2_querywords[x], x)) if carried
                else fam["lemma"])
        if NFC(seed) != seed:
            raise FatalError(f"guid_seed for family {fid} is not NFC: {seed!r}")
        new_rows[fid] = {"guid_seed": seed, "lemma_at_freeze": fam["lemma"],
                         "since": __version__, "carried_from_v2": bool(carried)}
    counts = registry.freeze_card_keys(new_rows, cfg.registry_local / "card_keys.json")
    for fid, fam in families.items():
        fam["guid_seed"] = registry.card_keys[fid]["guid_seed"]
    freeze_report = {
        "added": counts["added"],
        "total": counts["total"],
        "carried": sum(1 for f in families
                       if registry.card_keys[f].get("carried_from_v2")),
        "new": sum(1 for f in families
                   if not registry.card_keys[f].get("carried_from_v2")),
        "families_this_build": len(families),
        "registry_overlay": str(cfg.registry_local / "card_keys.json"),
        "note": "human review: diff this overlay, then copy it into "
                "src/ankidkdeck/registry/card_keys.json before release",
    }
    write_json(cfg.report_dir / "registry_freeze_report.json", freeze_report)

    # ---- 5. rank, searchable forms, dense freq_rank ------------------------
    for fam in families.values():
        fam["members"].sort(key=lambda m: (m["wiktionary_rank"], m["word"]))
        fam["rank"] = min(m["wiktionary_rank"] for m in fam["members"])
        fam["searchable_forms"] = _dedupe_ordered(
            [fam["display_headword"], fam["lemma"]]
            + [m["word"] for m in fam["members"]]
            + [c for e in fam["entry_ids"]
               for r in entries[e]["paradigm"]["rows"] for c in r["cells"]]
            + [a["form"] for e in fam["entry_ids"] for a in entries[e]["alt_spellings"]])
        if fam["merge_state"] is None:
            fam["merge_state"] = "merged" if len(fam["members"]) > 1 else "standalone"
        fam["family_sha"] = sha256_str(
            "".join(sorted(entries[e]["article_sha"] for e in fam["entry_ids"])))
        fam["freq_rank"] = None

    renderable = [f for f in families.values()
                  if any(entries[e]["senses"] or entries[e]["expressions"]
                         for e in f["entry_ids"])]
    for i, fam in enumerate(sorted(renderable,
                                   key=lambda f: (f["rank"], str(f["family_id"]))), 1):
        fam["freq_rank"] = i

    # ---- 6. sitemap homograph shortfall (report only at this stage) --------
    sm_lemmas = (sitemap or {}).get("lemmas", {})
    shortfall = []
    for fam in families.values():
        row = sm_lemmas.get(nk(fam["lemma"]))
        if not row:
            continue
        n_site = len(row.get("urls", []))
        if n_site > len(fam["entry_ids"]):
            shortfall.append({"family_id": fam["family_id"], "lemma": fam["lemma"],
                              "sitemap_urls": n_site,
                              "entry_ids": len(fam["entry_ids"]),
                              "shortfall": n_site - len(fam["entry_ids"])})
    shortfall.sort(key=lambda r: (-r["shortfall"], r["family_id"]))

    # ---- 7. gates ----------------------------------------------------------
    run_gates([
        Gate(G_RANK, "freq_rank is dense 1..N and unique over renderable families",
             lambda: dense_unique_ranks([f["freq_rank"] for f in renderable],
                                        len(renderable)), stage="30"),
        Gate(G_ASSIGN, "every wordlist word belongs to at most one family",
             lambda: unique_assignment(assignments, families), stage="30"),
        Gate(G_ANCHOR, "no family is anchored on a demoted or affix article",
             lambda: anchors_not_demoted_or_affix(families, entries, demoted_pos),
             stage="30"),
        Gate(G_SEED, "every carried guid_seed is NFC and byte-equal to a v2.1 QueryWord",
             lambda: registry_seed_bytes(registry.card_keys, v2_querywords,
                                         list(families)), stage="30"),
    ], cfg, stage="30")

    # ---- 8. outputs --------------------------------------------------------
    write_json(cfg.json_dir / "words.json", families)
    write_json(cfg.json_dir / "assignments.json", assignments)
    report.update({
        "components": len(comps),
        "refused_components": len(refused),
        "families": len(families),
        "merged": sum(1 for f in families.values() if f["merge_state"] == "merged"),
        "standalone": sum(1 for f in families.values() if f["merge_state"] == "standalone"),
        "refused_families": sum(1 for f in families.values() if f["merge_state"] == "refused"),
        "renderable_families": len(renderable),
        "cards": len(renderable),
        "largest_component_words": max((len(f["members"]) for f in families.values()),
                                       default=0),
        "registry_freeze": freeze_report,
        "sitemap_shortfall_families": len(shortfall),
        "sitemap_shortfall": shortfall[:200],
    })
    write_json(cfg.report_dir / "merge_report.json", report)
    return {k: v for k, v in report.items()
            if k not in ("sitemap_shortfall", "unplaced_sample",
                         "words_off_wordlist_ignored")}
