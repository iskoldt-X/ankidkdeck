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
    display_headword = the anchor's LEMMA, with its homograph index kept apart
                       in `super` and rendered as a superscript by stage 70
The seed is FROZEN here once per major version and never recomputed: it is the
users' study progress. guid_for() hashes the bytes, and 55 (headword, wordlist
word) pairs differ only by case -- ('Er','er') at rank 1.
"""

from collections import defaultdict

from ..config import Config
from ..gates import (G_ANCHOR, G_ASSIGN, G_CASE, G_RANK, G_REGKEY, G_SEED, Gate,
                     anchors_not_demoted_or_affix, case_only_members,
                     dense_unique_ranks, registry_family_ids,
                     registry_seed_bytes, run_gates, unique_assignment)
from ..util import NFC, FatalError, nk, read_json, sha256_str, write_json
from .s22_classify import BUCKET_ORDER, squash


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


def best_member(word: str, members: list, entries: dict) -> tuple:
    """The ONE article a word is assigned to: lowest bucket, own lemma first,
    then the lowest entry_id. Used both for the assignment and for deciding
    which head of a refused component a word joins, so the two can never
    disagree."""
    pick = min(
        (BUCKET_ORDER.get(m["bucket"], 99),
         0 if entries[m["entry_id"]]["lemma_key"] == nk(word) else 1,
         int(m["entry_id"]), m["entry_id"], m["bucket"])
        for m in members)
    return pick[3], pick[4]


def _relation(word: str, bucket: str, fam: dict) -> str:
    if nk(word) == nk(fam["lemma"]):
        return "anchor"
    if bucket == "form":
        return "inflection"
    if bucket == "variant":
        return "variant"
    return "alias"


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
    # EVERY word with classification members joins the graph, wordlist or not.
    # Phase B/C fetch the lemma pages precisely because the article set depends
    # on which form you query (/har returns 1 article, /have returns 3), so
    # excluding non-wordlist words made the ~275 phase-B requests buy nothing:
    # have(sb., garden, 5 senses) landed on no card at all. Off-wordlist words
    # join families but carry wiktionary_rank = null and never drive rank.
    wrank = {w["word"]: w["rank"] for w in wordlist}
    uf = UnionFind()
    off_wordlist = []
    for word, c in sorted(classification.items()):
        members = c.get("members") or []
        if not members:
            continue
        if word not in wrank:
            off_wordlist.append(word)
        for m in members:
            eid = m["entry_id"]
            if eid not in entries:
                raise FatalError(
                    f"classification references unknown entry_id {eid} for {word!r}")
            uf.union(("W", word), ("E", eid))
    report["words_off_wordlist"] = off_wordlist

    # A "head" is an ORTHOGRAPHIC IDENTITY (squash: casefold, drop spaces and
    # hyphens), not a raw lemma string. Two lemmas that differ only by case or
    # spacing are the same dictionary word -- that is exactly what buckets 3 and
    # 4 admit -- so comparing raw strings would refuse every component the two
    # buckets exist to build: `udenfor`+`uden for` (and indenfor, overfor,
    # ovenpaa, bagefter, bagved, nedenunder, indeni, udenom, udover), `i`+`I`,
    # `var`+`VAR`. That is also the only reading under which the guide's own two
    # statements agree: it names those words as content that must ship AND
    # claims 0 multi-headword components post-classifier. `hav`/`have` and
    # `kunne`/`khan` still squash apart and are still refused.
    comps, refused = [], []
    for words, eids in uf.components():
        by_head_key: dict[str, list] = {}
        for e in eids:
            if demoted(e):
                continue
            by_head_key.setdefault(squash(entries[e]["lemma"]), []).append(e)
        if len(by_head_key) > 1:
            # Refuse, never guess: a component holding two real lemmas would
            # merge two dictionary words onto one card.
            refused.append({
                "heads": sorted({entries[e]["lemma"] for e in eids
                                 if not demoted(e)}),
                "head_keys": sorted(by_head_key),
                "words": words, "entry_ids": eids})
            continue
        comps.append({"words": words, "entry_ids": eids, "refused": False})

    # SPLIT the refused component, one fallback family per distinct non-demoted
    # head lemma, each owning ONLY the entries whose own lemma is that head. The
    # previous fallback handed the word back all of its members, i.e. rebuilt
    # exactly the component it had just refused (khan rendered as a meaning
    # block on the rank-23 `kan` card) and, with two words in the component,
    # emitted two families with identical anchor, entry_ids and lemma -- two
    # byte-identical cards with two GUIDs. Entry sets are disjoint per head, so
    # two fallback families can never share an anchor and family_id stays a bare
    # entry_id. Demoted entries are left out and become GC orphans.
    for r in refused:
        by_head: dict[str, list] = {}
        left_out = []
        for e in r["entry_ids"]:
            if demoted(e):
                left_out.append(e)
                continue
            # Same head key as the refusal test, so a head can never be split
            # from an article the classifier called a variant of it.
            by_head.setdefault(squash(entries[e]["lemma"]), []).append(e)
        r["fallback_heads"] = sorted(by_head)
        r["entries_left_out"] = sorted(left_out)
        for head in sorted(by_head):
            eids = sorted(by_head[head])
            own = set(eids)
            words = sorted(
                w for w in r["words"]
                if any(m["entry_id"] in own
                       for m in ((classification.get(w) or {}).get("members") or [])))
            if not words:
                r.setdefault("heads_with_no_word", []).append(head)
                continue
            comps.append({"words": words, "entry_ids": eids, "refused": True})
    write_json(cfg.review_dir / "merge_conflicts.json", refused)
    comps.sort(key=lambda c: (c["words"], c["entry_ids"]))

    # ---- 2. family_id and anchor ----------------------------------------
    def edge_bucket(word: str, eid: str) -> str | None:
        """The classifier's verdict on this exact (word, article) pair."""
        for m in ((classification.get(word) or {}).get("members") or []):
            if m["entry_id"] == eid:
                return m.get("bucket")
        return None

    def best_bucket(words: list[str], eid: str) -> int:
        """The best (lowest) classifier bucket any of the family's words gave
        this article. 99 when no word claims it as a member."""
        ranks = [BUCKET_ORDER.get(edge_bucket(w, eid), 99) for w in words]
        return min(ranks) if ranks else 99

    def anchor_of(eids: list[str], words: list[str]) -> str:
        """The card headline, in the guide's own order (4.8).

        Key, after the non-empty filter:
          1. NOT DEMOTED. With sense count first a demoted article outranked a
             real one whenever it had more senses -- Cm(symbol, 4 senses) beat
             centimeter(sb., 1 sense) -- and G-ANCHOR then failed the whole build
             on a tie-break accident rather than on a defect. With demotion first
             demoted_anchored_with_alternative is unreachable by construction.
          2. BEST CLASSIFIER BUCKET (exact_cs > form > variant > exact_ci). Guide
             4.8 rules that bucket order decides the headline, Variants and
             Etymology, because stage 70 reads them off sorted_entries[0]. Without
             it `uden for` (praep., variant bucket, 4 senses) headed the card for
             `udenfor` (exact_cs, 3 senses) -- the card face showed a spelling no
             wordlist word has.
          3. THE WORDLIST SPELLING. Tie-break within a bucket: prefer the article
             whose own lemma IS the family's lowest-rank member word.
          4. sense count, then the lowest entry_id (oldest DDO article).

        `var`/`VAR` is deliberately NOT fixed here: the `var` adjective has 0
        senses, so it is `empty` and the filter one line below excludes it.
        Forcing an empty article to anchor would hand the family id, the
        Etymology field and a header-only meaning block to an article with
        nothing to render. That family's content genuinely belongs to a different
        spelling -- information for a human, which is what
        review/case_only_members.json is for.
        """
        lead = min(words, key=lambda w: (wrank.get(w, 10 ** 9), w)) if words else ""
        lead_key = nk(lead)
        ranked = [e for e in eids if not entries[e]["empty"]] or list(eids)
        return max(ranked, key=lambda e: (not demoted(e),
                                          -best_bucket(words, e),
                                          entries[e]["lemma_key"] == lead_key,
                                          len(entries[e]["senses"]), -int(e)))

    families: dict[str, dict] = {}
    fam_of: dict[tuple[str, str], str] = {}
    for comp in comps:
        if not comp["entry_ids"]:
            continue
        anchor = anchor_of(comp["entry_ids"], comp["words"])
        fid = anchor
        if fid in families:
            # Components have pairwise-disjoint entry sets (union-find gives
            # that, and the refused split preserves it), so this is now a real
            # invariant violation rather than an expected collision to paper
            # over with a "#word" suffix.
            raise FatalError(
                "two components claim anchor %s; family_id must stay a bare "
                "entry_id (words: %s)" % (anchor, comp["words"][:5]))
        others = sorted((e for e in comp["entry_ids"] if e != anchor),
                        key=lambda e: (-len(entries[e]["senses"]), e))
        families[fid] = {
            "family_id": fid,
            "anchor_entry_id": anchor,
            "lemma": entries[anchor]["lemma"],
            "display_headword": entries[anchor]["display_headword"],
            "super": entries[anchor].get("super"),
            "entry_ids": [anchor] + others,
            "members": [],
            "cross_refs": [],
            "merge_state": "refused" if comp["refused"] else None,
        }
        for w in comp["words"]:
            for e in comp["entry_ids"]:
                fam_of[(w, e)] = fid

    # ---- 3. UNIQUE ASSIGNMENT: one form -> exactly one family --------------
    def place(word: str, rank) -> tuple | None:
        """Join `word` to the one family its best surviving member points at.
        Returns (family_id, entry_id, bucket), or None if nothing survived."""
        c = classification.get(word) or {}
        members = c.get("members") or []
        # A member whose entry was left out of every family (demoted, or a head
        # no word claimed) cannot carry the word; the word joins on its best
        # SURVIVING member instead.
        placed = [m for m in members if (word, m["entry_id"]) in fam_of]
        if not placed:
            return None
        eid, bucket = best_member(word, placed, entries)
        fid = fam_of[(word, eid)]
        fam = families[fid]
        fam["members"].append({"word": word, "wiktionary_rank": rank,
                               "relation": _relation(word, bucket, fam)})
        seen_x = {x["entry_id"] for x in fam["cross_refs"]}
        for x in (c.get("xrefs") or []):
            if x in fam["entry_ids"] or x in seen_x or x not in entries:
                continue
            seen_x.add(x)
            fam["cross_refs"].append({"entry_id": x, "lemma": entries[x]["lemma"],
                                      "pos_key": entries[x].get("pos_key")})
        return fid, eid, bucket

    assignments: dict[str, dict] = {}
    unplaced: list[str] = []
    for w in wordlist:
        word = w["word"]
        got = place(word, w["rank"])
        if got is None:
            # Owner decision 2026-08-24: a word that resolves to nothing is
            # skipped and recorded (stage 21 wrote it to unresolved.json),
            # never a stop.
            unplaced.append(word)
            continue
        fid, eid, bucket = got
        members = (classification.get(word) or {}).get("members") or []
        assignments[word] = {"family_id": fid, "via": bucket, "entry_id": eid,
                             "evidence": next((m.get("evidence") for m in members
                                               if m["entry_id"] == eid), None)}
    for word in off_wordlist:
        place(word, None)
    report["unplaced_words"] = len(unplaced)
    report["unplaced_sample"] = unplaced[:25]

    # ---- 4. drop the components no wordlist word claims --------------------
    # This replaces the `memberless` FATAL, which is what forced non-wordlist
    # words out of the graph in the first place. A component nobody on the
    # wordlist reaches is not an error, it is an article we fetched and do not
    # need; its entries become GC orphans. Owner ruling: record, never stop.
    dropped_components = []
    for fid in sorted(families):
        fam = families[fid]
        if not any(m["wiktionary_rank"] is not None for m in fam["members"]):
            dropped_components.append({
                "family_id": fid, "lemma": fam["lemma"],
                "entry_ids": fam["entry_ids"],
                "off_wordlist_words": [m["word"] for m in fam["members"]],
                "why": "no wordlist word claims this component"})
            del families[fid]
    report["dropped_components"] = len(dropped_components)
    report["dropped_components_sample"] = dropped_components[:25]

    # ---- 5. PROPOSE the GUID seeds (nothing is written yet) ----------------
    # Runs AFTER assignment, not before: the seed is chosen from the family's
    # members, and members only exist once every word has been assigned. (The
    # guide's pseudocode puts freeze at step 3 and members at step 4; that
    # ordering cannot execute.)
    new_rows = {}
    for fid, fam in families.items():
        if fid in registry.card_keys:
            continue
        carried = [m["word"] for m in fam["members"] if m["word"] in v2_querywords]
        seed = (min(carried, key=lambda x: (v2_querywords[x], x)) if carried
                else fam["lemma"])
        if NFC(seed) != seed:
            raise FatalError(f"guid_seed for family {fid} is not NFC: {seed!r}")
        # `since` is a RELEASE label from config, never __version__: the rows are
        # immutable once frozen, so a dev pre-release stamp ("3.0.0a0") would
        # brand every v3.0 family forever in the file whose human-reviewed diff
        # IS the release artifact.
        new_rows[fid] = {"guid_seed": seed, "lemma_at_freeze": fam["lemma"],
                         "since": cfg.registry_version,
                         "carried_from_v2": bool(carried)}
    proposed = {**registry.card_keys, **new_rows}
    for fid, fam in families.items():
        fam["guid_seed"] = proposed[fid]["guid_seed"]

    # ---- 6. rank, searchable forms, dense freq_rank ------------------------
    for fam in families.values():
        # None sorts last: an off-wordlist lemma page never sets the tone.
        fam["members"].sort(key=lambda m: (m["wiktionary_rank"] is None,
                                           m["wiktionary_rank"] or 0, m["word"]))
        fam["rank"] = min(m["wiktionary_rank"] for m in fam["members"]
                          if m["wiktionary_rank"] is not None)
        # display_headword is the lemma now, never the glued 'udenfor1': that
        # form was a junk Anki search token on every homograph card.
        fam["searchable_forms"] = _dedupe_ordered(
            [fam["lemma"]]
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

    # ---- 6b. the case-only membership population, as a REVIEWED artifact ----
    # Bucket 4 (exact_ci) is card membership on purpose: making it xref-only
    # would delete `I` (pron., 2 senses + 7 expressions) from the deck, because
    # no wordlist word reaches it -- the wordlist's only capitalised entries are
    # names (Jack, Dr, Mrs, David...). The anchor rule above keeps such an
    # article from heading the card, which was the real risk. What remains is a
    # content change a human should see once: ~55 (headword, wordlist word) pairs
    # that differ only by case, plus the `var`/`VAR` class where the family's
    # whole content sits under a different spelling. Written out, and the COUNT
    # baselined in registry/gates.json so the population cannot grow silently.
    # The population is per EDGE, not per word: `i` matches i(praep.) exactly AND
    # I(pron.) by case only, and it is the second edge that changes what the card
    # shows. Two row kinds, both reviewable:
    #   case_only_member  a (word, article) pair admitted through bucket 4
    #   anchor_spelling   the anchor's own lemma differs from the family's
    #                     lowest-rank member word by case only (var/VAR)
    case_only = []
    for fid in sorted(families):
        fam = families[fid]
        lemma = fam["lemma"]
        own = set(fam["entry_ids"])
        for m in fam["members"]:
            word = m["word"]
            for x in ((classification.get(word) or {}).get("members") or []):
                eid = x["entry_id"]
                if eid not in own or x.get("bucket") != "exact_ci":
                    continue
                case_only.append({
                    "kind": "case_only_member", "family_id": fid,
                    "anchor_lemma": lemma, "word": word, "entry_id": eid,
                    "member_lemma": entries[eid]["lemma"],
                    "pos_key": entries[eid].get("pos_key"),
                    "senses": len(entries[eid]["senses"]),
                    "wiktionary_rank": m.get("wiktionary_rank"),
                    "why": "this article joins the family only through a "
                           "case-only (exact_ci) match, so its meaning block "
                           "appears on a card the wordlist reached under a "
                           "different capitalisation"})
            if nk(lemma) == nk(word) and NFC(lemma) != NFC(word):
                case_only.append({
                    "kind": "anchor_spelling", "family_id": fid,
                    "anchor_lemma": lemma, "word": word,
                    "wiktionary_rank": m.get("wiktionary_rank"),
                    "why": "the anchor article's own spelling differs from the "
                           "wordlist word by case only, so the meaning-block "
                           "header and the Etymology field come from a "
                           "capitalisation the wordlist does not have"})
    write_json(cfg.review_dir / "case_only_members.json", case_only)
    report["case_only_members"] = len(case_only)
    report["case_only_members_by_kind"] = {
        k: sum(1 for r in case_only if r["kind"] == k)
        for k in ("case_only_member", "anchor_spelling")}

    # ---- 7. sitemap homograph shortfall (report only at this stage) --------
    # Enforced at export (G-SITEMAP) and reported by phase C, where a shortfall
    # can actually be remedied by a fetch. Compared against max(homographs) or
    # 1, not len(urls): `urls` holds every URL under the lemma key including the
    # bare and alias slugs, so a lemma listed both ways showed a phantom gap.
    sm_lemmas = (sitemap or {}).get("lemmas", {})
    shortfall = []
    for fam in families.values():
        row = sm_lemmas.get(nk(fam["lemma"]))
        if not row:
            continue
        n_site = max(row.get("homographs") or [0]) or 1
        if n_site > len(fam["entry_ids"]):
            shortfall.append({"family_id": fam["family_id"], "lemma": fam["lemma"],
                              "sitemap_homographs": n_site,
                              "sitemap_urls": len(row.get("urls", [])),
                              "entry_ids": len(fam["entry_ids"]),
                              "shortfall": n_site - len(fam["entry_ids"])})
    # Lemmas the classifier rejected wholesale never became families, so they are
    # invisible above -- and they are the interesting ones (vinge, uanset, vove,
    # zone). Counted separately.
    fam_lemma_keys = {nk(f["lemma"]) for f in families.values()}
    rejected_keys = {entries[r["entry_id"]]["lemma_key"]
                     for c in classification.values()
                     for r in (c.get("rejected") or [])
                     if r["entry_id"] in entries}
    rejected_lemmas = sorted((rejected_keys & set(sm_lemmas)) - fam_lemma_keys)
    shortfall.sort(key=lambda r: (-r["shortfall"], r["family_id"]))
    write_json(cfg.json_dir / "sitemap_shortfall.json",
               {"rows": shortfall, "n_families": len(families),
                "lemmas_rejected_wholesale": rejected_lemmas,
                "note": "reported here, enforced at export (G-SITEMAP); the "
                        "remedy is a fetch, which stage 30 cannot do"})

    # ---- 8. gates ----------------------------------------------------------
    # Everything above is a proposal held in memory. The registry is frozen
    # only after every gate has passed (step 9): card_keys.json IS the users'
    # study progress, and a failed build used to leave a seed permanently frozen
    # onto whatever article the broken anchor rule had picked -- which the next
    # run then treats as reviewed truth and never revisits.
    run_gates([
        Gate(G_RANK, "freq_rank is dense 1..N and unique over renderable families",
             lambda: dense_unique_ranks([f["freq_rank"] for f in renderable],
                                        len(renderable)), stage="30"),
        Gate(G_ASSIGN, "every wordlist word belongs to at most one family",
             lambda: unique_assignment(assignments, families), stage="30"),
        Gate(G_ANCHOR, "no family is anchored on a demoted or affix article",
             lambda: anchors_not_demoted_or_affix(
                 families, entries, demoted_pos,
                 int(registry.gates.get("all_demoted_families_max", 40))),
             stage="30"),
        Gate(G_SEED, "every carried guid_seed is NFC and byte-equal to a v2.1 QueryWord",
             lambda: registry_seed_bytes(proposed, v2_querywords,
                                         list(families)), stage="30"),
        Gate(G_REGKEY, "every card_keys.json family_id is a bare DDO entry_id",
             lambda: registry_family_ids(proposed), stage="30"),
        Gate(G_CASE, "the case-only family membership population "
                     "(review/case_only_members.json) is inside its baseline",
             lambda: case_only_members(
                 case_only,
                 int(registry.gates.get("case_only_members_max", 80))),
             stage="30"),
    ], cfg, stage="30")

    # ---- 9. FREEZE the registry, then write the outputs -------------------
    counts = registry.freeze_card_keys(new_rows, cfg.registry_local / "card_keys.json")
    freeze_report = {
        "added": counts["added"],
        "total": counts["total"],
        "carried": sum(1 for f in families
                       if registry.card_keys[f].get("carried_from_v2")),
        "new": sum(1 for f in families
                   if not registry.card_keys[f].get("carried_from_v2")),
        "families_this_build": len(families),
        "registry_version": cfg.registry_version,
        "registry_overlay": str(cfg.registry_local / "card_keys.json"),
        "note": "written only after every stage-30 gate passed. Human review: "
                "diff this overlay, then copy it into "
                "src/ankidkdeck/registry/card_keys.json before release",
    }
    write_json(cfg.report_dir / "registry_freeze_report.json", freeze_report)
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
        "sitemap_lemmas_rejected_wholesale": len(rejected_lemmas),
        "sitemap_shortfall": shortfall[:200],
        "case_only_members_sample": case_only[:25],
    })
    write_json(cfg.report_dir / "merge_report.json", report)
    return {k: v for k, v in report.items()
            if k not in ("sitemap_shortfall", "unplaced_sample",
                         "words_off_wordlist", "dropped_components_sample",
                         "case_only_members_sample")}
