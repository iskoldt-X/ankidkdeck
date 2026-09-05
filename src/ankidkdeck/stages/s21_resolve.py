"""Stage 21: resolve wordlist words that classified to zero entries.

Layers, in order: (0) registry/wordlist_invalid.json drops OCR-damaged wordlist
rows before anything else; (1) forward page already gave members; (2) curated
registry/form_to_lemma.json overrides; (3) reverse form index over every kept
article's flex table (this is what recovers er -> vaere); (4) DDO's dotted
abbreviation entry; (5) known_no_entry.

LAYER 4 IS LAST FOR A REASON, AND THE REASON IS MEASURED. Owner policy B
(2026-08-26) adopts the 19 wordlist words whose only reachable DDO article is the
abbreviation entry spelled with a period -- mr, Dr, Mrs, hr, kl, nr, st, frk,
phil, ca, pga, mia, kr, th, mio, no, on, oz, ma, every one of which shipped a
v2.1 card. 42 words in the corpus reach such an article, not 19:

  * 20 of them are WORDLIST ROWS that already own a card through an exactness
    bucket (min/min., med/med., to/to., ti, tv, par, port, red, art, da, den,
    do, eks, el, fa, man, pr, soe, soen, aarh) -- 19 by `exact_cs` and `pr` by
    `exact_ci`, which binds the unrelated article `PR`(sb.). Admitting the dotted
    article for those would staple a second, unrelated dictionary word's meaning
    block onto 20 existing cards -- `min.` is `minut` -- and policy B says
    INDEPENDENT abbreviation cards. They never reach layer 4 at all, because
    layer 1 returns first: exclusive exactness is protected by the layer order,
    not by a new filter that could later be weakened.
  * `vaer` reaches `vaer.`(fork., 2 senses) AND is the imperative of `vaere`,
    which a curated override already binds at layer 2. An abbreviation layer
    placed before the override or the reverse index would steal it.
  * 2 more (`net`, `sen`) reach a dotted article but are NOT wordlist rows: they
    are off-list `source_words`. This loop iterates the wordlist, so those two
    are excluded by the LOOP and never depended on the layer order at all
    (reviewer A/B, round 4). 20 + 1 + 2 + 19 = 42.

Layer 4 therefore admits exactly the 19 words and nothing else, and that is a
property of the ORDER rather than of a hand-maintained word list.

Layer 4 also deliberately drops the demoted filter that layers 2 and 3 keep: 18
of the 19 dotted entries are `fork.` or `symbol`, i.e. demoted, which is the
whole reason the policy needed an owner decision. is_dotted_abbreviation() is
what makes that safe -- it admits ONLY a headword that is the query plus trailing
periods, and the element-symbol articles that put `symbol` in demoted_pos_keys
(`Th`, `No`, `Ca`, `Kr`, `mA` for th/no/ca/kr/ma) are case-only matches with no
period, so they are still rejected. An all-demoted family is not new either:
`cal`, `cm`, `kg`, `km`, `ms` and `www` are already six of them and they ship
cards; G-ANCHOR baselines the count.

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

THE OVERRIDE CHANNEL WAS SEALED SHUT (round 3, measured). Routing the override
through classify_one() unchanged made the registry useless for the one job it
exists to do. DDO's 2026 flex tables carry no imperative, no genitive, no
middle-voice -s, no present participle and no definite superlative, so
`vent`/`vente` walks the whole decision chain down to the final fallback and
comes back ("reject", "unrelated") -- and 140 hand-verified mappings recovered
exactly ZERO words while reporting themselves green. Two locks, both lifted
here and only for `evidence == "override"`:

  1. judged() dropped every candidate the classifier called `unrelated`.
     `unrelated` is the classifier's LAST fallback: it means "the automatic
     layers have nothing to say", which is precisely the case a curated mapping
     is for. It is now accepted as bucket `form` -- `form`, not `variant`, so
     s30._relation() calls it an inflection and s70.alt_forms_html keeps it out
     of the visible Variants line: the imperative becomes an Anki search token,
     not card-face text. affix / abbreviation / multiword_neighbour /
     case_only_demoted_pos still reject: those are the classifier making a
     positive claim about the article, not running out of ideas.
  2. indexable() also refused any article in `rejected_everywhere`, and the
     rejection that put the article there was the overriding word's OWN
     `unrelated` -- a self-referential lockout that judged() cannot see,
     because the candidate is filtered out before it is ever judged (no audit
     row either: 174 recovery rejections covered only 136 of 140 words). That
     filter exists to keep the erbium-class SYMBOL article out of the recovery
     door; symbol articles are demoted, and the demoted and affix filters both
     still apply on the override path.

Exclusive exactness (godt/god) is not reachable through either: layer 2 only
runs for a word whose forward classification produced NO members, and a word
that is its own dictionary headword always has one.

The remaining guard is the data itself, so every mapping that binds this way is
written to review/override_accepted.json and the count of mappings that did NOT
bind is a GATE (G-OVERRIDE). Before, override_problems could go 0 -> 140 with
the gate report still all green, because stage 21 had no gate at all.
"""

from collections import defaultdict

from ..config import Config
from ..gates import (G_ADMIT, G_OVERRIDE, G_SUPPRESS, Gate,
                     abbreviation_admissions, curated_overrides_bind,
                     is_affix_entry, run_gates, suppression_registries)
from ..util import NFC, nk, read_json, write_json
from .s22_classify import classify_one, is_dotted_abbreviation

# Reason codes for unresolved.json. Closed set, one row per skipped word.
REASON_NOHIT = "nohit"                     # DDO has no such word
REASON_ALL_REJECTED = "all_rejected"       # the page had articles; all rejected
REASON_OVERRIDE_NOT_CRAWLED = "override_lemma_not_crawled"
REASON_NO_SURVIVOR = "no_survivor"         # nothing left to attach
UNRESOLVED_REASONS = (REASON_NOHIT, REASON_ALL_REJECTED,
                      REASON_OVERRIDE_NOT_CRAWLED, REASON_NO_SURVIVOR)

# The one sentence every review/abbreviation_accepted.json row carries. Module
# level because the row is now DERIVED from classification.json rather than
# appended while layer 4 runs -- see abbreviation_admission_rows().
ABBREVIATION_WHY = ("DDO has no entry for this wordlist spelling; the only "
                    "article it answers with is the abbreviation written with "
                    "its period (owner policy B, 2026-08-26)")


def abbreviation_admission_rows(wordlist: list, classification: dict,
                                entries: dict, demoted_pos: set) -> list:
    """The layer-4 audit table, derived from the STATE rather than from the run.

    review/abbreviation_accepted.json is the sheet a human signs for the 19
    edges owner policy B admits against the classifier's own verdict, and the
    admission baseline gate (G-ADMIT) counts its rows. Built by appending while
    layer 4 ran, it emptied itself on a retry: a second standalone `resolve`
    reads back the classification.json this stage wrote, sees the members layer 4
    already attached, returns at layer 1 -- and wrote `[]` over all 19 rows with
    no gate looking (reviewer B, round 4, MINOR-4). override_accepted.json has
    the same shape and the same defect; this is the fix for the table round 4
    added.

    Deriving it also makes the file a function of what is on disk, so it is
    byte-identical across reruns, and it cannot disagree with the members it
    describes. Iterates the wordlist so the row order is rank order.
    """
    rows = []
    for w in wordlist:
        c = classification.get(w["word"]) or {}
        for m in (c.get("members") or ()):
            if m.get("evidence") != "abbreviation":
                continue
            e = entries[m["entry_id"]]
            rows.append({"word": w["word"], "entry_id": m["entry_id"],
                         "abbreviation_lemma": e["lemma"],
                         "pos_key": e.get("pos_key"),
                         "demoted_pos": e.get("pos_key") in demoted_pos,
                         "senses": len(e.get("senses") or ()),
                         "why": ABBREVIATION_WHY})
    return rows


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

    def indexable(eid: str, curated: bool = False) -> bool:
        """What layer 2 and layer 3 are allowed to hand a word.

        A demoted or affix article is never recovered: the classifier rejects it
        on the forward page for a reason, and re-admitting it here (with a
        fabricated demoted flag, as this stage used to do) is how the symbol
        article gets a chance to head a family. Those two filters apply to the
        curated layer too.

        `rejected_everywhere` does NOT (curated=True). It is an automatic-layer
        heuristic -- "no word that saw this article wanted it" -- and on the
        override path the word that did not want it is usually the very word the
        mapping is trying to rescue: `indstil` rejected `indstille` as unrelated,
        which then disqualified `indstille` from being handed to `indstil`. Four
        articles (planlaegge, oversaette, fascinere, indstille -- 10 senses, four
        retired v2.1 cards) were unreachable this way, and invisibly: the
        candidate never reaches judged(), so it leaves no `recovery_` audit row.
        """
        e = entries[eid]
        if not curated and eid in rejected_everywhere:
            return False
        return (e.get("pos_key") not in demoted_pos
                and not is_affix_entry(e))

    rev = defaultdict(set)
    for eid in entries:
        if not indexable(eid):
            continue
        for f in entries[eid]["form_index"]:
            rev[f].add(eid)

    # Every article parsed from the word's OWN page, which is the only place
    # layer 4 looks. Same construction stage 22 uses for its candidate lists: a
    # dotted abbreviation is adopted because it is what DDO answered for THAT
    # query, never because some article elsewhere in the corpus happens to be
    # spelled that way.
    own_page: dict[str, set] = defaultdict(set)
    for eid, e in entries.items():
        for w in e.get("source_words", []):
            own_page[w].add(eid)

    recovery_rejects: list = []
    override_accepted: list = []

    def judged(word: str, c: dict, eids, evidence: str) -> list:
        """classify_one() on every candidate, exactly as the forward page does.

        Returns [(entry_id, bucket, why)] for the survivors; a refusal is written
        into the word's own `rejected` list with a `recovery_` reason so the
        recovery door leaves the same audit trail the front door does.

        One exception, module docstring lock 1: on the CURATED path a bare
        `unrelated` is the classifier declining to have an opinion, so the human
        mapping wins and the pair enters as bucket `form` (hidden searchable
        form, never card-face Variants). Every other rejection still stands.
        """
        keep = []
        for eid in eids:
            bucket, why = classify_one(word, entries[eid], demoted_pos, alias_pairs)
            if bucket == "reject" and evidence == "override" and why == "unrelated":
                bucket, why = "form", "curated_override"
                override_accepted.append({
                    "word": word, "entry_id": eid,
                    "override_lemma": entries[eid]["lemma"],
                    "pos_key": entries[eid].get("pos_key"),
                    "senses": len(entries[eid].get("senses") or ()),
                    "in_paradigm_index": nk(word) in set(
                        entries[eid].get("paradigm_index") or ()),
                    "why": "registry/form_to_lemma.json says so; the classifier "
                           "had no automatic evidence either way"})
            if (bucket == "reject" and evidence == "abbreviation"
                    and why == "abbreviation"):
                # Owner policy B. The classifier's `abbreviation` rejection is a
                # POSITIVE claim -- "this headword is the query plus a period" --
                # so unlike the override channel there is no guessing here: the
                # reason code IS the evidence. Bucket `abbreviation`, which
                # s30._relation() renders as relation "abbreviation": not an
                # inflection (the dotless form is not a paradigm cell of the
                # dotted headword) and not a variant spelling (s70's Variants
                # line would then print `hr` on a card already headlined `hr.`,
                # which is noise, not information). It stays a hidden searchable
                # form, which is what makes Anki find the card by typing `hr`.
                # No audit row is appended here on purpose: it is DERIVED from
                # the resulting members afterwards, so a rerun that returns at
                # layer 1 cannot blank the table -- abbreviation_admission_rows().
                bucket, why = "abbreviation", "dotted_abbreviation_entry"
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
    invalid_rows_that_bound: list = []
    resolved = {"forward": 0, "reverse_index": 0, "override": 0,
                "abbreviation": 0, "wordlist_invalid": 0, "known_no_entry": 0}
    for w in wordlist:
        word = w["word"]
        c = classification.setdefault(
            word, {"members": [], "xrefs": [], "rejected": [], "resolved_by": None})
        # Layer 0: the row is not a word. OCR damage in the frozen wordlist
        # (registry/wordlist_invalid.json, owner decision 10.5). Checked BEFORE
        # layer 1 so an invalid row can never bind by any route; every one of the
        # 53 rows is a DDO nohit today, so this subtracts nothing, and a row that
        # DID have members is a curation bug G-SUPPRESS fails on rather than a
        # card this registry deletes quietly.
        #
        # THE EVIDENCE HAS TO OUTLIVE THE CLEARING. classification.json is this
        # stage's INPUT and also one of its outputs, and it is rewritten below --
        # before the gates run. Clearing `members` without recording what was
        # cleared therefore destroyed the only trace of the condition: a second
        # standalone `resolve` read the already-emptied row, found nothing to
        # report, and G-SUPPRESS PASSED on the retry with the binding still
        # deleted (reviewer A, round 4, MAJOR-1, reproduced end to end). The
        # suppressed entry_ids are written into the row instead, so the gate's
        # third assertion is re-derivable from the artifact on every later run.
        # Write-once: an id recorded here is never dropped, because the run that
        # can still see it is the first one.
        if word in registry.wordlist_invalid:
            suppressed = sorted(
                {m["entry_id"] for m in c["members"]}
                | set(c.get("suppressed_by_wordlist_invalid") or ()))
            if suppressed:
                c["suppressed_by_wordlist_invalid"] = suppressed
                invalid_rows_that_bound.append(
                    {"word": word, "rank": w["rank"], "entry_ids": suppressed})
            c["members"] = []
            c["resolved_by"] = "wordlist_invalid"
            resolved["wordlist_invalid"] += 1
            continue
        if c["members"]:
            resolved["forward"] += 1
            continue
        # Layer 2: the curated override, AHEAD of the automatic index.
        override_lemma = registry.form_to_lemma.get(word)
        override_reason = None
        if override_lemma:
            key = nk(NFC(override_lemma))
            of_lemma = [e for e, v in entries.items() if v["lemma_key"] == key]
            usable = judged(word, c,
                            sorted(e for e in of_lemma
                                   if indexable(e, curated=True)),
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
        # Layer 4: DDO's dotted abbreviation entry, from the word's OWN page.
        # LAST of the rescue layers on purpose -- see the module docstring for
        # the 20 + 1 words the order protects. The affix filter still applies;
        # the demoted filter deliberately does not, which is the whole content of
        # owner policy B, and is_dotted_abbreviation() is the guard that keeps it
        # from reaching an element-symbol article.
        abbr = judged(word, c,
                      sorted(eid for eid in own_page.get(word, ())
                             if is_dotted_abbreviation(word, entries[eid])
                             and not is_affix_entry(entries[eid])),
                      "abbreviation")
        if abbr:
            attach(c, abbr, "abbreviation")
            c["resolved_by"] = "abbreviation"
            resolved["abbreviation"] += 1
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
    # review/, not reports/: this is the audit face of the override channel, and
    # it belongs next to rejected.json for the same reason -- a human has to be
    # able to read what the human registry admitted. Accepting `unrelated` on the
    # curated path means the classifier is no longer a second opinion on these
    # edges, so the mappings that bound this way are the ones under review.
    write_json(cfg.review_dir / "override_accepted.json", override_accepted)
    # review/, same argument as override_accepted.json: layer 4 admits an article
    # the classifier rejects on the forward page, on the strength of an owner
    # policy rather than of automatic evidence, so the edges it admits are edges
    # a human signs for. 19 rows, one per abbreviation card. Derived from the
    # classification (not from this run's admissions) so a retry cannot empty it.
    abbreviation_accepted = abbreviation_admission_rows(
        wordlist, classification, entries, demoted_pos)
    write_json(cfg.review_dir / "abbreviation_accepted.json",
               abbreviation_accepted)
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
        if m.get("evidence") in ("reverse_index", "override", "abbreviation"))
    # An override that could not be used is a CURATION bug: reported even when
    # the reverse index rescued the word, because otherwise it is invisible.
    resolved["override_problems"] = len(override_problems)
    resolved["override_problems_sample"] = override_problems[:20]
    resolved["override_accepted"] = len(override_accepted)
    resolved["override_mappings"] = len(registry.form_to_lemma)
    resolved["abbreviation_accepted"] = len(abbreviation_accepted)
    resolved["abbreviation_words"] = sorted(
        {r["word"] for r in abbreviation_accepted})
    resolved["known_no_entry_registry_rows"] = len(registry.known_no_entry)
    resolved["wordlist_invalid_registry_rows"] = len(registry.wordlist_invalid)
    resolved["invalid_rows_that_bound"] = len(invalid_rows_that_bound)
    write_json(cfg.report_dir / "resolve_report.json", resolved)

    # G-OVERRIDE runs AFTER the writes, which is the opposite of stages 20/22/30
    # -- deliberately. Nothing this stage writes is immutable or human-ratified
    # (stage 30's card_keys.json is, which is why its gates run first), and the
    # three files above ARE the evidence for the bug this gate fires on. Failing
    # before writing them would leave the curator with a count and no rows.
    run_gates([
        Gate(G_OVERRIDE, "every registry/form_to_lemma.json mapping actually "
                         "bound a word to its lemma's article",
             lambda: curated_overrides_bind(
                 override_problems, len(override_accepted),
                 int(registry.gates.get("override_problems_max", 0))),
             stage="21"),
        Gate(G_SUPPRESS, "the two registries that remove words from "
                         "unresolved.json are inside their baselines, disjoint, "
                         "and delete no binding",
             lambda: suppression_registries(
                 registry.known_no_entry, registry.wordlist_invalid,
                 invalid_rows_that_bound,
                 int(registry.gates.get("known_no_entry_max", 0)),
                 int(registry.gates.get("wordlist_invalid_max", 0))),
             stage="21"),
        # Fail-closed the same way G-SUPPRESS is: a MISSING baseline key is an
        # empty word list, so every admission is "beyond the baseline" and the
        # gate fires. A channel that mints cards from rejected articles must not
        # be able to go unbaselined by deleting a registry line.
        Gate(G_ADMIT, "the layer-4 abbreviation admissions are the words the "
                      "owner signed for, and no more of them",
             lambda: abbreviation_admissions(
                 abbreviation_accepted,
                 registry.gates.get("abbreviation_accepted_words") or (),
                 int(registry.gates.get("abbreviation_accepted_max", 0))),
             stage="21"),
    ], cfg, stage="21")
    return resolved


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
