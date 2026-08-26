"""Stage 22: the four-bucket classifier.

Buckets, in headline order: exact_cs (NFC codepoint-equal) > form (in the
article's own flex table) > variant (orthographic variant / alias) > exact_ci
(equal only after casefold -- where Er(erbium)/er lives) > abbreviation (stage
21's last-resort layer only, never returned here). Decisive bucket-2
signal is flex-table containment; tab labels are refuted (kan->khan). Exactness
is EXCLUSIVE: a word that is its own dictionary headword is never absorbed as a
form of another lemma (godt has 15 senses of its own AND sits in god's flex
table). Affix/junk detection is by headword shape and data-pos-key, never by
sitemap shard.

DECISION ORDER (this is the part the guide's own pseudocode got wrong):

    affix reject          headword shape / data-pos-key
    abbreviation reject   min. (min), i.e. nk(hw).rstrip('.') == nk(q)
    exact_cs              NFC(hw) == q
    case-only             nk(hw) == nk(q)  ->  reject if demoted, else exact_ci
    form                  nk(q) in PARADIGM CELLS   (not the lemma key, not
                                                     alt spellings)
    variant               official alt spellings + alias pairs + squash equality
    multiword reject      'en bloc', 'alle sammen'  -- AFTER variant
    unrelated reject

Three things that order fixes, all measured:

  * `form` tested `form_index`, which always contains nk(lemma), so EVERY
    case-only pair matched bucket 2 and `exact_ci` / `case_only_demoted_pos`
    were unreachable for every possible input. `I`(pron., a real word with 2
    senses and 7 expressions) was classified as an inflection of itself and
    then thrown into xrefs; the erbium-class symbol article was kept as a
    visible cross-reference instead of being rejected.
  * `khan` carries the DEPRECATED spelling `kan`, which used to enter
    form_index; that is the sole reason the kan/khan component existed at all.
  * the multiword reject fired before the variant test, so /udenfor lost
    `uden for` (praep., 4 senses) -- and the same shape for indenfor, overfor,
    ovenpaa, bagefter, bagved, nedenunder, indeni, udenom, udover. `en bloc`
    and `alle sammen` still reject, because their squash forms differ from the
    query.
"""

from ..config import Config
from ..gates import G_AFFIX, Gate, no_affix_members, run_gates
from ..util import NFC, nk, read_json, write_json

# `abbreviation` is a stage-21 LAST-RESORT bucket, never returned by
# classify_one(); it is registered here because it is the weakest binding in the
# pipeline and stage 30 reads this table for both best_member() and the anchor
# rule. Ranking it after exact_ci is what guarantees an abbreviation article can
# never outrank a real one for the card headline. See s21_resolve layer 4.
BUCKET_ORDER = {"exact_cs": 0, "form": 1, "variant": 2, "exact_ci": 3,
                "abbreviation": 4}


def squash(s: str) -> str:
    """Orthographic identity: casefold, then drop spaces and hyphens.

    This is the relation that makes `uden for` a variant of `udenfor` and `I` the
    same headword as `i`. It is the FIRST of is_variant()'s three branches, not
    all of it -- stage 30 groups component heads with is_variant() so that the
    classifier cannot admit a member the merge then refuses. This docstring used
    to claim that property for squash() alone, and it was false for the alias and
    official-alt-spelling branches: three admitted pairs were split back apart.
    """
    return nk(s).replace(" ", "").replace("-", "")


_squash = squash    # the internal name this module used before stage 30 shared it


def _official_forms(e: dict) -> set:
    """Deprecated spellings (official: false) are searchable data, never
    classification evidence -- see the khan/kan note above."""
    return {nk(a["form"]) for a in (e.get("alt_spellings") or [])
            if a.get("official")}


def is_variant(word: str, e: dict, alias_pairs: set) -> bool:
    """The orthographic-variant relation, in three branches: squash equality,
    DDO's OFFICIAL alternative spellings, and the curated alias registry.

    Public, because stage 30 has to group component heads by exactly this
    relation. squash() alone is only the FIRST branch, and using it as the whole
    test is what made the merge refuse components the classifier had already
    admitted: `check`+`tjek`, `naeh`+`nae`, `ok`+`o.k.` squash apart, so each of
    the three registered alias pairs was split back into two heads and one side's
    article shipped on no card at all.
    """
    if nk(word) == nk(e["lemma"]):
        # A case-only pair is bucket 1's business. _squash() casefolds, so
        # without this guard every ('er','Er') pair is "already a variant" and
        # the case-only branch stays unreachable even after form_index is fixed.
        return False
    if _squash(word) == _squash(e["lemma"]):
        return True
    if nk(word) in _official_forms(e):
        return True
    pair = (nk(word), nk(e["lemma"]))
    return pair in alias_pairs or (pair[1], pair[0]) in alias_pairs


def is_dotted_abbreviation(word: str, e: dict) -> bool:
    """DDO's own abbreviation entry for `word`: the headword is the query plus
    one or more trailing periods and nothing else (`hr.` for `hr`, `dr.` for
    `Dr`, `st.` for `st`).

    Compared through nk(), which is the round-2 case-sensitivity fix (reviewer B
    N-1): DDO spells the doctor abbreviation `dr.` in lower case, so the raw-NFC
    test called `Dr`(rank 358) and `Mrs`(598) `unrelated` -- the classifier's
    "I have no opinion" fallback -- while claiming to have a rule for exactly
    that shape. Two rejection REASON codes change; no edge changes verdict, and
    the relation is strictly weaker than the raw test (a match under it implies
    nk(hw) == nk(q) + dots, so it can never steal an edge from exact_cs,
    case-only, form or variant).

    It also cannot admit an element-symbol article, which is the hazard that put
    `symbol` and `fork.` in demoted_pos_keys: `Th`(thorium), `No`(nobelium),
    `Ca`, `Kr` and `mA` are CASE-ONLY matches for th/no/ca/kr/ma and carry no
    period, so they still reject as case_only_demoted_pos.

    THIS IS A SHAPE TEST, NOT A POS TEST, and the difference has a live
    counterexample: `11040665 'pr.'` is `praep.` with 6 senses -- a full
    preposition article, not an abbreviation -- and it passes this relation,
    because the relation only asks whether the headword is the query plus
    periods. "A period inside a DDO headword IS the abbreviation mark" is
    therefore a claim about this corpus, not a property of this function. It is
    unreachable today: `pr` binds `PR` by exact_ci at layer 1, so stage 21's
    layer 4 never runs for it (reviewer A, round 4, minor). Anything that lets a
    word past layer 1 to layer 4 has to re-check that assumption.

    NOT o.k./ok: nk("o.k.").rstrip(".") is "o.k", which is why that pair lives
    in the alias registry instead.

    Public because stage 21's layer 4 (owner policy B, 2026-08-26) adopts
    exactly this relation, and the two must not be able to drift apart.
    """
    a, b = nk(e["lemma"]), nk(word)
    return a.rstrip(".") == b and a != b


def classify_one(word: str, e: dict, demoted_pos: set, alias_pairs: set):
    q, hw = NFC(word), e["lemma"]
    if hw.startswith("-") or hw.endswith("-"):
        return "reject", "affix"                      # -kvinde (kvinder), -ske (sker)
    if is_dotted_abbreviation(q, e):
        # min. (min), nr. (nr), dr. (Dr). The FORWARD page still rejects: an
        # abbreviation entry is a different dictionary word from the word that
        # happens to spell it without the period, and 20 WORDLIST ROWS that
        # already own a card (min, med, to, ti, tv, par, port, red, art, da, den,
        # do, eks, el, fa, man, pr, soe, soen, aarh) reach a dotted article this
        # way -- plus `net` and `sen`, which reach one but are off-list
        # source_words rather than wordlist rows. Owner policy B adopts the 19
        # words for which the dotted entry is the ONLY thing DDO offers, and
        # stage 21 does that as a last-resort layer so exclusive exactness is
        # untouched by construction.
        return "reject", "abbreviation"
    demoted = e.get("pos_key") in demoted_pos
    if NFC(hw) == q:
        return "exact_cs", ("demoted" if demoted else "exact")
    if nk(hw) == nk(q):
        # Case-only match: Er(erbium)/er, VAR/var, Se(selenium)/se, I/i.
        if demoted:
            return "reject", "case_only_demoted_pos"
        return "exact_ci", "case_only"
    if nk(q) in set(e.get("paradigm_index") or ()):
        return "form", "flex_table"
    if is_variant(q, e, alias_pairs):
        return "variant", "orthographic_variant"
    if " " in hw and " " not in q:
        return "reject", "multiword_neighbour"        # en bloc, alle sammen
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

    # G-AFFIX runs BEFORE the outputs are written: an affix member is not a
    # result to be reviewed afterwards. It used to be a bare AssertionError
    # after classification.json had already landed on disk, invisible to
    # gates_report.json.
    run_gates([
        Gate(G_AFFIX, "no accepted classification member is an affix-page "
                      "article (-kvinde, -ske, for-)",
             lambda: no_affix_members(classification, entries), stage="22"),
    ], cfg, stage="22")

    write_json(cfg.json_dir / "classification.json", classification)
    # review/, not reports/: these are the two files a human is REQUIRED to read
    # before a release (guide 4.8, plan section 4 step 12, and G-REVIEW's
    # definition names review/rejected.json by path).
    write_json(cfg.review_dir / "rejected.json", rejected_report)
    write_json(cfg.review_dir / "bucket2_accepted.json", bucket2_report)
    return {"words": len(classification),
            "with_members": sum(1 for c in classification.values() if c["members"]),
            "rejected_edges": len(rejected_report)}
