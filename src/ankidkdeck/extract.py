"""The text-extraction contract.

The 22,734 x 4 languages of existing Gemini translations are keyed by the exact
Danish strings the 2025 parser produced, and that parser mixed get_text()
separators per field. A one-character change in this table silently invalidates
the translation asset (that is how 2,007 English cards shipped untranslated).
Every row below is measured: the right separator reproduces the 2025 strings
100%, the wrong one provably does not (see danish_pipelines vault,
research/2026-08-v3-review/v3_final_guide.md section 1.1).
"""

from .util import collapse_ws

# The version of the PARSED-ARTICLE HASH contract (s20_parse.content_sha), not of
# this table. Bump it whenever the set of fields that feed article_sha changes,
# so the drift ledger can say "the parser schema changed" instead of reporting
# every article in the corpus as edited by DDO. Schema 1 hashed the whole entry
# dict, derived indexes and provenance included, so any parser refactor reset the
# whole ledger; schema 2 hashes content only; schema 3 expands DDO's own ".."
# elision inside paradigm.rows[].cells, which content_sha() hashes.
ARTICLE_SHA_SCHEMA = 3

# DDO writes a compound's inflected cell with its own shared prefix elided:
# julemand's plural cell is the literal string "..maend", oversaette's past tense
# is "..satte". Not a parser loss -- the raw HTML holds "<span
# class=mark-flex>..maend</span>" and the page shows those two dots (measured:
# 19 articles, 31 cells; the literal ".." appears 12 times on /julemand and the
# Unicode ellipsis 0 times).
ELISION = ".."

# field -> get_text separator. Source of truth: 02_generate_entries.py line refs.
SEP = {
    # space-joined (definition 388/388 space vs 330 nosep; etymology 49/49 vs 0)
    "definition": " ",       # 02:166
    "grammar": " ",          # 02:168
    "example": " ",          # 02:182
    "etymology_raw": " ",    # 02:125
    "expr_definition": " ",  # 02:211  (the row every reviewer but one missed)
    "expr_example": " ",     # 02:216
    "ipa": " ",              # 02:094
    "udtale_label": " ",     # 02:105
    # no-separator (expression 689/689 nosep vs 499 space; pos 59/59 vs 39)
    "expression": "",        # 02:198
    "pos_text": "",          # 02:247
    "headword": "",          # 02:243
    "sense_number": "",      # 02:158
    "example_source": "",    # 02:183
    "etymology_form": "",    # 02:134
    "etymology_desc": "",    # 02:140
    "onym_link": "",         # 02:051
    "orddannelse": "",       # 02:068
    "wordform_cell": "",     # <td>hus<span class=mark-flex>et</span></td> -> "huset"
}


def xt(node, field: str) -> str:
    """The ONLY text extractor allowed on DDO markup."""
    return node.get_text(SEP[field], strip=True)


def is_tag(node) -> bool:
    return getattr(node, "name", None) is not None


def expand_elision(cell: str, lemma: str) -> str:
    """Restore the prefix DDO elided with "..": "..maend" + julemand -> julemaend.

    The rule is DDO's own: the cell resumes the headword at the LAST occurrence
    of the elided form's first character, so the shared prefix is everything
    before it. Verified on all 31 elided cells in the corpus -- julemand/..maend
    -> julemaend, oversaette/..satte -> oversatte, efterlade/..lod -> efterlod,
    bedstefar/..faedre -> bedstefaedre, gennemgaa/..gik -> gennemgik, and the
    hyphenated compound "goer det selv-bog"/..boeger -> "goer det selv-boeger".

    Two reasons this must happen in the parser and not downstream. The cell is
    what the export renders (s70.paradigm_block_html reads paradigm.rows), so an
    unexpanded cell prints two dots on the card; and it is what feeds
    paradigm_index, so the real form (oversat, julemaend) can never be matched.

    A cell whose first character is not in the lemma is left EXACTLY as it is:
    guessing is worse than a visible ".." a human can report.
    """
    if not cell.startswith(ELISION):
        return cell
    body = cell[len(ELISION):].lstrip()
    if not body:
        return cell
    cut = lemma.rfind(body[0])
    if cut <= 0:
        return cell
    return lemma[:cut] + body


def cell_alternatives(td) -> list[str]:
    """Split one flex-table <td> into its alternative spellings of ONE slot.

    Alternatives are separated by <span class="diskret">eller ...</span>; a
    naive get_text("") glues them ('engroshandelenellerengroshandlen'). The
    separator prose itself ('eller', 'eller uofficiel form:') is discarded, and
    parts that are deprecated-spelling prose are dropped.
    """
    parts, cur = [], ""
    for ch in td.children:
        if is_tag(ch) and "diskret" in (ch.get("class") or []):
            parts.append(cur)
            cur = ""
        else:
            cur += xt(ch, "wordform_cell") if is_tag(ch) else str(ch)
    parts.append(cur)
    out = []
    for p in parts:
        p = p.strip().strip("()").strip()
        if not p or "uofficiel" in p or "form:" in p:
            continue
        out.append(collapse_ws(p))
    return out or [collapse_ws(xt(td, "wordform_cell"))]
