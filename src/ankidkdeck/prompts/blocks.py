"""The append-only RICH blocks. Frozen design: specs/v3-prompt-comparison.md 2b.

Every block is appended AFTER the frozen V4 core, in the fixed numeric order
below, and every block is independently droppable. Two properties follow, and
they are the whole reason for this shape:

  * `LEAN` (no blocks) is a pure BYTE PREFIX of `RICH` (all blocks). Rolling
    back a prompt is "assemble fewer blocks", never "maintain a second text",
    so the two variants cannot drift.
  * A block whose pack slot is missing is skipped, not rendered with an empty
    hole. That is what lets a language with no pack file run at all.

Danish metalanguage (`fx`, `el.lign.`, `isaer`, `det at`) appears here as
product text: these are DDO's own editorial formulas, they are what the model
has to read, and the frozen V4 core already carries Danish in its structure
example. No DDO definition text is stored here.

Sizes below are the measured character counts of the rendered skeleton, not
estimates; the option-A trims of patch plan 4.2 are already applied:
block 2 has no word-formation line (0 cells in the corpus carry those POS
keys), block 4 folds the parenthesis and compound rules into the semicolon
rule, and block 7 states its three shape rules as two sentences.
"""

def _lines(value) -> str:
    """A pack slot that is a list of lines, rendered one per line."""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def _json_block(value) -> str:
    import json                                # noqa: PLC0415 - local by intent
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=4)


def block_1_script(lang: str, pack: dict) -> str | None:
    """SCRIPT AND ORTHOGRAPHY CONTRACT.

    This block repairs a defect that is currently SHIPPING, which is why it is
    first and why the A/B ramp puts it in unconditionally: 325 Chinese
    definition cells (2.41%) and 22 expression cells carry Traditional
    characters, mixed into Simplified text at a median of 26% of the Han
    characters in the cell, and the failure is per-request -- 28 entries have
    every translated sense contaminated. 20 more Chinese expression lemmas
    carry pinyin. The 2025 prompts never said "Simplified" and never said
    "no phonetic transcription".

    `allowed_scripts` and `lemma_charset_rule` are read from the same pack
    fields the G-SCRIPT gate reads, so the instruction and the check cannot
    disagree.
    """
    if not pack.get("allowed_scripts"):
        return None
    out = ["### SCRIPT AND ORTHOGRAPHY CONTRACT",
           "Every character you emit in `lemma` and in `gloss` must belong to "
           "the allowed set", "for %s: %s." % (lang, pack["allowed_scripts"])]
    if pack.get("orthography_rules"):
        out.append(pack["orthography_rules"])
    out.append("Never emit Cyrillic, Greek, Arabic, Hebrew, Hangul, Hiragana, "
               "Katakana,")
    out.append("Devanagari or Thai characters, in either field, for any reason.")
    out.append("The one exception is an entry whose own subject IS such a "
               "character. Then it")
    out.append("may appear in the `gloss`, and in the `lemma` only inside a "
               "phrase that names")
    out.append("the letter -- never as the whole `lemma`, and never inside a "
               "word.")
    if pack.get("lemma_charset_rule"):
        out.append("`lemma`: %s" % pack["lemma_charset_rule"])
    if pack.get("punctuation_rule"):
        out.append("Punctuation: %s" % pack["punctuation_rule"])
    out.append("Never leave `lemma` or `gloss` empty. Never echo the Danish "
               "definition.")
    return "\n".join(out)


def block_2_pos(lang: str, pack: dict) -> str | None:
    """PART-OF-SPEECH NORMS. Twenty DDO pos_key values collapsed into groups.

    The keys below are the SOURCE DICTIONARY'S OWN pos_key strings, spelled the
    way the user message spells them. That is a correction, not a detail: this
    block used to enumerate ASCII-folded keys (`praep.`, `udraabsord`,
    `maengdetal`) while the payload's headword line carried `pos_text` -- 38
    long Danish forms such as "substantiv, faelleskoen" -- so the plural rule,
    whose whole criterion is the literal string `sb. pl.`, referred to a string
    that never appeared in any request. The payload now states the key
    (definition_user_payload's `Part of speech` line, 20 distinct values), and
    these lines are spelled to match it exactly.

    The word-formation group (foersteled / sidsteled / praefiks / suffiks) is
    deliberately absent: those four keys occur in zero cells of the shipped
    corpus, so a line about them would be paid for on every one of the 3,623
    definition requests and never be read.
    """
    out = ["### PART-OF-SPEECH NORMS",
           "The user message states the headword's part of speech on its `Part "
           "of speech` line,",
           "as the source dictionary's own key. Match the `lemma` to it.",
           "- vb.: bare infinitive, no infinitive particle (no Danish `at`, no "
           "English `to`,", "  no German `zu`)."]
    if pack.get("pos_vb"):
        out[-1] = out[-1] + " " + pack["pos_vb"]
    out.append("- sb. / sb. pl.: singular indefinite unless the key is `sb. "
               "pl.`, in which case")
    out.append("  use the target plural." +
               (" " + pack["pos_sb"] if pack.get("pos_sb") else ""))
    out.append("- adj. / adj. pl.: positive degree, base form. Do not emit a "
               "comparative.")
    out.append("- adv. / præp. / konj. / pron. / artikel / formelt subjekt / "
               "infinitivpartikel:")
    out.append("  a function word rarely has a substitutable equivalent. Give "
               "a content-word or")
    out.append("  short-phrase `lemma` that names the FUNCTION, and let the "
               "`gloss` state how the")
    out.append("  word is used." +
               (" " + pack["pos_function"] if pack.get("pos_function") else ""))
    out.append("- udråbsord / lydord: give the natural interjection or sound "
               "word itself, not a")
    out.append("  description of it.")
    out.append("- talord (mængdetal): a cardinal numeral word. talord "
               "(ordenstal): an ordinal.")
    out.append("- fork. / symbol / egennavn: expand an abbreviation into its "
               "meaning, never")
    out.append("  transliterate it; keep a proper name in its established "
               "target-language form.")
    out.append("Never change part of speech between the Danish headword and "
               "your `lemma`.")
    return "\n".join(out)


def block_3_example(lang: str, pack: dict) -> str | None:
    """WORKED EXAMPLE. The one block that buys quality AND style continuity.

    The frozen V4 core shows the model a JSON SHAPE whose every lemma and gloss
    is a placeholder: after 22,734 cells the model has never been shown one
    good answer. The Danish input side here is hand-authored DDO-style prose
    written for this repository; the target side is written to the register the
    shipped corpus actually uses (see the pack's measured basis).

    The example is chosen for a literal trap: `traek` has three senses on
    "action", "natural phenomenon" and "abstract property", and the literal
    meaning of the Danish headword ("pull") is right for only the first. That
    is exactly what V4 lemma rule 3e warns about and never demonstrates.
    """
    if not pack.get("worked_example_output"):
        return None
    return "\n".join([
        "### WORKED EXAMPLE (hand-authored; the target side shows the required "
        "register)",
        'Input Headword: "træk sb."',
        "Input Definitions: {",
        '    "0": "det at flytte noget hen imod sig selv ved at bruge kraft",',
        '    "1": "strøm af kold luft der bevæger sig gennem et rum, fx fra en '
        'åben dør el.lign.",',
        '    "2": "karakteristisk egenskab ved en person eller ting; ofte om '
        'noget der er let at genkende"',
        "}",
        "Correct output:",
        _json_block(pack["worked_example_output"]),
        "Why it is correct:",
        "- Sense 2 is NOT the literal meaning of the Danish headword. A `lemma` "
        "derived",
        '  from "pull" would be wrong for it. The definition decides, not the '
        "headword.",
        "- The three `lemma` values are distinct, and each `gloss` is "
        "independently",
        "  readable without the other two.",
        "- `fx` and `el.lign.` are rendered idiomatically, not transliterated.",
    ])


def block_4_reading(lang: str, pack: dict) -> str | None:
    """READING THE DANISH DEFINITION. Contents set by MEASURED frequency.

    All three audits asked for a register-marker block built around
    `i overfoert betydning`, `jf.`, `spoegende`, `nedsaettende` and
    `gammeldags`. Those five occur 14 times in 13,497 shipped Danish
    definitions, because DDO's register labels live in a field that
    `compute_todo` never sends. The
    formulas that DO have objects are `el.lign.` (11.1%), `fx` (10.3%),
    `ofte` (4.2%), `isaer` (3.4%) and the `person der` / `noget der` /
    `det at` frames, and this block is written around those instead.

    Item 8 is new relative to the frozen design and is here because the payload
    change landed: `sense["grammar"]` is non-empty on 38.8% of senses and had
    never reached the model at all.
    """
    return "\n".join([
        "### READING THE DANISH DEFINITION",
        "1. A definition usually opens with a hypernym and then narrows it. "
        "Your `lemma`",
        "   must name the NARROWED concept, not the hypernym.",
        "2. `fx` (10.3% of definitions) and `el.lign.` (11.1%) are "
        "abbreviations. Render",
        "   them idiomatically. Never transliterate them and never keep them "
        "in Latin",
        "   letters in a non-Latin target.",
        "3. `ofte` (4.2%) and `især` (3.4%) qualify the scope of the sense. "
        "Keep them in",
        "   the `gloss`; never drop them and never promote them into the "
        "`lemma`.",
        "4. `det at ...` marks a nominalised action. The `lemma` should be the "
        "action",
        "   noun, not the verb.",
        "5. `person der ...` / `noget der ...` mark an agent or a thing defined "
        "by what it",
        "   does. The `lemma` names the agent or thing.",
        "6. A semicolon (15.1% of definitions), a parenthesis (12.5%) and a "
        "compound or",
        "   particle verb all narrow ONE sense. None of them is a second "
        "sense: produce",
        "   one object, keep the narrowing in the `gloss`, and translate the "
        "sense given",
        "   rather than the parts.",
        "7. Grammar notes in the user message are the source dictionary's own "
        "labels for",
        "   that sense: `NOGEN`/`NOGET` marks the valency frame, and the rest "
        "mark number",
        "   or register. Let them shape the `lemma`; never translate the note "
        "itself.",
    ])


def block_5_renderings(lang: str, pack: dict) -> str | None:
    """FIXED RENDERINGS. The modal rendering this dictionary already uses.

    Mined by lift: for each Danish formula, the target n-grams whose rate
    inside the cells carrying that formula most exceeds their corpus-wide rate,
    kept when coverage is at least 5%. This is the cheapest block per token
    because it is DESCRIPTIVE -- it pins a habit the corpus already has, so it
    cannot make new cells stylistically foreign.
    """
    if not pack.get("fixed_renderings_table"):
        return None
    return "\n".join([
        "### FIXED RENDERINGS",
        "Render these recurring Danish formulas consistently. The right-hand "
        "form is the",
        "one this dictionary already uses; do not invent a synonym.",
        _lines(pack["fixed_renderings_table"]),
    ])


def block_6_anti(lang: str, pack: dict) -> str | None:
    """ANTI-PATTERNS. bad -> why -> good, on real failure modes only."""
    if not pack.get("anti_patterns"):
        return None
    return "\n".join([
        "### ANTI-PATTERNS",
        "Each line is a real failure mode. bad -> why -> good.",
        _lines(pack["anti_patterns"]),
    ])


def block_7_length(lang: str, pack: dict) -> str | None:
    """LENGTH AND SHAPE TARGETS.

    The last two sentences replace the "disambiguating parenthetical" rule the
    final audit rated highest. Measured: a parenthetical in the lemma is a
    0.27% (Chinese) to 3.43% (German) habit and its rate does NOT rise when a
    lemma is reused, so writing it as a rule would push new cells an order of
    magnitude away from the old ones. What the corpus actually does is rewrite
    the gloss: two senses of one headword never share a gloss, 0 of 111 in
    Chinese and 0.0% in all four languages.
    """
    if not pack.get("length_targets"):
        return None
    return "\n".join([
        "### LENGTH AND SHAPE TARGETS",
        "These are measured norms of this dictionary, not suggestions.",
        _lines(pack["length_targets"]),
        "Exactly one sentence per `gloss`, readable on its own without the "
        "other senses",
        "of the same headword, and never identical to another sense's `gloss`.",
    ])


def block_8_precedence(lang: str, pack: dict) -> str | None:
    """RULE PRECEDENCE. Negative thinking risk: it ends repeated re-weighing."""
    return "\n".join([
        "### RULE PRECEDENCE",
        "When these instructions conflict, resolve in this order:",
        "1. The script and orthography contract. It is absolute and has no "
        "exceptions.",
        "2. The semantics of THIS Danish definition.",
        "3. Naturalness in %s." % lang,
        "4. Brevity.",
        "5. The literal meaning of the Danish headword. This is the weakest "
        "signal.",
        "The number of objects to return comes from the user message, never "
        "from these",
        "instructions.",
    ])


# The fixed assembly order. The index is the block number in the frozen design,
# and the ORDER OF THIS TUPLE is the order of the text: any variant built from a
# prefix of this tuple is a byte prefix of the full RICH prompt.
DEFINITION_BLOCKS = (
    ("script", block_1_script),
    ("pos", block_2_pos),
    ("example", block_3_example),
    ("reading", block_4_reading),
    ("renderings", block_5_renderings),
    ("anti_patterns", block_6_anti),
    ("length", block_7_length),
    ("precedence", block_8_precedence),
)

BLOCK_NAMES = tuple(name for name, _ in DEFINITION_BLOCKS)


# --------------------------------------------------------------------------
# expression-prompt blocks (patch plan 4.3, frozen design section 3b)
# --------------------------------------------------------------------------

def expr_block_script(lang: str, pack: dict) -> str | None:
    """The generator half of the fix for the English-permission contradiction.

    In the 2025 pipeline the generator said "Never use English in the `lemma`"
    while the reviewer was told that English was allowed and asked to inspect
    "the lemma part" -- so an English lemma on a Chinese card was reported as
    clean. The two prompts now read the SAME pack fields, `lemma_allowed_set`
    and `gloss_allowed_set`, which is why the contradiction cannot come back by
    someone rewording one of two prose paragraphs.
    """
    if lang.lower() == "english":
        # English is the recorded special case: there is no CRITICAL RULES
        # block to extend, and "no other script" is already what Latin means.
        return None
    sets = {"lemma": pack.get("lemma_allowed_set") or "",
            "gloss": pack.get("gloss_allowed_set") or ""}
    if not sets["lemma"]:
        return None
    out = ["### SCRIPT CONTRACT",
           "Every character you emit must belong to the allowed set for %s."
           % lang,
           "- `lemma`: %s -- no romanisation, no pinyin, no phonetic "
           "transcription" % sets["lemma"],
           "  of any kind.",
           "- `gloss`: %s." % sets["gloss"],
           "Never emit Cyrillic, Greek, Arabic, Hebrew, Hangul, Hiragana, "
           "Katakana,",
           "Devanagari or Thai. Never leave `lemma` or `gloss` empty."]
    return "\n".join(out)


def expr_block_example(lang: str, pack: dict) -> str | None:
    """WORKED EXAMPLE for the expression prompt.

    The expression prompt had no example at all, while idioms are the task that
    most needs one -- and on the measured volumes the expression wave is the
    larger half of the spend. The Danish side is hand-authored: the idioms
    themselves are common property and the hints are written for this file.
    """
    if not pack.get("expr_worked_example_output"):
        return None
    return "\n".join([
        "### WORKED EXAMPLE (hand-authored)",
        "Input:",
        "{",
        '    "0": {"expr": "have ben i næsen",',
        '          "hint": "være viljestærk og turde sætte sig igennem"},',
        '    "1": {"expr": "gå agurk",',
        '          "hint": "blive vildt begejstret eller helt ude af kontrol"}',
        "}",
        "Correct output:",
        _json_block(pack["expr_worked_example_output"]),
        "Why it is correct:",
        "- Item 0 has a real equivalent idiom in %s; when one exists, use it."
        % lang,
        "- Item 1 is rendered idiomatically where %s has an idiom and as a "
        "concise" % lang,
        "  descriptive phrase where it does not. Inventing a fake idiom is "
        "always worse",
        "  than a plain description.",
        "- Neither `gloss` translates the `hint`. The `hint` only "
        "disambiguates.",
    ])


EXPRESSION_BLOCKS = (
    ("script", expr_block_script),
    ("example", expr_block_example),
)
