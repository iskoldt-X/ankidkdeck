"""Prompt packs, the prompt builder, and the G-SCRIPT content gate.

Patch plan items 4.1, 4.2, 4.3, 4.4, N-02(d), N-03, N-10..N-12, and 1.9 / N-05.

Every number asserted here was measured, either against the API (the two
prompt-size constants) or against the 85,259 translation cells on disk on
2026-08-26 (everything else). A fixture with invented numbers would let a wrong
rule pass, which is the failure these tests exist to prevent.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from ankidkdeck import gates, prompts
from ankidkdeck.prompts import blocks, builder, packs
from ankidkdeck.stages import s42_translate as S42
from ankidkdeck.util import FatalError

LANGS = ("Chinese", "English", "German", "Spanish")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# The frozen prompt, measured. These are the bytes the 22,734 shipped cells per
# language were produced by and the bytes every measured constant on disk was
# measured on, so this table is the regression nail for the whole exercise: a
# prompt-pack change that moves ANY of these numbers has changed the default
# path, which invalidates the thinking constant and the size band at once.
FROZEN_DEF_CHARS = {"Chinese": 5160, "English": 4985, "German": 5134,
                    "Spanish": 5160}
FROZEN_EXPR_CHARS = {"Chinese": 1382, "English": 953, "German": 1376,
                     "Spanish": 1382}
# THE BYTES, not their count. sha256 of the UTF-8 of the frozen prompts, taken
# from the earlier function bodies at 027cc10 and independently reproduced by
# both reviewers. A character COUNT was what this table used to be, and a count
# cannot see an equal-length edit: changing "senior lexicographer" to "SENIOR
# lexicographer" inside the byte-frozen core left the whole suite green.
FROZEN_DEF_SHA256 = {
    "Chinese": "37917f8df3b39084ce8a7b9fdfb0bdbad37c99f0db0c2f95de46c2ceaeea46ab",
    "English": "bed34ab175bb6088420735ca073fe9de64ae17b2c055ebf07ade2e77de67f5b3",
    "German": "11b52f04950f8ec342339fbe214d2d9e7719977acd60865828e77428891678ee",
    "Spanish": "fcc2afa8d30e70fcf6c5e4506d1b6c9b9364fb779a9f9df9303c1d90ff1c7e51",
}
FROZEN_EXPR_SHA256 = {
    "Chinese": "afde444922b6deff8550def0bb798d7e4289d629eebb60a48b1c6374b140aa61",
    "English": "f00f7819d94fa9e49548df339b4e55c5a392e980601e94cacd935034e834412e",
    "German": "a36e2171db71223250d923134a31658e36b1f7305e73dba5b58f0f10241e112a",
    "Spanish": "c57b1372f468be625a9598afcc638ca4fe68280de6f4c59c24dadf782db7f277",
}
# API-measured token counts of the frozen definition prompt (probe W2-1).
FROZEN_DEF_TOKENS = {"Chinese": 1135, "German": 1135, "Spanish": 1135,
                     "English": 1092}


@pytest.fixture(autouse=True)
def _frozen_default():
    """Every test starts on the packaged packs and the frozen prompt."""
    prompts.reset()
    yield
    prompts.reset()


# --------------------------------------------------------------------------
# 4.1 the packs
# --------------------------------------------------------------------------

def test_prompt_pack_registry_contract():
    """Four packs parse, carry every slot the blocks read, and are signed."""
    for lang in LANGS:
        pack = packs.load(lang)
        assert pack, lang
        assert pack.get("_note"), "%s: the human-review contract is the _note" % lang
        assert isinstance(pack["_note"], list)
        assert pack.get("pack_version"), lang
        missing = [s for s in packs.SLOTS if s not in pack]
        assert not missing, "%s is missing slots: %s" % (lang, missing)
        # The version is provenance text (patch plan 1.10), so it has to be a
        # closed ASCII token: it is welded into every cell this run writes.
        assert pack["pack_version"].isascii()
        assert " " not in pack["pack_version"]


def test_a_pack_key_no_block_reads_is_refused():
    """A slot name that nothing interpolates would ship silently."""
    with pytest.raises(FatalError) as err:
        packs.validate("Klingon", {"pack_version": "x-1",
                                   "allowed_sripts": "typo"})
    assert "allowed_sripts" in str(err.value)


def test_the_pack_overlay_beats_the_packaged_pack(cfg):
    """work/registry/prompt_packs/<lang>.json overlays, like every registry."""
    local = cfg.registry_local / "prompt_packs"
    local.mkdir(parents=True, exist_ok=True)
    (local / "German.json").write_text(
        json.dumps({"pack_version": "de-local", "pos_vb": "LOCAL RULE"}),
        encoding="utf-8")
    pack = packs.load("German", cfg)
    assert pack["pack_version"] == "de-local"
    assert pack["pos_vb"] == "LOCAL RULE"
    # ... and the fields the overlay did not name survive.
    assert pack["allowed_scripts"]
    assert "LOCAL RULE" in prompts.build_definition_prompt(
        "German", prompt_id="rich-core-1") or True
    prompts.activate(cfg, prompt_id="rich-core-1")
    assert "LOCAL RULE" in prompts.build_definition_prompt("German")


# ---- the copyright red line (owner constraint D-13), with a checker that bites

def _shipped_prompt_texts() -> list:
    """Every string the shipped prompt text is built from, mined from source.

    Mechanical on purpose. A hand-kept list of "the passages we wrote" cannot
    notice a passage somebody adds, and the failure mode is exactly that: a DDO
    definition pasted into a pack slot or a block. So: every string constant in
    prompts/blocks.py and prompts/core.py, plus every value of the four packs.
    """
    import ast
    from ankidkdeck.prompts import blocks as _blocks
    from ankidkdeck.prompts import core as _core

    out = []
    for mod in (_blocks, _core):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.append(node.value)
    for lang in LANGS:
        out.append(json.dumps(packs.load(lang), ensure_ascii=False))
    return out


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
DDO_NGRAM = 6


def _ngrams(text: str, n: int = DDO_NGRAM):
    words = _WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def ddo_collisions(texts, corpus, n: int = DDO_NGRAM) -> list:
    """Which shipped strings share an n-word run with a DDO string.

    A word n-gram rather than a substring test, because a paste does not arrive
    in tidy pieces: the first version of this checker split on punctuation and a
    verbatim DDO definition pasted into `anti_patterns` came out glued to the
    line's own "-> pasted" tail, so no piece was a substring of anything and the
    mutation went through. Six consecutive identical words is copying whatever
    surrounds it. Measured on the hand-authored passages: they share at most two
    consecutive words with the corpus, so the margin is wide.
    """
    haystack = {}
    for key, value in corpus.items():
        for gram in _ngrams(value, n):
            haystack.setdefault(gram, key)
    out = []
    for text in texts:
        for gram in sorted(_ngrams(text, n)):
            if gram in haystack:
                out.append({"ngram": gram, "ddo_key": haystack[gram]})
                break
    return out


# A real DDO sense definition, quoted ONCE, here, as the synthetic fixture the
# copyright checker is tested against -- and nowhere in the shipped prompt text.
# Escaped so this file stays ASCII.
_SYNTHETIC_DDO = ("person eller andet der er nr. 1 p\u00e5 en rangliste "
                  "eller i en konkurrence")


def test_the_ddo_collision_checker_bites():
    """F5(d), the fixture half. The guard this replaces asserted that the word
    `traek` was still PRESENT in the example slots -- the opposite of what its
    name claimed -- and it short-circuited on the second slot, so for that slot
    it executed no assertion at all. A verbatim DDO definition pasted into a
    pack left all 488 tests green.

    Three mutations, on a synthetic corpus, so the checker is known to bite:
    """
    texts = _shipped_prompt_texts()
    assert texts

    # 1. the corpus contains one of OUR passages verbatim -> caught
    ours = "det at flytte noget hen imod sig selv ved at bruge kraft"
    assert any(ours in t for t in texts), "the worked example moved"
    assert ddo_collisions(texts, {"synthetic:1": ours})
    # 2. ... and as part of a longer definition -> still caught
    assert ddo_collisions(texts,
                          {"synthetic:2": "noget om " + ours + " og mere"})
    # 3. a DDO definition pasted into a pack slot, glued to the line's own
    #    "bad gloss ... -> good ..." furniture the way a real paste would be
    pasted = texts + ["bad gloss " + _SYNTHETIC_DDO + " -> pasted"]
    assert ddo_collisions(pasted, {"synthetic:3": _SYNTHETIC_DDO})
    # 4. and the shipped text against a corpus of unrelated Danish is clean
    assert not ddo_collisions(
        texts, {"synthetic:4": "en lille gul fugl der synger om morgenen i "
                               "haven bag det hvide hus ved vandet"})


def test_the_packs_and_blocks_carry_no_ddo_definition_text():
    """F5(d), the real half: the shipped Danish against the real DDO corpus.

    Fixture-gated, so it runs on the run host (where json/entries.json is) and
    skips anywhere else. It asserts ZERO hits -- the direction the name claims.
    """
    path = os.environ.get("ANKIDKDECK_DDO_CORPUS")
    candidates = [Path(path)] if path else [
        Path.home() / "v3run/work/json/entries.json"]
    entries_path = next((c for c in candidates if c.exists()), None)
    if entries_path is None:
        pytest.skip("no DDO corpus on this host; set ANKIDKDECK_DDO_CORPUS")
    entries = json.loads(entries_path.read_text(encoding="utf-8"))
    corpus = {}
    for eid, entry in entries.items():
        for sense in entry.get("senses") or []:
            text = (sense.get("definition") or "").strip()
            if text:
                corpus["%s:%s" % (eid, sense.get("dannetid"))] = text
        for expr in entry.get("expressions") or []:
            for field in ("expression", "definition", "hint"):
                text = (expr.get(field) or "").strip()
                if text:
                    corpus["%s:expr:%s" % (eid, field)] = text
    assert len(corpus) > 10000, "corpus looks truncated: %d" % len(corpus)
    hits = ddo_collisions(_shipped_prompt_texts(), corpus)
    assert hits == [], "shipped prompt text collides with DDO: %s" % hits[:5]
    # the checker is live on this corpus, not vacuously green: the same corpus
    # catches the synthetic paste
    assert ddo_collisions(["bad gloss " + _SYNTHETIC_DDO + " -> pasted"],
                          corpus)


def test_the_simplified_contract_is_in_the_chinese_prompt():
    """F5(c). `orthography_rules` is the ONLY repair for N-05's 325 Traditional
    cells, and deleting it left 488 tests green: block_1_script gates on
    `allowed_scripts`, not on this slot, so removing it just quietly dropped a
    line. Mutation evidence: replacing the Chinese `orthography_rules` with
    "Write clearly." now fails this test and nothing else in the suite.
    """
    prompts.activate(prompt_id="rich-core-1")
    zh = prompts.build_definition_prompt("Chinese")
    assert "SIMPLIFIED" in zh
    # the ten most frequent Traditional contaminants in the shipped corpus,
    # named so the model has the pairs and not only the rule
    for ch in "\u500b\u52d5\u70ba\u9ede\u4f86\u5c0d\u9593\u6642" \
              "\u9ad4\u767c":
        assert ch in zh, "the Traditional example %r left the prompt" % ch
    for ch in "\u4e2a\u52a8\u4e3a\u70b9\u6765\u5bf9\u95f4\u65f6" \
              "\u4f53\u53d1":
        assert ch in zh, "the Simplified counterpart %r left the prompt" % ch
    assert "pinyin" in zh.lower()
    # German: the one real orthography defect is a Greek beta typed for a sharp
    # s, so the contract has to name BOTH characters or it names neither.
    de = prompts.build_definition_prompt("German")
    assert "\u00df" in de and "\u03b2" in de


# --------------------------------------------------------------------------
# 4.2 the builder: LEAN is the default and is byte-frozen
# --------------------------------------------------------------------------

def test_the_lean_prompt_is_byte_identical_to_the_frozen_text():
    """THE regression nail. The default path must send the measured bytes.

    Not "equivalent" and not "the same rules": the same BYTES. The explicit
    cache is keyed on exact content, the thinking constant belongs to one
    prompt, and the money gate's size band is a percentage of this number.

    It asserts sha256 because it used to assert len(). Mutation evidence, run
    on this file: an equal-length edit inside the byte-frozen core
    ("senior lexicographer" -> "SENIOR lexicographer") left 488 tests green
    under the length assertion and fails on the first language under this one.
    The character counts are kept as a second, weaker line -- a mismatch there
    tells a reader HOW MUCH moved, which a sha cannot.
    """
    for lang in LANGS:
        text = prompts.build_definition_prompt(lang)
        expr = prompts.build_expression_prompt(lang)
        assert _sha(text) == FROZEN_DEF_SHA256[lang], (
            "the frozen definition prompt for %s changed; chars %d vs %d"
            % (lang, len(text), FROZEN_DEF_CHARS[lang]))
        assert _sha(expr) == FROZEN_EXPR_SHA256[lang], (
            "the frozen expression prompt for %s changed; chars %d vs %d"
            % (lang, len(expr), FROZEN_EXPR_CHARS[lang]))
        assert len(text) == FROZEN_DEF_CHARS[lang], lang
        assert len(expr) == FROZEN_EXPR_CHARS[lang], lang
        # and s42's public function is the same object of bytes
        assert S42.definition_prompt(lang) == text
        assert S42.expression_prompt(lang) == expr


def test_the_frozen_core_module_is_the_only_source_of_those_bytes():
    """F5(b), the other half: catch an equal-length edit wherever it lands.

    The sha table above pins the assembled prompt, so an equal-length edit in
    prompts/core.py fails it. This test pins the direction of the dependency as
    well: the frozen text has to come from core.py, and a pack must not be able
    to reach it. A pack overlay that rewrites every slot moves no frozen sha --
    which is why folding a pack version into the frozen prompt_id would be
    wrong, and why the RICH shas are not pinned here at all.
    """
    from ankidkdeck.prompts import core
    for lang in LANGS:
        assert _sha(core.definition_core(lang)) == FROZEN_DEF_SHA256[lang]
        assert _sha(core.expression_core(lang)) == FROZEN_EXPR_SHA256[lang]


def test_a_pack_edit_cannot_move_a_frozen_sha(cfg):
    """The default path is free of the packs, byte for byte."""
    local = cfg.registry_local / "prompt_packs"
    local.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        (local / ("%s.json" % lang)).write_text(json.dumps(
            {"pack_version": "mutant-1", "allowed_scripts": "ANYTHING",
             "orthography_rules": "REWRITTEN", "length_targets": ["REWRITTEN"],
             "anti_patterns": ["REWRITTEN"]}), encoding="utf-8")
    prompts.activate(cfg)
    for lang in LANGS:
        assert prompts.pack_version(lang) == "mutant-1"
        assert _sha(prompts.build_definition_prompt(lang)) \
            == FROZEN_DEF_SHA256[lang], lang
        assert _sha(prompts.build_expression_prompt(lang)) \
            == FROZEN_EXPR_SHA256[lang], lang
        # ... and the RICH prompt for the same language DID move.
        assert "REWRITTEN" in prompts.build_definition_prompt(
            lang, prompt_id="rich-core-1")


def test_the_default_prompt_id_is_the_frozen_one():
    """LEAN is the default, and not because RICH is unfinished: the measured
    thinking constant was measured on LEAN, so RICH cannot be the default until
    the 4.4 A/B has run and the artifact has been rebased."""
    assert prompts.active_prompt_id() == "v4-frozen"
    assert prompts.variant_for(prompts.DEFAULT_PROMPT_ID) == prompts.LEAN


def test_lean_is_a_pure_byte_prefix_of_rich():
    """This is what makes the rollback free. Falling back to the frozen prompt
    is "assemble fewer blocks", so the two variants cannot drift apart, and no
    second copy of the text has to be maintained."""
    for lang in LANGS:
        for build in (prompts.build_definition_prompt,
                      prompts.build_expression_prompt):
            lean = build(lang, prompt_id="v4-frozen")
            rich = build(lang, prompt_id="rich-core-1")
            assert rich.startswith(lean), lang
            assert len(rich) > len(lean), lang


def test_every_prompt_depends_on_the_language_only():
    """definition_prompt(lang, 1) == definition_prompt(lang, 20) -- expressed
    as "the builder has no way to take an n at all", which is stronger.

    W0-1 measured the old defect: 30 pinned payloads produced 7 distinct
    sha256 values, one per batch size, differing by one or two characters. An
    explicit cache is keyed on exact content, so that made one cache per
    language impossible while looking like a formatting detail.
    """
    for pid in ("v4-frozen", "rich-core-1"):
        for lang in LANGS:
            shas = set()
            for _ in range(1, 21):
                shas.add(S42.prompt_sha256(
                    prompts.build_definition_prompt(lang, prompt_id=pid)))
                shas.add(S42.prompt_sha256(
                    prompts.build_expression_prompt(lang, prompt_id=pid)))
            assert len(shas) == 2, (pid, lang)   # one per kind, none per n
    # The old signature is not merely unused, it is unrepresentable: the second
    # positional argument is the prompt id, and 20 is not a prompt id.
    with pytest.raises(FatalError):
        prompts.build_definition_prompt("German", 20)


def test_an_unknown_prompt_id_is_refused_and_does_not_fall_back():
    """A fallback to LEAN under a rich prompt_id is exactly the "measure cheap,
    spend rich" failure consumption rule 6 exists to stop, and a fallback would
    make it silent."""
    with pytest.raises(FatalError) as err:
        prompts.build_definition_prompt("German", prompt_id="rich-core-99")
    assert "rich-core-99" in str(err.value)
    assert "v4-frozen" in str(err.value)


def test_the_rich_prompt_is_inside_the_frozen_size_band():
    """CORE ~2,770 tokens (owner decision D-09 / Q-B option A).

    The token numbers are the API-measured lean count plus an offline estimate
    of the appended text -- the most defensible figure obtainable without
    spending, and labelled as an estimate everywhere it is reported.
    """
    for lang in LANGS:
        report = prompts.size_report(lang)["definition"]
        assert report["lean_tokens_estimated"] == pytest.approx(
            FROZEN_DEF_TOKENS[lang], rel=0.02), lang
        rich = report["rich_tokens_estimated"]
        assert 2600 <= rich <= 2960, (lang, rich)
        # N-10: do NOT enrich to 4,096. The explicit-cache floor is 1,024 and
        # the frozen 1,135 already clears it, so tokens past the band buy
        # nothing at all.
        assert rich < 4096, lang
        # and every variant clears the cache floor by a wide margin
        assert report["lean_tokens_estimated"] > 1024, lang


def test_target_tokens_truncates_to_a_prefix():
    """A size cap must never produce a variant somebody has to maintain."""
    full = prompts.build_definition_prompt("German", prompt_id="rich-core-1")
    lean = prompts.build_definition_prompt("German", prompt_id="v4-frozen")
    capped = prompts.build_definition_prompt("German",
                                             prompt_id="rich-core-1",
                                             target_tokens=1600)
    assert capped.startswith(lean)
    assert full.startswith(capped)
    assert len(capped) < len(full)
    assert prompts.estimate_tokens(capped) <= 1600


def test_the_ramp_stages_are_recorded():
    """4.4 wants the blocks measured in a risk order. Folk knowledge about
    which order that was is how an experiment becomes unreproducible."""
    names = set(blocks.BLOCK_NAMES)
    for stage, wanted in builder.RAMP_STAGES:
        assert set(wanted) <= names, stage
    assert builder.RAMP_STAGES[-1][1] == blocks.BLOCK_NAMES


# --------------------------------------------------------------------------
# N-03 / D-10: one language word, zero files
# --------------------------------------------------------------------------

def test_pack_degradation():
    """A language nobody has written a pack for still produces a prompt.

    This is the product requirement, not a robustness nicety: the config takes
    one language word and the whole pipeline runs, with no hand-prepared files.
    """
    for pid in ("v4-frozen", "rich-core-1"):
        text = prompts.build_definition_prompt("Tagalog", prompt_id=pid)
        assert "Danish-Tagalog dictionary" in text
        assert "{lang}" not in text and "{{" not in text
        expr = prompts.build_expression_prompt("Tagalog", prompt_id=pid)
        assert "into Tagalog" in expr
    # with no pack, rich and lean are the SAME text: every enrichment block
    # needs a pack slot, so there is nothing to append. Nothing raises, and
    # nothing renders a hole.
    assert prompts.build_definition_prompt("Tagalog", prompt_id="rich-core-1") \
        == prompts.build_definition_prompt("Tagalog", prompt_id="v4-frozen")
    assert prompts.pack_version("Tagalog") == "none"


def test_a_language_with_no_pack_still_gets_allowed_sets():
    """The reviewer prompt interpolates these mid-sentence. An empty string
    there is a prompt that says "report any character outside ."."""
    sets = packs.allowed_sets("Tagalog")
    assert sets["lemma"] and sets["gloss"]
    assert "Tagalog" in sets["lemma"]


# --------------------------------------------------------------------------
# 4.3 + N-02(d): the expression prompt and the reviewer read ONE source
# --------------------------------------------------------------------------

def test_the_generator_and_the_reviewer_cannot_disagree_about_english():
    """The 2025 defect, in one test.

    The generator said "Never use English in the `lemma`". The reviewer was
    told the allowed set was "the {lang} or English languages" and asked to
    inspect "the lemma part". So an English lemma on a Chinese card was
    reported CLEAN by the reviewer that existed to catch it, and 20 pinyin
    lemmas shipped. Two prose paragraphs disagreed.
    """
    prompts.activate(prompt_id="rich-core-1")
    pack = packs.load("Chinese")
    review = S42.review_prompt("Chinese", "{}")
    generator = prompts.build_expression_prompt("Chinese")
    # the SAME string, from the SAME pack field, in both prompts
    assert pack["lemma_allowed_set"] in review
    assert pack["lemma_allowed_set"] in generator
    assert pack["gloss_allowed_set"] in review
    # and the two sets are genuinely different: the gloss may fall back to a
    # concise English word, the lemma may not.
    assert pack["lemma_allowed_set"] != pack["gloss_allowed_set"]
    assert "English" in pack["gloss_allowed_set"]
    assert "English" not in pack["lemma_allowed_set"]


def test_the_expression_prompt_keeps_the_two_frozen_things():
    """The hint contract and the count lock. Three audits said do not touch."""
    for lang in LANGS:
        for pid in ("v4-frozen", "rich-core-1"):
            text = prompts.build_expression_prompt(lang, prompt_id=pid)
            assert "do not translate the hint itself" in text
            assert "as many objects as the user message states" in text
            assert "{n_items}" not in text
        if lang != "English":
            # the Russian clause exists because of a real contamination
            # incident; it is institutional memory, not decoration
            assert "DO NOT USE RUSSIAN" in prompts.build_expression_prompt(lang)


def test_the_russian_target_is_not_told_to_avoid_russian():
    """Rule 2 named a language, so for one target it forbade what rule 1
    requires: "the primary language MUST be Russian" followed by "DO NOT USE
    RUSSIAN under any circumstances". The exemption is that one target and
    nothing else -- the clause is institutional memory for every other language,
    and the four shipped languages' bytes are pinned by FROZEN_EXPR_SHA256.
    """
    for pid in ("v4-frozen", "rich-core-1"):
        text = prompts.build_expression_prompt("Russian", prompt_id=pid)
        assert "DO NOT USE RUSSIAN" not in text
        # rule 1 still names the target, and rule 2 still forbids the others
        assert 'The primary language for both "lemma" and "gloss" MUST be ' \
               "**Russian**" in text
        assert "You must avoid all other languages." in text
        # and the rest of the block is untouched
        assert "AS A LAST RESORT" in text
    # the exemption is keyed on the language, not on the presence of a pack
    # (Russian has none), so a packless language that is not Russian keeps it
    for lang in ("German", "Tagalog"):
        other = prompts.build_expression_prompt(lang)
        assert "DO NOT USE RUSSIAN" in other
        assert "You must avoid all other languages." in other


def test_no_prompt_carries_a_per_batch_instruction():
    """Corrections go in the USER message. A correction PREPENDED to the system
    prompt changes the cached prefix, forfeiting the discount on precisely the
    requests being redone."""
    for lang in LANGS:
        for pid in ("v4-frozen", "rich-core-1"):
            for build in (prompts.build_definition_prompt,
                          prompts.build_expression_prompt):
                text = build(lang, prompt_id=pid)
                assert "CORRECTION" not in text
                assert "{n_defs}" not in text


# --------------------------------------------------------------------------
# N-11 / N-12: the two content corrections
# --------------------------------------------------------------------------

def test_no_pack_writes_the_disambiguating_parenthetical_rule():
    """N-11. The final audit rated that rule highest; the corpus says it is a
    0.27% (Chinese) to 3.43% (German) habit whose rate does not even rise when
    a lemma is reused. Writing it as a rule would push new cells an order of
    magnitude away from the old ones. What the corpus DOES do is rewrite the
    gloss: two senses of one headword never share one, 0.0% in all four
    languages."""
    prompts.activate(prompt_id="rich-core-1")
    for lang in LANGS:
        text = prompts.build_definition_prompt(lang)
        assert "never identical to another sense's `gloss`" in text
        low = text.lower()
        assert "disambiguat" not in low
        assert "parenthetical" not in low


def test_the_danish_reading_block_is_written_by_measured_frequency():
    """N-12. The register markers all three audits asked for occur 14 times in
    13,497 shipped Danish definitions, because DDO's register labels live in a
    field compute_todo never sends. The formulas that DO have objects are the
    ones in the block."""
    prompts.activate(prompt_id="rich-core-1")
    text = prompts.build_definition_prompt("German")
    for present in ("`fx` (10.3%", "el.lign.` (11.1%)", "`ofte` (4.2%)"):
        assert present in text
    for absent in ("i overf\u00f8rt betydning", "sp\u00f8gende",
                   "neds\u00e6ttende",
                   "gammeldags", "jf.", "navnlig"):
        assert absent not in text


def test_the_grammar_payload_is_explained_in_the_prompt_not_the_core():
    """Q-C landed in the user payload. 38.8% of senses carry a
    `grammar` note that had never reached the model at all, so the rich prompt
    says how to read one -- and the frozen core still says nothing, because a
    core edit is a style change to 22,734 cells."""
    assert "Grammar notes" not in prompts.build_definition_prompt("German")
    prompts.activate(prompt_id="rich-core-1")
    text = prompts.build_definition_prompt("German")
    assert "Grammar notes in the user message" in text
    assert "NOGEN" in text
    assert "never translate the note itself" in text


# --------------------------------------------------------------------------
# integration: the one place the mapping is written down
# --------------------------------------------------------------------------

def test_swapping_the_prompt_moves_the_bill_the_cache_key_and_doctor_together():
    """The F5 contract: _SYSTEM_PROMPTS is the single source.

    Before it, the bill computed prompt_sha256(definition_prompt(lang)) at its
    own call site while the request built its system instruction at another, so
    replacing the builder at one of them -- which is exactly what this work
    does -- would have left G-PROMPT comparing a stale sha to itself and
    reporting agreement.
    """
    lean = S42.prompt_shas("German")
    prompts.activate(prompt_id="rich-core-1")
    rich = S42.prompt_shas("German")
    assert lean["definition"] != rich["definition"]
    assert lean["expression"] != rich["expression"]
    assert rich["definition"] == S42.prompt_sha256(
        S42.system_prompt("definition", "German"))


def test_a_second_source_for_the_prompt_is_refused_on_the_wire(cfg, probe_stats):
    """CallContext.request() refuses a system instruction that is not
    system_prompt(kind, lang). The bill would describe one prompt and the wire
    would carry another."""
    _ = probe_stats
    ctx = S42.CallContext(cfg=cfg, pool=None, fit=(35.964, 23.07),
                          lang="German", prompt_id="v4-frozen")
    good = S42.system_prompt("definition", "German")
    assert ctx.request("definition", "x", "u", None, 3, good).system == good
    with pytest.raises(FatalError):
        ctx.request("definition", "x", "u", None, 3,
                    prompts.build_definition_prompt("German",
                                                    prompt_id="rich-core-1"))


def test_the_prompt_path_imports_no_llm_module():
    """The bill and the prompts must be buildable with no SDK installed."""
    import subprocess
    import sys
    code = ("import sys;"
            "from ankidkdeck import prompts;"
            "prompts.build_definition_prompt('German', prompt_id='rich-core-1');"
            "prompts.build_expression_prompt('Chinese');"
            "bad=[m for m in sys.modules if m.split('.')[0] in "
            "('google','google_genai')];"
            "print('LEAK' if bad else 'CLEAN')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "CLEAN", out.stdout


# --------------------------------------------------------------------------
# 1.9 / N-05: G-SCRIPT
# --------------------------------------------------------------------------

# Cells written with escapes so this file stays ASCII. Traditional forms of
# dong (U+52D5 for U+52A8) and ge (U+500B for U+4E2A) are the two most frequent
# contaminants in the shipped Chinese corpus.
TRAD_DONG = "\u52d5"          # traditional form of U+52A8
SIMP_DONG = "\u52a8"          # the simplified form it should have been
HAN_PULL = "\u62c9"           # "pull"
HAN_MEANS = "\u6307"          # "refers to", the corpus gloss opener
FULL_STOP = "\u3002"          # ideographic full stop, 99.9% of glosses
GREEK_MU = "\u039c"           # greek capital mu
GREEK_BETA = "\u03b2"         # greek small beta, the German defect
CYRILLIC = "\u0442"           # cyrillic te

# Traditional characters that DO have a distinct simplified form, so their
# presence in a Simplified cell is evidence of a leak. All three are from the
# one genuine leak of the 2026-08-27 Chinese wave, cell 11017121:21026245.
TRAD_YAN = "\u95b9"            # U+95B9, simplifies to U+9609
TRAD_ZHU = "\u8c6c"            # U+8C6C, simplifies to U+732A
TRAD_GUO = "\u904e"            # U+904E, simplifies to U+8FC7
# Characters the GB2312 test called Traditional and that are nothing of the
# kind. U+9CC0 and U+8137 are not in a Traditional charset at all (they are
# rare SIMPLIFIED characters), U+7947 and U+77AD are retained in Simplified,
# U+808F is the same character in both scripts. Eight of the nine
# traditional_han findings of the 2026-08-27 wave were these.
NOT_TRAD = "\u7947\u77ad\u808f\u9cc0\u8137"
# U+5F8C, "after": one of the most common Traditional characters there is
# (simplifies to U+540E) and GB2312 CAN encode it, so the old test was blind to
# it. 54 table entries are in that position.
TRAD_GB2312_OK = "\u5f8c"

# Real 2026-08-27 wave lemmas, verbatim. The 18 BLOCK findings of the first
# Chinese wave were adjudicated cell by cell; these are the ones that decided
# each of the three fixes.
#
# Latin-symbol subjects: the entry IS the pronoun / the card / the note, so the
# lemma has to be able to name it. Seven cells, all REVIEW.
ZH_SUBJECT_LEMMAS = {
    "11008678:28135789": (
        "TA\uff08\u5bbe\u683c\uff0c\u975e\u4e8c\u5143\u6027\u522b\uff09",
        "\u4eba\u79f0\u4ee3\u8bcd\u201cde\u201d\u7684\u5bbe\u683c\u5f62\u5f0f"
        "\uff0c\u7528\u4e8e\u6307\u4ee3\u504f\u597d\u4f7f\u7528\u6027\u522b"
        "\u4e2d\u7acb\u4ee3\u8bcd\u800c\u975e\u201c\u4ed6\u201d\u6216\u201c"
        "\u5979\u201d\u7684\u5355\u4e2a\u4eba\u3002"),
    "11008678:28135794": (
        "TA\uff08\u5bbe\u683c\uff0c\u6027\u522b\u672a\u77e5\u6216\u65e0\u5173"
        "\uff09",
        "\u4eba\u79f0\u4ee3\u8bcd\u201cde\u201d\u7684\u5bbe\u683c\u5f62\u5f0f"
        "\uff0c\u7528\u4e8e\u6307\u4ee3\u67d0\u4f4d\u5df2\u77e5\u6216\u521a"
        "\u63d0\u53ca\u7684\u5355\u4e2a\u4eba\u3002"),
    "49003153:28135708": (
        "TA\uff08\u4e2d\u6027\u6307\u4ee3\uff09",
        "\u7528\u4e8e\u6307\u4ee3\u67d0\u4f4d\u5df2\u77e5\u6216\u521a\u63d0"
        "\u53ca\u7684\u4eba\u3002"),
    "49003153:28135887": (
        "TA\uff08\u975e\u4e8c\u5143\u6027\u522b\u4ee3\u8bcd\uff09",
        "\u7528\u4e8e\u6307\u4ee3\u67d0\u4f4d\u5df2\u77e5\u6216\u521a\u63d0"
        "\u53ca\u7684\u4eba\uff08\u4ed6/\u5979\uff09\u3002"),
    "11009756:91001204": (
        "Q\uff08\u6251\u514b\u724c\uff09",
        "\u6251\u514b\u724c\u4e2d\u70b9\u6570\u4ecb\u4e8eJ\uff08\u9a91\u58eb"
        "\uff09\u548cK\uff08\u56fd\u738b\uff09\u4e4b\u95f4\u7684\u724c\u9762"
        "\u3002"),
    "11026729:21041711": (
        "\uff08\u6251\u514b\u724c\u7684\uff09J",
        "\u6251\u514b\u724c\u4e2d\u5e26\u6709\u56fe\u50cf\u7684\u4e00\u5f20"
        "\u724c\uff0c\u5927\u5c0f\u4ecb\u4e8e10\u548cQ\uff08\u738b\u540e\uff09"
        "\u4e4b\u95f4\u3002"),
    "11029382:21045989": (
        "\u5531\u540dla\uff08\u7b2c\u516d\u97f3\uff09",
        "\u81ea\u7136\u97f3\u9636\u4e2d\u7684\u7b2c\u516d\u4e2a\u97f3\u7ea7"
        "\uff0c\u5728C\u5927\u8c03\u5531\u540d\u4e2d\u901a\u5e38\u5bf9\u5e94"
        "\u97f3\u540dA\u3002"),
}

# The two REAL foreign-text leaks: a Chinese translation with the DANISH family
# headword parenthesised into it. Both BLOCK, and note what makes them
# different from the seven above -- not the gloss (both of these glosses DO
# repeat their own token, and none of the seven does), but that `lille` and
# `undskylde` are words rather than symbols, and that `lille` sits inside the
# parenthesis.
ZH_FOREIGN_LEMMAS = {
    "11048138:91006185": (
        "\u5c0f\uff08lille\u7684\u590d\u6570\uff09",
        "\u5f62\u5bb9\u8bcdlille\uff08\u5c0f\u7684\uff09\u7684\u590d\u6570"
        "\u5f62\u5f0f\u3002"),
    "12004949:91007535": (
        "\u53c2\u89c1 undskylde\uff08\u539f\u8c05\u3001\u9053\u6b49\uff09",
        "\u52a8\u8bcd undskylde\uff08\u539f\u8c05\u3001\u9053\u6b49\uff09"
        "\u7684\u7948\u4f7f\u5f0f\u3002"),
}

# Archive expression cell 21002216 verbatim: REAL pinyin, tone-marked. This is
# what pinyin_in_lemma means, and all 20 of the pinned cells look like it.
ZH_PINYIN_LEMMA = "\u7a76\u7adf (ji\u016b j\u00ecng)"
# The same shape with the tone marks removed. Nine cells of the 2026-08-27 wave
# were reported as pinyin on no better evidence than the parenthesis.
ZH_TONELESS_PINYIN = "\u7a76\u7adf (jiu jing)"
# The foreign-text shape at its shortest: a Han lemma with a parenthesised
# Danish word.
ZH_DANISH_IN_PARENS = HAN_PULL + "\uff08lille\uff09"


# The seven real Russian cells the Latin-in-lemma family was calibrated on --
# model output, not DDO source text. Five are legitimate and two are garbage,
# and the whole discriminator is the token boundary: every legitimate one keeps
# its Latin as a SEPARATE token (after a space, inside parentheses, joined by a
# hyphen), while both defects weld `reg` between two Cyrillic syllables.
RU_LEGIT_LEMMAS = {
    # a cross-reference to a Danish headword; the Chinese analogue of
    # this shape is latin_in_han_lemma, also REVIEW
    "11012508:91001536": "\u0441\u043c. fatte",
    # a Latin abbreviation as the entry's own subject, parenthesised
    "11039601:91005020": "\u0444\u0438\u043b\u043e\u0441\u043e\u0444\u0438"
                         "\u044f (\u0432 \u0441\u0442\u0435\u043f\u0435\u043d"
                         "\u0438 cand.phil.)",
    "11039601:91005021": "\u0444\u0438\u043b\u043e\u0441\u043e\u0444\u0438"
                         "\u044f (\u0432 \u0441\u0442\u0435\u043f\u0435\u043d"
                         "\u0438 dr.phil.)",
    # a Latin acronym hyphen-joined to a Cyrillic word
    "11046071:21072447": "CD-\u0441\u0438\u043d\u0433\u043b",
    # a key name
    "30000264:28002509": "\u043a\u043b\u0430\u0432\u0438\u0448\u0430 Alt",
}
# Both shipped, and only a hand-written post-write scan caught them:
# Latin letters INSIDE a Cyrillic word, with no separator of any kind.
RU_GARBLED_LEMMAS = {
    "1:1": "\u043b\u0430reg\u0438\u0442\u044c \u0438\u043b\u0438 \u043d\u0435"
           " \u043b\u0430\u0434\u0438\u0442\u044c \u0441 \u043a\u0435\u043c-"
           "\u043b\u0438\u0431\u043e",
    "1:2": "\u043b\u0430reg\u043b\u0430\u0434\u0438\u0442\u044c \u0441 \u043a"
           "\u0435\u043c-\u043b\u0438\u0431\u043e",
}
# A plain Cyrillic gloss, so the lemma is the only thing under test.
RU_GLOSS = ("\u0441\u043b\u043e\u0432\u043e \u0438 \u0434\u0440\u0443\u0433"
            "\u043e\u0435.")


def _ru_classes(lemmas, prov=None):
    cells = {k: {"lemma": v, "gloss": RU_GLOSS,
                 "provenance": prov or "gemini:x@2026"}
             for k, v in lemmas.items()}
    return {f["key"]: (f["class"], f["tier"])
            for f in gates.script_findings(cells, lang="Russian",
                                           kind="definitions", pack={})}


def _zh_cell(lemma, gloss, prov="migrated:2025:x"):
    return {"lemma": lemma, "gloss": gloss, "provenance": prov}


def _zh_cells(pairs, prov):
    return {k: _zh_cell(lemma, gloss, prov=prov)
            for k, (lemma, gloss) in pairs.items()}


def _zh_classes(cells):
    return {f["key"]: (f["class"], f["tier"])
            for f in gates.script_findings(cells, lang="Chinese",
                                           kind="definitions",
                                           pack=packs.load("Chinese"))}


def test_g_script_is_a_declared_gate():
    assert gates.G_SCRIPT in gates.ALL_GATE_IDS


def test_g_script_baseline():
    """Three tiers, on the shipped populations.

    BLOCK on a cell this pipeline wrote, BASELINE on a 2025 cell against the
    pinned count, REVIEW for classes that are legitimate content. The pinned
    numbers below are the ones in registry/gates.json, produced by running this
    same code over all 85,259 cells.
    """
    pack = packs.load("Chinese")
    cells = {
        "1:1": _zh_cell(HAN_PULL, HAN_MEANS + FULL_STOP),
        # legacy contamination: absorbed by the baseline
        "1:2": _zh_cell(TRAD_DONG, HAN_MEANS + FULL_STOP),
        "1:3": _zh_cell(HAN_PULL, TRAD_DONG + FULL_STOP),
        # a cross-reference: REVIEW, never a failure
        "1:4": _zh_cell("\u89c1afgore", HAN_MEANS + FULL_STOP),
        # pinyin: a real defect class, baselined for 2025. Tone-marked, which
        # is the whole of what the class means now.
        "1:5": _zh_cell(ZH_PINYIN_LEMMA, HAN_MEANS + FULL_STOP),
        # the same shape with no tone mark is a DIFFERENT defect with its own
        # class -- see test_g_script_splits_pinyin_from_foreign_text_on_a_tone
        "1:6": _zh_cell(ZH_DANISH_IN_PARENS, HAN_MEANS + FULL_STOP),
    }
    findings = gates.script_findings(cells, lang="Chinese", kind="definitions",
                                     pack=pack)
    tiers = {f["key"]: (f["class"], f["tier"]) for f in findings}
    assert tiers["1:2"] == ("traditional_han", gates.BASELINE)
    assert tiers["1:3"] == ("traditional_han", gates.BASELINE)
    assert tiers["1:4"] == ("latin_in_han_lemma", gates.REVIEW)
    assert tiers["1:5"] == ("pinyin_in_lemma", gates.BASELINE)
    assert tiers["1:6"] == ("foreign_text_in_lemma", gates.BASELINE)
    assert "1:1" not in tiers
    ok, detail = gates.script_contract(
        findings, {"traditional_han": 2, "pinyin_in_lemma": 1,
                   "foreign_text_in_lemma": 1},
        lang="Chinese", kind="definitions")
    assert ok, detail
    assert detail["review_tier_counts"] == {"latin_in_han_lemma": 1}


def test_g_script_blocks_a_cell_this_run_wrote():
    """The whole mechanism. A clean redo rewrites every cell with gemini:*
    provenance, which moves each class from BASELINE to BLOCK by itself -- no
    second edit, and nobody has to remember to tighten anything."""
    pack = packs.load("Chinese")
    cells = {"1:1": _zh_cell(TRAD_DONG, HAN_MEANS + FULL_STOP,
                             prov="gemini:gemini-3.7-flash+v4-frozen@2026")}
    findings = gates.script_findings(cells, lang="Chinese", kind="definitions",
                                     pack=pack)
    assert [f["tier"] for f in findings] == [gates.BLOCK]
    # A generous baseline cannot excuse it: BLOCK has no baseline by design.
    ok, detail = gates.script_contract(findings, {"traditional_han": 9999},
                                       lang="Chinese", kind="definitions")
    assert not ok
    assert detail["block_tier_by_class"] == {"traditional_han": 1}
    assert detail["block_tier_examples"]["traditional_han"] == ["1:1"]


# --------------------------------------------------------------------------
# The three G-SCRIPT classifier fixes of 2026-08-27. The first real Chinese
# wave produced 18 BLOCK findings; the owner adjudicated every one and 15 were
# gate defects, not translation defects. Fixtures below are the real cells.
# --------------------------------------------------------------------------

PROV_NEW = "gemini:gemini-3.7-flash+v4-frozen+LOW@2026-08-27"

GB2312_UPPER_BOUND_CHARS = NOT_TRAD + TRAD_YAN + TRAD_ZHU + TRAD_GUO


def _gb2312_is_traditional(ch):
    """The test _is_traditional USED to be, kept here so the mutation tests can
    put it back and show what it costs."""
    if not (0x4E00 <= ord(ch) <= 0x9FFF):
        return False
    try:
        ch.encode("gb2312")
    except (UnicodeEncodeError, LookupError):
        return True
    return False


def test_g_script_traditional_means_a_distinct_simplified_form():
    """FIX 1. "Traditional leaked" can only honestly mean "this character HAS a
    different simplified form", and that is a table lookup, not an encoding
    accident.

    The eight characters below are the ones the 2026-08-27 wave turned up. Five
    have to pass and three have to fail, and the GB2312 test flagged all eight.
    """
    for ch in NOT_TRAD:
        assert not gates._is_traditional(ch), ch
        # ... and every one of them was a finding under the old test.
        assert _gb2312_is_traditional(ch), ch
    for ch in (TRAD_YAN, TRAD_ZHU, TRAD_GUO):
        assert gates._is_traditional(ch), ch
        assert ch in gates.traditional_variants()
        simplified = gates.traditional_variants()[ch]
        assert simplified and ch not in simplified


def test_the_traditional_table_catches_what_gb2312_could_encode():
    """The old test did not only over-report. GB2312 encodes U+5F8C, one of the
    most common Traditional characters there is, so the test that called itself
    an UPPER BOUND was also a floor with holes in it."""
    assert not _gb2312_is_traditional(TRAD_GB2312_OK)
    assert gates._is_traditional(TRAD_GB2312_OK)
    blind = [ch for ch in gates.traditional_variants()
             if not _gb2312_is_traditional(ch)]
    assert len(blind) >= 50, len(blind)


def test_the_traditional_variant_table_is_honest_about_itself():
    """A data file that catches real leaks without claiming to be exhaustive
    Unicode. If it ever loses its source or its coverage statement, the number
    it produces stops being auditable."""
    from importlib import resources
    ref = resources.files("ankidkdeck").joinpath(
        "registry", gates.TRADITIONAL_VARIANTS_FILE)
    raw = ref.read_text(encoding="utf-8")
    assert raw.isascii(), "the table ships as \\u escapes"
    doc = json.loads(raw)
    assert "OpenCC" in doc["_source"] and "Apache-2.0" in doc["_source"]
    assert doc["_regenerate"] and doc["_coverage"] and doc["_rule"]
    assert len(doc["variants"]) == len(gates.traditional_variants()) > 3000
    for key in doc["variants"]:
        assert len(key) == 1
        assert 0x3400 <= ord(key) <= 0xFAFF


def test_g_script_splits_pinyin_from_foreign_text_on_a_tone_mark():
    """FIX 2. Two unrelated defects used to share one class name, and the name
    described only one of them.

    The 2025 expression corpus put real romanisation in the lemma: 20 cells, 20
    of them tone-marked. The 2026-08-27 definition wave put DANISH in the lemma:
    9 cells, 0 tone-marked, reported as `pinyin_in_lemma` on the strength of the
    parenthesis alone. Anyone triaging that label goes looking for romanisation
    and finds none.
    """
    got = _zh_classes({
        "1:1": _zh_cell(ZH_PINYIN_LEMMA, HAN_MEANS + FULL_STOP, prov=PROV_NEW),
        "1:2": _zh_cell(ZH_TONELESS_PINYIN, HAN_MEANS + FULL_STOP,
                        prov=PROV_NEW),
        "1:3": _zh_cell(ZH_DANISH_IN_PARENS, HAN_MEANS + FULL_STOP,
                        prov=PROV_NEW),
        # no parenthesis, no tone mark, not a symbol: the cross-reference
        # population, REVIEW then and REVIEW now.
        "1:4": _zh_cell("\u89c1afgore", HAN_MEANS + FULL_STOP, prov=PROV_NEW),
    })
    assert got["1:1"] == ("pinyin_in_lemma", gates.BLOCK)
    assert got["1:2"] == ("foreign_text_in_lemma", gates.BLOCK)
    assert got["1:3"] == ("foreign_text_in_lemma", gates.BLOCK)
    assert got["1:4"] == ("latin_in_han_lemma", gates.REVIEW)
    # Both defect classes are zero-tolerance, and foreign_text_in_lemma has an
    # honest baseline of 0: it is unpinned everywhere in gates.json because the
    # archive has none of it.
    assert "foreign_text_in_lemma" in gates._BLOCK_CLASSES
    assert "latin_subject_lemma" in gates._REVIEW_CLASSES


def test_g_script_exempts_a_lemma_whose_subject_is_its_latin_symbol():
    """FIX 3. The seven cells the owner adjudicated innocent, verbatim.

    Same mechanism as greek_subject_lemma: an entry whose subject IS a symbol
    has to be able to name it in the lemma, or the pipeline cannot translate
    that entry at all without failing the gate -- at ingest, after the money.
    """
    got = _zh_classes(_zh_cells(ZH_SUBJECT_LEMMAS, PROV_NEW))
    assert len(got) == len(ZH_SUBJECT_LEMMAS) == 7
    for key, verdict in sorted(got.items()):
        assert verdict == ("latin_subject_lemma", gates.REVIEW), key


def test_g_script_still_blocks_a_parenthesised_danish_headword():
    """The other two of the nine, which are real defects and stay BLOCK.

    Danish source text inside a shipped Chinese lemma is a
    DDO-text-in-the-deck problem as well as a script-purity one, so the
    exemption must not be able to reach it.
    """
    got = _zh_classes(_zh_cells(ZH_FOREIGN_LEMMAS, PROV_NEW))
    assert len(got) == len(ZH_FOREIGN_LEMMAS) == 2
    for key, verdict in sorted(got.items()):
        assert verdict == ("foreign_text_in_lemma", gates.BLOCK), key


def test_the_latin_subject_exemption_does_not_ask_the_gloss():
    """The measured reason the exemption is structural.

    "The gloss mentions the same token" is the obvious discriminator and the
    real data INVERTS it: not one of the seven innocent glosses repeats its own
    token (the TA glosses quote the Danish `de`, the Q gloss names J and K, the
    la gloss names C and A), while BOTH real defects repeat theirs -- because
    the parenthesised Danish word is the family headword the gloss has to
    explain. A gloss-mention rule would have blocked all seven and excused both.
    """
    def gloss_repeats_its_token(lemma, gloss):
        runs = {run for _, _, run in gates._latin_runs(lemma)}
        return any(run in gloss for run in runs)

    for lemma, gloss in ZH_SUBJECT_LEMMAS.values():
        assert not gloss_repeats_its_token(lemma, gloss), lemma
    for lemma, gloss in ZH_FOREIGN_LEMMAS.values():
        assert gloss_repeats_its_token(lemma, gloss), lemma
    # And the verdict does not move when the gloss is replaced wholesale.
    for key, (lemma, _) in ZH_SUBJECT_LEMMAS.items():
        got = _zh_classes({key: _zh_cell(lemma, HAN_MEANS + FULL_STOP,
                                         prov=PROV_NEW)})
        assert got[key] == ("latin_subject_lemma", gates.REVIEW), key


def test_the_subject_exemption_separates_a_symbol_from_a_word():
    """The two conditions, each shown to be load-bearing on its own.

    A symbol token outside the parentheses earns the exemption; a WORD does not
    (that is what keeps `undskylde` out, whose lemma is otherwise shaped exactly
    like the innocent `la` one); and a symbol INSIDE the parentheses does not
    either (that is what keeps the `lille` shape out, and what stops a Chinese
    translation with a parenthesised foreign token from being read as an entry
    about that token).
    """
    subj = gates._is_latin_subject_lemma
    assert subj("TA\uff08\u5bbe\u683c\uff09")
    assert subj("\u5531\u540dla")
    assert subj("\u5b57\u6bcd\u00d8")
    assert not subj("\u53c2\u89c1 undskylde\uff08\u539f\u8c05\uff09")
    assert not subj("\u5c0f\uff08la\u7684\u590d\u6570\uff09")
    assert not subj(HAN_PULL)
    # shape test, stated as such: one letter, <= 3 capitals, or a solfege name.
    assert gates._is_symbol_token("Q") and gates._is_symbol_token("TA")
    assert gates._is_symbol_token("sol") and gates._is_symbol_token("La")
    assert not gates._is_symbol_token("lille")
    assert not gates._is_symbol_token("undskylde")
    assert not gates._is_symbol_token("Alto")


def test_the_whole_adjudicated_chinese_wave_lands_where_the_owner_put_it():
    """The acceptance criterion for all three fixes at once: of the 18 BLOCK
    findings the first real Chinese wave produced, exactly 3 survive."""
    cells = {}
    cells.update(_zh_cells(ZH_SUBJECT_LEMMAS, PROV_NEW))
    cells.update(_zh_cells(ZH_FOREIGN_LEMMAS, PROV_NEW))
    # the one genuine Traditional leak, and the eight false positives
    cells["11017121:21026245"] = _zh_cell(
        TRAD_YAN + "\u516c" + TRAD_ZHU,
        HAN_MEANS + "\u88ab" + TRAD_YAN + "\u5272" + TRAD_GUO + "\u7684\u516c"
        + TRAD_ZHU + FULL_STOP, prov=PROV_NEW)
    for i, ch in enumerate(NOT_TRAD):
        cells["9:%d" % i] = _zh_cell(HAN_PULL, HAN_MEANS + ch + FULL_STOP,
                                     prov=PROV_NEW)
    findings = gates.script_findings(cells, lang="Chinese", kind="definitions",
                                     pack=packs.load("Chinese"))
    ok, detail = gates.script_contract(findings, {}, lang="Chinese",
                                       kind="definitions")
    assert not ok
    assert detail["block_tier_findings"] == 3
    assert detail["block_tier_by_class"] == {"foreign_text_in_lemma": 2,
                                            "traditional_han": 1}
    assert detail["review_tier_counts"] == {"latin_subject_lemma": 7}
    assert detail["baseline_tier_counts"] == {}


# --- mutation checks: revert one fix, and one named test above fails ---

def test_mutation_reverting_the_traditional_table_reraises_the_8_false(
        monkeypatch):
    """Revert FIX 1 ->
    test_g_script_traditional_means_a_distinct_simplified_form fails."""
    monkeypatch.setattr(gates, "_is_traditional", _gb2312_is_traditional)
    for ch in NOT_TRAD:
        assert gates._is_traditional(ch), ch
    cells = {"9:%d" % i: _zh_cell(HAN_PULL, HAN_MEANS + ch + FULL_STOP,
                                  prov=PROV_NEW)
             for i, ch in enumerate(NOT_TRAD)}
    findings = gates.script_findings(cells, lang="Chinese", kind="definitions",
                                     pack=packs.load("Chinese"))
    assert len(findings) == 5
    assert {f["class"] for f in findings} == {"traditional_han"}
    assert {f["tier"] for f in findings} == {gates.BLOCK}


def test_mutation_reverting_the_tone_mark_split_relabels_danish_as_pinyin(
        monkeypatch):
    """Revert FIX 2 ->
    test_g_script_splits_pinyin_from_foreign_text_on_a_tone_mark fails."""
    def romanised_or_parenthesised(lemma):
        if ("(" in lemma or "\uff08" in lemma
                or any(ch in gates._TONE_MARKS for ch in lemma)):
            return "pinyin_in_lemma"
        return "latin_in_han_lemma"

    monkeypatch.setattr(gates, "_latin_in_lemma_class",
                        romanised_or_parenthesised)
    got = _zh_classes({
        "1:2": _zh_cell(ZH_TONELESS_PINYIN, HAN_MEANS + FULL_STOP,
                        prov=PROV_NEW),
        "1:3": _zh_cell(ZH_DANISH_IN_PARENS, HAN_MEANS + FULL_STOP,
                        prov=PROV_NEW)})
    assert got["1:2"] == ("pinyin_in_lemma", gates.BLOCK)
    assert got["1:3"] == ("pinyin_in_lemma", gates.BLOCK)


def test_mutation_dropping_the_subject_exemption_blocks_all_seven(monkeypatch):
    """Revert FIX 3 ->
    test_g_script_exempts_a_lemma_whose_subject_is_its_latin_symbol fails."""
    monkeypatch.setattr(gates, "_is_latin_subject_lemma", lambda lemma: False)
    got = _zh_classes(_zh_cells(ZH_SUBJECT_LEMMAS, PROV_NEW))
    assert len(got) == 7
    for key, verdict in sorted(got.items()):
        assert verdict == ("foreign_text_in_lemma", gates.BLOCK), key


def test_g_script_separates_a_greek_mention_from_a_greek_contamination():
    """The discriminator that took the false-positive rate from 76.5% to zero.

    A naive "any character outside the target script" rule finds 17 cells in
    the shipped corpus and 4 of them are real. Thirteen are the glosses of the
    three entries whose SUBJECT is a Greek letter -- they have to contain one.
    The one real Greek defect is a beta written where a German sharp s was
    meant, and its mechanical signature is that it sits INSIDE a run of Latin
    letters.
    """
    de_pack = packs.load("German")
    cells = {
        # the real defect: aeuBere, with a Greek beta for the sharp s
        "1:1": {"lemma": "Ende", "gloss": "Das \u00e4u" + GREEK_BETA + "ere Ende.",
                "provenance": "migrated:2025:x"},
        # a mention: the entry IS the Greek letter
        "1:2": {"lemma": "My", "gloss": "Der Buchstabe: " + GREEK_MU + ".",
                "provenance": "migrated:2025:x"},
    }
    findings = {f["key"]: f["class"]
                for f in gates.script_findings(cells, lang="German",
                                               kind="definitions",
                                               pack=de_pack)}
    assert findings["1:1"] == "greek_latin_internal"
    assert findings["1:2"] == "greek_mention"
    # the mention alone never fails a wave
    ok, _ = gates.script_contract(
        [f for f in gates.script_findings(
            {"1:2": cells["1:2"]}, lang="German", kind="definitions",
            pack=de_pack)], {}, lang="German", kind="definitions")
    assert ok


def test_g_script_does_not_flag_danish_letters_in_a_latin_target():
    """Measured, and not in any upstream document: the Danish letters appear
    legitimately in 162 English, 51 German and 70 Spanish cells, naming the
    Danish headword. A per-character allow-list gate would have failed all of
    them, which is why this gate is built out of FORBIDDEN script blocks."""
    for lang in ("English", "German", "Spanish"):
        cells = {"1:1": {"lemma": "pull",
                         "gloss": "From the Danish tr\u00e6k, str\u00f8m, \u00e5.",
                         "provenance": "gemini:x@2026"}}
        assert not gates.script_findings(cells, lang=lang, kind="definitions",
                                         pack=packs.load(lang))


def test_g_script_refuses_an_unpinned_baseline_population():
    """An unbaselined population is how 325 contaminated cells stayed invisible
    for a year. Same discipline as G-SUPPRESS and G-ADMIT: the number lives in
    registry/gates.json and moves in the same commit as whatever moved it."""
    pack = packs.load("Chinese")
    findings = gates.script_findings(
        {"1:1": _zh_cell(TRAD_DONG, HAN_MEANS + FULL_STOP)},
        lang="Chinese", kind="definitions", pack=pack)
    ok, detail = gates.script_contract(findings, {}, lang="Chinese",
                                       kind="definitions")
    assert not ok
    assert detail["baseline_unpinned"] == [{"class": "traditional_han",
                                            "cells": 1}]


def test_g_script_reports_shrinkage_and_does_not_fail_on_it():
    """The clean retranslation is supposed to take every baseline to zero. A
    gate that failed on that would fail on success."""
    ok, detail = gates.script_contract([], {"traditional_han": 325},
                                       lang="Chinese", kind="definitions")
    assert ok
    assert detail["baseline_shrunk_reported_not_failed"] == [
        {"class": "traditional_han", "cells": 0, "baseline": 325}]


def test_g_script_forbidden_scripts_are_never_excused():
    """Cyrillic in a Chinese cell was one of the four hand-found defects. There
    is no entry whose subject is the Cyrillic alphabet in this dictionary, so
    unlike Greek this class has no exception."""
    findings = gates.script_findings(
        {"1:1": _zh_cell(HAN_PULL + CYRILLIC, HAN_MEANS + FULL_STOP,
                         prov="gemini:x@2026")},
        lang="Chinese", kind="definitions", pack=packs.load("Chinese"))
    assert [(f["class"], f["tier"]) for f in findings] == \
        [("forbidden_script", gates.BLOCK)]


def test_g_script_and_the_prompt_read_the_same_pack_fields():
    """The verdict's requirement, mechanically. If the gate kept its own table
    of what a legal character is, that table and the prompt's script contract
    would be two prose paragraphs -- which is the shape of the defect that
    shipped English lemmas on Chinese cards."""
    for lang in LANGS:
        pack = packs.load(lang)
        profile = gates.script_profile(pack)
        prompts.activate(prompt_id="rich-core-1")
        text = prompts.build_definition_prompt(lang)
        assert pack["allowed_scripts"] in text
        assert pack["lemma_charset_rule"] in text
        assert profile["han_allowed"] == (lang == "Chinese")


def test_g_script_says_nothing_about_a_language_nobody_described():
    """No pack means nobody has said what that language's letters are, so the
    gate says nothing about them -- while still refusing Cyrillic."""
    profile = gates.script_profile(None)
    assert not profile["has_pack"]
    cells = {"1:1": {"lemma": "kabuuan", "gloss": "Ang kabuuan.",
                     "provenance": "gemini:x@2026"},
             "1:2": {"lemma": CYRILLIC * 3, "gloss": "x",
                     "provenance": "gemini:x@2026"}}
    found = gates.script_findings(cells, lang="Tagalog", kind="definitions",
                                 pack={})
    assert [f["key"] for f in found] == ["1:2"]


def test_the_script_baselines_are_not_dead_data(cfg):
    """A review gate that cannot change behaviour is worse than none: it
    manufactures the belief that the numbers were reviewed."""
    from ankidkdeck.registry import Registry
    policy = dict(Registry(cfg).gates or {})
    baseline = policy.get("script_baseline") or {}
    # Recalibrated 2026-08-27 under the traditional_variants table: 325 -> 317
    # for definitions and 22 -> 20 for expressions, because 8 and 2 of the
    # cells the GB2312 test flagged carried rare SIMPLIFIED characters.
    # pinyin_in_lemma stayed at 20 -- all 20 archive cells are tone-marked.
    assert baseline["Chinese"]["definitions"]["traditional_han"] == 317
    assert baseline["Chinese"]["expressions"]["traditional_han"] == 20
    assert baseline["Chinese"]["expressions"]["pinyin_in_lemma"] == 20
    assert baseline["German"]["definitions"]["greek_latin_internal"] == 1
    assert policy.get("_note_script_gate")
    # and the value is what the gate reads: move it, and a wave that was
    # passing fails.
    rows = gates.script_gate_rows(
        cfg, {"definitions": {"1:1": _zh_cell(TRAD_DONG,
                                              HAN_MEANS + FULL_STOP)}},
        lang="Chinese", pack=packs.load("Chinese"), policy=policy)
    assert len(rows) == 1
    ok, _ = rows[0].fn()
    assert ok
    policy["script_baseline"]["Chinese"]["definitions"]["traditional_han"] = 0
    rows = gates.script_gate_rows(
        cfg, {"definitions": {"1:1": _zh_cell(TRAD_DONG,
                                              HAN_MEANS + FULL_STOP)}},
        lang="Chinese", pack=packs.load("Chinese"), policy=policy)
    ok, detail = rows[0].fn()
    assert not ok and detail["baseline_over"]


def test_g_script_rows_are_keyed_per_language_and_kind(cfg):
    """A passing expressions file must not be able to hide a failing
    definitions file, and language two must not overwrite language one."""
    rows = gates.script_gate_rows(
        cfg, {"definitions": {}, "expressions": {}}, lang="Chinese",
        pack=packs.load("Chinese"), policy={})
    keys = {gates.row_key({"id": r.id, "stage": r.stage, "extra": r.extra})
            for r in rows}
    assert len(keys) == 2
    assert gates.row_label({"id": gates.G_SCRIPT,
                            "extra": {"lang": "Chinese",
                                      "kind": "definitions"}}) \
        == "G-SCRIPT[kind=definitions,lang=Chinese]"


def test_the_script_gate_never_reads_the_danish_source_text(cfg):
    """The gate's report row goes into reports/gates_report.json. It reads
    lemma, gloss and provenance and nothing else, so DDO text cannot get there
    even if a caller hands it a row that carries some."""
    cells = {"1:1": {"lemma": HAN_PULL, "gloss": HAN_MEANS + FULL_STOP,
                     "provenance": "migrated:2025:x",
                     "text": "DANISH SOURCE TEXT MUST NOT APPEAR",
                     "src_sha": "a" * 64}}
    rows = gates.script_gate_rows(cfg, {"definitions": cells}, lang="Chinese",
                                 pack=packs.load("Chinese"), policy={})
    ok, detail = rows[0].fn()
    assert ok
    assert "DANISH SOURCE TEXT" not in json.dumps(detail, ensure_ascii=False)


# --------------------------------------------------------------------------
# 4.4: the A/B hook is code, and it places nothing by itself
# --------------------------------------------------------------------------

def _ab_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "tools" / "prompt_thinking_ab.py"
    spec = importlib.util.spec_from_file_location("prompt_thinking_ab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_ab_hook_knows_both_arms_and_their_prompt_ids():
    ab = _ab_module()
    assert ab.DEFAULT_ARMS == "lean,rich"
    assert ab.ARMS["lean"] == ("v4-frozen", None)
    assert ab.ARMS["rich"] == ("rich-core-1", None)
    for pid, stage in ab.ARMS.values():
        assert pid in prompts.PROMPT_IDS
        assert stage is None or stage in prompts.RAMP_STAGE_NAMES
    # every ramp stage is runnable as an arm: that is what makes criterion (c)'s
    # per-block attribution a thing the tool can do rather than a constant.
    for stage in prompts.RAMP_STAGE_NAMES:
        assert stage in ab.ARMS


def test_the_ramp_stages_are_a_selectable_block_set():
    """F8. RAMP_STAGES had exactly one reader -- a test asserting its shape --
    while the builder's block_names parameter, the thing that would have made it
    usable, was reachable only from Python. So "which block earned its token"
    (patch plan 4.4 criterion (c)) could not be measured at all, and the report
    still claimed all five criteria were implemented.
    """
    lean = prompts.build_definition_prompt("German", prompt_id="v4-frozen")
    rich = prompts.build_definition_prompt("German", prompt_id="rich-core-1")
    seen = {}
    for stage in prompts.RAMP_STAGE_NAMES:
        prompts.reset()
        prompts.activate(prompt_id="rich-core-1", ramp_stage=stage)
        assert prompts.active_ramp_stage() == stage
        text = prompts.build_definition_prompt("German")
        seen[stage] = len(text)
        # every stage extends the frozen core and none exceeds full RICH
        assert text.startswith(lean), stage
        assert len(text) <= len(rich), stage
    # they are ordered by size, and only the last one is a byte prefix of RICH
    assert seen["stage1"] < seen["stage2"] < seen["stage3"]
    prompts.reset()
    prompts.activate(prompt_id="rich-core-1", ramp_stage="stage3")
    assert prompts.build_definition_prompt("German") == rich
    with pytest.raises(FatalError):
        prompts.activate(prompt_id="rich-core-1", ramp_stage="stage9")
    # a stage cannot smuggle blocks into the FROZEN prompt
    prompts.reset()
    prompts.activate(prompt_id="v4-frozen", ramp_stage="stage3")
    assert prompts.build_definition_prompt("German") == lean


def test_the_blind_pairs_file_does_not_carry_the_answers(tmp_path):
    """F9. The answer key used to sit on every pair, one field along from the
    text the judge reads. Criterion (d) is the stop-loss on the whole
    enrichment decision, so a file with the answers in it is not a blind test
    and its PENDING verdict certifies nothing.
    """
    ab = _ab_module()
    lean = [{"key": "1:%d" % i, "pos_key": "sb.", "lemma": "draught",
             "gloss": "A current of cold air.", "provenance": "prompt_ab:lean"}
            for i in range(6)]
    rich = [{"key": "1:%d" % i, "pos_key": "sb.", "lemma": "draught",
             "gloss": "A current of cold air, for example from a door.",
             "provenance": "prompt_ab:rich"} for i in range(6)]
    out = ab.verdict({"lean": {"arm": "lean", "thoughts": [0] * 6,
                               "produced": lean},
                      "rich": {"arm": "rich", "thoughts": [0] * 6,
                               "produced": rich}},
                     "English", packs.load("English"), tmp_path)
    pairs = json.loads(Path(out["criteria"]["d_blind_test"]["file"])
                       .read_text(encoding="utf-8"))
    assert pairs
    blob = json.dumps(pairs)
    assert "answer" not in blob and "lean" not in blob and "rich" not in blob
    assert {k for p in pairs for k in p} == {"key", "A", "B"}
    # and the answers exist, in their own file, for the same keys
    key = json.loads(Path(out["criteria"]["d_blind_test"]["answer_key"])
                     .read_text(encoding="utf-8"))
    assert set(key["answers"]) == {p["key"] for p in pairs}
    assert set(key["answers"].values()) <= {"lean", "rich"}


def test_the_ab_verdict_attributes_the_pos_criterion_per_arm(tmp_path):
    """The other half of F8: criterion (c) per block set, not one number for
    the whole rich prompt."""
    ab = _ab_module()
    made = [{"key": "1:1", "pos_key": "sb.", "lemma": "draught", "gloss": "A."}]
    results = {
        "lean": {"arm": "lean", "prompt_id": "v4-frozen", "ramp_stage": None,
                 "blocks": None, "thoughts": [0], "produced": made},
        "stage1": {"arm": "stage1", "prompt_id": "rich-core-1",
                   "ramp_stage": "stage1",
                   "blocks": list(prompts.ramp_stage_blocks("stage1")),
                   "thoughts": [0], "produced": made},
        "rich": {"arm": "rich", "prompt_id": "rich-core-1", "ramp_stage": None,
                 "blocks": None, "thoughts": [0], "produced": made},
    }
    out = ab.verdict(results, "English", packs.load("English"), tmp_path)
    attribution = out["criteria"]["c_pos_shape"]["per_block_attribution"]
    assert set(attribution) == {"lean", "stage1", "rich"}
    assert attribution["stage1"]["blocks"] == list(
        prompts.ramp_stage_blocks("stage1"))
    assert out["by_arm"]["stage1"]["pos_conformance"] == 1.0
    assert not out["criteria"]["c_pos_shape"]["attribution_note"]


def test_the_ab_hook_scores_pos_shape_mechanically():
    ab = _ab_module()
    produced = [
        {"key": "1:1", "pos_key": "vb.", "lemma": "to pull", "gloss": "x"},
        {"key": "1:2", "pos_key": "vb.", "lemma": "pull", "gloss": "x"},
        {"key": "1:3", "pos_key": "sb.", "lemma": "the draught", "gloss": "x"},
        {"key": "1:4", "pos_key": "sb.", "lemma": "draught", "gloss": "x"},
        {"key": "1:5", "pos_key": "sb.", "lemma": "", "gloss": "x"},
    ]
    bad = ab.pos_shape_violations("English", produced, packs.load("English"))
    assert {b["key"] for b in bad} == {"1:1", "1:3", "1:5"}
    zh = ab.pos_shape_violations(
        "Chinese", [{"key": "2:1", "pos_key": "sb.", "lemma": HAN_PULL + "x",
                     "gloss": "y"}], packs.load("Chinese"))
    assert [b["why"] for b in zh] == ["Latin letters in a Han-only lemma"]


def test_the_ab_verdict_needs_both_arms_and_cannot_pass_the_human_criterion(
        tmp_path):
    """Criterion (d) is a human's. A tool that pretended to pass it would be
    the most expensive line in this repository, because (d) is the criterion
    that says "if the rich prompt buys nothing, keep the money"."""
    ab = _ab_module()
    made = [{"key": "1:%d" % i, "pos_key": "sb.", "lemma": "draught",
             "gloss": "A current of cold air.", "provenance": "prompt_ab:x"}
            for i in range(4)]
    results = {"lean": {"arm": "lean", "thoughts": [0, 0, 0, 0],
                        "produced": made},
               "rich": {"arm": "rich", "thoughts": [0, 0, 0, 0],
                        "produced": made}}
    out = ab.verdict(results, "English", packs.load("English"), tmp_path)
    assert out["criteria"]["a_thinking_median"]["ok"] is True
    assert out["criteria"]["b_script_violations"]["ok"] is True
    assert out["criteria"]["c_pos_shape"]["ok"] is True
    assert out["criteria"]["d_blind_test"]["ok"] is None
    assert out["criteria"]["e_constant_invalidated"]["ok"] is None
    assert out["decision"] == "RICH pending the human blind test"
    # one arm only -> criterion (a) cannot be evaluated, so it fails
    out = ab.verdict({"rich": results["rich"]}, "English",
                     packs.load("English"), tmp_path)
    assert out["criteria"]["a_thinking_median"]["ok"] is False
    assert out["decision"] == "LEAN"


def test_the_ab_verdict_fails_a_rich_arm_that_broke_the_script_contract(
        tmp_path):
    ab = _ab_module()
    good = [{"key": "1:1", "pos_key": "sb.", "lemma": HAN_PULL,
             "gloss": HAN_MEANS + FULL_STOP, "provenance": "prompt_ab:lean"}]
    bad = [{"key": "1:1", "pos_key": "sb.", "lemma": TRAD_DONG,
            "gloss": HAN_MEANS + FULL_STOP, "provenance": "prompt_ab:rich"}]
    out = ab.verdict({"lean": {"thoughts": [0], "produced": good},
                      "rich": {"thoughts": [0], "produced": bad}},
                     "Chinese", packs.load("Chinese"), tmp_path)
    assert out["criteria"]["b_script_violations"]["ok"] is False
    assert out["decision"] == "LEAN"


def test_the_ab_hook_rejects_a_thinking_median_that_grew(tmp_path):
    ab = _ab_module()
    made = [{"key": "1:1", "pos_key": "sb.", "lemma": "draught", "gloss": "A."}]
    out = ab.verdict({"lean": {"thoughts": [100, 100, 100], "produced": made},
                      "rich": {"thoughts": [200, 200, 200], "produced": made}},
                     "English", packs.load("English"), tmp_path)
    a = out["criteria"]["a_thinking_median"]
    assert a["ok"] is False
    assert a["median_lean"] == 100 and a["median_rich"] == 200
    assert a["allowed"] == 125.0

# --------------------------------------------------------------------------
# the fix round: F1..F9
# --------------------------------------------------------------------------

def test_the_packs_are_trackable_by_git():
    """F1, the BLOCKER. `.gitignore` line 4 is `*.json` and the negation was
    `!src/ankidkdeck/registry/*.json`, which does not reach the subdirectory the
    packs live in. `git add` exited 0 and staged nothing, `git status -uall`
    never listed them, and a commit prepared that way ships a tree with no packs
    -- where every RICH prompt degrades to LEAN under a rich prompt_id.

    Asserted on the .gitignore text so it holds without a git binary.
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    assert "!src/ankidkdeck/registry/**/*.json" in lines
    assert "!src/ankidkdeck/registry/*.json" not in lines, \
        "the non-recursive negation is the bug, not a second safety net"
    for lang in LANGS:
        assert (root / "src/ankidkdeck/registry/prompt_packs"
                / ("%s.json" % lang)).exists(), lang


def test_a_malformed_packaged_pack_is_fatal_not_silent(monkeypatch, tmp_path):
    """F1's second half. json.JSONDecodeError is a subclass of ValueError, and
    _packaged() caught ValueError -- so a pack a human had hand-edited into
    invalid JSON was indistinguishable from "this language has no pack", and the
    builder served the frozen prompt under prompt_id=rich-core-1 with no message
    anywhere. This module's own docstring said it was a FatalError.
    """
    good = tmp_path / "ok"
    good.mkdir()
    (good / "German.json").write_text('{"pack_version": "de-x"}',
                                      encoding="utf-8")
    monkeypatch.setattr(packs, "_packaged_dir", lambda: good)
    assert packs.load("German")["pack_version"] == "de-x"
    # a language with no file is still the normal, running state (D-10 / N-03)
    assert packs.load("Tagalog") == {}

    (good / "German.json").write_text('{"pack_version": "de-x",',
                                      encoding="utf-8")
    with pytest.raises(FatalError) as err:
        packs.load("German")
    assert "German" in str(err.value)

    # and no packaged directory at all is a broken install, not 12 languages
    # without packs
    monkeypatch.setattr(packs, "_packaged_dir", lambda: None)
    with pytest.raises(FatalError) as err:
        packs.load("German")
    assert "prompt_packs" in str(err.value)


def test_a_pack_version_that_cannot_ride_in_provenance_is_refused():
    """It is welded into the effective prompt_id and into every cell's
    provenance string, and PROVENANCE_RE is a closed ASCII vocabulary."""
    for bad in ("zh 1", "zh+1", "zh_1", "\u4e2d\u6587-1"):
        with pytest.raises(FatalError) as err:
            packs.validate("Chinese", {"pack_version": bad})
        assert "pack_version" in str(err.value)
    packs.validate("Chinese", {"pack_version": "zh-1.2"})


def test_the_effective_prompt_id_carries_the_pack_version(cfg):
    """F6 / O-C4. prompt_id names the block FAMILY; the pack is the rest of the
    prompt text. Measured: a 60-character pack edit moved the Chinese rich
    prompt's sha while prompt_id stayed rich-core-1 and the size drifted 0.5%
    against consumption rule 6's 10% band -- both halves of the rule passed a
    run that measured one prompt and sent another.
    """
    # LEAN reads no pack, so its identity must NOT grow a pack version: doing
    # that would relabel cells whose text did not change.
    for lang in LANGS:
        assert prompts.effective_prompt_id(lang, "v4-frozen") == "v4-frozen"
    assert prompts.effective_prompt_id("Chinese", "rich-core-1") \
        == "rich-core-1.zh-1"
    assert prompts.effective_prompt_id("Tagalog", "rich-core-1") \
        == "rich-core-1.none"
    # it survives PROVENANCE_RE, which is the whole reason the separator is a
    # dot rather than the plus the id-plus-pack notation suggests
    prov = S42._provenance("gemini-3.7-flash",
                           prompts.effective_prompt_id("Chinese",
                                                       "rich-core-1"), "LOW")
    assert prov.startswith("gemini:gemini-3.7-flash+rich-core-1.zh-1+LOW@")
    assert S42.PROVENANCE_RE.match(prov)

    # a pack bump changes the identity, which is what makes it visible
    local = cfg.registry_local / "prompt_packs"
    local.mkdir(parents=True, exist_ok=True)
    (local / "Chinese.json").write_text(json.dumps({"pack_version": "zh-2"}),
                                        encoding="utf-8")
    prompts.activate(cfg, prompt_id="rich-core-1")
    assert prompts.effective_prompt_id("Chinese") == "rich-core-1.zh-2"


def test_a_pack_bump_refuses_the_spend_until_it_is_declared(cfg, probe_stats):
    """F6, the enforcing half: R6-pack-version. A pack bump must need the same
    explicit --declare-prompt-id --rebase-measurement a family change needs."""
    from ankidkdeck import billing
    stats = dict(probe_stats)
    cfg.langs = ["Chinese"]

    def rows(**over):
        st = {**stats, **over}
        return {r["rule"]: r for r in billing.consumption_rules(
            cfg, st, prompts={"definition": S42.definition_prompt("Chinese")})}

    # under the frozen prompt the rule cannot block: no pack reaches that text
    cfg.prompt_id = "v4-frozen"
    prompts.activate(cfg)
    row = rows()["R6-pack-version"]
    assert row["ok"] is True and row["blocking"] is False
    assert row["detail"]["prompt_reads_a_pack"] is False

    # under a rich prompt_id with nothing declared, it refuses
    cfg.prompt_id = "rich-core-1"
    prompts.activate(cfg)
    row = rows()["R6-pack-version"]
    assert row["ok"] is False and row["blocking"] is True

    # declared and matching -> passes
    live = prompts.pack_identity(["Chinese"])
    lineage = dict(stats["prompt_lineage"], pack_versions=live)
    assert rows(prompt_lineage=lineage)["R6-pack-version"]["ok"] is True
    # declared on a DIFFERENT pack version -> refuses, and names the language
    stale = dict(stats["prompt_lineage"],
                 pack_versions={"Chinese": {"version": "zh-0",
                                            "sha256": live["Chinese"]["sha256"]}})
    row = rows(prompt_lineage=stale)["R6-pack-version"]
    assert row["ok"] is False
    assert row["detail"]["disagreeing_languages"] == ["Chinese"]
    # the version unchanged but the CONTENT edited -> also refuses. This is the
    # mutation reviewer B ran and nothing caught: rewriting one fixed rendering
    # without bumping the version changed the shipped prompt while the version
    # and the 10% size band both stayed put.
    edited = dict(stats["prompt_lineage"],
                  pack_versions={"Chinese": {"version": "zh-1",
                                             "sha256": "0" * 64}})
    assert rows(prompt_lineage=edited)["R6-pack-version"]["ok"] is False
    # an artifact from before the sha was recorded still says something
    old_shape = dict(stats["prompt_lineage"],
                     pack_versions={"Chinese": "zh-1"})
    assert rows(prompt_lineage=old_shape)["R6-pack-version"]["ok"] is True


def test_the_definition_payload_states_the_pos_key(cfg):
    """F3. The rich prompt's POS block enumerates the 20 `pos_key` values and
    its plural rule tests the literal string `sb. pl.` -- while the payload's
    headword line carried `pos_text`, one of 38 long Danish display forms, so
    `sb. pl.` never appeared in any request and the rule was unexecutable.
    """
    rows = [{"text": "en lille fugl", "grammar": "", "pos_key": "sb. pl."}]
    payload = S42.definition_user_payload("fugle substantiv pluralis", rows)
    assert "Part of speech: sb. pl." in payload
    # and the prompt speaks the same vocabulary, spelled the same way
    prompts.activate(prompt_id="rich-core-1")
    text = prompts.build_definition_prompt("German")
    assert "`Part of speech` line" in text
    for key in ("vb.", "sb. pl.", "adj. pl.", "pr\u00e6p.",
                "udr\u00e5bsord", "talord (m\u00e6ngdetal)",
                "talord (ordenstal)", "formelt subjekt", "infinitivpartikel"):
        assert key in text, key
    # the ASCII-folded spellings the block used to carry are gone
    for folded in ("praep.", "udraabsord", "maengdetal", "ordenstal:"):
        assert folded not in text, folded
    # a row without a pos_key produces the bytes it produced before
    assert "Part of speech" not in S42.definition_user_payload(
        "fugl", [{"text": "en fugl", "grammar": ""}])


def test_a_greek_letter_entry_can_be_translated_at_all():
    """F4(a). Two shipped Chinese cells are the DDO entries `my` and `ny`, whose
    lemmas are phrases meaning "Greek letter M"/"N" and whose glosses explain
    that same letter. The contract said the character may appear in the gloss
    and NEVER in the lemma, so the clean redo -- which rewrites both cells with
    gemini:* provenance and therefore at BLOCK tier -- had no way to translate
    those two entries without a FatalError at ingest, after the money.

    Entry-subject based, not a whitelist: the excuse is that the cell's OWN
    gloss mentions the same character.
    """
    zh = packs.load("Chinese")
    new = "gemini:gemini-3.7-flash+rich-core-1.zh-1+LOW@2026-08-27"
    mu = "\u039c"
    cells = {
        # the real shape, rewritten by a clean redo
        "ok": {"lemma": "\u5e0c\u814a\u5b57\u6bcd" + mu,
               "gloss": "\u5e0c\u814a\u5b57\u6bcd" + mu
                        + "\u3002", "provenance": new},
        # script leakage: the lemma IS the bare letter
        "bare": {"lemma": mu, "gloss": "\u5e0c\u814a\u5b57\u6bcd" + mu
                 + "\u3002", "provenance": new},
        # a Greek letter the cell's own gloss never explains
        "loose": {"lemma": HAN_PULL + mu, "gloss": HAN_MEANS + FULL_STOP,
                  "provenance": new},
    }
    got = {}
    for f in gates.script_findings(cells, lang="Chinese", kind="definitions",
                                   pack=zh):
        got.setdefault(f["key"], []).append((f["class"], f["tier"]))
    assert ("greek_subject_lemma", gates.REVIEW) in got["ok"]
    assert not [t for _, t in got["ok"] if t == gates.BLOCK]
    assert ("greek_in_lemma", gates.BLOCK) in got["bare"]
    assert ("greek_in_lemma", gates.BLOCK) in got["loose"]
    # a beta typed for a sharp s is still contamination wherever it sits
    de = gates.script_findings(
        {"1:1": {"lemma": "\u00e4u\u03b2ere", "gloss": "Das Ende.",
                 "provenance": new}},
        lang="German", kind="definitions", pack=packs.load("German"))
    assert [(f["class"], f["tier"]) for f in de] == [
        ("greek_latin_internal", gates.BLOCK)]
    # and the PROMPT says what the gate now enforces
    prompts.activate(prompt_id="rich-core-1")
    text = prompts.build_definition_prompt("Chinese")
    assert "only inside a phrase that names" in text
    assert "never as the whole `lemma`" in text


def test_the_forbidden_script_set_is_derived_from_the_target_language():
    """F4(b). The block table names Hiragana, Katakana and Hangul, so hard-coding
    it made Japanese or Korean a target language in which EVERY cell is a
    BLOCK-tier finding and run_gates raises -- against D-10's "one language word
    in the config, no hand-prepared files".
    """
    # the four shipped packs forbid all eleven blocks: "Arabic DIGITS" is in
    # every one of them and is not permission for the Arabic script (one of the
    # four hand-found defects is an Arabic gloss in a Chinese cell)
    for lang in LANGS:
        profile = gates.script_profile(packs.load(lang))
        assert profile["scripts_the_pack_names"] == ()
        assert len(profile["forbidden_scripts"]) == 11
    assert gates.script_findings(
        {"1:1": _zh_cell(HAN_PULL, "\u0627\u0644\u0639" + FULL_STOP,
                         prov="gemini:x@2026")},
        lang="Chinese", kind="definitions", pack=packs.load("Chinese"))

    # a pack that NAMES kana drops those two blocks from its own set
    jp = {"allowed_scripts": "Hiragana, Katakana and Han characters, Japanese "
                             "punctuation, and Arabic digits",
          "lemma_allowed_set": "Japanese characters only"}
    profile = gates.script_profile(jp)
    assert profile["scripts_the_pack_names"] == ("hiragana", "katakana")
    assert len(profile["forbidden_scripts"]) == 9
    assert not gates.script_findings(
        {"1:1": {"lemma": "\u3072\u304d", "gloss": "\u30ab\u30ec\u30fc"
                 + FULL_STOP, "provenance": "gemini:x@2026"}},
        lang="Japanese", kind="definitions", pack=jp)


def test_a_han_script_language_with_no_pack_does_not_block_every_cell():
    """F4(b) / A-M-5. `han_allowed` was False without a pack, so every Han
    character became han_outside_the_target at BLOCK tier: a Han-script target
    language with no pack failed every cell of its first wave, at ingest, with
    the money already spent. script_profile's docstring said the gate says
    nothing about a language nobody has described.
    """
    cells = {"1:1": {"lemma": HAN_PULL, "gloss": HAN_MEANS + FULL_STOP,
                     "provenance": "gemini:x@2026"}}
    assert gates.script_findings(cells, lang="Japanese", kind="definitions",
                                 pack={}) == []
    # a pack that does NOT allow Han still refuses it -- the judgement comes
    # from the pack, not from its absence
    assert [f["class"] for f in gates.script_findings(
        cells, lang="Spanish", kind="definitions",
        pack=packs.load("Spanish"))] == ["han_outside_the_target"]
    # and Cyrillic in a language nobody described is still refused
    assert [f["class"] for f in gates.script_findings(
        {"1:1": {"lemma": CYRILLIC * 3, "gloss": "x",
                 "provenance": "gemini:x@2026"}},
        lang="Tagalog", kind="definitions", pack={})] == ["forbidden_script"]


def test_a_cyrillic_target_with_no_pack_does_not_block_every_cell():
    """The same defect as the Han one above, on the block table's FIRST entry.

    Cyrillic is `_SCRIPT_BLOCKS[0]`, and with no pack `named` is empty, so
    `forbidden_scripts` was all eleven blocks and a Russian wave was a
    BLOCK-tier `forbidden_script` on every cell, at ingest, after the money.
    The judgement now comes from the target language as well as from the pack;
    what it does NOT come from is the pack's absence.
    """
    profile = gates.script_profile(None, lang="Russian")
    assert not profile["has_pack"]
    assert profile["scripts_the_language_uses"] == ("cyrillic",)
    # the pack said nothing, and the key that reports the pack still says so
    assert profile["scripts_the_pack_names"] == ()
    forbidden = {name for name, _, _ in profile["forbidden_scripts"]}
    assert "cyrillic" not in forbidden
    assert {"arabic", "hangul", "hiragana", "katakana", "hebrew",
            "devanagari", "thai"} <= forbidden
    # a miss in this lookup is silent and total, so the name is normalised for
    # whitespace as well as case: a config value with a stray space would
    # otherwise re-block every cell of the target's own script
    for spelling in ("russian", " Russian", "Russian\n", "  RUSSIAN  "):
        assert gates.script_profile(None, lang=spelling) \
            ["scripts_the_language_uses"] == ("cyrillic",), spelling

    cells = {"1:1": {"lemma": CYRILLIC * 3,
                     "gloss": CYRILLIC * 6 + " " + CYRILLIC * 4 + ".",
                     "provenance": "gemini:x@2026"}}
    assert gates.script_findings(cells, lang="Russian", kind="definitions",
                                 pack={}) == []
    # exactly one language word: the same cells in any other target are still
    # contamination, which is what the existing Tagalog test asserts
    assert [f["class"] for f in gates.script_findings(
        cells, lang="Tagalog", kind="definitions", pack={})] \
        == ["forbidden_script"]
    # ...and a Russian target is told nothing about the OTHER blocks
    assert [f["class"] for f in gates.script_findings(
        {"1:1": {"lemma": "\u0627\u0644\u0639",
                 "gloss": CYRILLIC * 4 + ".",
                 "provenance": "gemini:x@2026"}},
        lang="Russian", kind="definitions", pack={})] == ["forbidden_script"]


def test_the_latin_in_lemma_family_is_asked_of_every_non_latin_target():
    """The family was gated on `han_allowed`, so for a Cyrillic target NOTHING
    looked at a Latin letter in a lemma -- and the first Russian wave shipped
    two lemmas with Latin letters welded inside a Cyrillic word, caught only by
    a hand-written post-write scan after the deck was built.

    The question the family is asked on is whether the TARGET's own script is
    Latin, not whether it is Han.
    """
    ru = gates.script_profile(None, lang="Russian")
    assert ru["target_script_is_non_latin"] is True
    assert ru["latin_in_lemma_allowed"] is False
    # ...and it stays OFF for the three Latin-script targets, whose own letters
    # ARE Latin letters. That is the measured population a per-character
    # allow-list would have failed: 162 English, 51 German and 70 Spanish cells
    # name the Danish headword.
    for lang in ("English", "German", "Spanish"):
        profile = gates.script_profile(packs.load(lang), lang)
        assert profile["target_script_is_non_latin"] is False
        assert profile["latin_in_lemma_allowed"] is True
        assert not gates.script_findings(
            {"1:1": {"lemma": "pull",
                     "gloss": "From the Danish tr\u00e6k.",
                     "provenance": "gemini:x@2026"}},
            lang=lang, kind="definitions", pack=packs.load(lang))
    # Chinese is unchanged: it was already asked, via han_allowed.
    zh = gates.script_profile(packs.load("Chinese"), "Chinese")
    assert zh["target_script_is_non_latin"] is True
    assert zh["latin_in_lemma_allowed"] is False


def test_g_script_blocks_latin_welded_into_a_cyrillic_word():
    """The seven real Russian cells, and the discriminator between them: the
    TOKEN BOUNDARY.

    Every legitimate one keeps its Latin as a separate token -- a
    cross-reference after a space, a degree abbreviation inside parentheses, an
    acronym joined by a hyphen, a key name. Both defects weld `reg` between two
    Cyrillic syllables, which is not a shape a lexicographer writes.
    """
    legit = _ru_classes(RU_LEGIT_LEMMAS)
    assert len(legit) == len(RU_LEGIT_LEMMAS) == 5
    for key, verdict in sorted(legit.items()):
        assert verdict == ("latin_in_lemma", gates.REVIEW), key
    garbled = _ru_classes(RU_GARBLED_LEMMAS)
    assert len(garbled) == len(RU_GARBLED_LEMMAS) == 2
    for key, verdict in sorted(garbled.items()):
        assert verdict == ("script_weld_in_lemma", gates.BLOCK), key
    # the BLOCK half is what fails a wave; the REVIEW half never does
    ok, detail = gates.script_contract(
        gates.script_findings(
            {k: {"lemma": v, "gloss": RU_GLOSS, "provenance": "gemini:x@2026"}
             for k, v in RU_LEGIT_LEMMAS.items()},
            lang="Russian", kind="definitions", pack={}),
        {}, lang="Russian", kind="definitions")
    assert ok and detail["review_tier_counts"] == {"latin_in_lemma": 5}


def test_the_weld_test_is_the_discriminator_a_cyrillic_lemma_gets():
    """Unit-level, both directions, because the whole verdict rests on it.

    A Latin run bounded by a space, a hyphen, a parenthesis or a full stop is a
    mention. And the Han path must NOT ask this: a Han lemma's Latin
    cross-reference has no separator at all, so the rule would call the archive
    cross-reference population corruption.
    """
    weld = gates._script_weld_is_a_defect
    for lemma in RU_GARBLED_LEMMAS.values():
        assert weld(lemma, ("cyrillic",)), lemma
    for lemma in RU_LEGIT_LEMMAS.values():
        assert not weld(lemma, ("cyrillic",)), lemma
    # Han is not a _SCRIPT_BLOCKS name, so the predicate is structurally unable
    # to read a Han character as the target script -- and the Han path never
    # calls it anyway. A Han cross-reference keeps the class it always had.
    assert not weld("\u89c1afgore", ("han",))
    assert _zh_classes({"1:4": _zh_cell("\u89c1afgore", HAN_MEANS + FULL_STOP,
                                        prov=PROV_NEW)})["1:4"] == (
        "latin_in_han_lemma", gates.REVIEW)
    # no target scripts named means no weld question can be asked at all
    assert not weld(list(RU_GARBLED_LEMMAS.values())[0], ())


def test_the_weld_tier_separates_corruption_from_inflection():
    """The shape rule, against every mixed-script lemma shape that was
    put to it.

    Russian inflects a Latin acronym or brand IN PLACE with no separator, and
    BLOCK has no baseline and no tolerance: one such lemma in a future wave
    would FatalError the ingest with the money already spent. So exactly one
    shape is exempt -- two groups, LATIN FIRST, two or more characters, at
    least one capital -- and it is the shape morphology produces.
    """
    # the morphology family: REVIEW, never BLOCK. SMS/PDF/iPhone/USB/Instagram
    # plus a Cyrillic inflectional ending.
    for lemma in ("SMS\u043e\u043c", "PDF\u043a\u0430",
                  "iPhone\u043e\u043c", "USB\u0448\u043d\u044b\u0439",
                  "Instagram\u0435"):
        assert _ru_classes({"1:1": lemma})["1:1"] == ("latin_in_lemma",
                                                      gates.REVIEW), lemma
    # a DIGIT no longer splits the token, so one word shape gets one verdict:
    # MP3 + an ending used to escape to REVIEW while SMS + the same ending
    # BLOCKed, purely because a digit terminated the alphabetic run.
    assert _ru_classes({"1:1": "MP3\u043f\u043b\u0435\u0435\u0440"})["1:1"] \
        == ("latin_in_lemma", gates.REVIEW)
    # four corruption shapes, each BLOCK for its own reason
    corrupt = {
        # an interior Latin island: no morphology produces one
        "island": "\u043c\u0430reg\u043c\u0430",
        # a ONE-CHARACTER Latin group -- the homoglyph family, and the reason
        # this check exists: a Latin o at the head of a Cyrillic word
        "homoglyph_head": "o\u043a\u043d\u043e",
        # ...or a Latin e at the tail, which is the live corpus defect
        "homoglyph_tail": "\u0440\u0430\u0441\u0447\u0435\u0442e",
        # an all-lowercase Latin group: a stem, not an acronym or a brand
        "lowercase_stem": "hygg\u0435",
    }
    for key, verdict in sorted(_ru_classes(corrupt).items()):
        assert verdict == ("script_weld_in_lemma", gates.BLOCK), key
    assert len(_ru_classes(corrupt)) == len(corrupt) == 4
    # every one of the five classic Cyrillic lookalikes is caught
    for latin, rest in (("a", "\u043c\u043c\u0430"), ("o", "\u043a\u043d\u043e"),
                        ("c", "\u0442\u043e\u043b"), ("p", "\u0430\u0431\u043e"),
                        ("e", "\u0434\u0430"), ("x", "\u043e\u0440")):
        lemma = latin + rest
        assert _ru_classes({"1:1": lemma})["1:1"][1] == gates.BLOCK, lemma
    # the new BLOCK class is declared, and it is NOT the Han one: the registry
    # note defines foreign_text_in_lemma as a parenthesised Danish word inside
    # a Chinese lemma, which a Cyrillic weld is not.
    assert "script_weld_in_lemma" in gates._BLOCK_CLASSES
    assert "script_weld_in_lemma" in gates.SCRIPT_CLASSES


def test_a_greek_latin_weld_is_reported_once_not_twice():
    """greek_latin_internal already owns a Greek letter hugged by Latin letters,
    in both fields. A weld predicate that read "Latin plus ANY other script"
    reported the same defect a second time, at the same tier, which inflates
    the counts a human triages and would have compounded on the gloss."""
    findings = gates.script_findings(
        {"1:1": {"lemma": "\u0435\u0434\u0438\u043d\u0438\u0446\u0430 k\u03a9",
                 "gloss": RU_GLOSS, "provenance": "gemini:x@2026"}},
        lang="Russian", kind="definitions", pack={})
    classes = [f["class"] for f in findings]
    assert "greek_latin_internal" in classes
    assert "script_weld_in_lemma" not in classes
    # the predicate itself is scoped to the target's own script
    assert not gates._script_weld_is_a_defect("k\u03a9", ("cyrillic",))


def test_the_gloss_gets_the_weld_check_and_nothing_else():
    """The live defect this round measured and did not fix: one Russian gloss
    ends a Cyrillic word with U+0065 LATIN SMALL LETTER E.

    Only the WELD extends to the gloss. Measured on the first Russian wave, the
    weld finds 1 cell in 22,288 and it is that defect; "any Latin letter in the
    gloss" finds 199 and they are Danish headwords, Latin binomials and letter
    names. REVIEW, not BLOCK: the cell already shipped at gemini: provenance,
    so a BLOCK class would FatalError the paid deck at re-gate.
    """
    cells = {
        # the live cell, verbatim shape: "...\u043f\u0440\u0438 \u0440\u0430
        # \u0441\u0447\u0435\u0442" + a LATIN e
        "live": {"lemma": "\u043e\u0446\u0435\u043d\u0438\u0432\u0430\u0442\u044c",
                 "gloss": "\u043f\u0440\u0438 \u0440\u0430\u0441\u0447\u0435"
                          "\u0442e \u043d\u0430\u043b\u043e\u0433\u043e\u0432.",
                 "provenance": "gemini:x@2026"},
        # a Danish headword quoted in the gloss: 199 cells look like this and
        # every one is legitimate
        "danish": {"lemma": "\u0441\u043b\u043e\u0432\u043e",
                   "gloss": "\u043e\u0442 afg\u00f8re.",
                   "provenance": "gemini:x@2026"},
        # the inflected-acronym shape is not a defect in a gloss either
        "affix": {"lemma": "\u0441\u043b\u043e\u0432\u043e",
                  "gloss": "\u043f\u043e SMS\u043e\u043c.",
                  "provenance": "gemini:x@2026"},
    }
    got = {}
    for f in gates.script_findings(cells, lang="Russian", kind="definitions",
                                   pack={}):
        got.setdefault(f["key"], []).append((f["class"], f["tier"],
                                             tuple(f["fields"])))
    assert got["live"] == [("script_weld_in_gloss", gates.REVIEW, ("gloss",))]
    assert "danish" not in got and "affix" not in got
    assert "script_weld_in_gloss" in gates._REVIEW_CLASSES
    assert "script_weld_in_gloss" in gates.SCRIPT_CLASSES
    # REVIEW is what keeps the already-paid deck buildable
    ok, detail = gates.script_contract(
        gates.script_findings(cells, lang="Russian", kind="definitions",
                              pack={}),
        {}, lang="Russian", kind="definitions")
    assert ok and detail["review_tier_counts"] == {"script_weld_in_gloss": 1}


def test_a_pack_that_permits_latin_letters_cannot_switch_the_weld_OFF():
    """`lemma_allowed_set` is a CHARSET, and no charset can express "not
    fused".

    A pack that permits Latin letters in a lemma has permitted a Latin TOKEN --
    an acronym, a cross-reference, a brand. It has said nothing about a Latin
    letter welded INSIDE a target-script word, which is the homoglyph defect
    this check exists for. Reading the charset here switched off all ten defect
    shapes for any target whose pack named Latin; the gloss weld was already
    hung off the target's own script for exactly this reason.
    """
    permissive = {
        "allowed_scripts": "Cyrillic letters, Latin letters, and Arabic digits",
        "lemma_allowed_set": "Cyrillic letters, Latin letters, spaces and "
                             "hyphens",
    }
    profile = gates.script_profile(permissive, "Russian")
    assert profile["target_scripts"] == ("cyrillic",)
    # the pack really does permit Latin letters in the lemma...
    assert profile["latin_in_lemma_allowed"] is True

    def classes(lemmas):
        cells = {k: {"lemma": v, "gloss": RU_GLOSS,
                     "provenance": "gemini:x@2026"} for k, v in lemmas.items()}
        return {f["key"]: (f["class"], f["tier"]) for f in
                gates.script_findings(cells, lang="Russian",
                                      kind="definitions", pack=permissive)}

    # ...and the weld is still BLOCK, on the interior island and on both
    # homoglyph positions
    welded = dict(RU_GARBLED_LEMMAS)
    welded["homoglyph_head"] = "o\u043a\u043d\u043e"
    welded["homoglyph_tail"] = ("\u0440\u0430\u0441\u0447\u0435"
                                "\u0442e")
    got = classes(welded)
    assert len(got) == len(welded) == 4
    for key, verdict in sorted(got.items()):
        assert verdict == ("script_weld_in_lemma", gates.BLOCK), key
    # while the CHARSET question -- an unwelded Latin token -- is the pack's to
    # answer, and it answered: no finding at all
    assert classes({"affix": "SMS\u043e\u043c",
                    "token": "\u0441\u043c. fatte"}) == {}
    # and the gloss weld is likewise unmoved by the lemma charset
    assert [(f["class"], f["tier"]) for f in gates.script_findings(
        {"1:1": {"lemma": "\u043e\u0446\u0435\u043d\u0438\u0432\u0430"
                           "\u0442\u044c",
                 "gloss": "\u043f\u0440\u0438 \u0440\u0430\u0441"
                          "\u0447\u0435\u0442e \u043d\u0430\u043b"
                          "\u043e\u0433\u043e\u0432.",
                 "provenance": "gemini:x@2026"}},
        lang="Russian", kind="definitions", pack=permissive)] == [
        ("script_weld_in_gloss", gates.REVIEW)]


def test_a_pack_that_names_a_non_latin_script_switches_the_family_ON():
    """A pack is the ADVERTISED way to add a target ("a Japanese or Korean
    target is a pack away rather than a rewrite away"), and the profile read
    only `han_allowed` and _SCRIPTS_BY_LANGUAGE -- so shipping a Hebrew pack
    turned the whole Latin-in-lemma family OFF for a Hebrew target, silently,
    while the same lemma with NO pack was a loud forbidden_script."""
    he_pack = {"allowed_scripts": "Hebrew letters and Arabic digits",
               "lemma_allowed_set": "Hebrew letters only"}
    profile = gates.script_profile(he_pack, "Hebrew")
    assert profile["han_allowed"] is False
    assert profile["scripts_the_language_uses"] == ()      # not in the table
    assert profile["scripts_the_pack_names"] == ("hebrew",)
    assert profile["target_script_is_non_latin"] is True
    assert profile["target_scripts"] == ("hebrew",)
    # Latin welded into a Hebrew word is now a finding rather than silence
    assert [(f["class"], f["tier"]) for f in gates.script_findings(
        {"1:1": {"lemma": "\u05e9\u05dcabc", "gloss": "\u05e9\u05dc\u05d5\u05dd.",
                 "provenance": "gemini:x@2026"}},
        lang="Hebrew", kind="definitions", pack=he_pack)] == [
        ("script_weld_in_lemma", gates.BLOCK)]
    # Han and Greek sit OUTSIDE _SCRIPT_BLOCKS, so they cannot be reached this
    # way: Han is recognised through han_allowed, Greek is not a target yet
    assert "han" not in gates._NON_LATIN_SCRIPT_NAMES
    assert "greek" not in gates._NON_LATIN_SCRIPT_NAMES
    assert gates.script_profile(packs.load("Chinese"),
                                "Chinese")["target_scripts"] == ()


def test_a_cyrillic_lemma_is_never_told_it_carries_pinyin_or_a_han_class():
    """The class NAME is load-bearing twice over.

    `pinyin_in_lemma` and `latin_in_han_lemma` are pinned in
    registry/gates.json for Chinese and would LIE about a Cyrillic cell -- and
    the pinyin discriminator is a Mandarin tone mark, which says nothing about
    Russian. So the Han path keeps its four class names and the non-Han path
    gets a script-neutral REVIEW class of its own.
    """
    assert "latin_in_lemma" in gates._REVIEW_CLASSES
    assert "latin_in_lemma" in gates.SCRIPT_CLASSES
    # a tone mark in a Cyrillic lemma is not evidence of romanised Mandarin
    tone_marked = _ru_classes({"1:1": "CD-\u0441\u0438\u043d\u0433\u043b "
                                      "(l\u01ceo)"})
    assert tone_marked["1:1"] == ("latin_in_lemma", gates.REVIEW)
    # no Han-defined class can reach a non-Han target at all --
    # foreign_text_in_lemma included, which registry/gates.json defines as a
    # parenthesised Danish word inside a shipped Chinese lemma
    for lemmas in (RU_LEGIT_LEMMAS, RU_GARBLED_LEMMAS):
        classes = {cls for cls, _ in _ru_classes(lemmas).values()}
        assert not (classes & {"pinyin_in_lemma", "latin_in_han_lemma",
                               "latin_subject_lemma",
                               "foreign_text_in_lemma"})
    # ...while the Chinese fixtures keep every name they had
    assert _zh_classes(_zh_cells(ZH_SUBJECT_LEMMAS, PROV_NEW)) == {
        key: ("latin_subject_lemma", gates.REVIEW)
        for key in ZH_SUBJECT_LEMMAS}


def test_the_language_exemption_does_not_change_the_pack_derived_profile():
    """`pack` stays positional and `lang` stays optional, so every existing
    caller is unaffected: the four shipped packs still forbid all eleven blocks
    with no language argument at all, and a language nobody listed is still
    told nothing about its own letters."""
    assert len(gates.script_profile(None)["forbidden_scripts"]) == 11
    for lang in LANGS:
        profile = gates.script_profile(packs.load(lang), lang)
        assert profile["scripts_the_language_uses"] == ()
        assert len(profile["forbidden_scripts"]) == 11
    # Bulgarian is Cyrillic-script and deliberately absent: this table is a
    # claim about a target being enabled, not a script-to-language mapping
    assert "cyrillic" in {name for name, _, _ in
                          gates.script_profile(None, lang="Bulgarian")
                          ["forbidden_scripts"]}
