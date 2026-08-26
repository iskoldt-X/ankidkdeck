"""Prompt assembly. ONE function per prompt kind, selected by `prompt_id`.

Two variants ship:

    v4-frozen    LEAN. The frozen V4 core and nothing else. THE DEFAULT.
    rich-core-1  RICH. The core plus the eight append-only definition blocks
                 and the two expression blocks.

LEAN is the default on purpose, and not because RICH is unfinished. The measured
constant that prices a wave -- thinking tokens per request -- is only valid for
the prompt it was measured on (consumption rule 6). It was measured on the
frozen prompt. Changing `prompt_id` therefore refuses every `--confirm-spend`
until someone runs the A/B of patch plan 4.4 and rebases the measurement, which
is the intended behaviour and not an obstacle: see tools/prompt_thinking_ab.py.

An unknown `prompt_id` is a FatalError rather than a fallback to LEAN. Serving
LEAN under an id that says RICH is precisely the "measure cheap, spend rich"
failure that rule 6 exists to prevent, and a fallback would make it silent.

`target_tokens` assembles blocks in order while they fit. Because the order of
`DEFINITION_BLOCKS` is the order of the text, every prompt this produces is a
byte prefix of the full RICH prompt, so a size cap can never produce a variant
that has to be maintained separately.
"""

import math

from ..util import FatalError
from . import blocks as _blocks
from . import core, packs

LEAN = "lean"
RICH = "rich"

DEFAULT_PROMPT_ID = "v4-frozen"

# prompt_id -> (variant, definition blocks, expression blocks).
# A new row here is a new prompt family: it changes prompt_sha256, which
# invalidates the measured thinking constant, which blocks spending until a new
# measurement exists. That chain is the point.
PROMPT_IDS = {
    "v4-frozen": (LEAN, (), ()),
    "rich-core-1": (RICH, _blocks.BLOCK_NAMES,
                    tuple(n for n, _ in _blocks.EXPRESSION_BLOCKS)),
}

# The order patch plan 4.4 wants the blocks measured in, by risk: no-risk and
# negative-risk blocks first, the two medium-thinking-risk blocks last. These
# are BLOCK SETS for the A/B, not shipping variants -- a stage is only a byte
# prefix of RICH when its set is a prefix of DEFINITION_BLOCKS, which stage 3
# alone is.
#
# CONSUMED, not recorded: activate(ramp_stage=...) makes a stage the
# process-wide block set, `tools/prompt_thinking_ab.py --arms stage1,stage2`
# measures them, and criterion (c) is then reported per arm -- which is the
# "per-block attribution: which block earned its tokens" half of 4.4 (c).
# It used to be a constant with one shape test and no reader at all, which is
# the same "a review gate that cannot change behaviour" failure this file's own
# tests are written against.
RAMP_STAGES = (
    ("stage1", ("script", "precedence", "length", "renderings")),
    ("stage2", ("script", "pos", "reading", "renderings", "length",
                "precedence")),
    ("stage3", _blocks.BLOCK_NAMES),
)

RAMP_STAGE_NAMES = tuple(name for name, _ in RAMP_STAGES)
_RAMP_BY_NAME = dict(RAMP_STAGES)

_ACTIVE = {"prompt_id": DEFAULT_PROMPT_ID, "cfg": None, "ramp_stage": None}
_PACK_MEMO = {}
_TEXT_MEMO = {}


def activate(cfg=None, prompt_id=None, ramp_stage=None) -> str:
    """Point the builders at a run's config and prompt_id. Returns the id.

    Called once per run, before the first request, by whoever knows the config.
    Everything downstream -- the wire, the bill's prompt_sha256 block, doctor's
    printout, the cache key, the ledger row -- then reads one string, because
    they all go through s42's `_SYSTEM_PROMPTS`.

    Not calling it is safe: the default is the frozen prompt, which is the
    prompt every measured constant on disk was measured on.

    `ramp_stage` selects one row of RAMP_STAGES as the process-wide block set.
    It exists for the 4.4 per-block attribution and for nothing else: a stage
    is a MEASUREMENT arm, not a shippable variant, because only stage 3 is a
    byte prefix of RICH. Putting it in the activated state rather than passing
    block names down to each call site is deliberate -- CallContext.request()
    asserts that the system instruction is system_prompt(kind, lang), so an arm
    has to be a property of the process to be measurable at all.
    """
    if cfg is not None:
        _ACTIVE["cfg"] = cfg
        if prompt_id is None:
            prompt_id = getattr(cfg, "prompt_id", None)
    if prompt_id:
        variant_for(prompt_id)                 # validate before storing
        _ACTIVE["prompt_id"] = prompt_id
    if ramp_stage is not None:
        if ramp_stage not in _RAMP_BY_NAME:
            raise FatalError(
                "unknown ramp stage %r. Known stages: %s. A stage is one row of "
                "prompts.builder.RAMP_STAGES, the risk order patch plan 4.4 "
                "measures the blocks in." % (ramp_stage,
                                             ", ".join(RAMP_STAGE_NAMES)))
        _ACTIVE["ramp_stage"] = ramp_stage
    _TEXT_MEMO.clear()
    _PACK_MEMO.clear()
    return _ACTIVE["prompt_id"]


def reset() -> None:
    """Back to packaged packs and the frozen prompt. For tests."""
    _ACTIVE["prompt_id"] = DEFAULT_PROMPT_ID
    _ACTIVE["cfg"] = None
    _ACTIVE["ramp_stage"] = None
    _TEXT_MEMO.clear()
    _PACK_MEMO.clear()


def active_prompt_id() -> str:
    return _ACTIVE["prompt_id"]


def active_ramp_stage():
    return _ACTIVE["ramp_stage"]


def ramp_stage_blocks(stage: str) -> tuple:
    """The definition block names one ramp stage assembles."""
    if stage not in _RAMP_BY_NAME:
        raise FatalError("unknown ramp stage %r (known: %s)"
                         % (stage, ", ".join(RAMP_STAGE_NAMES)))
    return tuple(_RAMP_BY_NAME[stage])


def effective_prompt_id(lang: str, prompt_id: str | None = None) -> str:
    """The prompt identity that a CELL of `lang` was actually translated under.

    `prompt_id` alone names the FAMILY -- which blocks are assembled. It does
    not name the pack, and the pack IS the prompt: editing
    `fixed_renderings_table` changes what the model is told, changes
    prompt_sha256, and changed nothing that any artifact, gate or ledger row
    could see. Measured: a 60-character pack edit moved the Chinese rich prompt
    from sha d7d6d6e8ea2a to a951a5590d0f while prompt_id stayed rich-core-1
    and the size drifted 0.5% against a 10% tolerance, so both halves of
    consumption rule 6 passed a run that measured one prompt and sent another.

    So the identity that travels is `<prompt_id>.<pack_version>` -- for the
    RICH variants only. LEAN reads no pack (verified: a pack overlay does not
    move any of the four frozen shas), so folding a pack version into the frozen
    id would invalidate every measured constant on disk for a text that did not
    change.

    The separator is a DOT, not a plus: PROVENANCE_RE allows letters, digits,
    dot and dash inside the prompt_id slot and uses `+` as its field separator,
    and s41's PAID_PROVENANCE_RE matches on `[^+@]+` fields. A dot rides through
    both unchanged; a plus would silently stop matching paid rows.
    """
    pid = prompt_id or _ACTIVE["prompt_id"]
    if variant_for(pid) == LEAN:
        return pid
    return "%s.%s" % (pid, pack_version(lang))


def effective_prompt_ids(langs, prompt_id: str | None = None) -> dict:
    """{lang: effective_prompt_id} -- for the bill and the report."""
    return {lang: effective_prompt_id(lang, prompt_id) for lang in langs}


def variant_for(prompt_id: str) -> str:
    """LEAN or RICH for a prompt_id. FatalError on anything else."""
    row = PROMPT_IDS.get(prompt_id)
    if row is None:
        raise FatalError(
            "unknown prompt_id %r. Known ids: %s. A prompt_id that no builder "
            "recognises cannot be served: falling back to the lean prompt "
            "would let a run measure thinking on one prompt and spend on "
            "another, which is the failure consumption rule 6 exists to stop. "
            "Add a row to prompts.builder.PROMPT_IDS, or set prompt_id back."
            % (prompt_id, ", ".join(sorted(PROMPT_IDS))))
    return row[0]


def _pack(lang: str) -> dict:
    cfg = _ACTIVE["cfg"]
    where = str(getattr(cfg, "registry_local", "")) if cfg is not None else ""
    key = (lang, where)
    if key not in _PACK_MEMO:
        _PACK_MEMO[key] = packs.load(lang, cfg)
    return _PACK_MEMO[key]


def pack_version(lang: str) -> str:
    return _pack(lang).get("pack_version") or "none"


def pack_sha256(lang: str) -> str:
    """The identity of the pack TEXT, not of the version somebody typed."""
    return packs.content_sha256(lang, _ACTIVE["cfg"])


def pack_identity(langs) -> dict:
    """{lang: {"version":..., "sha256":...}} -- what the artifact declares and
    what consumption rule 6's pack half compares against."""
    return {lang: {"version": pack_version(lang), "sha256": pack_sha256(lang)}
            for lang in langs}


def estimate_tokens(text: str) -> int:
    """An OFFLINE token estimate. Not a tokenizer -- there is no offline
    tokenizer for this model, and calling the API to count is spending.

    Calibrated against the two API measurements we have: the frozen definition
    prompt is 5,134 characters / 1,135 tokens in German and 4,985 / 1,092 in
    English, i.e. 4.52 and 4.57 characters per token for this English-language
    instruction text. CJK is far denser, so pack slots written in Han
    characters are counted at 0.7 tokens per character instead; plain chars/4
    under-reports a Chinese pack by roughly a third.

    Use it for assembly decisions and for reporting a DELTA against a measured
    number. Never present it as a measurement.
    """
    cjk = sum(1 for ch in text
              if 0x3000 <= ord(ch) <= 0x9FFF or 0xFF00 <= ord(ch) <= 0xFF65)
    return int(math.ceil(cjk * 0.7 + (len(text) - cjk) / 4.52))


def _selected(available: tuple, wanted, target_tokens, base: str,
              lang: str, pack: dict, table) -> list:
    """The rendered blocks to append, in table order, honouring a size cap.

    NO PACK MEANS NO BLOCKS, whatever the prompt_id says (patch plan N-03). Not
    because the three language-independent blocks would fail to render -- they
    render fine -- but because they are not independent of the others: the rule
    precedence block's first line is "the script and orthography contract is
    absolute", and the script contract is the block that needs a pack. A prompt
    whose precedence list points at instructions that are not in it is worse
    than the frozen prompt, which is complete.

    So a brand-new target language gets exactly the text that produced 22,734
    cells in four languages, and gets it with zero hand-prepared files. The
    degradation is silent by design and visible in one place: `doctor` prints
    the pack version, which reads "none".
    """
    out = []
    if not pack:
        return out
    budget = None if target_tokens is None else target_tokens
    if budget is not None:
        budget -= estimate_tokens(base)
    for name, fn in table:
        if name not in wanted:
            continue
        text = fn(lang, pack)
        if not text:
            continue
        if budget is not None:
            cost = estimate_tokens("\n\n" + text)
            if cost > budget:
                break
            budget -= cost
        out.append(text)
    return out


def build_definition_prompt(lang: str, prompt_id: str | None = None,
                            target_tokens: int | None = None,
                            block_names=None) -> str:
    """The definition system prompt for one language.

    Depends on the LANGUAGE ONLY once a run has activated its prompt_id: no
    count, no batch size, no correction instruction. That is what makes one
    explicit cache per language possible, and test_prompt_is_constant is the
    thing that keeps it true.
    """
    pid = prompt_id or _ACTIVE["prompt_id"]
    stage = _ACTIVE["ramp_stage"] if block_names is None else None
    memo = ("definition", lang, pid, target_tokens, block_names, stage,
            str(getattr(_ACTIVE["cfg"], "registry_local", "")))
    if memo in _TEXT_MEMO:
        return _TEXT_MEMO[memo]
    _, def_blocks, _ = PROMPT_IDS[pid] if pid in PROMPT_IDS else (
        variant_for(pid), (), ())
    if stage is not None and def_blocks:
        # A ramp stage narrows the RICH block set. It cannot widen the LEAN
        # one: the frozen prompt has no blocks, and a stage that added some
        # would be serving enriched text under the frozen id.
        def_blocks = tuple(n for n in def_blocks
                           if n in ramp_stage_blocks(stage))
    base = core.definition_core(lang)
    wanted = tuple(block_names) if block_names is not None else def_blocks
    parts = _selected(_blocks.BLOCK_NAMES, wanted, target_tokens, base, lang,
                      _pack(lang), _blocks.DEFINITION_BLOCKS)
    text = base if not parts else base + "\n\n" + "\n\n".join(parts)
    _TEXT_MEMO[memo] = text
    return text


def build_expression_prompt(lang: str, prompt_id: str | None = None,
                            block_names=None) -> str:
    """The expression system prompt for one language.

    Deliberately NOT cached at the API level even in the rich variant: the rich
    expression prompt is around 500 tokens and the explicit-cache floor is
    1,024, so it would have to be padded to twice its size to qualify. Padding
    to reach a discount buys nothing (patch plan N-10).
    """
    pid = prompt_id or _ACTIVE["prompt_id"]
    memo = ("expression", lang, pid, block_names,
            str(getattr(_ACTIVE["cfg"], "registry_local", "")))
    if memo in _TEXT_MEMO:
        return _TEXT_MEMO[memo]
    if pid in PROMPT_IDS:
        expr_blocks = PROMPT_IDS[pid][2]
    else:
        variant_for(pid)
        expr_blocks = ()
    base = core.expression_core(lang)
    wanted = tuple(block_names) if block_names is not None else expr_blocks
    parts = _selected(tuple(n for n, _ in _blocks.EXPRESSION_BLOCKS), wanted,
                      None, base, lang, _pack(lang), _blocks.EXPRESSION_BLOCKS)
    text = base if not parts else base + "\n\n" + "\n\n".join(parts)
    _TEXT_MEMO[memo] = text
    return text


def build_review_prompt(lang: str, json_string: str) -> str:
    """The hand-run inspector prompt (patch plan N-02d).

    The 2025 inspector was told the allowed set was "the {lang} or English
    languages" while the generator was told "never use English in the lemma".
    An English lemma on a Chinese card was therefore reported as clean, and 20
    pinyin lemmas shipped. Both prompts now read the same two pack fields, and
    lemma and gloss are judged SEPARATELY -- which is the actual policy: the
    gloss may fall back to a concise English word, the lemma may not.
    """
    sets = packs.allowed_sets(lang, _ACTIVE["cfg"])
    return f"""
You are a meticulous language quality inspector. You are given the `lemma` and
`gloss` values of one batch.

- For every `lemma`: report any character outside {sets["lemma"]}.
- For every `gloss`: report any character outside {sets["gloss"]}.
- You MUST ignore JSON structure characters like {{, }}, " and the keys "lemma"
  and "gloss".
- If nothing is found, "contains_other_languages" must be false and
  "detected_words" must be an empty list.

Text to inspect:
---
{json_string}
---
"""


def size_report(lang: str) -> dict:
    """chars and estimated tokens for both variants of both prompts.

    The lean numbers have API measurements behind them (1,135 tokens for
    zh/de/es, 1,092 for en); the rich numbers are the lean measurement plus an
    estimate of the appended text, which is the most defensible number
    obtainable without spending.
    """
    out = {}
    for kind, fn in (("definition", build_definition_prompt),
                     ("expression", build_expression_prompt)):
        lean = fn(lang, prompt_id="v4-frozen")
        rich = fn(lang, prompt_id="rich-core-1")
        if not rich.startswith(lean):
            # Not an assert: `python -O` strips those, and this is the property
            # that makes the documented rollback free.
            raise FatalError(
                "the rich %s prompt for %s is not a byte-prefix extension of "
                "the frozen one; a size cap would then produce a variant "
                "somebody has to maintain" % (kind, lang))
        out[kind] = {
            "lean_chars": len(lean), "rich_chars": len(rich),
            "lean_tokens_estimated": estimate_tokens(lean),
            "rich_tokens_estimated": estimate_tokens(rich),
            "appended_tokens_estimated": estimate_tokens(rich[len(lean):]),
        }
    out["pack_version"] = pack_version(lang)
    out["effective_prompt_id"] = effective_prompt_id(lang)
    return out
