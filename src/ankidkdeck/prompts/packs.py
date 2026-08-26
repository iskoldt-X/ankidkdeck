"""Per-language prompt packs: registry/prompt_packs/<Language>.json.

A pack only ENRICHES a prompt. Four languages ship one; a language with no pack
still runs, and that is the product requirement (patch plan D-10 / N-03): the
user writes one language word in the config and the whole pipeline runs with no
hand-prepared files. The builder degrades to the frozen V4 skeleton, POS falls
back to the LLM path that is already built, and caching, JSONL and wave
splitting are unaffected. A missing pack must never be an error.

Loading follows the same packaged-default-plus-work-overlay rule as every other
registry file (registry.py), with one difference: the packaged file is allowed
to be absent, because "there is no pack for Tagalog" is a normal state and not a
broken install.

Two of these fields are read by BOTH the prompt and the G-SCRIPT gate:
`lemma_allowed_set` and `gloss_allowed_set`. That is deliberate. The 2025 corpus
shipped English lemmas in Chinese cards because the generator prompt said "never
use English in the lemma" while the reviewer prompt was told English was
allowed -- two prose sentences that disagreed. One pack field cannot disagree
with itself.
"""

import hashlib
import json
import re
from importlib import resources
from pathlib import Path

from ..util import FatalError, read_json

PACK_DIR = "prompt_packs"

# A pack version is welded into provenance and into the effective prompt_id, so
# it is a closed ASCII token: the same alphabet PROVENANCE_RE allows in the
# prompt_id slot (letters, digits, dot, dash).
PACK_VERSION_RE = re.compile(r"^[A-Za-z0-9.\-]+$")

# Every slot the RICH blocks can interpolate. A pack that is missing one of
# these is not an error: the block that needs it is dropped, which is the same
# graceful degradation a language with no pack at all gets. A pack carrying a
# key that is NOT here IS an error -- it is a typo that would silently do
# nothing, and silence is what this whole module is written against.
SLOTS = (
    "allowed_scripts",
    "orthography_rules",
    "lemma_charset_rule",
    "punctuation_rule",
    "pos_vb",
    "pos_sb",
    "pos_function",
    "worked_example_output",
    "fixed_renderings_table",
    "anti_patterns",
    "length_targets",
    "lemma_allowed_set",
    "gloss_allowed_set",
    "expr_worked_example_output",
)

# Keys that carry provenance rather than prompt text.
META = ("_note", "pack_version")


def _packaged_dir():
    """The packaged prompt_packs directory, or None when the install has none.

    Addressed through the top-level package on purpose: this package's sibling
    `registry` is a module (registry.py), so files("ankidkdeck.registry")
    resolves to the module and hands back the wrong directory.
    """
    try:
        ref = resources.files("ankidkdeck").joinpath("registry", PACK_DIR)
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None
    try:
        if not ref.is_dir():
            return None
    except (FileNotFoundError, OSError):
        return None
    return ref


def _packaged(lang: str):
    """The pack that ships inside the wheel, or None if this language has none.

    TWO failure modes that used to look identical, and separating them is the
    point of this function:

      * the directory is there and has no file for `lang`. That is the NORMAL
        state of a brand-new target language (D-10 / N-03): return None and let
        the builder serve the frozen skeleton.
      * the directory itself is absent, or the file is present and unparseable.
        Both mean the install is broken, and both used to be swallowed --
        json.JSONDecodeError is a subclass of ValueError, so a pack a human had
        hand-edited into invalid JSON was indistinguishable from "no pack", and
        the builder then served LEAN under a rich prompt_id with no message
        anywhere. That is a FatalError, which is what this module's own
        docstring promised and the code did not do.
    """
    ref = _packaged_dir()
    if ref is None:
        raise FatalError(
            "the installed package carries no registry/%s directory. The four "
            "hand-reviewed prompt packs are part of the install, not optional "
            "data: without them every enriched prompt silently degrades to the "
            "frozen skeleton while prompt_id still reports the rich variant. "
            "Check that %s/*.json is tracked by git (a non-recursive .gitignore "
            "negation hid them once) and included in the wheel." % (PACK_DIR,
                                                                    PACK_DIR))
    path = ref.joinpath("%s.json" % lang)
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = fh.read()
    except (FileNotFoundError, OSError):
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise FatalError(
            "the packaged prompt pack for %s is not valid JSON (%s). This is a "
            "human-review file somebody edited by hand and got wrong, which is "
            "exactly the case where continuing quietly loses the edit: the "
            "builder would serve the frozen prompt under a rich prompt_id."
            % (lang, exc)) from exc


def available(cfg=None) -> list:
    """The language names that have a pack, packaged or local. Sorted."""
    names = set()
    ref = _packaged_dir()
    if ref is not None:
        try:
            for child in ref.iterdir():
                if child.name.endswith(".json"):
                    names.add(child.name.removesuffix(".json"))
        except (FileNotFoundError, OSError):
            pass
    local = _local_dir(cfg)
    if local is not None and local.is_dir():
        for path in local.glob("*.json"):
            names.add(path.name.removesuffix(".json"))
    return sorted(names)


def _local_dir(cfg):
    if cfg is None or not hasattr(cfg, "registry_local"):
        return None
    return Path(cfg.registry_local) / PACK_DIR


def load(lang: str, cfg=None) -> dict:
    """The pack for one language, or {} when there is none.

    An empty dict is the graceful-degradation signal and every caller must be
    able to take it. A pack that exists but is malformed is a FatalError,
    because that is a human-review file someone edited by hand and got wrong --
    exactly the case where continuing quietly loses the edit.
    """
    pack = _packaged(lang)
    local = _local_dir(cfg)
    if local is not None:
        path = local / ("%s.json" % lang)
        if path.exists():
            try:
                overlay = read_json(path)
            except ValueError as exc:
                # A bare JSONDecodeError here was the one place this module
                # raised something other than the FatalError it promises.
                raise FatalError(
                    "prompt pack overlay %s is not valid JSON (%s)"
                    % (path, exc)) from exc
            if not isinstance(overlay, dict):
                raise FatalError(
                    "prompt pack overlay %s is not a JSON object" % path)
            pack = {**(pack or {}), **overlay}
    if pack is None:
        return {}
    if not isinstance(pack, dict):
        raise FatalError("prompt pack for %s is not a JSON object" % lang)
    validate(lang, pack)
    return pack


def validate(lang: str, pack: dict) -> None:
    """Reject a pack whose keys are not slots. See SLOTS for why."""
    unknown = sorted(set(pack) - set(SLOTS) - set(META))
    if unknown:
        raise FatalError(
            "prompt pack for %s has %d key(s) that no prompt block reads: %s. "
            "A slot name that nothing interpolates is a typo that would ship "
            "silently -- fix the name or delete the key. Known slots: %s"
            % (lang, len(unknown), ", ".join(unknown), ", ".join(SLOTS)))
    if "pack_version" in pack:
        version = pack["pack_version"]
        if not isinstance(version, str):
            raise FatalError(
                "prompt pack for %s has a non-string pack_version: it goes "
                "into provenance as text" % lang)
        if not PACK_VERSION_RE.match(version):
            raise FatalError(
                "prompt pack for %s has pack_version %r, which is not a closed "
                "ASCII token (letters, digits, dot, dash). The version is "
                "welded into the effective prompt_id and into every cell's "
                "provenance string, and a provenance a later audit cannot "
                "filter on is worse than no provenance at all."
                % (lang, version))
    for slot in ("worked_example_output", "expr_worked_example_output"):
        if slot in pack and not isinstance(pack[slot], (dict, str)):
            raise FatalError(
                "prompt pack for %s: %s must be the JSON object the model is "
                "meant to emit (or a string), not %s"
                % (lang, slot, type(pack[slot]).__name__))
    for slot in ("fixed_renderings_table", "anti_patterns", "length_targets"):
        if slot in pack and not isinstance(pack[slot], (list, str)):
            raise FatalError(
                "prompt pack for %s: %s must be a list of lines (or a string), "
                "not %s" % (lang, slot, type(pack[slot]).__name__))


def pack_version(lang: str, cfg=None) -> str:
    """The pack version for provenance, or "none" when the language has no pack.

    "none" is a real, reportable state, not a hole: it is what a brand-new
    target language legitimately reports, and it belongs in provenance next to
    the model and the thinking level.
    """
    return (load(lang, cfg).get("pack_version") or "none")


def content_sha256(lang: str, cfg=None) -> str:
    """sha256 of the pack TEXT THE MODEL READS, or "" when there is no pack.

    The version string is a human's claim; this is the fact. A pack edit that
    does not bump the version -- rewriting one fixed rendering, say -- changes
    what the model is told, changes prompt_sha256, and leaves both the version
    and the 10% size band exactly where they were. Recording only the version
    would have made R6-pack-version refuse the honest case (somebody bumped) and
    pass the dishonest one (somebody did not).

    `_note` is excluded on purpose: it is the human-review contract, no block
    interpolates it, and forcing a re-measurement because a comment was improved
    is how a gate gets switched off.
    """
    pack = load(lang, cfg)
    if not pack:
        return ""
    material = {k: pack[k] for k in SLOTS if k in pack}
    material["pack_version"] = pack.get("pack_version") or "none"
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def allowed_sets(lang: str, cfg=None) -> dict:
    """{"lemma": str, "gloss": str} -- the ONE source for the generator prompt,
    the hand-run reviewer prompt and the G-SCRIPT gate.

    Falls back to a description of the frozen V4 rule when there is no pack, so
    a pack-less language still gets a coherent (if unspecific) contract instead
    of an empty string in the middle of a sentence.
    """
    pack = load(lang, cfg)
    if lang.lower() == "english":
        default_gloss = "Latin letters, ASCII punctuation, and Arabic digits"
        default_lemma = "Latin letters, spaces, and hyphens"
    else:
        default_gloss = ("characters of %s, its punctuation, Arabic digits, "
                         "and rarely a concise English word" % lang)
        default_lemma = "characters of %s only" % lang
    return {"lemma": pack.get("lemma_allowed_set") or default_lemma,
            "gloss": pack.get("gloss_allowed_set") or default_gloss}
