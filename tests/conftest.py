"""Shared test scaffolding. Everything here is offline.

Two classes of test live in this suite:

  * PURE LOGIC (urls, extract, registry, classifier, merge, gates) -- runs
    anywhere, on any checkout, with no DDO content on disk. These are the tests
    CI must keep green.
  * FIXTURE-DEPENDENT (separators, parse, results label) -- needs saved DDO
    pages, which are gitignored and never committed. Those modules skip
    themselves when the fixtures are absent.

Fixtures live in $ANKIDKDECK_FIXTURES, or work/fixtures/ under the repo root;
build them with tools/build_fixtures.py on the run host.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # src layout: the tests must run from a plain checkout, not only from an
    # installed wheel.
    sys.path.insert(0, str(SRC_DIR))

FIXTURES_DIR = Path(os.environ.get("ANKIDKDECK_FIXTURES")
                    or (REPO_ROOT / "work" / "fixtures"))
FIXTURE_MANIFEST = FIXTURES_DIR / "manifest.json"

NO_FIXTURES_REASON = (
    "DDO page fixtures are not present (%s). They are gitignored by policy; "
    "build them with tools/build_fixtures.py on the host that holds the "
    "corpus." % FIXTURE_MANIFEST)


def fixtures_available() -> bool:
    return FIXTURE_MANIFEST.exists()


def require_fixtures() -> dict:
    """Module-level guard for the fixture-dependent test files."""
    if not fixtures_available():
        pytest.skip(NO_FIXTURES_REASON, allow_module_level=True)
    import json
    with open(FIXTURE_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


requires_fixtures = pytest.mark.skipif(not fixtures_available(),
                                       reason=NO_FIXTURES_REASON)


@pytest.fixture
def fixtures_env(monkeypatch):
    """For tests that run a full stage-70 export.

    G-SEP is an EXPORT gate now, and "fixtures unavailable" is a FAILURE there
    on purpose: a release host that cannot check the separator table has not
    checked it. So a test that drives s70_export.run() to completion needs the
    fixture set, and points the gate at it through the same environment variable
    a release build would use.
    """
    if not fixtures_available():
        pytest.skip(NO_FIXTURES_REASON)
    monkeypatch.setenv("ANKIDKDECK_FIXTURES", str(FIXTURES_DIR))
    return FIXTURES_DIR


# --------------------------------------------------------------------------
# config / registry
# --------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    from ankidkdeck.config import Config
    c = Config(work_dir=tmp_path / "work")
    for d in (c.json_dir, c.report_dir, c.review_dir, c.audio_dir,
              c.registry_local):
        d.mkdir(parents=True, exist_ok=True)
    return c


@pytest.fixture
def registry(cfg):
    from ankidkdeck.registry import Registry
    return Registry(cfg)


@pytest.fixture
def shipped_card_keys_empty(monkeypatch):
    """The SHIPPED card_keys.json, emptied.

    Every count freeze_card_keys returns -- `added`, `total`, `unchanged`,
    `not_in_the_committed_registry` -- is relative to what the package ships,
    and G-SEED's duplicate-seed check runs over the merged view. The package
    shipped `{}` until the 2026-08-27 release refreeze and now ships 2,927 real
    rows, so a test that appends one synthetic family and asserts `total == 1`,
    or one that replays kan/kunne/khan on two synthetic articles, was quietly
    relying on the registry being empty: `11021722` (hus) and `11028611`
    (kunne, seed `kan`) are both real rows now.

    Emptying it keeps those tests about the append-only mechanism they were
    written for. The shipped table itself is covered by its own tests and by
    the stage-30 gates on the real corpus, which is where it belongs.

    Patches registry._package_default -- both what Registry() reads and what
    freeze_card_keys re-reads for the uncommitted count.
    """
    from ankidkdeck import registry as _registry
    real = _registry._package_default
    monkeypatch.setattr(
        _registry, "_package_default",
        lambda name: {} if name == "card_keys.json" else real(name))


@pytest.fixture
def registry_empty_card_keys(cfg, shipped_card_keys_empty):
    """A Registry built while `shipped_card_keys_empty` is in force.

    Depending on that fixture rather than being listed beside it is deliberate:
    the patch has to be applied BEFORE Registry() reads the file, and fixture
    argument order is a weak thing to hang that on.
    """
    from ankidkdeck.registry import Registry
    return Registry(cfg)


# --------------------------------------------------------------------------
# the refreeze signature, faked for every test
# --------------------------------------------------------------------------
#
# G-SCOPE-FROZEN is wired into every confirmed path (stage 42 run and review,
# stage 50), and it REFUSES on a real checkout today: the packaged
# registry/card_keys.json is `{}` and the refreeze -- 22 guid_seed reselections
# plus three alias merges -- has not happened. That refusal is the correct
# behaviour for this program, so it is pinned by its own tests
# (test_a_confirmed_run_refuses_until_the_scope_is_refrozen and the unit tests in
# test_money.py) rather than by making every other test carry it.
#
# This fixture stands in for the state AFTER the freeze. It is autouse because
# the alternative is the same four lines in forty-odd tests, and it synthesises
# the stamp AT CALL TIME from words.json, because the count the stamp signs for
# is len(words.json) and each test writes its own workspace after the fixtures
# have run.
CARD_KEYS_ROWS = 2922

# Captured at import, BEFORE any fixture has patched it, so the two stand-in
# _packaged_registry functions below can still reach the genuine one for every
# file except card_keys.json.
from ankidkdeck import gates as _gates                          # noqa: E402
_REAL_PACKAGED_REGISTRY = _gates._packaged_registry


def _refrozen_stamp(cfg):
    from ankidkdeck.util import read_json
    families = read_json(Path(cfg.json_dir) / "words.json", default={}) or {}
    return ({"refrozen_at": "2026-08-27", "families": len(families),
             "card_keys_rows": CARD_KEYS_ROWS,
             "by": "tests/conftest.py refrozen fixture"},
            "tests/conftest.py refrozen fixture")


def _refrozen_packaged(name):
    if name == "card_keys.json":
        return {"card-%05d" % i: i for i in range(CARD_KEYS_ROWS)}
    return _REAL_PACKAGED_REGISTRY(name)


@pytest.fixture(autouse=True)
def refrozen(monkeypatch):
    """Pretend the release refreeze has happened. See the note above."""
    monkeypatch.setattr(_gates, "read_refreeze_stamp", _refrozen_stamp)
    monkeypatch.setattr(_gates, "_packaged_registry", _refrozen_packaged)
    return _refrozen_stamp


def _unfrozen_stamp(cfg):
    """read_refreeze_stamp with the PACKAGED stamp absent.

    The <work>/registry copy still wins when a test wrote one: that is the real
    function's first branch, and a test that signs its own stamp is exercising
    the reader, not the package. Only the packaged half is suppressed.
    """
    from ankidkdeck.util import read_json
    local = Path(cfg.registry_local) / _gates.REFREEZE_STAMP
    if local.exists():
        return read_json(local, default=None), str(local)
    return None, "%s or package registry/%s" % (local, _gates.REFREEZE_STAMP)


def _unfrozen_packaged(name):
    if name == "card_keys.json":
        return {}
    return _REAL_PACKAGED_REGISTRY(name)


@pytest.fixture
def not_refrozen(monkeypatch):
    """The state of the package BEFORE the release refreeze: no stamp, `{}`
    card_keys.

    Until 2026-08-27 this fixture put the REAL functions back, because a real
    checkout WAS in that state and the refusal these tests pin was the one it
    produced. The refreeze has now happened -- the package ships 2,927
    card_keys rows and a signed refreeze_stamp.json -- so restoring them would
    hand these tests a REFROZEN package and the refusal would either not fire
    or fire for the wrong reason (`the stamp signed for 2927 families,
    words.json has 1`, from a synthetic one-family workspace).

    The refusal still has to be pinned: an unsigned stamp is the state this
    program returns to the moment the scope moves again. So the state is
    synthesised instead, returning exactly what read_refreeze_stamp and
    _packaged_registry return when the two files are absent and empty.
    """
    monkeypatch.setattr(_gates, "read_refreeze_stamp", _unfrozen_stamp)
    monkeypatch.setattr(_gates, "_packaged_registry", _unfrozen_packaged)


# --------------------------------------------------------------------------
# the measured LLM constants, and a fake SDK
# --------------------------------------------------------------------------
#
# Nothing measured is hard-coded in the package: the stage reads
# work/probes/stats.json and REFUSES to spend if it is missing, if it was
# measured on another model, or if a key it needs is absent. So any test that
# drives a confirmed run needs the artifact, and it must carry the real measured
# values -- a test fixture with invented numbers would let a wrong formula pass.
#
# Values: work/probes/stats.json, schema 3, measured 2026-08-26 on
# gemini-3.7-flash across the three probe keys.
MEASURED_CONSTANTS = {
    "schema": 3,
    "measured_at": "2026-08-26T01:45+02:00",
    "model": "gemini-3.7-flash",
    "EXPECTED_OUTPUT": {"a": 35.964, "b": 23.07, "r2": 0.985, "points": 62},
    # Total prompt tokens ~= a*n + b for a DEFINITION request (system prompt +
    # schema + n Danish payload rows). Subtracting the system half is how the
    # bill separates "cacheable" from "uncached payload".
    "PROMPT_TOKENS_fit": {"a": 23.917, "b": 1164.2, "r2": 0.9325, "points": 75},
    "PROMPT_TOKENS_system_only": {"Chinese": 1135, "English": 1092,
                                  "German": 1135, "Spanish": 1135},
    # Measured chars/token, per language. Present in the real artifact and it was
    # missing here, which was invisible until the paid paths started sizing
    # prompts offline: billing.estimated_prompt_tokens returns None without it,
    # so anything that has to quote a prompt (the explicit-cache floor check, the
    # ranking wave's forecast, N-09 rule 6's size band) answered "unknown" in the
    # test suite and "a number" in production. Verbatim from
    # work/probes/stats.json, measured 2026-08-26.
    "CHARS_PER_TOKEN": {"Chinese": 4.314, "English": 4.322, "German": 4.295,
                        "Spanish": 4.314},
    "thinking": {
        "THINKING_SOURCE": "derived",
        "THINKING_PER_REQUEST_LOW": {"mean": 0, "p95": 0.0, "max": 0,
                                     "n_observations": 38},
        "THINKING_PER_REQUEST_MEDIUM": {"mean": 578.7, "p95": 1042.0,
                                        "max": 1156, "n_observations": 13},
        # WHAT G-THINK ADJUDICATES AGAINST, rebased from the PRODUCTION usage
        # ledger by tools/backfill_probe_stats.py --rebase-thinking-from. The
        # probe's 62 LOW observations were all zero, the gate treated that as a
        # per-row absolute, and the first real definition wave -- 3,648 paid
        # requests, mean 1.939145, p95 0, max 797, 44 rows (1.21%) non-zero,
        # every one finishReason=STOP -- could not pass it. Verbatim from
        # work/probes/stats.json, and it has to be verbatim: a fixture with
        # invented margins would let a wrong criterion pass.
        "THINKING_PER_REQUEST_LOW_BY_KIND": {
            "definition": {
                "mean": 1.939145, "p95": 0.0, "max": 797,
                "nonzero_rows": 44, "nonzero_share": 0.012061,
                "n_observations": 3648,
                "source": "production_wave batches/ocdjpkwxlz65w8pm4eapouuqmlb"
                          "rovbwd9h4, batches/u46gmzvoblokstnh5e0q5dd023z6vrri"
                          "brqs"},
            # The expression canary: 20 paid rows, one of which thought 97
            # tokens. This is the measurement the canary was paid for, and it is
            # what replaced the ranking prompt's 275 as the expression prior.
            "expression": {
                "mean": 4.85, "p95": 4.85, "max": 97,
                "nonzero_rows": 1, "nonzero_share": 0.05,
                "n_observations": 20,
                "source": "production_wave Chinese-expr-w0-00"},
        },
    },
    "budget": {"MAX_OUTPUT_FORMULA": "ceil(a*n + b) * 1.5 with NO thinking term",
               "MIN_SAFE_BUDGET": {"8": 1024}, "UNSET_BUDGET_HANGS": False},
    "wave2": {"EXPLICIT_CACHE_FLOOR": 1024, "PRODUCTION_PROMPT_TOKENS": 1135,
              # The enrichment probe: 4,564 cacheable tokens, so this is the
              # size the FORBIDDEN "rich prompt, no cache" figure is priced at.
              "W2_2_rich": {"cached": 4564, "prompt": 4779, "verdict": "PASS"}},
    # NOTE: there is deliberately no IMPLICIT_CACHE_FLOOR here. The real
    # artifact does not carry that key -- the documented 4096 is a docs value,
    # not a measurement -- and a fixture that invents it would let code depend
    # on a number no probe produced.
    "TEMPERATURE": None,
    "SCHEMA_SURFACE_VERIFIED": "responseSchema",
    "RANK_ENUM_HONOURED": True,
    # WHICH PROMPT the constants above were measured on. These two were missing
    # from this fixture while the real work/probes/stats.json has carried them
    # since the N-09 backfill, and the difference stopped mattering the moment
    # the paid paths started asking billing.assert_ready_to_spend: without them
    # every confirmed run refuses itself on R6-prompt-id, which is the correct
    # verdict for an artifact that does not say what it measured.
    "prompt_id": "v4-frozen",
    "prompt_lineage": {
        "prompt_id": "v4-frozen",
        "measured_prompt_chars": [5123, 5124],
        "size_band_basis": {"prompt_id": "v4-frozen",
                            "by_family": {"definition": [5123, 5124]}},
    },
}


@pytest.fixture
def probe_stats(cfg):
    """work/probes/stats.json in the test workspace, with the measured values."""
    from ankidkdeck.util import write_json
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)
    return MEASURED_CONSTANTS


class FakeUsage:
    """usageMetadata, SDK-shaped (snake_case attributes).

    thoughtsTokenCount is deliberately ABSENT: protobuf omits zero-valued
    fields, so the real object has no such attribute when thinking is 0, and
    that is exactly why the code derives thinking from the identity instead of
    reading the field.
    """

    # The defaults are the MEASURED model's own answer for a one-cell definition
    # request: prompt = the measured system-prompt size, candidates =
    # ceil(EXPECTED_OUTPUT.a * 1 + b) = ceil(35.964 + 23.07) = 60. They used to
    # be 1135/120, and the 120 was twice what the fit predicts -- invisible until
    # the four money gates were wired onto the interactive path, at which point
    # G-BILL correctly reported that a "healthy" one-request wave cost 18% more
    # than its own quote. A fake whose token counts contradict the measured token
    # model cannot be used to test a gate that compares the two.
    def __init__(self, prompt=1135, cached=0, candidates=60, thoughts=0,
                 tool_use=0):
        self.prompt_token_count = prompt
        self.cached_content_token_count = cached
        self.candidates_token_count = candidates
        self.tool_use_prompt_token_count = tool_use
        self.total_token_count = prompt + candidates + thoughts + tool_use


class FakeCandidate:
    def __init__(self, finish_reason="STOP"):
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, text, finish_reason="STOP", usage=None):
        self.text = text
        self.candidates = [FakeCandidate(finish_reason)]
        self.usage_metadata = usage if usage is not None else FakeUsage()


class FakeGenai:
    """A recording stand-in for google.genai. No network, no key, no spend.

    `responder(call) -> FakeResponse | str | dict` decides what comes back; the
    recorded calls carry the model, the contents and the config object the stage
    actually built, which is what makes "temperature is never sent" and
    "thinkingLevel is always sent" checkable.
    """

    def __init__(self):
        self.calls: list = []
        self.responder = None

    def respond(self, fn):
        self.responder = fn
        return fn

    def handle(self, call) -> FakeResponse:
        out = self.responder(call) if self.responder else "{}"
        if isinstance(out, FakeResponse):
            return out
        if isinstance(out, (dict, list)):
            import json as _json
            return FakeResponse(_json.dumps(out))
        return FakeResponse(out)

    @property
    def models(self) -> list:
        return [c["model"] for c in self.calls]

    @property
    def configs(self) -> list:
        return [c["config"] for c in self.calls]


@pytest.fixture
def fake_genai(monkeypatch):
    """Install the fake SDK under google.genai and hand back the recorder."""
    import sys
    import types as _types

    recorder = FakeGenai()

    class _Models:
        def generate_content(self, model=None, contents=None, config=None):
            call = {"model": model, "contents": contents, "config": config}
            recorder.calls.append(call)
            return recorder.handle(call)

    class _Client:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _Models()

    class _Config:
        def __init__(self, **kw):
            self.kwargs = kw

    class _ThinkingConfig:
        def __init__(self, **kw):
            self.kwargs = kw

    google = _types.ModuleType("google")
    genai = _types.ModuleType("google.genai")
    gtypes = _types.ModuleType("google.genai.types")
    genai.Client = _Client
    gtypes.GenerateContentConfig = _Config
    gtypes.ThinkingConfig = _ThinkingConfig
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setenv("GEMINI_API_KEYS", "fake-key")
    return recorder


@pytest.fixture
def no_sleep(monkeypatch):
    """The throttle intervals are real seconds; a test does not wait them out."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *a, **k: None)


# --------------------------------------------------------------------------
# synthetic entries: the same shape stage 20 writes, minus the HTML
# --------------------------------------------------------------------------

def make_sense(dannetid: str, definition: str, examples=(), number=None) -> dict:
    from ankidkdeck.util import NFC, sha256_str
    return {
        "dannetid": dannetid,
        "sense_path": "betydning-%s" % dannetid,
        "number": number,
        "definition": definition,
        "grammar": None,
        "examples": [{"text": t, "source_short": None, "source_full": None}
                     for t in examples],
        "eksempler": [],
        "onyms": {"synonym": [], "antonym": [], "see_also": [], "abbrev_of": []},
        "related": [],
        "src_sha": sha256_str(NFC(definition)),
        "sense_sha": sha256_str(NFC(definition)),
    }


def make_expression(dannetid: str, expression: str, definition: str = "",
                    variants=()) -> dict:
    from ankidkdeck.util import NFC, sha256_str
    senses = []
    if definition:
        senses.append({"dannetid": dannetid, "definition": definition,
                       "examples": [], "src_sha": sha256_str(NFC(definition))})
    return {"dannetid": dannetid, "expression": expression,
            "variants": list(variants), "senses": senses}


def make_entry(entry_id: str, lemma: str, *, pos_key="sb.", pos_text="substantiv",
               senses=(), expressions=(), forms=(), alt_spellings=(),
               display_headword=None, super_=None, paradigm_rows=None,
               paradigm_short=None, udtale=(), source_words=(),
               orddannelser=None, etymology=None) -> dict:
    """One entries.json row, exactly as stage 20 writes it.

    `forms` are extra flex-table cells. The two indexes are built the same way
    the parser builds them, and the difference matters: `paradigm_index` is real
    inflection cells only (bucket 2's evidence), `form_index` adds the lemma key
    and the OFFICIAL alternative spellings (stage 21's reverse index). A
    deprecated spelling (`{"form": "kan", "official": False}`) reaches neither.
    """
    from ankidkdeck.util import NFC, canonical_json, nk, sha256_str
    rows = paradigm_rows
    if rows is None:
        rows = ([{"table": 0, "row": i, "cells": [f], "slot_label": None}
                 for i, f in enumerate(forms)] if forms else [])
    alt = [a if isinstance(a, dict) else {"form": a, "official": True}
           for a in alt_spellings]
    e = {
        "entry_id": entry_id,
        # display_headword IS the lemma; the homograph index lives in `super`
        # and is rendered as a superscript by stage 70.
        "display_headword": display_headword or NFC(lemma),
        "headword_glued": NFC(lemma) + (super_ or ""),
        "lemma": NFC(lemma),
        "lemma_key": nk(lemma),
        "super": super_,
        "pos_text": pos_text,
        "pos_key": pos_key,
        "ddo_lastmod": None,
        "paradigm": {"short": paradigm_short, "rows": rows},
        "alt_spellings": alt,
        "udtale": [dict(u) for u in udtale],
        "etymology": etymology,
        "senses": list(senses),
        "expressions": list(expressions),
        "orddannelser": orddannelser or {},
        "source_words": list(source_words),
    }
    e["paradigm_index"] = sorted({nk(c) for r in rows for c in r["cells"]})
    e["form_index"] = sorted(set(e["paradigm_index"])
                             | {nk(lemma)}
                             | {nk(a["form"]) for a in alt if a.get("official")})
    e["empty"] = not e["senses"] and not e["expressions"]
    e["article_sha"] = sha256_str(canonical_json(e))
    return e


def parse_fixture_page(path, registry) -> dict:
    """{entry_id: entry} for one saved page, through the real stage-20 parser.

    bs4 is imported here so the pure-logic tests never need it.
    """
    from bs4 import BeautifulSoup

    from ankidkdeck.stages.s20_parse import parse_article, slice_articles

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    report: dict = {}
    out = {}
    for eid, scope, art in slice_articles(soup):
        out[eid] = parse_article(eid, scope, art, registry, report)
    return out


def write_workspace(cfg, entries: dict, wordlist: list, classification=None,
                    v2_querywords=None) -> None:
    """Lay down the json inputs a stage expects, so a stage can be run for
    real in a tmp dir instead of being reimplemented by the test."""
    from ankidkdeck.util import write_json
    write_json(cfg.json_dir / "entries.json", entries)
    write_json(cfg.json_dir / "wordlist.json",
               {"source": "test", "sha256": "test",
                "words": [{"rank": r, "raw": w, "word": w}
                          for r, w in wordlist]})
    if classification is not None:
        write_json(cfg.json_dir / "classification.json", classification)
    if v2_querywords is not None:
        write_json(cfg.json_dir / "legacy" / "v2_querywords.json", v2_querywords)
