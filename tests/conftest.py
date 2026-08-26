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
    "thinking": {
        "THINKING_SOURCE": "derived",
        "THINKING_PER_REQUEST_LOW": {"mean": 0, "p95": 0.0, "max": 0,
                                     "n_observations": 38},
        "THINKING_PER_REQUEST_MEDIUM": {"mean": 578.7, "p95": 1042.0,
                                        "max": 1156, "n_observations": 13},
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

    def __init__(self, prompt=1135, cached=0, candidates=120, thoughts=0,
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
