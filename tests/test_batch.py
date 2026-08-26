"""The batch transport: the JSONL row, the keys, the fingerprint, the wave
splitter, the cache lifecycle, the positional reconciliation and the retry bound.

EVERY TEST HERE IS OFFLINE. The service is a fake that reads the JSONL this
package actually wrote and answers from it, so the round trip exercised is the
real one: build a request through CallContext, serialize it, "upload" it,
"download" a result file, reconcile it by position and write the cells.

The invariants being pinned were measured on a real 32-row batch job (probe wave
3) and are catalogued in specs/v3-translate-patch-plan.md section 1.6. Where a
number appears in an assertion it comes from work/probes/stats.json, not from
this file.
"""

import json
import sys
import types as pytypes

import pytest
from conftest import MEASURED_CONSTANTS, make_entry, make_expression, make_sense

from ankidkdeck.batch import jsonl as BJ
from ankidkdeck.batch import caches as BC
from ankidkdeck.batch import keys as BK
from ankidkdeck.batch import reconcile as BR
from ankidkdeck.batch import registry as BREG
from ankidkdeck.batch import transport as BT
from ankidkdeck.batch import waves as BW
from ankidkdeck.stages import s42_translate as S42
from ankidkdeck.util import FatalError, read_json, write_json

# The two keys the real work/probes/stats.json carries and the shared test
# fixture does not. Verbatim from the artifact (measured 2026-08-26): the
# transport reads both off disk rather than hard-coding them, so a fixture
# without them would make the transport refuse rather than test it.
BATCH_CONSTANTS = dict(MEASURED_CONSTANTS)
BATCH_CONSTANTS["CHARS_PER_TOKEN"] = {"Chinese": 4.314, "English": 4.322,
                                      "German": 4.295, "Spanish": 4.314}
# The measured expiry behaviour is what licenses a 1.5x TTL margin instead of
# the 3x the audit wanted: the margin was insurance against a SILENT full-price
# wave, and expiry is measurably loud and free.
BATCH_CONSTANTS["wave2"] = dict(MEASURED_CONSTANTS["wave2"],
                                CACHE_EXPIRY_BEHAVIOUR="LOUD_ERROR_403")
BATCH_CONSTANTS["wave3"] = {
    "W3_5_ENQUEUED": {"ENQUEUED_COUNTS_CACHED": False,
                      "cache_tokens": 150503, "nominal_enqueued": 3160563,
                      "rows": 21},
    "W3_1_ARCHITECTURE": {"declared_cache_tokens": 1135, "rows": 30,
                          "cached_equals_declared_rows": 30, "verdict": "GO"},
}

DECLARED_CACHE_TOKENS = 1135


@pytest.fixture
def batch_stats(cfg):
    write_json(cfg.probe_stats_path, BATCH_CONSTANTS)
    return BATCH_CONSTANTS


# --------------------------------------------------------------------------
# a fake batch service
# --------------------------------------------------------------------------

class FakeUploaded:
    def __init__(self, name):
        self.name = name


class FakeDest:
    def __init__(self, file_name):
        self.file_name = file_name


class FakeJob:
    def __init__(self, name, src, state, dest=None, stats=None):
        self.name = name
        self.src = src
        self.state = state
        self.dest = dest
        self.batch_stats = stats


class FakeCached:
    def __init__(self, name, tokens, expire_time=None):
        self.name = name
        self.usage_metadata = {"totalTokenCount": tokens}
        self.expire_time = expire_time


class FakeService:
    """files / batches / caches, backed by the JSONL the transport wrote.

    `responder(row, pos)` decides what one input row comes back as, so a test
    can inject an error object, a truncation or a short array without touching
    anything else. The default honours the count lock.
    """

    no_dest = False

    def __init__(self):
        self.uploads = {}
        self.outputs = {}
        self.jobs = {}
        self.caches_created = []
        self.caches_deleted = []
        self.caches_updated = []
        self.cache_tokens = DECLARED_CACHE_TOKENS
        self.responder = None
        self.create_raises = None
        # Injected once, then cleared: a crash between "the job is SUBMITTED on
        # the service" and "its results are on our disk" is the window a
        # 24-to-50-hour foreground drain actually dies in.
        self.download_raises = None
        # What the service reports once the job stops running. SUCCEEDED with
        # no_dest is "we could not fetch the results of a job that ran";
        # EXPIRED or FAILED is "the service says it produced nothing". Those are
        # opposite answers to "may this wave be submitted again?".
        self.terminal_state = "JOB_STATE_SUCCEEDED"
        self.polls_before_terminal = 1
        self.list_returns_existing = False
        self.cache_expired = False
        self._n = 0
        self.files = _Files(self)
        self.batches = _Batches(self)
        self.caches = _Caches(self)
        self.models = _Models(self)
        self.model_calls = []

    def next_name(self, prefix):
        self._n += 1
        return "%s/%d" % (prefix, self._n)

    def answer(self, row, pos):
        if self.responder is not None:
            out = self.responder(row, pos)
            if out is not None:
                return out
        return ok_line(row, self.cache_tokens)


class _Files:
    def __init__(self, svc):
        self.svc = svc

    def upload(self, file=None, config=None):
        name = self.svc.next_name("files")
        with open(file, encoding="utf-8") as f:
            self.svc.uploads[name] = f.read()
        return FakeUploaded(name)

    def download(self, file=None):
        if self.svc.download_raises is not None:
            exc = self.svc.download_raises
            self.svc.download_raises = None
            raise exc
        return self.svc.outputs[file].encode("utf-8")


class _Batches:
    def __init__(self, svc):
        self.svc = svc

    def create(self, model=None, src=None, config=None):
        if self.svc.create_raises is not None:
            exc = self.svc.create_raises
            self.svc.create_raises = None
            if self.svc.list_returns_existing:
                self._make(model, src)
            raise exc
        return self._make(model, src)

    def _make(self, model, src):
        rows = [json.loads(x) for x in self.svc.uploads[src].splitlines()
                if x.strip()]
        out_name = self.svc.next_name("files")
        self.svc.outputs[out_name] = "".join(
            json.dumps(self.svc.answer(row, i)) + "\n"
            for i, row in enumerate(rows))
        name = self.svc.next_name("batches")
        job = FakeJob(name, src, "JOB_STATE_PENDING",
                      None if self.svc.no_dest else FakeDest(out_name),
                      {"failed_request_count": 0, "request_count": len(rows)})
        self.svc.jobs[name] = {"job": job, "polls": 0}
        return job

    def get(self, name=None):
        record = self.svc.jobs[name]
        record["polls"] += 1
        if record["polls"] > self.svc.polls_before_terminal:
            record["job"].state = self.svc.terminal_state
        else:
            record["job"].state = "JOB_STATE_RUNNING"
        return record["job"]

    def list(self):
        return [r["job"] for r in self.svc.jobs.values()]


class _Caches:
    def __init__(self, svc):
        self.svc = svc

    def create(self, model=None, config=None):
        name = self.svc.next_name("cachedContents")
        self.svc.caches_created.append({"name": name, "config": config})
        return FakeCached(name, self.svc.cache_tokens)

    def get(self, name=None):
        if self.svc.cache_expired:
            raise RuntimeError("403 PERMISSION_DENIED: CachedContent not found "
                               "(or permission denied)")
        return FakeCached(name, self.svc.cache_tokens,
                          expire_time="2099-01-01T00:00:00Z")

    def update(self, name=None, config=None):
        if self.svc.cache_expired:
            raise RuntimeError("403 PERMISSION_DENIED: CachedContent not found "
                               "(or permission denied)")
        self.svc.caches_updated.append(name)
        return FakeCached(name, self.svc.cache_tokens)

    def delete(self, name=None):
        self.svc.caches_deleted.append(name)


class _Models:
    """Only the POS fallback reaches this: one interactive call per language."""

    def __init__(self, svc):
        self.svc = svc

    def generate_content(self, model=None, contents=None, config=None):
        self.svc.model_calls.append({"model": model, "config": config})
        props = config.kwargs["response_schema"]["properties"]
        return _Resp(json.dumps({k: "POS-%s" % k for k in props}))


class _Resp:
    def __init__(self, text):
        self.text = text
        self.candidates = [pytypes.SimpleNamespace(finish_reason="STOP")]
        self.usage_metadata = {"promptTokenCount": 400,
                              "candidatesTokenCount": 40,
                              "totalTokenCount": 440}


def n_of(row, kind=None):
    props = row["request"]["generationConfig"]["responseSchema"]["properties"]
    array = "definitions" if "definitions" in props else "fixed_expressions"
    return array, int(props[array]["minItems"])


def ok_line(row, cache_tokens, *, items=None, finish="STOP"):
    array, n = n_of(row)
    body = {array: (items if items is not None
                    else [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                          for i in range(n)])}
    if array == "definitions":
        body["headword"] = "hus"
    cached = (cache_tokens if row["request"].get("cachedContent") else 0)
    prompt = 1350 if cached else 1350
    return {"key": row["key"],
            "response": {"candidates": [{"content": {"parts": [
                {"text": json.dumps(body)}]}, "finishReason": finish}],
                "usageMetadata": {"promptTokenCount": prompt,
                                  "cachedContentTokenCount": cached,
                                  "candidatesTokenCount": 60,
                                  "totalTokenCount": prompt + 60}}}


def error_line(row, code=7, message="The caller does not have permission",
               with_key=True):
    """A failed row, in the shape the probe measured: a bare gRPC status."""
    line = {"error": {"code": code, "message": message}}
    if with_key:
        line["key"] = row["key"]
    return line


@pytest.fixture
def batch_genai(monkeypatch):
    """google.genai, faked to the surfaces the batch transport touches."""
    service = FakeService()

    class _Config:
        def __init__(self, **kw):
            self.kwargs = kw

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    gtypes = pytypes.ModuleType("google.genai.types")
    genai.Client = lambda api_key=None: service
    for name in ("UploadFileConfig", "CreateBatchJobConfig",
                 "CreateCachedContentConfig", "UpdateCachedContentConfig",
                 "GenerateContentConfig", "ThinkingConfig"):
        setattr(gtypes, name, _Config)

    class _Schema:
        """Identity stand-in. The REAL camelCase conversion is checked against
        the installed SDK in test_the_count_lock_survives_the_sdk_schema_model;
        here the row has to stay readable so the fake service can answer it."""

        @staticmethod
        def model_validate(schema):
            return _Schema._Held(schema)

        class _Held:
            def __init__(self, schema):
                self.schema = schema

            def model_dump(self, **kw):
                return self.schema

    gtypes.Schema = _Schema
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setenv("GEMINI_API_KEYS", "fake-key")
    return service


# --------------------------------------------------------------------------
# workspace helpers
# --------------------------------------------------------------------------

def _entry(entry_id="11021722", n_senses=2, n_exprs=1):
    senses = [make_sense("2100000%d" % (i + 1), "definition nummer %d" % i)
              for i in range(n_senses)]
    exprs = [make_expression("2200000%d" % (i + 1), "hus forbi %d" % i,
                             "helt forkert")
             for i in range(n_exprs)]
    return make_entry(entry_id, "hus", pos_key="sb.", pos_text="substantiv",
                      senses=senses, expressions=exprs, source_words=["hus"])


def _workspace(cfg, entries=None):
    rows = entries or [_entry()]
    write_json(cfg.json_dir / "entries.json", {e["entry_id"]: e for e in rows})
    write_json(cfg.json_dir / "words.json",
               {e["entry_id"]: {"family_id": e["entry_id"],
                                "anchor_entry_id": e["entry_id"],
                                "entry_ids": [e["entry_id"]], "freq_rank": i + 1}
                for i, e in enumerate(rows)})
    return rows


def _batch_cfg(cfg, cache=True):
    cfg.mode = "batch"
    cfg.cache_enabled = bool(cache)
    return cfg


def _todo(cfg, entries, lang="German", **kw):
    scope, _ = S42.renderable_scope(cfg, entries, {}, True)
    return S42.compute_todo(cfg, entries, {"definitions": {},
                                           "expressions": {}}, lang, scope,
                            **kw)


def _fake_request(cfg, lang="German", kind="definition", n=3, cache=None):
    """One LlmRequest through the production constructors."""
    entries = {e["entry_id"]: e for e in [_entry(n_senses=n, n_exprs=n)]}
    todo = _todo(cfg, entries, lang)
    rows = [r for r in todo if r["kind"] == kind][:n]
    ctx = S42.CallContext(cfg=cfg, pool=None, fit=(35.964, 23.07), lang=lang,
                          cache_name=cache)
    if kind == "definition":
        user = S42.definition_user_payload("hus substantiv", rows)
        schema = S42.definition_schema(len(rows))
    else:
        user = S42.expression_user_payload(lang, "hus substantiv", rows)
        schema = S42.expression_schema(len(rows))
    return ctx.request(kind, "%s batch hus" % kind, user, schema, len(rows),
                       S42.system_prompt(kind, lang))


def _identity_schema(schema):
    return schema


# ==========================================================================
# 5.2 -- the key scheme
# ==========================================================================

def test_the_key_carries_the_chunk_because_the_entry_id_alone_collides():
    """5.2 / W0-3. `def__{entry_id}` collides 42 times on the real corpus and
    `{lemma+pos}` 186 times, because _group_by_entry splits an entry with more
    than 20 rows into several REQUESTS. Reproduced here on the MECHANISM: three
    chunks of one entry."""
    chunks = range(3)
    retired_entry_scheme = {"def__%s" % "11000283" for _ in chunks}
    retired_label_scheme = {"def__%s" % "hus substantiv" for _ in chunks}
    shipped = {BK.make_key("definition", "German", "11000283", c) for c in chunks}
    assert len(retired_entry_scheme) == 1     # 3 requests, 1 key: 2 collisions
    assert len(retired_label_scheme) == 1
    assert len(shipped) == 3
    assert shipped == {"def__German__11000283__00", "def__German__11000283__01",
                       "def__German__11000283__02"}


def test_every_key_of_a_four_language_wave_is_unique_and_ascii(cfg):
    entries = {e["entry_id"]: e
               for e in [_entry("11000001", 3, 2), _entry("11000002", 1, 1)]}
    allkeys = []
    for lang in ("Chinese", "English", "German", "Spanish"):
        todo = _todo(cfg, entries, lang)
        for kind in ("definition", "expression"):
            for i, (eid, _rows) in enumerate(
                    S42._group_by_entry(todo, kind, 20)):
                allkeys.append(BK.make_key(kind, lang, eid, i))
    assert allkeys
    assert BK.validate_keys(allkeys) == allkeys
    assert all(k.isascii() and BK.KEY_RE.match(k) for k in allkeys)


def test_a_duplicate_key_is_refused_before_anything_is_uploaded():
    with pytest.raises(FatalError) as exc:
        BK.validate_keys(["def__German__1__00", "def__German__1__00"])
    assert "duplicate" in str(exc.value)


def test_a_language_that_cannot_be_a_key_segment_is_refused():
    with pytest.raises(FatalError):
        BK.language_tag("---")
    assert BK.language_tag("Brazilian Portuguese") == "BrazilianPortuguese"


# ==========================================================================
# N-07 / W0-2 -- the JSONL row shape
# ==========================================================================

def test_jsonl_field_placement(cfg):
    """W0-2, on the PRODUCTION constructors and the production writer.

    Five shapes, and the one that matters is the first: cachedContent as a
    top-level sibling of generationConfig. The wrong placement is rejected at
    submit ("no such field: 'cachedContent'"), which is the good outcome; the
    reason this test exists is that nobody knew that until it was measured.
    """
    cached = BJ.build_row("def__German__1__00",
                          _fake_request(cfg, cache="cachedContents/abc"),
                          schema_serializer=_identity_schema)
    plain = BJ.build_row("def__German__1__00", _fake_request(cfg),
                         schema_serializer=_identity_schema)
    expr = BJ.build_row("expr__German__1__00",
                        _fake_request(cfg, kind="expression"),
                        schema_serializer=_identity_schema)
    req, gen = cached["request"], cached["request"]["generationConfig"]
    assert "cachedContent" in req
    assert "cachedContent" not in gen
    assert "systemInstruction" not in req
    assert "responseSchema" in gen
    assert "thinkingConfig" in gen
    assert gen["thinkingConfig"]["thinkingLevel"] == "LOW"
    assert "maxOutputTokens" in gen
    assert "model" not in req
    assert "serviceTier" not in json.dumps(cached)
    assert "temperature" not in json.dumps(cached)
    # the uncached row DOES carry the system prompt, and it is the one from
    # system_prompt() rather than a second copy of the text
    assert plain["request"]["systemInstruction"]["parts"][0]["text"] == \
        S42.system_prompt("definition", "German")
    assert "cachedContent" not in plain["request"]
    assert "fixed_expressions" in json.dumps(
        expr["request"]["generationConfig"]["responseSchema"])
    for row in (cached, plain, expr):
        BJ.assert_batch_row_shape(row)


def test_the_row_is_the_shape_the_probe_writer_produced(cfg):
    """The reference implementation's row, transcribed, against ours.

    home-vm:~/v3run/probes/batch_jsonl.py passed 16/16 offline checks and wrote
    every line of the 32-row job that came back 30/30 cached == declared. Its
    build_row is reproduced here literally, and the production writer has to
    agree with it key for key.
    """
    req = _fake_request(cfg, cache="cachedContents/abc")
    gen = {"responseMimeType": "application/json",
           "thinkingConfig": {"thinkingLevel": "LOW"},
           "maxOutputTokens": req.max_output_tokens,
           "responseSchema": req.schema}
    reference = {"key": "def__German__1__00",
                 "request": {"contents": [{"role": "user",
                                           "parts": [{"text": req.user}]}],
                             "generationConfig": gen,
                             "cachedContent": "cachedContents/abc"}}
    ours = BJ.build_row("def__German__1__00", req,
                        schema_serializer=_identity_schema)
    assert ours == reference


def test_a_batch_row_never_carries_a_service_tier(cfg):
    """flex is a property of the INTERACTIVE surface. cfg.effective_service_tier
    is None under mode=batch by construction, so this is the belt on top of it:
    a row built from a flex request is refused rather than sent."""
    cfg.mode = "flex"
    req = _fake_request(cfg)
    assert req.service_tier == "flex"
    with pytest.raises(FatalError) as exc:
        BJ.build_row("def__German__1__00", req,
                     schema_serializer=_identity_schema)
    assert "serviceTier" in str(exc.value)


def test_a_row_with_both_a_cache_and_a_system_prompt_cannot_be_built(cfg):
    req = _fake_request(cfg)
    both = type(req)(kind=req.kind, label=req.label, user=req.user,
                     schema=req.schema, n_expected=req.n_expected,
                     max_output_tokens=req.max_output_tokens,
                     thinking_level=req.thinking_level, system=None,
                     cache_name="cachedContents/abc")
    object.__setattr__(both, "system", "SYSTEM TEXT")
    with pytest.raises(FatalError) as exc:
        BJ.build_row("k", both, schema_serializer=_identity_schema)
    assert "XOR" in str(exc.value)


def test_the_wrong_placement_is_caught_by_the_shape_assertion(cfg):
    row = BJ.build_row("def__German__1__00",
                       _fake_request(cfg, cache="cachedContents/abc"),
                       schema_serializer=_identity_schema)
    moved = json.loads(json.dumps(row))
    moved["request"]["generationConfig"]["cachedContent"] = \
        moved["request"].pop("cachedContent")
    with pytest.raises(FatalError) as exc:
        BJ.assert_batch_row_shape(moved)
    assert "cachedContent" in str(exc.value)


def test_the_count_lock_survives_the_sdk_schema_model(cfg):
    """N-13 on the WIRE, not only on our dict: minItems == maxItems == n after
    the SDK's own alias conversion, inside the row this package uploads."""
    pytest.importorskip("google.genai.types",
                        reason="the real SDK is not installed here")
    for n in (1, 3, 20):
        row = BJ.build_row("def__German__1__00", _fake_request(cfg, n=n))
        lock = row["request"]["generationConfig"]["responseSchema"][
            "properties"]["definitions"]
        assert lock["minItems"] in (n, str(n))
        assert lock["maxItems"] in (n, str(n))


def test_the_same_wave_written_twice_is_byte_identical(cfg, tmp_path):
    rows = [BJ.build_row("def__German__1__00",
                         _fake_request(cfg, cache="cachedContents/abc"),
                         schema_serializer=_identity_schema)]
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    BJ.write_jsonl(first, rows)
    BJ.write_jsonl(second, rows)
    assert first.read_bytes() == second.read_bytes()


# ==========================================================================
# 5.1 -- the fingerprint and the job registry
# ==========================================================================

def test_the_fingerprint_refuses_a_second_submission_of_the_same_wave(cfg):
    """batches.create is NOT idempotent: the same job submitted back to back was
    accepted twice, ran twice and was billed twice."""
    reg = BREG.JobRegistry(cfg)
    fp = BREG.wave_fingerprint(model="gemini-3.7-flash", prompt_id="v4-frozen",
                               lang="German", keys=["a", "b"],
                               cache_name="cachedContents/1")
    reg.plan("j1", fingerprint=fp, lang="German", kind="definition",
             model="gemini-3.7-flash", prompt_id="v4-frozen",
             cache_name="cachedContents/1", declared_cache_tokens=1135,
             cache_prompt_sha256="sha", jsonl_path="x.jsonl",
             plan=[], enqueued_tokens=10, wave=0)
    with pytest.raises(FatalError) as exc:
        reg.plan("j2", fingerprint=fp, lang="German", kind="definition",
                 model="gemini-3.7-flash", prompt_id="v4-frozen",
                 cache_name="cachedContents/1", declared_cache_tokens=1135,
                 cache_prompt_sha256="sha", jsonl_path="y.jsonl",
                 plan=[], enqueued_tokens=10, wave=0)
    assert "NOT idempotent" in str(exc.value)


def test_the_fingerprint_uses_the_effective_prompt_id_not_the_bare_one():
    """A pack change alters the prompt TEXT while cfg.prompt_id stays
    the same, so under the bare id two genuinely different waves would share a
    fingerprint and the second would be refused as a duplicate."""
    base = dict(model="gemini-3.7-flash", lang="Chinese", keys=["a"],
                cache_name=None)
    assert BREG.wave_fingerprint(prompt_id="rich-core-1.zh-1", **base) != \
        BREG.wave_fingerprint(prompt_id="rich-core-1.zh-2", **base)
    # and the cache name is part of the identity, because a recreated cache has
    # a new resource name and its rows really are different rows
    assert BREG.wave_fingerprint(prompt_id="v4-frozen", model="m",
                                 lang="German", keys=["a"],
                                 cache_name="cachedContents/1") != \
        BREG.wave_fingerprint(prompt_id="v4-frozen", model="m", lang="German",
                              keys=["a"], cache_name="cachedContents/2")
    # and a RETRY wave of the same keys is a different wave, which is the only
    # way the retry loop and the duplicate refusal can coexist
    assert BREG.wave_fingerprint(prompt_id="v4-frozen", model="m",
                                 lang="German", keys=["a"], wave=0) != \
        BREG.wave_fingerprint(prompt_id="v4-frozen", model="m", lang="German",
                              keys=["a"], wave=1)


def test_the_registry_states_walk_forward_and_survive_a_reload(cfg):
    reg = BREG.JobRegistry(cfg)
    reg.plan("j1", fingerprint="fp", lang="German", kind="definition",
             model="m", prompt_id="v4-frozen", cache_name=None,
             declared_cache_tokens=None, cache_prompt_sha256=None,
             jsonl_path="x", plan=[{"pos": 0, "key": "k", "kind": "definition",
                                    "n": 1, "cells": []}],
             enqueued_tokens=1, wave=0)
    assert reg.in_state(BREG.PLANNED)
    reg.mark_uploaded("j1", "files/1")
    reg.mark_submitted("j1", "batches/1", "JOB_STATE_PENDING")
    reg.mark_downloaded("j1", "res.jsonl", job_state="JOB_STATE_SUCCEEDED")
    # a fresh process sees "downloaded but not absorbed" and finishes the job
    again = BREG.JobRegistry(cfg)
    assert [j["job_id"] for j in again.in_state(BREG.DOWNLOADED)] == ["j1"]
    assert again.get("j1")["uploaded_file"] == "files/1"
    again.mark_recovered("j1", {"written": 1})
    assert not again.in_state(BREG.DOWNLOADED)
    assert BREG.JobRegistry(cfg).unfinished() == []


def test_the_terminal_states_are_read_from_the_enum_not_hard_coded():
    """Two spellings exist (JOB_STATE_* in the guide, BATCH_STATE_* in the API
    reference) and NEITHER list has PARTIALLY_SUCCEEDED -- that is a Vertex enum,
    and the GPCR reference implementation's succeeded_states includes it."""
    assert BREG.job_state_name("JOB_STATE_SUCCEEDED") == "SUCCEEDED"
    assert BREG.job_state_name("BATCH_STATE_EXPIRED") == "EXPIRED"
    assert BREG.is_terminal("JOB_STATE_CANCELLED")
    assert not BREG.is_terminal("JOB_STATE_RUNNING")
    assert not BREG.is_terminal("PARTIALLY_SUCCEEDED")
    assert "PARTIALLY_SUCCEEDED" not in BREG.TERMINAL_JOB_STATES


def test_the_attempt_counter_is_per_cell_and_survives_the_process(cfg):
    reg = BREG.JobRegistry(cfg)
    reg.bump_attempts(["def__German__1__00"])
    reg.bump_attempts(["def__German__1__00"])
    assert BREG.JobRegistry(cfg).attempts_for("def__German__1__00") == 2


def test_a_results_line_that_does_not_parse_is_fatal(cfg, tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"key": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(FatalError) as exc:
        BREG.read_results(path)
    assert "positional" in str(exc.value)


# ==========================================================================
# 5.3 -- positional reconciliation
# ==========================================================================

def _plan(n_rows=3, kind="definition", n=2):
    return [{"pos": i, "key": "def__German__1__%02d" % i, "kind": kind,
             "n": n, "label": "hus", "cap": 1024,
             "cells": [{"key": "c%d-%d" % (i, j), "src_sha": "s%d-%d" % (i, j)}
                       for j in range(n)]}
            for i in range(n_rows)]


def _line(key, n=2, finish="STOP", items=None):
    body = {"headword": "hus",
            "definitions": (items if items is not None
                            else [{"lemma": "L", "gloss": "G"}] * n)}
    return {"key": key,
            "response": {"candidates": [{"content": {"parts": [
                {"text": json.dumps(body)}]}, "finishReason": finish}],
                "usageMetadata": {"promptTokenCount": 100,
                                  "candidatesTokenCount": 10,
                                  "totalTokenCount": 110}}}


def test_positional_reconciliation(cfg):
    """W0-4. N-1 good rows plus one BARE gRPC status with no key at all: the
    error is attributed by POSITION, because the key echo on an error row is
    undocumented (it was present in the probe wave, which is not a contract)."""
    plan = _plan(3)
    lines = [_line(plan[0]["key"]),
             {"error": {"code": 7,
                        "message": "The caller does not have permission"}},
             _line(plan[2]["key"])]
    out = BR.reconcile(plan, lines)
    assert [o.ok for o in out] == [True, False, True]
    assert out[1].key == plan[1]["key"]          # by position, not by key
    assert out[1].key_echoed is None
    assert out[1].error["code"] == 7
    assert "does not have permission" in out[1].why()


def test_a_length_mismatch_is_fatal_and_writes_nothing():
    with pytest.raises(FatalError) as exc:
        BR.reconcile(_plan(3), [_line("a"), _line("b")])
    assert "POSITIONAL" in str(exc.value)


def test_a_key_that_disagrees_with_its_position_is_fatal():
    plan = _plan(2)
    with pytest.raises(FatalError) as exc:
        BR.reconcile(plan, [_line(plan[1]["key"]), _line(plan[0]["key"])])
    assert "cross-check" in str(exc.value)


def test_the_count_lock_is_checked_on_our_side_of_the_wire():
    plan = _plan(1, n=3)
    short = BR.reconcile(plan, [_line(plan[0]["key"],
                                      items=[{"lemma": "L", "gloss": "G"}])])
    assert not short[0].ok
    assert not short[0].count_ok
    assert short[0].got == 1
    assert "count lock" in short[0].why()


def test_a_truncated_row_is_read_as_a_cap_error_not_a_parse_error():
    plan = _plan(1)
    line = {"key": plan[0]["key"],
            "response": {"candidates": [{"content": {"parts": [
                {"text": '{"definitions": [{"lemma"'}]},
                "finishReason": "MAX_TOKENS"}],
                "usageMetadata": {"promptTokenCount": 10,
                                  "candidatesTokenCount": 1024,
                                  "totalTokenCount": 1034}}}
    out = BR.reconcile(plan, [line])
    assert out[0].finish_reason == "MAX_TOKENS"
    assert out[0].parse_error is None
    assert "MAX_TOKENS" in out[0].why()


def test_failed_request_count_is_read_but_is_not_the_criterion():
    """PARTIALLY_SUCCEEDED does not exist on this API, so a job with failed rows
    reports SUCCEEDED. batchStats is recorded as evidence; the rows decide."""
    job = FakeJob("batches/1", "files/1", "JOB_STATE_SUCCEEDED",
                  stats={"failed_request_count": 2, "request_count": 32})
    assert BR.failed_request_count(job) == 2
    assert BR.batch_stats_dict(job)["failed_request_count"] == 2
    assert BR.failed_request_count(FakeJob("b", "f", "x")) is None


# ==========================================================================
# 5.5 -- the wave splitter
# ==========================================================================

def test_wave_splitter_counts_uncached_only(batch_stats):
    """5.5. 21 rows x 150,503 cached tokens = 3.16M nominal enqueued tokens were
    ACCEPTED against a 3M limit, so cached tokens do not count. Counting them
    would make the definition wave 6 jobs per language instead of 2."""
    counts_cached, why = BW.enqueued_counts_cached(batch_stats)
    assert counts_cached is False
    assert "3,160,563" in why
    requests = [{"entry_id": "e%d" % i, "cached_tokens": 150_503,
                 "uncached_tokens": 30_000} for i in range(100)]
    target = BW.job_token_target()
    uncached_only = BW.split_into_jobs(requests, target_tokens=target,
                                       counts_cached=False)
    everything = BW.split_into_jobs(requests, target_tokens=target,
                                    counts_cached=True)
    assert len(uncached_only) == 2                 # 3.0M uncached / 2.4M target
    assert len(everything) == 8                    # 18.05M nominal
    assert sum(len(j) for j in uncached_only) == 100


def test_a_missing_measurement_makes_the_splitter_conservative():
    """Absence of the measurement counts everything: more jobs than necessary is
    a cost, a rejected submit in the middle of a drain is an incident."""
    counts_cached, why = BW.enqueued_counts_cached({"wave3": {}})
    assert counts_cached is True
    assert "not on disk" in why


def test_the_requests_of_one_entry_are_never_split_across_jobs():
    requests = ([{"entry_id": "e1", "cached_tokens": 0, "uncached_tokens": 900}]
                * 3
                + [{"entry_id": "e2", "cached_tokens": 0,
                    "uncached_tokens": 900}])
    jobs = BW.split_into_jobs(requests, target_tokens=2000,
                              counts_cached=False)
    assert [len(j) for j in jobs] == [3, 1]
    assert {r["entry_id"] for r in jobs[0]} == {"e1"}


def test_the_job_target_leaves_headroom_under_the_documented_limit():
    assert BW.ENQUEUED_TOKEN_LIMIT == 3_000_000
    assert BW.job_token_target() == 2_400_000
    assert BW.MAX_JOBS_IN_FLIGHT == 1


def test_the_splitter_and_the_bill_size_a_request_the_same_way(cfg,
                                                              batch_stats):
    """One arithmetic. The splitter's answer decides the job count and the
    enqueued limit is a hard refusal at submit, so a second copy of the formula
    would be discovered as a rejected submit in the middle of a paid drain."""
    entries = {e["entry_id"]: e for e in [_entry(n_senses=4, n_exprs=2)]}
    todo = _todo(cfg, entries)
    tokens = S42.bill_tokens(todo, [], "German", batch_stats)
    pfit = S42.prompt_token_fit(batch_stats)
    lean = S42.system_prompt_tokens(batch_stats, "German")
    by_hand = 0
    for kind, cap in (("definition", 20), ("expression", 20)):
        for _eid, batch in S42._group_by_entry(todo, kind, cap):
            chars = sum(len(r["text"]) + len(r.get("grammar") or "")
                        + len(r["hint"]) for r in batch)
            part = S42.request_input_tokens(
                kind, len(batch), chars, system_tokens=lean,
                measured_system_tokens=lean, prompt_fit=pfit,
                cached=(kind in S42.CACHEABLE_KINDS))
            by_hand += part["uncached"]
    assert by_hand == tokens["cache_works"]["uncached_input_tokens"]


# ==========================================================================
# 5.4 -- the cache lifecycle
# ==========================================================================

def test_the_cache_floor_comes_off_the_artifact(batch_stats):
    assert BC.cache_floor(batch_stats) == 1024
    with pytest.raises(FatalError):
        BC.cache_floor({"wave2": {}})
    # the implicit floor is a DIFFERENT number and is deliberately not consulted
    assert "IMPLICIT_CACHE_FLOOR" not in json.dumps(batch_stats)


def test_the_expression_prompt_is_refused_a_cache_object(batch_genai,
                                                         batch_stats):
    """~640 tokens against a measured 1,024 floor. Padding it to qualify would
    buy a discount on tokens that only exist because of the padding (N-10)."""
    ok, tokens, floor = BC.qualifies_for_cache(
        S42.system_prompt("expression", "German"), batch_stats)
    assert not ok and tokens < floor
    with pytest.raises(FatalError) as exc:
        BC.create(batch_genai, model="gemini-3.7-flash",
                  system_text=S42.system_prompt("expression", "German"),
                  lang="German", kind="expression", ttl=3600,
                  stats=batch_stats)
    assert "explicit-cache floor" in str(exc.value)
    assert batch_genai.caches_created == []


def test_the_definition_prompt_qualifies_and_the_declared_size_is_recorded(
        batch_genai, batch_stats):
    handle = BC.create(batch_genai, model="gemini-3.7-flash",
                       system_text=S42.system_prompt("definition", "German"),
                       lang="German", kind="definition", ttl=5400,
                       stats=batch_stats)
    assert handle.declared_tokens == DECLARED_CACHE_TOKENS
    assert handle.prompt_sha256 == S42.prompt_sha256(
        S42.system_prompt("definition", "German"))
    assert handle.ttl_seconds == 5400


def test_the_ttl_is_the_configured_multiple_of_the_estimated_drain():
    """1.5x, not the 3x the audit wanted: that margin was insurance against
    SILENT full-price billing, and expiry is measurably loud and free (every row
    fails with gRPC code 7 at prompt=0, billed $0)."""
    drain = BW.wall_clock_estimate_seconds(3623)
    assert BC.ttl_seconds(drain, 1.5) == int(drain * 1.5)
    assert BW.wall_clock_estimate_seconds(1) == 3600.0


def test_an_expired_cache_cannot_be_extended_only_recreated(batch_genai,
                                                            batch_stats):
    handle = BC.create(batch_genai, model="gemini-3.7-flash",
                       system_text=S42.system_prompt("definition", "German"),
                       lang="German", kind="definition", ttl=3600,
                       stats=batch_stats)
    batch_genai.cache_expired = True
    with pytest.raises(S42.CacheUnavailable) as exc:
        BC.extend(batch_genai, handle, 7200)
    assert "recreated" in str(exc.value)


def test_a_free_tier_key_is_told_why_it_can_never_cache(batch_genai,
                                                        batch_stats):
    def _raise(model=None, config=None):
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED: quota_metric "
            "TotalCachedContentStorageTokensPerModelFreeTier limit=0")
    batch_genai.caches.create = _raise
    with pytest.raises(FatalError) as exc:
        BC.create(batch_genai, model="gemini-3.7-flash",
                  system_text=S42.system_prompt("definition", "German"),
                  lang="German", kind="definition", ttl=3600,
                  stats=batch_stats)
    assert "free tier" in str(exc.value)
    assert "only discount path" in str(exc.value)


def test_the_cache_is_deleted_at_the_end_of_the_wave_and_priced(batch_genai,
                                                               batch_stats):
    handle = BC.create(batch_genai, model="gemini-3.7-flash",
                       system_text=S42.system_prompt("definition", "German"),
                       lang="German", kind="definition", ttl=3600,
                       stats=batch_stats)
    assert BC.delete(batch_genai, handle)["deleted"] is True
    assert batch_genai.caches_deleted == [handle.name]
    cost = BC.storage_cost(handle, 24.0, "batch")
    assert cost["usd"] == pytest.approx(0.0136, abs=0.001)


# ==========================================================================
# 5.6 -- the retry decision
# ==========================================================================

def _outcome(**kw):
    base = dict(pos=0, key="def__German__1__00", kind="definition",
                n_expected=3)
    base.update(kw)
    return BR.RowOutcome(**base)


def test_a_dead_cache_reference_is_recoverable_and_a_refused_request_is_not():
    dead = BW.retry_decision(_outcome(error={"code": 7, "message": "no perm"}),
                             cap=1024, ceiling=8192)
    refused = BW.retry_decision(
        _outcome(error={"code": 3, "message": "invalid"}), cap=1024,
        ceiling=8192)
    assert dead["retry"] and dead["why"] == "cache_missing"
    assert not refused["retry"] and refused["why"] == "invalid_argument"


def test_a_truncation_raises_the_budget_once_and_only_once():
    """The interactive path raises the budget inside _generate. The batch path
    never goes through _generate, so this is the ONLY raise on this path and the
    two cannot stack."""
    first = BW.retry_decision(_outcome(finish_reason="MAX_TOKENS"), cap=1024,
                              ceiling=8192)
    assert first["retry"] and first["cap"] == 2048
    at_ceiling = BW.retry_decision(_outcome(finish_reason="MAX_TOKENS"),
                                   cap=8192, ceiling=8192)
    assert not at_ceiling["retry"]
    assert at_ceiling["why"] == "max_tokens_at_ceiling"


def test_a_count_lock_violation_is_retried_with_a_correction_in_the_payload():
    """The correction travels at the END of the user message. Prepending it to
    the system prompt would forfeit the cache discount on precisely the requests
    being redone."""
    decision = BW.retry_decision(
        _outcome(finish_reason="STOP", items=[{"lemma": "a", "gloss": "b"}]),
        cap=1024, ceiling=8192)
    assert decision["retry"] and decision["why"] == "count_lock"
    assert "exactly 3" in decision["correction"]
    assert decision["cap"] == 1024


def test_the_retry_loop_has_a_bound():
    assert BW.MAX_RETRY_WAVES == 3
    assert BW.within_retry_bound(3)
    assert not BW.within_retry_bound(4)


# ==========================================================================
# end to end, through stage 42
# ==========================================================================

@pytest.fixture
def wave(cfg, batch_genai, batch_stats, no_sleep):
    _workspace(cfg)
    return _batch_cfg(cfg)


def test_a_batch_wave_writes_the_cells_it_paid_for(wave, cfg, registry,
                                                   batch_genai):
    report = S42.run(cfg, registry, lang="German", confirm=True)
    defs = read_json(cfg.json_dir / "translations" / "German"
                     / "definitions.json")
    exprs = read_json(cfg.json_dir / "translations" / "German"
                      / "expressions.json")
    assert len(defs) == 2 and len(exprs) == 1
    assert report["languages"]["German"]["written"] == {"definitions": 2,
                                                        "expressions": 1}
    assert report["languages"]["German"]["transport"] == "batch"
    prov = report["languages"]["German"]["provenance"]
    assert prov.startswith("gemini:gemini-3.7-flash+v4-frozen+LOW@")
    assert all(row["provenance"] == prov for row in defs.values())
    # two jobs: one definition (cached), one expression (never cached)
    assert len(report["batch"]["languages"]["German"]["jobs"]) == 2
    assert batch_genai.caches_created and len(batch_genai.caches_deleted) == 1


def test_the_ledger_rows_of_a_batch_wave_go_through_normalize_usage(wave, cfg,
                                                                    registry):
    """A hand-rolled row loses the (ts, seq) stamp, and then the ledger
    files rows by INGEST day instead of call day, its rotation detection
    degrades to "cannot reconcile", and the sink and the ingest double-count."""
    S42.run(cfg, registry, lang="German", confirm=True)
    rows = [json.loads(x) for x in
            (cfg.report_dir / "translate_usage.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows
    assert all(r["mode"] == "batch" for r in rows)
    assert all(r["ts"] and r["seq"] for r in rows)
    assert all(r["cached_tokens"] <= r["prompt_tokens"] for r in rows)
    assert all(r["thinking_tokens"] == 0 for r in rows)
    assert all(r["prompt_id"] == cfg.prompt_id for r in rows)
    definition = [r for r in rows if r["kind"] == "definition"]
    expression = [r for r in rows if r["kind"] == "expression"]
    # the cached path: no sha of its own (systemInstruction XOR cachedContent),
    # so the identity of the prompt comes from the CACHE
    assert definition[0]["prompt_sha256"] is None
    assert definition[0]["cache_prompt_sha256"] == S42.prompt_sha256(
        S42.system_prompt("definition", "German"))
    assert definition[0]["cached_tokens"] == DECLARED_CACHE_TOKENS
    # the expression path is uncached BY DESIGN and carries its own sha
    assert expression[0]["cached_tokens"] == 0
    assert expression[0]["prompt_sha256"] == S42.prompt_sha256(
        S42.system_prompt("expression", "German"))


def test_the_money_gates_adjudicate_the_wave_that_was_paid_for(wave, cfg,
                                                               registry):
    """G-BILL / G-THINK / G-PROMPT / G-CACHE, with the two inputs only the
    transport can supply: declared_cache_tokens and the {cache_name: sha} map."""
    report = S42.run(cfg, registry, lang="German", confirm=True)
    gates = report["batch"]["gates"]
    assert gates["ok"], gates["error"]
    assert gates["declared_cache_tokens_by_language"] == {
        "German": DECLARED_CACHE_TOKENS}
    assert gates["cache_prompt_shas"]
    ids = {row["id"] for row in gates["rows"]}
    assert {"G-BILL", "G-THINK", "G-PROMPT", "G-CACHE"} <= ids
    recorded = read_json(cfg.report_dir / "gates_report.json")["results"]
    assert {"G-CACHE", "G-PROMPT"} <= {r["id"] for r in recorded}


def test_a_submit_phase_places_the_jobs_and_an_ingest_writes_the_rows(
        wave, cfg, registry):
    """One wave, two --confirm-spend invocations. The submit must not write a
    translation row and the ingest must not resubmit a job."""
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    tdir = cfg.json_dir / "translations" / "German"
    assert read_json(tdir / "definitions.json", default={}) == {}
    reg = BREG.JobRegistry(cfg)
    assert len(reg.in_state(BREG.DOWNLOADED)) == 2
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    assert len(read_json(tdir / "definitions.json")) == 2
    reg = BREG.JobRegistry(cfg)
    assert len(reg.in_state(BREG.RECOVERED)) == 2
    assert reg.unfinished() == []


def test_a_second_ingest_does_not_book_the_same_rows_twice(wave, cfg, registry):
    S42.run(cfg, registry, lang="German", confirm=True)
    first = len((cfg.report_dir / "translate_usage.jsonl")
                .read_text(encoding="utf-8").splitlines())
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    after = len((cfg.report_dir / "translate_usage.jsonl")
                .read_text(encoding="utf-8").splitlines())
    assert after == first


def test_a_row_that_fails_is_retried_in_a_new_job_on_batch(wave, cfg, registry,
                                                           batch_genai):
    """D-06: the retry stays on batch. Nothing downgrades to standard or flex by
    itself, because an automatic downgrade doubles the rate silently."""
    seen = {"n": 0}

    def responder(row, pos):
        if row["key"].startswith("def__") and seen["n"] < 1:
            seen["n"] += 1
            return error_line(row, code=13, message="internal")
        return None

    batch_genai.responder = responder
    report = S42.run(cfg, registry, lang="German", confirm=True)
    lang = report["batch"]["languages"]["German"]
    assert lang["retry_waves"] == 1
    assert not lang.get("unrecovered")
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2
    jobs = [j["job_id"] for j in lang["jobs"]]
    assert any("w1" in j for j in jobs)
    modes = {json.loads(x)["mode"] for x in
             (cfg.report_dir / "translate_usage.jsonl")
             .read_text(encoding="utf-8").splitlines() if x.strip()}
    assert modes == {"batch"}


def test_a_row_that_keeps_failing_is_recorded_and_the_wave_continues(
        wave, cfg, registry, batch_genai):
    """A FatalError in the middle of a wave throws away a job that has already
    been paid for. The cells stay missing, they are NAMED, and the next run's
    compute_todo picks them up as `missing`."""
    batch_genai.responder = lambda row, pos: (
        error_line(row, code=13, message="internal")
        if row["key"].startswith("def__") else None)
    # The wave itself does not raise -- it records and keeps going, because a
    # FatalError mid-wave discards a job that has already been paid for. What
    # raises is G-CACHE at the END, after the report is on disk: a wave that
    # declared a cache and never got a cacheable request through is a failure,
    # and it is a failure a human has to see the report of.
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "G-CACHE" in str(exc.value)
    report = read_json(cfg.report_dir / "translate_report.json")
    lang = report["batch"]["languages"]["German"]
    assert lang["retry_waves"] == BW.MAX_RETRY_WAVES
    assert lang["unrecovered"]
    assert report["languages"]["German"]["unrecovered_requests"] == 1
    # the expressions of the same wave were still written
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "expressions.json")) == 1
    assert read_json(cfg.json_dir / "translations" / "German"
                     / "definitions.json", default={}) == {}


def test_a_count_lock_violation_records_its_finish_reason(wave, cfg, registry,
                                                          batch_genai):
    """N-08. "The model dropped a sense" and "the cap truncated the JSON" were
    the same log line and they need opposite fixes."""
    batch_genai.responder = lambda row, pos: (
        ok_line(row, batch_genai.cache_tokens,
                items=[{"lemma": "L", "gloss": "G"}])
        if row["key"].startswith("def__") else None)
    # Four attempts of one request against a quote that priced one: G-BILL is
    # what raises, at the end, after the report and the violations are on disk.
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "G-BILL" in str(exc.value)
    rows = read_json(cfg.review_dir / "count_lock_violations_German.json")
    assert rows
    assert rows[0]["finish_reason"] == "STOP"
    assert rows[0]["expected"] == 2 and rows[0]["got"] == 1
    assert rows[0]["max_output_tokens"]
    # ...and the cells were NOT written from a short array: zip(rows, items)
    # would have shifted every gloss onto the wrong sense.
    assert read_json(cfg.json_dir / "translations" / "German"
                     / "definitions.json", default={}) == {}
    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["batch"]["languages"]["German"]["unrecovered"]
    assert [r["kind"] for r in
            report["batch"]["languages"]["German"]["failed_requests"]] \
        == ["definition"] * 4


def test_the_wave_refuses_to_submit_the_same_job_twice(wave, cfg, registry,
                                                        batch_genai):
    """The fingerprint is checked BEFORE the submit, because batches.create is
    not idempotent and a repeat is a repeat charge."""
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    created = len(batch_genai.jobs)
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert len(batch_genai.jobs) == created


def test_a_create_that_fails_with_a_5xx_adopts_the_existing_job(wave, cfg,
                                                                registry,
                                                                batch_genai):
    """NEVER blind-retry create. The recovery is batches.list matched on the
    input file resource name."""
    batch_genai.create_raises = RuntimeError("503 UNAVAILABLE: high demand")
    batch_genai.list_returns_existing = True
    S42.run(cfg, registry, lang="German", confirm=True)
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2


def test_a_create_that_fails_with_no_matching_job_refuses_to_resubmit(
        wave, cfg, registry, batch_genai):
    batch_genai.create_raises = RuntimeError("503 UNAVAILABLE: high demand")
    batch_genai.list_returns_existing = False
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True)
    assert "NOT idempotent" in str(exc.value)
    reg = BREG.JobRegistry(cfg)
    assert reg.in_state(BREG.FAILED)


def test_the_cache_ttl_is_extended_before_a_later_submit(wave, cfg, registry,
                                                         batch_genai):
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    # the ingest reuses the cache the submit created rather than making a second
    assert len(batch_genai.caches_created) == 1


def test_the_uploaded_jsonl_is_what_the_writer_produced(wave, cfg, registry,
                                                         batch_genai):
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert batch_genai.uploads
    for text in batch_genai.uploads.values():
        for line in text.splitlines():
            row = json.loads(line)
            BJ.assert_batch_row_shape(row)
            assert BK.KEY_RE.match(row["key"])


def test_the_pos_fallback_stays_on_the_interactive_surface(cfg, batch_genai,
                                                           batch_stats,
                                                           no_sleep, registry):
    """One request per language, and only for a pos_key the checked-in registry
    does not cover. Its ledger row says standard because that is what it is --
    filing it as batch would price it at half."""
    entry = _entry()
    entry["pos_key"] = "vidunderlig-ordklasse"
    _workspace(cfg, [entry])
    _batch_cfg(cfg)
    S42.run(cfg, registry, lang="German", confirm=True)
    assert len(batch_genai.model_calls) == 1
    rows = [json.loads(x) for x in
            (cfg.report_dir / "translate_usage.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    pos = [r for r in rows if r["kind"] == "pos"]
    assert len(pos) == 1
    assert pos[0]["mode"] == "standard"


def test_the_expression_canary_measures_thinking_and_writes_it_to_disk(
        cfg, batch_genai, batch_stats, no_sleep, registry):
    """The bill books 275 unmeasured thought tokens per expression
    request (the ranking prompt's measured maximum at the same LOW) because the
    expression prompt has never been probed. One small job measures it."""
    _workspace(cfg)
    _batch_cfg(cfg)
    S42.run(cfg, registry, lang="German", confirm=True)
    canary = read_json(cfg.report_dir
                       / "expression_thinking_canary_German.json")
    assert canary["clean"] is True
    assert canary["measurement"]["max"] == 0
    assert canary["measurement"]["n_observations"] == 1
    assert canary["written_to_artifact"] is True
    disk = read_json(cfg.probe_stats_path)
    family = disk["thinking"]["THINKING_AT_LOW_BY_PROMPT_FAMILY"]
    key = "LOW|expression(%d)" % len(S42.system_prompt("expression", "German"))
    assert family[key]["max"] == 0
    assert family[key]["measured_by"].startswith("batch expression canary")


def test_a_clean_redo_archives_on_the_submit_phase(wave, cfg, registry):
    """N-01 on the batch path: the archive happens on the invocation that
    commits the money (the submit), not on the ingest hours later."""
    tdir = cfg.json_dir / "translations" / "German"
    legacy = {"11021722:21000001": {"lemma": "alt", "gloss": "alt",
                                    "src_sha": "stale",
                                    "provenance": "legacy:2025"}}
    write_json(tdir / "definitions.json", legacy)
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit",
            retranslate_all=True, do_gc=False)
    archive = read_json(tdir / "archive.json")
    assert list(archive["definitions"]) == ["11021722:21000001"]
    assert archive["definitions"]["11021722:21000001"]["reason"] == "clean_redo"
    # the submit archived and placed the jobs; it wrote no new cell
    assert read_json(tdir / "definitions.json") == {}
    report = S42.run(cfg, registry, lang="German", confirm=True,
                     phase="ingest", retranslate_all=True, do_gc=False)
    assert len(read_json(tdir / "definitions.json")) == 2
    assert report["languages"]["German"]["written"]["definitions"] == 2


def test_a_resumed_redo_does_not_buy_the_rows_it_already_produced(wave, cfg,
                                                                  registry):
    """The resume half. A row that already carries this run's provenance is
    neither re-billed nor re-archived, so a crash halfway through a wave costs
    the remainder and nothing else."""
    S42.run(cfg, registry, lang="German", confirm=True)
    tdir = cfg.json_dir / "translations" / "German"
    before = read_json(tdir / "definitions.json")
    report = S42.run(cfg, registry, lang="German", confirm=True,
                     retranslate_all=True)
    assert read_json(tdir / "archive.json")["definitions"] == {}
    assert read_json(tdir / "definitions.json") == before
    assert report["bill"]["German"]["resume"][
        "definitions_already_redone"] == 2
    assert report["languages"]["German"]["written"]["definitions"] == 0


def test_a_job_that_ends_with_no_result_file_is_recorded_not_raised(
        wave, cfg, registry, batch_genai):
    """An EXPIRED or FAILED job is terminal and produces nothing. Recorded as
    FAILED so the wave keeps its place; the cells stay missing and the next
    run's compute_todo picks them up."""
    batch_genai.no_dest = True
    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True)
    reg = BREG.JobRegistry(cfg)
    failed = reg.in_state(BREG.FAILED)
    assert failed and "no result file" in failed[0]["failure"]
    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["batch"]["languages"]["German"]["jobs"][0]["state"] == "FAILED"


def test_waiting_for_a_job_that_never_drains_gives_up_without_resubmitting():
    """The documented target is 24h and the HARD expiry is 48. Past that this run
    stops waiting; the registry has kept its place and nothing is resubmitted."""
    service = FakeService()
    service.polls_before_terminal = 10 ** 9
    service.uploads["files/x"] = ""
    made = service.batches._make("gemini-3.7-flash", "files/x")
    clock = {"t": 0.0}
    with pytest.raises(FatalError) as exc:
        BT.poll_until_terminal(service, made.name, timeout_s=10,
                               interval_s=1,
                               sleep=lambda s: clock.update(t=clock["t"] + 100),
                               now=lambda: clock["t"])
    assert "still" in str(exc.value) and "resubmitted" in str(exc.value)


def test_an_expired_cache_is_recreated_and_the_next_jsonl_carries_the_new_name(
        wave, cfg, registry, batch_genai):
    """This is why the JSONL is written per job JUST BEFORE the submit: an
    expired cache cannot be updated, only recreated, and a recreate changes the
    resource name. A file written in advance would carry a dead reference on
    every line."""
    fail_once = {"done": False}

    def responder(row, pos):
        if row["key"].startswith("def__") and not fail_once["done"]:
            fail_once["done"] = True
            return error_line(row, code=7)
        return None

    batch_genai.responder = responder
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    first_cache = batch_genai.caches_created[0]["name"]
    batch_genai.cache_expired = True
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    assert len(batch_genai.caches_created) == 2
    second_cache = batch_genai.caches_created[1]["name"]
    assert second_cache != first_cache
    retry = [t for name, t in batch_genai.uploads.items()
             if "cachedContent" in t and second_cache in t]
    assert retry, "the retry job's JSONL must reference the RECREATED cache"
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2


def test_two_languages_in_one_invocation_are_gated_separately(cfg, batch_genai,
                                                              batch_stats,
                                                              no_sleep,
                                                              registry):
    """G-CACHE's denominator is per language. One verdict over both would make a
    healthy wave look half-cached, and G-BILL would count one language's cost
    against the other's quote."""
    _workspace(cfg)
    _batch_cfg(cfg)
    cfg.langs = ["German", "Spanish"]
    report = S42.run(cfg, registry, confirm=True)
    langs = report["batch"]["languages"]
    assert set(langs) == {"German", "Spanish"}
    german = langs["German"]["usage_rows"]
    spanish = langs["Spanish"]["usage_rows"]
    assert german[1] == spanish[0]                 # contiguous, not overlapping
    assert german[0] < german[1] <= spanish[1]
    extras = {row["extra"]["lang"] for row in report["batch"]["gates"]["rows"]}
    assert extras == {"German", "Spanish"}
    assert report["batch"]["gates"]["ok"], report["batch"]["gates"]["error"]
    assert len(batch_genai.caches_created) == 2    # one cache per language


# ==========================================================================
# the guards that changed
# ==========================================================================

def test_the_transport_guard_no_longer_refuses_batch(cfg):
    cfg.mode = "batch"
    cfg.cache_enabled = True
    S42.transport_guard(cfg)                      # the two branches are gone
    cfg.mode = "standard"
    with pytest.raises(FatalError) as exc:
        S42.transport_guard(cfg)
    assert "driven by the BATCH wave" in str(exc.value)
    cfg.cache_enabled = False
    S42.transport_guard(cfg)
    cfg.mode = "flex"
    S42.transport_guard(cfg)


def test_flex_is_the_interactive_surface_plus_a_service_tier(cfg):
    """D-05/D-06: three modes, all built, none automatic. flex rides the
    interactive path and IS serviceTier=flex; batch rows never carry the field."""
    cfg.mode = "flex"
    assert cfg.effective_service_tier == "flex"
    assert _fake_request(cfg).service_tier == "flex"
    cfg.mode = "batch"
    assert cfg.effective_service_tier is None
    assert _fake_request(cfg).service_tier is None


def test_the_dry_path_imports_no_llm_module_under_mode_batch(cfg, registry,
                                                             batch_stats,
                                                             monkeypatch):
    """4.2(4). The bill has to be readable on a machine with no SDK and no key,
    and mode=batch must not change that: the transport is reached only past the
    line where money is spent."""
    for name in list(sys.modules):
        if name == "google" or name.startswith("google."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    _workspace(cfg)
    _batch_cfg(cfg)
    S42.run(cfg, registry, lang="German", confirm=False)
    assert not [n for n in sys.modules if n == "google"
                or n.startswith("google.")]


def test_the_bill_of_a_batch_run_is_deterministic(cfg, registry, batch_stats):
    _workspace(cfg)
    _batch_cfg(cfg)
    S42.run(cfg, registry, lang="German", confirm=False)
    first = (cfg.report_dir / "translate_bill_German.json").read_bytes()
    S42.run(cfg, registry, lang="German", confirm=False)
    assert (cfg.report_dir / "translate_bill_German.json").read_bytes() == first


# ==========================================================================
# FIXER round -- the crash-recovery path, the TTL, the FAILED release
# ==========================================================================
#
# Both acceptance reviewers landed on the same BLOCKER: registry.py described a
# recovery for the PLANNED window in three paragraphs, transport.py's poll error
# told the operator the wait "should be resumed with --phase ingest later", and
# unfinished() / in_flight() / find_job_by_input_file had zero callers outside
# this file. A `--phase submit` is a 24-to-50-hour FOREGROUND call with no
# daemon, so dying mid-drain is not a corner case, and it landed in the one
# window nothing could recover.


def _die_after_the_submit(service, why="ssh died mid-drain"):
    """Leave the wave exactly where a broken foreground drain leaves it:
    the job is SUBMITTED on the service and its results are not on our disk."""
    service.download_raises = RuntimeError(why)


def test_a_submit_that_died_mid_drain_is_resumed_by_the_next_ingest(
        wave, cfg, registry, batch_genai):
    """The reviewers' repro, end to end: every cell ingested, nothing re-billed.

    Before the fix this run wrote ZERO definition rows, left the paid job at
    SUBMITTED with its job_name and uploaded_file sitting unused in jobs.json,
    and had no third entry point that would ever look at them again.
    """
    _die_after_the_submit(batch_genai)
    with pytest.raises(RuntimeError):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    stranded = reg.in_flight()
    assert [j["state"] for j in stranded] == [BREG.SUBMITTED]
    assert stranded[0]["job_name"] and stranded[0]["uploaded_file"]
    tdir = cfg.json_dir / "translations" / "German"
    assert read_json(tdir / "definitions.json", default={}) == {}

    jobs_on_the_service = len(batch_genai.jobs)
    report = S42.run(cfg, registry, lang="German", confirm=True,
                     phase="ingest")

    # every cell THE WAVE PAID FOR is on disk...
    assert len(read_json(tdir / "definitions.json")) == 2
    # ...the stranded job was DRAINED, not resubmitted, so nothing new was
    # created on the service and nothing was billed a second time...
    reg = BREG.JobRegistry(cfg)
    assert reg.in_flight() == []
    assert reg.unfinished() == []
    assert len(reg.in_state(BREG.RECOVERED)) == 1
    assert len(batch_genai.jobs) == jobs_on_the_service
    resumed = report["batch"]["languages"]["German"]["resumed"]
    assert [r["action"] for r in resumed] == ["resumed"]

    # The expression job was never submitted (the crash happened before it), so
    # its cells are simply still missing -- which is the state the next run
    # handles, and the phases stay explicit rather than an ingest silently
    # placing new orders.
    assert read_json(tdir / "expressions.json", default={}) == {}
    S42.run(cfg, registry, lang="German", confirm=True)
    assert len(read_json(tdir / "expressions.json")) == 1
    assert len(read_json(tdir / "definitions.json")) == 2


def test_a_planned_job_whose_upload_landed_is_adopted_and_never_created_twice(
        wave, cfg, registry, batch_genai, monkeypatch):
    """The PLANNED window: the JSONL is written, the file is uploaded, and the
    process dies before batches.create returns. The first call may have been
    accepted and the answer lost, so the recovery is a batches.list match on the
    input file resource name -- the one identifier both sides agree on."""
    real_create = BT.create_job

    def create_then_die(client, **kw):
        job = real_create(client, **kw)
        raise KeyboardInterrupt("power cut between create and mark_submitted")

    monkeypatch.setattr(BT, "create_job", create_then_die)
    with pytest.raises(KeyboardInterrupt):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    planned = reg.in_state(BREG.PLANNED)
    assert len(planned) == 1 and planned[0]["uploaded_file"]
    assert planned[0]["job_name"] is None

    monkeypatch.setattr(BT, "create_job", real_create)
    on_the_service = len(batch_genai.jobs)
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")

    reg = BREG.JobRegistry(cfg)
    assert reg.in_flight() == []
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2
    # the adopted job is the one that already existed
    assert len(batch_genai.jobs) == on_the_service
    report = read_json(cfg.report_dir / "translate_report.json")
    assert "adopted" in [r["action"] for r
                         in report["batch"]["languages"]["German"]["resumed"]]


def test_a_planned_job_that_never_uploaded_is_released_for_replanning(
        wave, cfg, registry, batch_genai, monkeypatch):
    """Nothing was uploaded, so nothing was submitted and nothing was billed.
    The reservation is released and the wave plans the job again."""
    real_upload = BT.upload_jsonl

    def upload_then_die(client, path, display_name):
        raise KeyboardInterrupt("power cut before the upload finished")

    monkeypatch.setattr(BT, "upload_jsonl", upload_then_die)
    with pytest.raises(KeyboardInterrupt):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    assert [j["state"] for j in reg.in_flight()] == [BREG.PLANNED]
    assert reg.in_flight()[0]["uploaded_file"] is None

    # restored by name, NOT monkeypatch.undo(): the monkeypatch fixture is shared
    # with conftest's autouse `refrozen`, so undo() would also un-sign the
    # refreeze and the next run would refuse on G-SCOPE-FROZEN instead.
    monkeypatch.setattr(BT, "upload_jsonl", real_upload)
    S42.run(cfg, registry, lang="German", confirm=True)
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2
    reg = BREG.JobRegistry(cfg)
    released = [j for j in reg.in_state(BREG.FAILED)]
    assert released and released[0]["resubmittable"] is True
    assert "never uploaded" in released[0]["failure"]


def test_a_planned_job_whose_upload_cannot_be_matched_is_never_resubmitted(
        wave, cfg, registry, batch_genai, monkeypatch):
    """The one state where guessing costs money: the file is on the service, no
    job references it, and the create may or may not have been accepted."""
    real_create = BT.create_job

    def create_then_die(client, **kw):
        raise KeyboardInterrupt("died before create was even attempted")

    monkeypatch.setattr(BT, "create_job", create_then_die)
    with pytest.raises(KeyboardInterrupt):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    monkeypatch.setattr(BT, "create_job", real_create)
    reg = BREG.JobRegistry(cfg)
    stuck = reg.in_state(BREG.PLANNED)[0]["job_id"]

    # the resume refuses to guess, records it, and the run says so LOUDLY -- an
    # uploaded input file with no job behind it is the one state where a wrong
    # guess is a duplicate charge rather than a wasted round trip
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    assert "no job on the service references" in str(exc.value)
    reg = BREG.JobRegistry(cfg)
    job = reg.get(stuck)
    assert job["state"] == BREG.FAILED
    assert job["resubmittable"] is False
    assert "no job on the service references" in job["failure"]
    # and the next submit refuses to reopen it, with an error that says what to do
    with pytest.raises(FatalError) as exc:
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert "not marked resubmittable" in str(exc.value)
    assert "second charge" in str(exc.value)


def test_one_job_in_flight_holds_across_processes_too(wave, cfg, registry,
                                                      batch_genai):
    """MAX_JOBS_IN_FLIGHT used to be a constant written into the report with no
    enforcement point: a second `--phase submit` while the first job was still
    SUBMITTED went ahead and opened another one. The enqueued ceiling is summed
    across ACTIVE jobs, so that is a refused submit in the middle of a paid
    drain. The resume at the head of the wave is what makes it true."""
    _die_after_the_submit(batch_genai)
    with pytest.raises(RuntimeError):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    assert len(reg.in_flight()) == 1
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    # the stranded job was drained first, so at no point were two in flight
    assert len(reg.in_flight()) == 0
    assert len(reg.in_state(BREG.DOWNLOADED)) == 2


# ---- F2: the cache outlives the jobs that reference it --------------------

def _one_planned_job(reg, cfg, *, job_id="German-def-w0-00", wave=0):
    fp = BREG.wave_fingerprint(model=cfg.gemini_model, prompt_id="v4-frozen",
                               lang="German", keys=[job_id],
                               cache_name="cachedContents/1", wave=wave)
    return reg.plan(job_id, fingerprint=fp, lang="German", kind="definition",
                    model=cfg.gemini_model, prompt_id="v4-frozen",
                    cache_name="cachedContents/1",
                    declared_cache_tokens=DECLARED_CACHE_TOKENS,
                    cache_prompt_sha256="sha", jsonl_path="x.jsonl",
                    plan=[{"key": job_id, "cells": []}], enqueued_tokens=1,
                    wave=wave)


def test_the_cache_is_not_deleted_while_a_job_is_still_in_flight(
        cfg, batch_genai, batch_stats):
    """Probe W3-4 paid to learn that a cache deleted immediately after a
    successful submit fails ALL 21 rows of that job with gRPC code 7: batch
    resolves the cache when each row EXECUTES. The end-of-wave cleanup deleted
    unconditionally, which re-introduces exactly that failure -- and turns
    "stranded but recoverable" into "certain to be void" for the job the resume
    above exists to rescue. Both orders are asserted."""
    _batch_cfg(cfg)
    reg = BREG.JobRegistry(cfg)
    _one_planned_job(reg, cfg)
    reg.mark_uploaded("German-def-w0-00", "files/1")
    reg.mark_submitted("German-def-w0-00", "batches/1", "JOB_STATE_RUNNING")
    handle = BC.CacheHandle(name="cachedContents/1",
                            declared_tokens=DECLARED_CACHE_TOKENS,
                            prompt_sha256="sha", model=cfg.gemini_model,
                            lang="German", kind="definition",
                            ttl_seconds=187200, created_at="now")
    reg.remember_cache("German/definition", handle.as_record())

    # order 1: a job is still in flight -> the cache stays
    report: dict = {}
    BT._end_of_wave_cache(cfg, "German", reg, batch_genai, report)
    assert batch_genai.caches_deleted == [], \
        "a job of this language is in flight; its rows would all be code 7"
    assert reg.cache_record("German/definition") is not None
    kept = report["cache"]["kept"]
    assert kept and kept[0]["jobs_in_flight"] == ["German-def-w0-00"]

    # order 2: nothing in flight -> the cache is deleted AND priced, because
    # storage is billed by the token-hour for as long as the TTL runs
    reg.mark_downloaded("German-def-w0-00", "r.jsonl",
                        job_state="JOB_STATE_SUCCEEDED")
    reg.mark_recovered("German-def-w0-00", {"written": {}})
    BT._end_of_wave_cache(cfg, "German", reg, batch_genai, report)
    assert batch_genai.caches_deleted == ["cachedContents/1"]
    assert reg.cache_record("German/definition") is None
    deleted = report["cache"]["deleted"]
    assert deleted[0]["deleted"] is True
    assert deleted[0]["storage_cost"]["usd"] > 0


def test_a_crashed_submit_leaves_its_cache_alive_for_the_resume(
        wave, cfg, registry, batch_genai):
    """The same rule, end to end: the job the next ingest is going to resume
    still has a cache to resolve when its rows execute."""
    _die_after_the_submit(batch_genai)
    with pytest.raises(RuntimeError):
        S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert batch_genai.caches_created
    assert batch_genai.caches_deleted == []
    reg = BREG.JobRegistry(cfg)
    assert reg.in_flight() and reg.cache_record("German/definition")
    # and once the resume has drained it, the cache is released
    S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    assert len(batch_genai.caches_deleted) == 1


def test_no_cache_object_is_created_for_a_wave_with_nothing_cacheable(
        cfg, batch_genai, batch_stats, no_sleep, registry):
    """An incremental wave with only expression cells left has nothing to cache,
    and a cache nobody references is storage billed by the token-hour. It also
    used to make G-CACHE adjudicate -- and fail -- a wave that never claimed a
    discount, which is the wall every re-run after an unrecovered definition
    request walked into."""
    entries = _workspace(cfg)
    _batch_cfg(cfg)
    tdir = cfg.json_dir / "translations" / "German"
    prov = S42._provenance(cfg.gemini_model, "v4-frozen", cfg.thinking_level)
    todo = _todo(cfg, {e["entry_id"]: e for e in entries})
    write_json(tdir / "definitions.json",
               {r["key"]: {"lemma": "L", "gloss": "G", "src_sha": r["src_sha"],
                           "provenance": prov}
                for r in todo if r["kind"] == "definition"})
    report = S42.run(cfg, registry, lang="German", confirm=True, do_gc=False)
    assert report["bill"]["German"]["definition_requests"] == 0
    assert batch_genai.caches_created == []
    assert report["batch"]["cache"]["skipped"][0]["lang"] == "German"
    # ...and the wave is not condemned for having nothing to cache
    assert report["batch"]["gates_ok"] is True
    assert len(read_json(tdir / "expressions.json")) == 1


# ---- F3: the TTL covers the window this transport actually waits ----------

def test_the_cache_ttl_covers_the_drain_window_this_transport_waits_for():
    """The artifact's TTL_POLICY is the criterion: "the cache must be alive when
    each ROW EXECUTES, so TTL must cover the whole drain window".

    The shipped estimate was 6 seconds per unit x a 1.5 margin, extrapolated from
    a 32-row probe job. Two things were wrong with it and the reviewers found one
    each: the unit was a CELL where the variable said `requests` (13,632 against
    3,642, so the answer was 34.1h -- adequate by accident), and the formula the
    variable name described gives 9.11h, which covers neither the documented 48h
    hard expiry nor the 50h this transport itself waits.
    """
    plan = BC.cache_ttl_plan(3642, poll_deadline_s=BT.JOB_WAIT_SECONDS,
                             factor=1.5)
    assert plan["requests"] == 3642
    assert plan["ttl_seconds"] >= BT.JOB_WAIT_SECONDS
    assert plan["ttl_seconds"] >= BW.DOCUMENTED_JOB_HARD_EXPIRY_SECONDS
    assert plan["decided_by"] == "documented drain window + margin"
    assert plan["ttl_seconds"] == BT.JOB_WAIT_SECONDS + BW.TTL_MARGIN_SECONDS

    # the two figures the reviewers measured on the old formula, pinned as the
    # numbers this TTL is NOT
    assert BC.ttl_seconds(BW.wall_clock_estimate_seconds(13632), 1.5) == 122688
    assert BC.ttl_seconds(BW.wall_clock_estimate_seconds(3642), 1.5) == 32778
    assert plan["ttl_seconds"] > 122688 and plan["ttl_seconds"] > 32778


def test_the_ttl_is_sized_from_requests_and_the_factor_still_bites():
    """`cfg.cache_ttl_factor` is not decoration: it governs a wave so large that
    even the documented window is optimistic. At the shipped size the window
    wins, which is why the throughput term alone could rot unnoticed."""
    small = BC.cache_ttl_plan(3642, poll_deadline_s=BT.JOB_WAIT_SECONDS,
                              factor=1.5)
    assert small["throughput_term"] < small["documented_window_term"]
    huge = BC.cache_ttl_plan(100000, poll_deadline_s=BT.JOB_WAIT_SECONDS,
                             factor=1.5)
    assert huge["decided_by"].endswith("throughput extrapolation")
    assert huge["ttl_seconds"] == BC.ttl_seconds(
        BW.wall_clock_estimate_seconds(100000), 1.5)
    # and the factor changes that answer, so removing it fails here
    assert BC.cache_ttl_plan(100000, poll_deadline_s=BT.JOB_WAIT_SECONDS,
                             factor=3.0)["ttl_seconds"] \
        > huge["ttl_seconds"]


def test_the_ttl_the_transport_asks_for_is_priced_and_affordable():
    """The cost of covering the window, computed rather than called negligible:
    a 1,135-token definition prompt held for the whole drain window."""
    plan = BC.cache_ttl_plan(3642, poll_deadline_s=BT.JOB_WAIT_SECONDS,
                             factor=1.5)
    handle = BC.CacheHandle(name="cachedContents/1",
                            declared_tokens=DECLARED_CACHE_TOKENS,
                            prompt_sha256="sha", model="gemini-3.7-flash",
                            lang="German", kind="definition",
                            ttl_seconds=plan["ttl_seconds"], created_at="now")
    cost = BC.storage_cost(handle, plan["ttl_seconds"] / 3600.0, "batch")
    assert cost["usd"] is not None
    assert cost["usd"] < 0.031, cost
    assert cost["usd"] * 4 < 0.13, "four languages, still under a tenth of a cent"


def test_the_cache_is_asserted_before_every_submit_not_once_per_language(
        wave, cfg, registry, batch_genai, monkeypatch):
    """Patch plan 5.4 asks for `caches.get` before EVERY submit and
    `caches.update` when the remaining life is short. Measured call sequence for
    an invocation that submitted two jobs: ['create', 'delete'] -- zero gets,
    zero updates. With one job in flight and a 50-hour poll deadline, "once per
    language" and "once per submit" are up to two days apart."""
    calls = []
    real = BC.remaining_seconds

    def watched(client, handle, **kw):
        calls.append("get")
        return real(client, handle, **kw)

    monkeypatch.setattr(BC, "remaining_seconds", watched)
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    # one create for the language, then a get before the definition job's submit
    assert batch_genai.caches_created and calls.count("get") >= 1
    report = read_json(cfg.report_dir / "translate_report.json")
    assert report["batch"]["cache"]["verified"], \
        "the cache's remaining life was never looked at before a submit"


def test_a_cache_whose_life_is_short_is_extended_before_the_submit(
        wave, cfg, registry, batch_genai, monkeypatch):
    """The caches.update half of 5.4, which had no test at all: the one named
    test_the_cache_ttl_is_extended_before_a_later_submit only asserted that a
    second cache was not CREATED, and the fake reported a life long enough that
    the extend branch could never run."""
    class _NearlyExpired:
        def __init__(self, name, tokens):
            self.name = name
            self.usage_metadata = {"totalTokenCount": tokens}
            # 10 minutes left, far under the TTL the transport wants
            self.expire_time = (
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc)
                + __import__("datetime").timedelta(minutes=10)).isoformat()

    monkeypatch.setattr(batch_genai.caches, "get",
                        lambda name=None: _NearlyExpired(
                            name, batch_genai.cache_tokens))
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    assert batch_genai.caches_updated, "a nearly-expired cache was not extended"
    report = read_json(cfg.report_dir / "translate_report.json")
    extended = report["batch"]["cache"]["extended"]
    assert extended and extended[0]["ttl"] >= BT.JOB_WAIT_SECONDS


# ---- F4: a FAILED job releases its wave, on purpose -----------------------

def test_a_terminal_failed_job_releases_its_wave_for_a_corrected_resubmission(
        wave, cfg, registry, batch_genai):
    """find_by_fingerprint excluded FAILED, which read like "a failed wave may be
    resubmitted" -- but the job id is deterministic and plan()'s first line
    refuses on it, so the (lang, kind, wave, index) slot was wedged for good and
    the error message named the collision without saying what to do."""
    # the service says the job EXPIRED: it produced nothing and billed nothing
    batch_genai.no_dest = True
    batch_genai.terminal_state = "JOB_STATE_EXPIRED"
    with pytest.raises(FatalError):
        S42.run(cfg, registry, lang="German", confirm=True)
    reg = BREG.JobRegistry(cfg)
    failed = reg.in_state(BREG.FAILED)
    assert failed and all(j["resubmittable"] is True for j in failed)
    assert reg.is_resubmittable("German-def-w0-00")
    # the fingerprint is released, and the reopened job takes an attempt suffix
    # rather than colliding with the record of what failed
    assert reg.next_job_id("German-def-w0-00") == "German-def-w0-00-a2"

    batch_genai.no_dest = False
    batch_genai.terminal_state = "JOB_STATE_SUCCEEDED"
    S42.run(cfg, registry, lang="German", confirm=True)
    assert len(read_json(cfg.json_dir / "translations" / "German"
                         / "definitions.json")) == 2
    reg = BREG.JobRegistry(cfg)
    assert any(j.endswith("-a2") for j in reg.jobs())


def test_a_job_whose_results_we_could_not_fetch_is_never_reopened(cfg):
    """The distinction mark_failed used to lose. "The service says FAILED,
    CANCELLED or EXPIRED" means nothing ran and nothing was billed. "We did not
    get the results of a job the service says SUCCEEDED" is money already
    committed, and batches.create is not idempotent, so reopening it is a second
    charge for the same rows."""
    reg = BREG.JobRegistry(cfg)
    fp = BREG.wave_fingerprint(model="m", prompt_id="v4-frozen", lang="German",
                               keys=["a"], cache_name=None)
    reg.plan("German-def-w0-00", fingerprint=fp, lang="German",
             kind="definition", model="m", prompt_id="v4-frozen",
             cache_name=None, declared_cache_tokens=None,
             cache_prompt_sha256=None, jsonl_path="x.jsonl",
             plan=[{"key": "a"}], enqueued_tokens=1, wave=0)
    reg.mark_failed("German-def-w0-00", "results could not be downloaded",
                    job_state="SUCCEEDED", resubmittable=False)
    assert reg.is_resubmittable("German-def-w0-00") is False
    # the fingerprint stays reserved...
    assert reg.find_by_fingerprint(fp) == "German-def-w0-00"
    # ...and the id refuses, with the console lookup spelled out
    with pytest.raises(FatalError) as exc:
        reg.next_job_id("German-def-w0-00")
    assert "not marked resubmittable" in str(exc.value)
    assert "resubmittable" in str(exc.value) and "console" in str(exc.value)


# ---- F5b: the denominator is this language's own declaration --------------

def test_the_g_cache_denominator_is_each_languages_own_declaration(
        cfg, batch_genai, batch_stats, no_sleep, registry):
    """The English definition prompt is 1,092 tokens against 1,135 for the other
    three, and the old code sent min() of everything it could find to every
    language's gate. Two consequences, both measured: `cached == declared` per
    row -- the criterion the docstring calls primary -- was dead for three
    languages out of four, and a stale small declaration from another run
    RAISED the share, which is the unsafe direction."""
    _workspace(cfg)
    _batch_cfg(cfg)
    cfg.langs = ["German", "English"]
    sizes = {"German": 1135, "English": 1092}

    real_create = BC.create

    def per_language(client, *, model, system_text, lang, kind, ttl, stats,
                     display_name=""):
        batch_genai.cache_tokens = sizes[lang]
        return real_create(client, model=model, system_text=system_text,
                           lang=lang, kind=kind, ttl=ttl, stats=stats,
                           display_name=display_name)

    import ankidkdeck.batch.transport as _t
    _t.cache_lifecycle.create = per_language
    try:
        report = S42.run(cfg, registry, confirm=True)
    finally:
        _t.cache_lifecycle.create = real_create
    seen = report["batch"]["gates"]["declared_cache_tokens_by_language"]
    assert seen == sizes, seen
    assert report["batch"]["gates_ok"] is True
    rows = [r for r in report["batch"]["gates"]["rows"]
            if r["id"] == "G-CACHE"]
    assert {r["extra"]["lang"] for r in rows} == {"German", "English"}


def test_a_stale_declaration_in_the_registry_cannot_reach_this_waves_gate(
        wave, cfg, registry, batch_genai):
    """The unsafe direction, and the one the old fallback took.

    G-CACHE's share is sum(cached) / (declared x requests), so a SMALLER
    denominator raises it: a stale `declared_cache_tokens` from another language
    or another run does not fail a healthy wave, it passes a broken one. The old
    fallback was min() over reg.jobs() with no language filter -- and because the
    cache record is deleted before the gates run, that fallback was the normal
    path rather than the exceptional one.

    A left-over job with a tiny declaration is enough to demonstrate it, and the
    ordering is why the two-language test above cannot: within one invocation the
    languages are drained one at a time, so min() happens to give each language
    its own number until something older is on file.
    """
    reg = BREG.JobRegistry(cfg)
    _one_planned_job(reg, cfg, job_id="Faroese-def-w9-00", wave=9)
    reg.jobs()["Faroese-def-w9-00"]["lang"] = "Faroese"
    reg.jobs()["Faroese-def-w9-00"]["declared_cache_tokens"] = 5
    reg.save()

    report = S42.run(cfg, registry, lang="German", confirm=True)
    seen = report["batch"]["gates"]["declared_cache_tokens_by_language"]
    assert seen == {"German": DECLARED_CACHE_TOKENS}, seen
    assert report["batch"]["gates_ok"] is True
    # and the number that would have been used, kept as the record of the risk
    assert min(5, DECLARED_CACHE_TOKENS) == 5


def test_the_declared_size_survives_the_gap_between_submit_and_ingest(
        wave, cfg, registry, batch_genai):
    """A wave is TWO invocations and the cache object is deleted at the end of the
    second one, BEFORE the gates run. Reading the denominator off the live cache
    record therefore read it off an empty dict, and the fallback was min() over
    every job the workspace ever ran -- so the fallback was the normal path."""
    S42.run(cfg, registry, lang="German", confirm=True, phase="submit")
    reg = BREG.JobRegistry(cfg)
    assert reg.declared_for("German") == DECLARED_CACHE_TOKENS
    report = S42.run(cfg, registry, lang="German", confirm=True, phase="ingest")
    # the cache object is gone by the time the gates ran...
    reg = BREG.JobRegistry(cfg)
    assert reg.cache_record("German/definition") is None
    # ...and the denominator was still this language's own number
    assert report["batch"]["gates"]["declared_cache_tokens_by_language"] == {
        "German": DECLARED_CACHE_TOKENS}
    assert report["batch"]["gates_ok"] is True


# ---- F7: the request ceiling belongs to a transport -----------------------

def test_the_batch_request_ceiling_matches_the_transports_own_bound():
    """s42.BATCH_REQUEST_CEILING_FACTOR is spelled out rather than imported, so
    that bill_row does not drag ankidkdeck.batch onto the dry path. This is the
    pin that keeps the duplicate honest."""
    assert S42.BATCH_REQUEST_CEILING_FACTOR == 1 + BW.MAX_RETRY_WAVES


def test_the_request_ceiling_on_the_bill_is_the_one_this_transport_has(
        cfg, registry, batch_stats):
    """The batch path never goes through _generate, so the interactive 25x
    transport ladder does not apply to it. Quoting it anyway over-stated the
    ceiling about 6x on the one line a human reads before --confirm-spend."""
    _workspace(cfg)
    _batch_cfg(cfg)
    batch = S42.run(cfg, registry, lang="German", confirm=False)["bill"]["German"]
    cfg.mode = "standard"
    cfg.cache_enabled = False
    plain = S42.run(cfg, registry, lang="German", confirm=False)["bill"]["German"]
    assert batch["requests_min"] == plain["requests_min"]
    assert batch["requests_max"] == batch["requests_min"] * 4
    assert plain["requests_max"] == plain["requests_min"] * 25
    assert batch["requests_max_transport"] == "batch"
    assert "does not go through _generate" in batch["requests_max_basis"]
    assert "transport ladder" in plain["requests_max_basis"]


# ---- F8: a measurement of the expression family reaches the bill ----------

def test_a_measured_expression_family_reaches_the_bill(cfg):
    """The canary's whole purpose, and it was disconnected. The prior was the
    highest max across every non-definition family, so `LOW|other(1503) = 275`
    (the ranking prompt) dominated for ever: the canary could write
    `LOW|expression(...) = 0` to disk and the bill would not move a cent."""
    stats = dict(BATCH_CONSTANTS)
    stats["thinking"] = dict(BATCH_CONSTANTS["thinking"])
    stats["thinking"]["THINKING_AT_LOW_BY_PROMPT_FAMILY"] = {
        "LOW|definition(5123)": {"max": 0, "n_observations": 62},
        "LOW|other(1503)": {"max": 275, "n_observations": 2},
    }
    before, why = S42.unmeasured_thinking_prior(stats, "expression")
    assert before == 275 and "highest non-definition" in why

    key = S42.thinking_family_key("expression", "German")
    stats["thinking"]["THINKING_AT_LOW_BY_PROMPT_FAMILY"][key] = {"max": 0}
    after, after_why = S42.unmeasured_thinking_prior(stats, "expression")
    assert after == 0, "the measurement did not reach the bill"
    assert "measured for this request kind" in after_why
    # the ranking prompt's 275 is still the answer for a kind nobody measured
    assert S42.unmeasured_thinking_prior(stats, "pos")[0] == 275


def test_the_definition_family_is_matched_without_its_character_count():
    """The artifact carries `LOW|definition(5123)` while the four live definition
    prompts are 5,160 / 4,985 / 5,134 / 5,160 characters. An exact-key lookup
    misses all four and throws away a measured zero from 62 observations, so the
    key is matched on (level, family) and the count is disclosure."""
    stats = {"thinking": {"THINKING_AT_LOW_BY_PROMPT_FAMILY": {
        "LOW|definition(5123)": {"max": 0, "n_observations": 62},
        "LOW|other(1503)": {"max": 275}}}}
    live = {lang: len(S42.system_prompt("definition", lang))
            for lang in ("Chinese", "English", "German", "Spanish")}
    assert 5123 not in live.values(), \
        "the artifact's key no longer matches ANY live prompt; that is the point"
    value, why = S42.unmeasured_thinking_prior(stats, "definition")
    assert value == 0 and "measured for this request kind" in why


def test_an_unmeasured_kind_falls_back_and_never_to_zero():
    """A miss must return the highest LOW anybody has measured. Returning 0 would
    let a re-worded prompt (a new family key) silently book zero thinking on
    1,922 requests per language, which is $0.99 per language of the one column
    that decides go/no-go."""
    stats = {"thinking": {"THINKING_AT_LOW_BY_PROMPT_FAMILY": {
        "LOW|other(1503)": {"max": 275}}}}
    assert S42.unmeasured_thinking_prior(stats, "expression")[0] == 275
    assert S42.unmeasured_thinking_prior(stats, "brand_new_kind")[0] == 275
    # and with no artifact at all it is the module constant, still not zero
    value, why = S42.unmeasured_thinking_prior({}, "expression")
    assert value == S42.UNMEASURED_THINKING_PRIOR > 0
    assert "UNMEASURED_THINKING_PRIOR" in why


def test_the_canary_lowers_the_expression_half_of_the_bill(
        cfg, batch_genai, batch_stats, no_sleep, registry):
    """End to end: the canary job measures the family, writes it to the artifact,
    and the NEXT bill books the measurement instead of the prior."""
    stats = dict(BATCH_CONSTANTS)
    stats["thinking"] = dict(BATCH_CONSTANTS["thinking"],
                             THINKING_AT_LOW_BY_PROMPT_FAMILY={
                                 "LOW|definition(5123)": {"max": 0},
                                 "LOW|other(1503)": {"max": 275}})
    write_json(cfg.probe_stats_path, stats)
    _workspace(cfg)
    _batch_cfg(cfg)
    before = S42.run(cfg, registry, lang="German", confirm=False)
    assert before["bill"]["German"]["tokens"][
        "thinking_per_request_by_kind"]["expression"] == 275

    S42.run(cfg, registry, lang="German", confirm=True)
    after = S42.run(cfg, registry, lang="German", confirm=False)
    tokens = after["bill"]["German"]["tokens"]
    assert tokens["thinking_per_request_by_kind"]["expression"] == 0
    # ...and the definition side is still the measured zero it always was
    assert tokens["thinking_basis"]["definition"] == S42.MEASURED_THINKING_BASIS
    # ...while pos, which nobody measured, still carries the prior
    assert tokens["thinking_per_request_by_kind"]["pos"] == 275


# ---- F9: the determinism claim, with teeth -------------------------------

def test_the_jsonl_is_byte_identical_even_when_the_keys_arrive_in_another_order(
        cfg, tmp_path):
    """`sort_keys` is the determinism, and nothing pinned it: removing it left
    all 572 tests green, because writing the SAME dict twice in one process
    cannot detect it -- CPython preserves insertion order. This builds an
    equivalent row with a DIFFERENT insertion order and compares raw bytes."""
    row = BJ.build_row("def__German__1__00",
                       _fake_request(cfg, cache="cachedContents/abc"),
                       schema_serializer=_identity_schema)
    shuffled = {k: row[k] for k in reversed(list(row))}
    shuffled["request"] = {k: row["request"][k]
                           for k in reversed(list(row["request"]))}
    assert list(shuffled) != list(row), "the fixture has to differ in ORDER"
    assert shuffled == row, "...and in nothing else"
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    BJ.write_jsonl(first, [row])
    BJ.write_jsonl(second, [shuffled])
    assert first.read_bytes() == second.read_bytes()


def test_the_production_writer_is_semantically_identical_to_the_probe_writer(
        cfg, tmp_path):
    """The honest form of the claim. The probe's reference writer used a plain
    json.dumps, so its bytes depend on insertion order; ours sorts. The rows are
    the same OBJECT and NOT the same bytes, and "byte-identical to the probe
    writer" was wrong in the reports even though the conclusion it supported --
    the shape on the wire is the shape the API accepted -- is right."""
    row = BJ.build_row("def__German__1__00",
                       _fake_request(cfg, cache="cachedContents/abc"),
                       schema_serializer=_identity_schema)
    probe_style = json.dumps(row, ensure_ascii=False)
    ours = json.dumps(row, ensure_ascii=False, sort_keys=True)
    assert json.loads(probe_style) == json.loads(ours)
    unsorted = {"request": row["request"], "key": row["key"]}
    assert json.dumps(unsorted, ensure_ascii=False) != ours, \
        "with a different insertion order the unsorted bytes differ"
    assert json.dumps(unsorted, ensure_ascii=False, sort_keys=True) == ours
