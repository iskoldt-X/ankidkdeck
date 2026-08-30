"""The batch transport: the JSONL row, the keys, the fingerprint, the wave
splitter, the cache lifecycle, the key-based reconciliation and the retry bound.

EVERY TEST HERE IS OFFLINE. The service is a fake that reads the JSONL this
package actually wrote and answers from it, so the round trip exercised is the
real one: build a request through CallContext, serialize it, "upload" it,
"download" a result file, reconcile it on the echoed key and write the cells.

The invariants being pinned were measured on a real 32-row batch job (probe wave
3) and are catalogued in specs/v3-translate-patch-plan.md section 1.6. Where a
number appears in an assertion it comes from work/probes/stats.json, not from
this file.
"""

import json
import os
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
        # The real service does NOT return the rows in input order: measured
        # 2026-08-27, it concatenates ~1000-row shards out of order. With this
        # set, the fake reverses its answer lines -- the smallest permutation
        # that breaks the position on a job too small to have shards.
        self.shuffle_output = False
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
        answers = [json.dumps(self.svc.answer(row, i)) + "\n"
                   for i, row in enumerate(rows)]
        if self.svc.shuffle_output:
            answers.reverse()
        self.svc.outputs[out_name] = "".join(answers)
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
    assert "cannot be skipped" in str(exc.value)


# ==========================================================================
# 5.3 -- reconciliation BY KEY
#
# The output order is NOT the input order. Measured 2026-08-27 on the first real
# paid wave (Chinese-def-w0-00, 3,644 rows): the service returned four ~1000-row
# shards, each internally in input order, concatenated in the order
# [0-999], [3000-3643], [2000-2999], [1000-1999]. Positions 0-999 agreed, so the
# divergence starts at exactly row 1000 -- invisible to every probe, all of
# which were <= 32 rows. The fixtures below use that exact shape.
# ==========================================================================

# The real wave's shard geometry, verbatim: sizes in input order, then the order
# the service concatenated them in.
REAL_SHARDS = (1000, 1000, 1000, 644)
REAL_SHARD_ORDER = (0, 3, 2, 1)


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


def _wide_plan(n_rows, kind="definition", n=2):
    """A plan big enough to cross the ~1000-row shard boundary, unique keys."""
    return [{"pos": i, "key": "def__Chinese__%d__00" % (11000000 + i),
             "kind": kind, "n": n, "label": "hus", "cap": 1024,
             "cells": [{"key": "c%d-%d" % (i, j), "src_sha": "s%d-%d" % (i, j)}
                       for j in range(n)]}
            for i in range(n_rows)]


def _shard_shuffled(plan, sizes=REAL_SHARDS, order=REAL_SHARD_ORDER):
    """The result file as the service really produced it: shards permuted.

    Each shard stays internally in input order -- that part of the docs held --
    and the shards are concatenated in `order`.
    """
    bounds, at = [], 0
    for size in sizes:
        bounds.append((at, at + size))
        at += size
    assert at == len(plan), "fixture shard sizes must cover the plan"
    lines = []
    for shard in order:
        lo, hi = bounds[shard]
        lines.extend(_line(row["key"]) for row in plan[lo:hi])
    return lines


def test_the_result_file_is_joined_on_the_key_not_on_the_position():
    """THE REFUTED ASSUMPTION. 3,644 rows in the real wave's shard order.

    The old code did zip(plan, lines). On the real file it REFUSED at row 1000,
    because every line happened to carry its key and the cross-check fired --
    that refusal is how the false guarantee was found. Had the key echo been
    absent it would have written 2,644 of the 3,644 rows onto the wrong senses
    in silence. The join is now the echoed key and the position is a note.
    """
    plan = _wide_plan(sum(REAL_SHARDS))
    lines = _shard_shuffled(plan)
    assert len(lines) == 3644
    # The defect's shape, asserted so the fixture cannot drift into agreeing:
    # position 999 still lines up, position 1000 does not, and a positional zip
    # would pair 2,644 rows wrongly -- the count the real file produced.
    assert lines[999]["key"] == plan[999]["key"]
    assert lines[1000]["key"] != plan[1000]["key"]
    assert sum(1 for row, line in zip(plan, lines)
               if row["key"] != line["key"]) == 2644

    report = {}
    out = BR.reconcile(plan, lines, report=report)

    # One outcome per PLANNED row, in PLAN order, each carrying its own answer.
    assert len(out) == len(plan)
    assert [o.key for o in out] == [r["key"] for r in plan]
    assert [o.pos for o in out] == [r["pos"] for r in plan]
    assert all(o.ok for o in out)
    assert all(o.key_echoed == o.key for o in out)
    # Every row's answer really came from the shuffled line that echoed its key.
    assert out[1000].result_pos == 2644
    assert out[3000].result_pos == 1000
    # The order cross-check reports the shuffle and does NOT gate on it.
    assert report["in_input_order"] is False
    assert report["agreeing_prefix"] == 1000
    assert report["first_divergence"] == 1000
    assert report["positions_agreeing"] == 1000
    assert report["joined_by_key"] == 3644
    assert report["joined_without_key"] == 0
    # 3, not 4: shard 0 and shard 3 happen to ascend across their boundary, so
    # they read as one run. This is the number the real file produced.
    assert report["ascending_runs"] == 3
    assert "NOT in input order" in BR.order_note(report)


def test_a_probe_scale_batch_that_does_arrive_in_order_still_works():
    """Regression at probe scale: 32 rows, order preserved, which is what every
    probe wave measured and why the false guarantee survived to a paid job."""
    plan = _wide_plan(32)
    report = {}
    out = BR.reconcile(plan, [_line(row["key"]) for row in plan], report=report)
    assert all(o.ok for o in out)
    assert [o.result_pos for o in out] == list(range(32))
    assert report["in_input_order"] is True
    assert report["agreeing_prefix"] == 32
    assert report["ascending_runs"] == 1
    assert report["first_divergence"] is None
    assert "arrived in input order" in BR.order_note(report)


def test_a_length_mismatch_is_fatal_and_writes_nothing():
    with pytest.raises(FatalError) as exc:
        BR.reconcile(_plan(3), [_line("a"), _line("b")])
    assert "Nothing was written" in str(exc.value)
    assert "2 line(s) for a job of 3 row(s)" in str(exc.value)


def test_a_duplicate_key_in_the_results_is_fatal():
    """Two answers for one request. Which one is the answer? Refuse."""
    plan = _plan(3)
    lines = [_line(plan[0]["key"]), _line(plan[1]["key"]),
             _line(plan[1]["key"])]
    with pytest.raises(FatalError) as exc:
        BR.reconcile(plan, lines)
    assert "duplicate key" in str(exc.value)
    assert plan[1]["key"] in str(exc.value)
    assert "Nothing was written" in str(exc.value)


def test_a_key_the_job_never_sent_is_fatal_and_names_what_is_missing():
    """Extra and missing are one break: with the count equal, a key that is not
    in the plan means a planned key has no answer."""
    plan = _plan(3)
    lines = [_line(plan[0]["key"]), _line("def__German__999999__00"),
             _line(plan[2]["key"])]
    with pytest.raises(FatalError) as exc:
        BR.reconcile(plan, lines)
    assert "never sent" in str(exc.value)
    assert "def__German__999999__00" in str(exc.value)
    assert plan[1]["key"] in str(exc.value)      # the missing one is named


def test_a_planned_key_with_no_result_is_fatal():
    """Missing on its own: the count matches because an unkeyed SUCCESS row took
    the slot, and an unkeyed success row would write cells on an assumption."""
    plan = _plan(3)
    payload = _line(plan[1]["key"])
    payload.pop("key")
    with pytest.raises(FatalError) as exc:
        BR.reconcile(plan, [_line(plan[0]["key"]), payload,
                            _line(plan[2]["key"])])
    assert "no key and no error" in str(exc.value)
    assert "Nothing was written" in str(exc.value)


def test_one_keyless_error_row_is_attributable_because_the_slot_is_forced(cfg):
    """W0-4, re-derived. N-1 keyed rows plus one BARE gRPC status with no key at
    all: the keyed rows claim their own plan rows and exactly one slot is left,
    so the attribution is forced rather than assumed."""
    plan = _plan(3)
    lines = [_line(plan[0]["key"]),
             {"error": {"code": 7,
                        "message": "The caller does not have permission"}},
             _line(plan[2]["key"])]
    report = {}
    out = BR.reconcile(plan, lines, report=report)
    assert [o.ok for o in out] == [True, False, True]
    assert out[1].key == plan[1]["key"]
    assert out[1].key_echoed is None
    assert out[1].error["code"] == 7
    assert "does not have permission" in out[1].why()
    assert report["joined_by_key"] == 2
    assert report["joined_without_key"] == 1


def test_a_keyless_error_row_survives_the_shuffle_too():
    """The forced slot is forced by the KEY SET, not by the position -- so it
    still works when the keyed rows arrive in a different order entirely."""
    plan = _plan(3)
    lines = [_line(plan[2]["key"]),
             _line(plan[0]["key"]),
             {"error": {"code": 7, "message": "no permission"}}]
    out = BR.reconcile(plan, lines)
    assert [o.ok for o in out] == [True, False, True]
    assert out[1].error["code"] == 7
    assert out[0].result_pos == 1 and out[2].result_pos == 0


def test_two_different_keyless_error_rows_are_ambiguous_and_fatal():
    """Nothing places them: the order is not a contract, so a plausible guess is
    exactly the mis-attribution the guard exists to prevent."""
    plan = _plan(4)
    lines = [_line(plan[0]["key"]),
             {"error": {"code": 7, "message": "The caller does not have "
                                              "permission"}},
             {"error": {"code": 3, "message": "Request contains an invalid "
                                             "argument."}},
             _line(plan[3]["key"])]
    with pytest.raises(FatalError) as exc:
        BR.reconcile(plan, lines)
    assert "not identical to one another" in str(exc.value)
    assert "line(s) 1, 2" in str(exc.value)      # the rows are named
    assert plan[1]["key"] in str(exc.value)      # so are the candidate slots
    assert "Nothing was written" in str(exc.value)


def test_a_dead_cache_fails_every_row_the_same_way_and_stays_recoverable():
    """Code 7 on every row (prompt=0, billed $0) is the failure this has to
    survive: the rows are keyless AND byte-identical, so no assignment can
    change any outcome and the whole job is attributed and retryable."""
    plan = _plan(5)
    dead = {"error": {"code": 7, "message": "The caller does not have "
                                           "permission"}}
    report = {}
    out = BR.reconcile(plan, [dict(dead) for _ in plan], report=report)
    assert [o.key for o in out] == [r["key"] for r in plan]
    assert all(o.error["code"] == 7 and not o.ok for o in out)
    assert report["joined_without_key"] == 5
    for outcome in out:
        assert BW.retry_decision(outcome, cap=1024,
                                 ceiling=8192)["why"] == "cache_missing"


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


def test_the_cells_land_on_their_own_senses_when_the_output_is_out_of_order(
        wave, cfg, registry, batch_genai):
    """END TO END at the real defect. The reconcile tests above prove the join;
    this one runs the whole wave with the service returning the rows in another
    order and checks the CELLS -- which is where the damage would have been,
    because _absorb zips the plan row's cells against the outcome's items. Every
    gloss names the request it came from, so a shifted attribution is loud."""
    batch_genai.shuffle_output = True

    def responder(row, pos):
        array, n = n_of(row)
        return ok_line(row, batch_genai.cache_tokens,
                       items=[{"lemma": "L%d" % i,
                               "gloss": "G-%s-%d" % (row["key"], i)}
                              for i in range(n)])

    batch_genai.responder = responder
    S42.run(cfg, registry, lang="German", confirm=True)
    tables = {
        "definition": read_json(cfg.json_dir / "translations" / "German"
                                / "definitions.json"),
        "expression": read_json(cfg.json_dir / "translations" / "German"
                                / "expressions.json")}
    jobs = read_json(cfg.work_dir / "batch" / "jobs.json")["jobs"]
    checked = 0
    for job in jobs.values():
        table = tables[job["kind"]]
        for row in job["plan"]:
            for i, cell in enumerate(row["cells"]):
                assert (table[cell["key"]]["gloss"]
                        == "G-%s-%d" % (row["key"], i))
                checked += 1
    assert checked == 3, "the fixture wave is 2 definitions and 1 expression"


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


# ==========================================================================
# The real paid artifact
#
# Everything above is synthetic. This one reads the actual 3,644-row result file
# the first paid wave produced and the actual plan it was submitted from, and is
# the reason the fix is believed rather than argued. Read-only: reconcile()
# writes nothing, and neither does this test.
#
# It SKIPS when the artifact is not on the box (CI, a fresh clone). Paths are
# overridable so the evidence can live anywhere:
#   ANKIDKDECK_PAID_JOBS_JSON  ANKIDKDECK_PAID_RESULTS
# ==========================================================================

PAID_JOBS_JSON = os.environ.get(
    "ANKIDKDECK_PAID_JOBS_JSON",
    os.path.expanduser("~/v3run/work/batch/jobs.json"))
PAID_RESULTS = os.environ.get(
    "ANKIDKDECK_PAID_RESULTS",
    os.path.expanduser(
        "~/v3run/PAID-RESULTS-BACKUP-Chinese-def-w0-00_results.jsonl"))
PAID_JOB_ID = "Chinese-def-w0-00"


@pytest.mark.skipif(
    not (os.path.exists(PAID_JOBS_JSON) and os.path.exists(PAID_RESULTS)),
    reason="the paid Chinese-def-w0-00 artifact is not on this box")
def test_the_real_paid_result_file_reconciles_completely_on_the_key():
    """THE MONEY TEST. 3,644 paid rows, out of order, joined with zero residue.

    Measured 2026-08-27 against work/batch/jobs.json and the read-only backup of
    the downloaded results. The old positional code refused this exact file at
    row 1000, which is what exposed the false order guarantee; a positional zip
    of this file pairs 2,644 of its 3,644 rows with the wrong request.
    """
    jobs = read_json(PAID_JOBS_JSON)["jobs"]
    plan = jobs[PAID_JOB_ID]["plan"]
    lines = BREG.read_results(PAID_RESULTS)
    assert len(plan) == 3644 and len(lines) == 3644

    report = {}
    out = BR.reconcile(plan, lines, report=report)

    # Complete join, zero residue: every planned row got its own paid answer.
    assert len(out) == 3644
    assert report["joined_by_key"] == 3644
    assert report["joined_without_key"] == 0
    assert all(o.key_echoed == o.key for o in out)
    assert [o.key for o in out] == [row["key"] for row in plan]
    assert len({o.result_pos for o in out}) == 3644
    assert sum(1 for row, line in zip(plan, lines)
               if row["key"] != line["key"]) == 2644, \
        "the positional zip the old code used pairs 2,644 rows wrongly"

    # The defect, on the real bytes: the file is NOT in input order, and the
    # first 1000 rows are exactly why no probe could have caught it.
    assert report["in_input_order"] is False
    assert report["agreeing_prefix"] == 1000
    assert report["first_divergence"] == 1000
    assert report["positions_agreeing"] == 1000
    assert report["ascending_runs"] == 3

    # What the ingest will do with it: 3,640 rows land, 4 are MAX_TOKENS
    # truncations for the retry ladder to re-buy at a higher cap. No error rows
    # at all -- the cache held on every row (cachedContentTokenCount 1139).
    assert sum(1 for o in out if o.ok) == 3640
    assert [o.finish_reason for o in out if not o.ok] == ["MAX_TOKENS"] * 4
    assert not [o for o in out if o.error]
    assert {o.usage.get("cachedContentTokenCount") for o in out} == {1139}


# ==========================================================================
# tools/prompt_thinking_ab.py --mode batch
# ==========================================================================
#
# The A/B moved to the batch surface because the interactive one 503-storms
# (83.9% of requests during the storm measured in the Chinese month), and an A/B
# whose two arms meet a storm at different rates measures the weather. These
# tests drive the tool against the SAME fake service the transport's own tests
# use, so what is exercised is the real round trip: build the request through
# CallContext, serialize it, "upload" it, "download" a result file, join it on
# the echoed key.

def _ab_tool():
    """tools/prompt_thinking_ab.py, imported by path.

    It is a script rather than a package module and there is no other importer,
    so the import is done here rather than by putting tools/ on sys.path for the
    whole suite.
    """
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "tools"
            / "prompt_thinking_ab.py")
    spec = importlib.util.spec_from_file_location("prompt_thinking_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backfill_tool():
    """tools/backfill_probe_stats.py, imported by path.

    Imported so the calls.jsonl rows the batch arm writes are checked against
    the READER that consumes them, not against a copy of its rules.
    """
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "tools"
            / "backfill_probe_stats.py")
    spec = importlib.util.spec_from_file_location("backfill_probe_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ab_workspace(cfg, n_entries=3, n_senses=3):
    """Entries the A/B's own picker will accept: 2-8 senses, all with text."""
    rows = [_entry(entry_id="1102172%d" % i, n_senses=n_senses, n_exprs=1)
            for i in range(n_entries)]
    _workspace(cfg, rows)
    return rows


def _ab_setup(cfg, tool, lang="German", n_entries=3):
    from ankidkdeck import prompts
    _ab_workspace(cfg, n_entries=n_entries)
    _batch_cfg(cfg, cache=True)
    prompts.reset()
    prompts.activate(cfg, prompt_id=tool.ARMS["lean"][0])
    batches = tool.pick_entries(cfg, lang, n_entries)
    assert batches, "the A/B picker found no entry with 2-8 senses"
    return batches, cfg.work_dir / "probes" / "calls.jsonl"


def _no_sleep(_seconds):
    return None


def test_the_ab_batch_run_is_one_job_per_arm_with_that_arms_prompt_cached(
        cfg, registry, batch_genai, batch_stats):
    """The end-to-end run: two jobs, two caches, two system prompts, and every
    downstream artifact in the shape the interactive arm produced."""
    from ankidkdeck import prompts
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)

    results, report = tool.run_batch_ab(cfg, "German", ["lean", "rich"],
                                        batches, ledger, client=batch_genai,
                                        sleep=_no_sleep)

    # ---- one job per arm, and the arms are told apart by the wave tag that
    # transport._resume_in_flight and registry.wave_fingerprint both read. The
    # tag also carries the CELL SET's digest, so it is derived here rather than
    # spelled out -- a hard-coded tag would pin this fixture's digest.
    todo = tool.todo_rows_for(cfg, "German", batches)
    sel = tool.ab_selection_id(todo)
    lean_tag = tool.ab_wave_tag("German", "lean", sel)
    rich_tag = tool.ab_wave_tag("German", "rich", sel)
    assert lean_tag == "AB-German-lean-%s" % sel and len(sel) == 8
    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    jobs = reg.jobs()
    assert len(jobs) == 2
    assert {j["lang"] for j in jobs.values()} == {lean_tag, rich_tag}
    assert sorted(jobs) == sorted("%s-def-w0-00" % t
                                  for t in (lean_tag, rich_tag))
    assert all(j["state"] == BREG.RECOVERED for j in jobs.values())
    # two distinct fingerprints: the cache name and the row keys both differ
    assert len({j["fingerprint"] for j in jobs.values()}) == 2

    # ---- each arm's own system prompt went INTO that arm's own cache
    assert len(batch_genai.caches_created) == 2
    systems = [c["config"].kwargs["system_instruction"]
               for c in batch_genai.caches_created]
    assert len(set(systems)) == 2
    lean_text = prompts.build_definition_prompt("German",
                                                prompt_id="v4-frozen")
    rich_text = prompts.build_definition_prompt("German",
                                                prompt_id="rich-core-1")
    assert systems == [lean_text, rich_text]
    assert len(rich_text) > len(lean_text)
    # ...and both caches were deleted once their job was terminal and downloaded
    assert len(batch_genai.caches_deleted) == 2

    # ---- the request bodies are the batch shape, cache at the TOP LEVEL
    uploaded = [json.loads(x)
                for blob in batch_genai.uploads.values()
                for x in blob.splitlines() if x.strip()]
    assert len(uploaded) == 2 * len(batches)
    for row in uploaded:
        req = row["request"]
        assert req["cachedContent"].startswith("cachedContents/")
        assert "systemInstruction" not in req          # hard 400 if both
        gen = req["generationConfig"]
        assert gen["thinkingConfig"] == {"thinkingLevel": "LOW"}
        assert gen["responseSchema"]["properties"]["definitions"]["minItems"] \
            == gen["responseSchema"]["properties"]["definitions"]["maxItems"]
        assert "temperature" not in gen
        assert "serviceTier" not in json.dumps(row)
    # each arm referenced its OWN cache, never the other's
    assert len({row["request"]["cachedContent"] for row in uploaded}) == 2

    # ---- calls.jsonl, in the schema backfill_probe_stats actually reads
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8")
            .splitlines() if x.strip()]
    assert len(rows) == 2 * len(batches)
    assert {r["arm"] for r in rows} == {"lean", "rich"}
    derived = _backfill_tool().rederive(rows)
    assert derived["calls_with_usage"] == len(rows)
    assert len(derived["prompt_families"]) == 2      # one per arm's prompt

    # ---- the usage ledger says BATCH, because that is what it was
    usage_rows = [json.loads(x) for x in
                  (cfg.report_dir / "prompt_ab_usage.jsonl")
                  .read_text(encoding="utf-8").splitlines() if x.strip()]
    assert usage_rows and all(r["mode"] == "batch" for r in usage_rows)
    assert all(r["cached_tokens"] == DECLARED_CACHE_TOKENS for r in usage_rows)
    # cached == declared, per row: the criterion the cache check applies
    for check in report["cache_check"].values():
        assert check["rows"] == check["cached_equals_declared"] > 0
        assert check["declared"] == DECLARED_CACHE_TOKENS

    # ---- the pre-spend block ran BEFORE any job or cache existed
    # PER ARM, with that arm's own prompt active -- see the R6-exemption test.
    assert sorted(report["pre_spend_gates"]) == ["lean", "rich"]
    assert sorted(report["consumption_rules"]) == ["lean", "rich"]
    for arm_rows in report["pre_spend_gates"].values():
        assert {row["id"] for row in arm_rows} == {"G-SCOPE-FROZEN", "G-BUDGET"}
        assert all(row["ok"] for row in arm_rows)
    for arm_rows in report["consumption_rules"].values():
        assert all(r["ok"] for r in arm_rows if r["blocking"])
    # quoted per ARM, because forecast() sums over these keys and the A/B places
    # the same cells once per arm
    assert sorted(report["bill"]) == sorted([lean_tag, rich_tag])
    assert report["surface"] == "batch"

    # ---- the criteria and the blind pairs are computed identically
    pack = prompts.packs.load("German", cfg)
    out = tool.verdict(results, "German", pack, cfg.work_dir / "review")
    assert set(out["criteria"]) == {"a_thinking_median", "b_script_violations",
                                    "c_pos_shape", "d_blind_test",
                                    "e_constant_invalidated"}
    assert out["criteria"]["a_thinking_median"]["ok"] is True
    assert out["criteria"]["b_script_violations"]["ok"] is True
    assert out["criteria"]["c_pos_shape"]["ok"] is True
    assert set(out["by_arm"]) == {"lean", "rich"}
    pairs = read_json(cfg.work_dir / "review" / "prompt_ab_blind_pairs.json")
    key = read_json(cfg.work_dir / "review" / "prompt_ab_blind_key.json")
    assert pairs and all(set(q) == {"key", "A", "B"} for q in pairs)
    assert set(key["answers"].values()) <= {"lean", "rich"}


def test_a_production_translate_resume_never_adopts_an_ab_job(
        cfg, registry, batch_genai, batch_stats):
    """Production isolation, and WHICH layer actually does it.

    Two layers, and this pins them in the order they are load-bearing, because
    an earlier version of the shipped comment had them the other way round:

      the MECHANISM is the wave tag. transport._resume_in_flight and
      _ingest_ready select on EXACT equality of the `lang` field, and an A/B job
      carries "AB-German-lean-<sel>" there rather than "German" -- so adoption
      cannot happen even out of a SHARED registry, which is asserted directly
      below by handing the production resume the A/B's own registry object.
      DEFENCE IN DEPTH is the separate file, which covers everything the
      registry does WITHOUT filtering by lang (find_by_fingerprint, next_job_id,
      summary, cache_prompt_shas) and keeps jobs.json readable by a human.
    """
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    # Kill the drain after the submit, so an AB job is left SUBMITTED -- the
    # exact state a production resume would find and adopt.
    batch_genai.download_raises = RuntimeError("connection reset")
    with pytest.raises(Exception):
        tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)

    todo = tool.todo_rows_for(cfg, "German", batches)
    lean_tag = tool.ab_wave_tag("German", "lean", tool.ab_selection_id(todo))
    ab_reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    assert [j["job_id"] for j in ab_reg.in_flight()] == \
        ["%s-def-w0-00" % lean_tag]

    # ---- THE MECHANISM: even asked to resume "German" out of the A/B's OWN
    # registry -- the worst case, a shared file -- the production scan adopts
    # nothing, because no A/B job's `lang` field is ever a language.
    assert BT._resume_in_flight(cfg, ab_reg, batch_genai, "German",
                                summary={"jobs": []}, sleep=_no_sleep) == []
    assert [j["job_id"] for j in ab_reg.in_flight()] == \
        ["%s-def-w0-00" % lean_tag]

    # ---- DEFENCE IN DEPTH: the production registry is a different file and
    # does not even see the record
    prod = BREG.JobRegistry(cfg)
    assert prod.path != ab_reg.path
    assert prod.jobs() == {} and prod.in_flight() == []
    summary = {"jobs": []}
    assert BT._resume_in_flight(cfg, prod, batch_genai, "German",
                                summary=summary, sleep=_no_sleep) == []
    # ...and the AB job is untouched: still in flight, still ours
    assert BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE).in_flight()

    # ---- and the other direction: a production job is invisible to the A/B
    prod.plan("German-def-w0-00", fingerprint="prod-fp", lang="German",
              kind="definition", model=cfg.gemini_model, prompt_id="v4-frozen",
              cache_name=None, declared_cache_tokens=None,
              cache_prompt_sha256=None, jsonl_path="x.jsonl", plan=[],
              enqueued_tokens=0, wave=0)
    prod.mark_submitted("German-def-w0-00", "batches/prod")
    fresh_ab = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    assert "German-def-w0-00" not in fresh_ab.jobs()
    assert [j["job_id"] for j in fresh_ab.in_flight()] == \
        ["%s-def-w0-00" % lean_tag]


def test_the_ab_batch_run_writes_nothing_into_the_translation_tables(
        cfg, registry, batch_genai, batch_stats):
    """The A/B is a measurement, not a wave: its answers are compared, never
    shipped. The production ingest writes json/translations/<lang>/, and this
    path must not go near it."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    tdir = cfg.json_dir / "translations" / "German"
    assert not tdir.exists()
    tool.run_batch_ab(cfg, "German", ["lean", "rich"], batches, ledger,
                      client=batch_genai, sleep=_no_sleep)
    assert not (tdir / "definitions.json").exists()
    assert not (tdir / "expressions.json").exists()
    # nothing at all under json/translations, for any language
    root = cfg.json_dir / "translations"
    assert not root.exists() or not list(root.rglob("*.json"))


def test_an_interrupted_ab_drain_is_resumed_and_never_creates_a_second_job(
        cfg, registry, batch_genai, batch_stats):
    """batches.create is NOT idempotent -- the same job submitted twice is
    accepted twice, runs twice and is billed twice. A drain that dies between
    the submit and the download is a normal event on a wait of up to 50 hours,
    so the re-run has to finish the job that exists."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    batch_genai.download_raises = RuntimeError("connection reset")
    with pytest.raises(Exception):
        tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)
    created_after_crash = len(batch_genai.jobs)
    caches_after_crash = len(batch_genai.caches_created)
    assert created_after_crash == 1

    # the SAME command again
    results, _ = tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                                   client=batch_genai, sleep=_no_sleep)
    assert len(batch_genai.jobs) == created_after_crash      # no second create
    assert len(batch_genai.caches_created) == caches_after_crash
    assert results["lean"]["calls"] == len(batches)
    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    assert [j["state"] for j in reg.jobs().values()] == [BREG.RECOVERED]

    # and a THIRD run ingests nothing again: RECOVERED is the guard, because the
    # ledger cannot dedupe a row a second process wrote
    rows_before = len(ledger.read_text(encoding="utf-8").splitlines())
    again, _ = tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                                 client=batch_genai, sleep=_no_sleep)
    assert again["lean"]["resumed"] is True
    assert len(batch_genai.jobs) == created_after_crash
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == rows_before


def test_a_shuffled_ab_result_file_still_joins_on_the_key(
        cfg, registry, batch_genai, batch_stats):
    """60 rows will not shard, but the guard stays: the real service was
    measured concatenating ~1000-row shards out of order, and the A/B's join is
    the same reconcile() with the same bijection hard guard."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    batch_genai.shuffle_output = True
    results, report = tool.run_batch_ab(cfg, "German", ["lean"], batches,
                                        ledger, client=batch_genai,
                                        sleep=_no_sleep)
    assert list(report["order_cross_check"]) == \
        ["AB-German-lean-%s"
         % tool.ab_selection_id(tool.todo_rows_for(cfg, "German", batches))]
    order = list(report["order_cross_check"].values())[0]
    assert order["in_input_order"] is False
    assert order["joined_by_key"] == order["rows"] == len(batches)
    assert order["joined_without_key"] == 0
    # every cell still came back, attributed to its own key
    assert results["lean"]["calls"] == len(batches)
    keys_sent = {r["key"] for b in batches for r in b["rows"]}
    assert {p["key"] for p in results["lean"]["produced"]} == keys_sent


def test_the_ab_default_surface_is_still_the_interactive_one(
        cfg, registry, fake_genai, no_sleep, probe_stats, capsys):
    """--mode batch is an ADDITION. The interactive arm is unchanged: one
    synchronous call per entry per arm, through _translate_definition_batch, and
    its ledger rows say standard because _generate is the standard surface."""
    from ankidkdeck import prompts
    tool = _ab_tool()
    _ab_workspace(cfg, n_entries=2)
    prompts.reset()
    prompts.activate(cfg, prompt_id=tool.ARMS["lean"][0])
    batches = tool.pick_entries(cfg, "German", 2)

    @fake_genai.respond
    def _answer(call):
        props = call["config"].kwargs["response_schema"]["properties"]
        n = props["definitions"]["minItems"]
        return {"headword": "hus",
                "definitions": [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                                for i in range(n)]}

    # plan() with no `mode` argument still describes the interactive surface
    tool.plan(cfg, "German", batches, ["lean", "rich"])
    assert "surface             interactive" in capsys.readouterr().out

    ledger = cfg.work_dir / "probes" / "calls.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    usage = S42.UsageLog(path=cfg.report_dir / "prompt_ab_usage.jsonl")
    got = tool.run_arm(cfg, "German", "lean", batches, ledger, usage)
    assert got["calls"] == len(batches) == len(fake_genai.calls)
    assert all(r["mode"] == "standard" for r in usage.rows)
    # no batch machinery was touched at all
    assert not (cfg.work_dir / "batch" / tool.AB_REGISTRY_FILE).exists()
    assert not (cfg.work_dir / "batch" / "jobs.json").exists()


def test_the_ab_batch_gates_refuse_before_a_cache_or_a_job_exists(
        cfg, registry, batch_genai, batch_stats):
    """G-BUDGET is only worth having where it can still refuse something nobody
    has paid for. The cache is a billable object and batches.create is not
    idempotent, so the gate block runs in front of BOTH."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    cfg.spend_cap_usd = 0.0001
    with pytest.raises(FatalError) as exc:
        tool.run_batch_ab(cfg, "German", ["lean", "rich"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)
    assert "G-BUDGET" in str(exc.value)
    assert batch_genai.caches_created == []
    assert batch_genai.jobs == {}
    assert batch_genai.uploads == {}
    assert not (cfg.work_dir / "batch" / tool.AB_REGISTRY_FILE).exists()
    assert not ledger.exists()


def test_the_ab_cli_defaults_to_interactive_and_accepts_mode_batch(
        cfg, registry, batch_genai, batch_stats, monkeypatch, capsys):
    """The flag itself: default interactive, `batch` accepted, and neither plan
    path sends anything."""
    import ankidkdeck.config as _config
    tool = _ab_tool()
    _ab_setup(cfg, tool)
    monkeypatch.setattr(_config, "load_config", lambda *a, **kw: cfg)

    assert tool.main(["--lang", "German", "--entries", "3"]) == 0
    out = capsys.readouterr().out
    assert "surface             interactive" in out
    assert "nothing has been sent" in out.lower() or "plan (nothing has been " \
                                                     "sent)" in out

    assert tool.main(["--lang", "German", "--entries", "3",
                      "--mode", "batch"]) == 0
    out = capsys.readouterr().out
    assert "surface             batch" in out
    assert tool.AB_REGISTRY_FILE in out
    # a plan is a plan on both surfaces: no job, no cache, no upload
    assert batch_genai.jobs == {} and batch_genai.caches_created == []


# --------------------------------------------------------------------------
# fix round 1: the scenarios the first round's tests never constructed
# --------------------------------------------------------------------------

def _ab_varied_workspace(cfg, tool, lang="German"):
    """Entries with DIFFERENT sense counts, so the probe ledger the A/B writes
    has more than one point on the x-axis of backfill's two fits."""
    from ankidkdeck import prompts
    rows = [_entry(entry_id="1102172%d" % i, n_senses=n, n_exprs=1)
            for i, n in enumerate((2, 3, 4))]
    _workspace(cfg, rows)
    _batch_cfg(cfg, cache=True)
    prompts.reset()
    prompts.activate(cfg, prompt_id=tool.ARMS["lean"][0])
    batches = tool.pick_entries(cfg, lang, 3)
    assert len({len(b["rows"]) for b in batches}) > 1
    return batches, cfg.work_dir / "probes" / "calls.jsonl"


def test_a_stale_resubmittable_failure_never_makes_a_finished_arm_pay_again(
        cfg, registry, batch_genai, batch_stats):
    """BLOCKER 1. ab_job_of used to return the FIRST record for an arm, and
    reg.jobs() iterates in INSERTION order.

    The state it takes is ordinary: a job hits the documented 48h expiry
    (EXPIRED is terminal and is recorded resubmittable, correctly -- nothing was
    billed), the operator re-runs, the arm succeeds as `-a2` and its cache is
    deleted at end of wave. From then on the arm's records are
    [FAILED, RECOVERED] and every later run matched the FAILED one, minted a
    fresh cache (new name -> new wave_fingerprint -> dedup defeated), opened
    `-a3` and PAID FOR THE ARM AGAIN -- reporting resumed=False, so nothing said
    it had already been measured.
    """
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)

    # 1. the service says the job produced nothing (EXPIRED, no result file)
    batch_genai.terminal_state = "JOB_STATE_EXPIRED"
    batch_genai.no_dest = True
    with pytest.raises(FatalError):
        tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)
    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    first = list(reg.jobs().values())
    assert [j["state"] for j in first] == [BREG.FAILED]
    assert first[0]["resubmittable"] is True

    # 2. the correct re-run: a second job, which succeeds
    batch_genai.terminal_state = "JOB_STATE_SUCCEEDED"
    batch_genai.no_dest = False
    tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                      client=batch_genai, sleep=_no_sleep)
    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    states = {j["job_id"]: j["state"] for j in reg.jobs().values()}
    assert sorted(states.values()) == [BREG.FAILED, BREG.RECOVERED]
    jobs_after_2 = len(batch_genai.jobs)
    caches_after_2 = len(batch_genai.caches_created)
    rows_after_2 = len(ledger.read_text(encoding="utf-8").splitlines())

    # 3. THE BUG: the same command a third time. It must adopt the RECOVERED
    #    record, not the FAILED one that sits in front of it.
    results, _ = tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                                   client=batch_genai, sleep=_no_sleep)
    assert results["lean"]["resumed"] is True             # truthfully reported
    assert len(batch_genai.jobs) == jobs_after_2          # no third job
    assert len(batch_genai.caches_created) == caches_after_2   # no new cache
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == rows_after_2
    assert len(BREG.JobRegistry(cfg,
                                file=tool.AB_REGISTRY_FILE).jobs()) == 2
    # ...and the lookup is where the whole thing turns. This is the exact
    # difference between the two implementations, asserted directly: iteration
    # order still puts the stale FAILED record first, and ab_job_of must not
    # take it.
    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    tag = tool.ab_wave_tag(
        "German", "lean",
        tool.ab_selection_id(tool.todo_rows_for(cfg, "German", batches)))
    first_match = next(j for j in reg.jobs().values() if j["lang"] == tag)
    assert first_match["state"] == BREG.FAILED          # what it used to return
    assert tool.ab_job_of(reg, tag)["state"] == BREG.RECOVERED


def test_a_different_entry_selection_never_adopts_the_previous_measurement(
        cfg, registry, batch_genai, batch_stats):
    """BLOCKER 2. The wave tag used to be lang+arm and nothing else, so a re-run
    with a different --entries adopted the old job's stored outcome while
    report["cells"] was computed from the NEW selection: prompt_ab_verdict.json
    then stated an n it did not have, and the LEAN-vs-RICH decision was read off
    the wrong sample."""
    tool = _ab_tool()
    _ab_setup(cfg, tool)
    ledger = cfg.work_dir / "probes" / "calls.jsonl"
    three = tool.pick_entries(cfg, "German", 3)
    two = tool.pick_entries(cfg, "German", 2)
    assert len(three) == 3 and len(two) == 2

    first, rep1 = tool.run_batch_ab(cfg, "German", ["lean"], three, ledger,
                                    client=batch_genai, sleep=_no_sleep)
    assert first["lean"]["calls"] == 3 and rep1["requests_per_arm"] == 3

    # a DIFFERENT selection: must place its own job, not serve the old outcome
    second, rep2 = tool.run_batch_ab(cfg, "German", ["lean"], two, ledger,
                                     client=batch_genai, sleep=_no_sleep)
    assert second["lean"]["resumed"] is False
    assert second["lean"]["calls"] == 2
    # the report's n and the adopted outcome's n agree, which is the property
    assert rep2["requests_per_arm"] == len(second["lean"]["thoughts"]) == 2
    assert len(batch_genai.jobs) == 2
    assert second["lean"]["job_id"] != first["lean"]["job_id"]
    assert len(BREG.JobRegistry(cfg,
                                file=tool.AB_REGISTRY_FILE).jobs()) == 2

    # ...and the SAME selection still adopts, which is what makes a re-run cheap
    again, rep3 = tool.run_batch_ab(cfg, "German", ["lean"], two, ledger,
                                    client=batch_genai, sleep=_no_sleep)
    assert again["lean"]["resumed"] is True
    assert again["lean"]["calls"] == rep3["requests_per_arm"] == 2
    assert len(batch_genai.jobs) == 2


def test_the_rich_arm_is_adjudicated_by_its_own_named_r6_exemption(
        cfg, registry, batch_genai, batch_stats):
    """BLOCKER 3. The N-09 rules used to be evaluated once, with whatever pack
    was active -- always LEAN, because main() activates it first -- and the RICH
    arm then spent behind a green verdict the gate had never formed an opinion
    on. R6 measures drift against the size the constants were taken on, and the
    rich prompt drifts far past the 10% tolerance, so it would refuse every rich
    arm for ever: the A/B has to be exempt, but BY NAME and on the record."""
    from ankidkdeck import billing, prompts
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)

    _, report = tool.run_batch_ab(cfg, "German", ["lean", "rich"], batches,
                                  ledger, client=batch_genai, sleep=_no_sleep)
    for arm in ("lean", "rich"):
        rules = {r["rule"]: r for r in report["consumption_rules"][arm]}
        assert "R6-prompt-size" not in rules          # replaced, not skipped
        row = rules[tool.AB_R6_EXEMPTION]
        assert row["ok"] is True and row["blocking"] is True
        assert row["spec_rule"] == "6"
        assert row["detail"]["replaces"] == "R6-prompt-size"
        assert row["detail"]["arm"] == arm
        assert row["detail"]["declared_sha256"] == row["detail"]["live_sha256"]
    # the two arms really were adjudicated on DIFFERENT prompts
    lean_row = {r["rule"]: r for r in
                report["consumption_rules"]["lean"]}[tool.AB_R6_EXEMPTION]
    rich_row = {r["rule"]: r for r in
                report["consumption_rules"]["rich"]}[tool.AB_R6_EXEMPTION]
    assert rich_row["detail"]["live_chars"] > lean_row["detail"]["live_chars"]
    assert lean_row["detail"]["live_sha256"] != rich_row["detail"]["live_sha256"]

    # ...and the exemption is load-bearing: the SHIPPED R6, asked about the rich
    # prompt, refuses it. That is the refusal the A/B exists to make obsolete.
    prompts.reset()
    prompts.activate(cfg, prompt_id="rich-core-1")
    stats = read_json(cfg.probe_stats_path)
    texts = {kind: S42.system_prompt(kind, "German")
             for kind in ("definition", "expression")}
    shipped = {r["rule"]: r
               for r in billing.consumption_rules(cfg, stats, prompts=texts)}
    assert shipped["R6-prompt-size"]["ok"] is False
    drift = [p for p in shipped["R6-prompt-size"]["detail"]["prompts"]
             if p["kind"] == "definition"][0]
    assert drift["drift"] > 1.0        # measured about 159%


def test_the_r6_exemption_still_refuses_a_prompt_the_arm_did_not_declare(
        cfg, registry, batch_genai, batch_stats):
    """The exemption is a swap, not a hole: R6's size band is replaced by an
    IDENTITY check, and a live prompt that is not the arm's declared prompt is
    still a blocking refusal."""
    from ankidkdeck import prompts
    tool = _ab_tool()
    _ab_setup(cfg, tool)
    stats = read_json(cfg.probe_stats_path)

    # the arm's own prompt active: passes
    prompts.reset()
    prompts.activate(cfg, prompt_id="rich-core-1")
    rows = tool.ab_consumption_rules(cfg, "German", "rich", stats)
    assert {r["rule"] for r in rows} >= {tool.AB_R6_EXEMPTION}

    # the WRONG pack active for this arm: refuses, naming the exemption
    prompts.reset()
    prompts.activate(cfg, prompt_id="v4-frozen")
    with pytest.raises(FatalError) as exc:
        tool.ab_consumption_rules(cfg, "German", "rich", stats)
    assert tool.AB_R6_EXEMPTION in str(exc.value)


def test_the_interactive_ab_path_now_refuses_when_g_budget_would(
        cfg, registry, fake_genai, no_sleep, probe_stats):
    """BLOCKER 3, third part. The interactive arms used to run behind no
    pre-spend gate at all -- up to 100 paid requests with no G-BUDGET, no
    G-SCOPE-FROZEN and no N-09 rule."""
    from ankidkdeck import prompts
    tool = _ab_tool()
    _ab_workspace(cfg, n_entries=2)
    prompts.reset()
    prompts.activate(cfg, prompt_id=tool.ARMS["lean"][0])
    batches = tool.pick_entries(cfg, "German", 2)
    ledger = cfg.work_dir / "probes" / "calls.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    cfg.spend_cap_usd = 0.0001

    with pytest.raises(FatalError) as exc:
        tool.run_interactive_ab(cfg, "German", ["lean", "rich"], batches,
                                ledger)
    assert "G-BUDGET" in str(exc.value)
    assert fake_genai.calls == []              # refused before the first call


def test_the_interactive_ab_path_quotes_the_surface_it_runs_on(
        cfg, registry, fake_genai, no_sleep, probe_stats):
    """AB_INTERACTIVE_MODE on the RANK_MODE / REVIEW_MODE template: one constant
    drives the ledger label, the rate card AND the request ceiling. Quoting off
    cfg.mode would price a synchronous arm at batch rates during a batch month
    and quote it the batch ceiling of 4 for a path that takes the interactive
    count-lock x transport ladder of 25 -- the review() defect, one file over."""
    from ankidkdeck import prompts
    tool = _ab_tool()
    _ab_workspace(cfg, n_entries=2)
    prompts.reset()
    prompts.activate(cfg, prompt_id=tool.ARMS["lean"][0])
    batches = tool.pick_entries(cfg, "German", 2)
    ledger = cfg.work_dir / "probes" / "calls.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    @fake_genai.respond
    def _answer(call):
        props = call["config"].kwargs["response_schema"]["properties"]
        n = props["definitions"]["minItems"]
        return {"headword": "hus",
                "definitions": [{"lemma": "L%d" % i, "gloss": "G%d" % i}
                                for i in range(n)]}

    # the operator's config says batch; the interactive arms do not run there
    cfg.mode = "batch"
    cfg.cache_enabled = True
    results, report = tool.run_interactive_ab(cfg, "German", ["lean"], batches,
                                              ledger)
    assert tool.AB_INTERACTIVE_MODE == "standard"
    assert report["surface"] == "standard"
    assert report["config_overrides"]["mode"] == {"was": "batch",
                                                  "now": "standard"}
    assert report["config_overrides"]["cache_enabled"] == {"was": True,
                                                           "now": False}
    row = list(report["bill"].values())[0]
    assert row["surface"] == "standard"
    assert row["dollars"]["rate_card_source"].endswith(", standard)")
    # the INTERACTIVE ladder (25x), not the batch ceiling of 4
    assert row["requests_max_transport"] == "standard"
    assert "INCLUDES transport retries" in row["requests_max_basis"]
    # and the rows really were placed and labelled standard
    assert results["lean"]["calls"] == len(batches)
    usage = [json.loads(x) for x in
             (cfg.report_dir / "prompt_ab_usage.jsonl")
             .read_text(encoding="utf-8").splitlines() if x.strip()]
    assert usage and all(r["mode"] == "standard" for r in usage)


def test_an_ab_run_leaves_the_production_gates_report_untouched(
        cfg, registry, batch_genai, batch_stats):
    """S1. s42._pre_spend goes through gates.run_gates, which persists into
    cfg.report_dir/gates_report.json and merges by (id, stage, extra) with "a
    later run of the SAME scope wins" -- and pre_spend_gates emits
    G-SCOPE-FROZEN and G-BUDGET at stage 42, the very rows a production stage-42
    run emits. One A/B run turned a red report green. A measurement tool must
    not participate in the release artifact."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    report_path = cfg.report_dir / "gates_report.json"
    write_json(report_path, {"results": [
        {"id": "G-BUDGET", "description": "d", "stage": "42", "extra": {},
         "ok": False, "detail": {"why": "the production wave is over budget"}},
        {"id": "G-SCOPE-FROZEN", "description": "d", "stage": "42",
         "extra": {}, "ok": False, "detail": {"why": "no refreeze stamp"}}]})
    before = report_path.read_bytes()

    tool.run_batch_ab(cfg, "German", ["lean", "rich"], batches, ledger,
                      client=batch_genai, sleep=_no_sleep)

    assert report_path.read_bytes() == before
    assert [r["ok"] for r in read_json(report_path)["results"]] == [False,
                                                                   False]
    # the A/B's own verdict is still on disk, in its OWN file
    own = read_json(cfg.report_dir / "prompt_ab_gates.json")
    assert {r["id"] for r in own["results"]} == {"G-SCOPE-FROZEN", "G-BUDGET"}
    assert all(r["ok"] for r in own["results"])
    assert all(r["stage"] == "42-prompt-ab" for r in own["results"])


def test_a_crash_inside_the_ab_ingest_does_not_duplicate_a_single_row(
        cfg, registry, batch_genai, batch_stats, monkeypatch):
    """S2. The DOWNLOADED -> RECOVERED guard closes the whole-file case and
    cannot close this one: the state only advances after the LAST row, so a
    crash after row 1 of 3 left the job DOWNLOADED and the next run re-ingested
    everything. billing.usage_row_uid dedupes on (ts, seq), which a second
    process cannot reproduce, so the duplicates counted as real spend and
    over-weighted one arm in the artifact backfill re-derives constants from."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    usage_path = cfg.report_dir / "prompt_ab_usage.jsonl"

    real_append = S42.append_jsonl
    state = {"n": 0}

    def _explode(path, row):
        # Only the PROBE ledger is counted: UsageLog.record goes through this
        # same function, and the crash has to land BETWEEN the two files so the
        # re-run sees a genuinely torn ingest.
        if str(path) == str(ledger):
            state["n"] += 1
            if state["n"] == 2:             # mid-ingest, after row 1 landed
                raise RuntimeError("disk went away")
        return real_append(path, row)

    monkeypatch.setattr(S42, "append_jsonl", _explode)
    with pytest.raises(RuntimeError):
        tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)
    monkeypatch.setattr(S42, "append_jsonl", real_append)

    reg = BREG.JobRegistry(cfg, file=tool.AB_REGISTRY_FILE)
    assert [j["state"] for j in reg.jobs().values()] == [BREG.DOWNLOADED]
    probe_rows = [x for x in ledger.read_text(encoding="utf-8").splitlines()
                  if x.strip()]
    usage_rows = [x for x in usage_path.read_text(encoding="utf-8").splitlines()
                  if x.strip()]
    assert len(probe_rows) == 1 and len(usage_rows) == 2   # torn mid-ingest

    # the re-run completes it and appends NOTHING twice
    results, _ = tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                                   client=batch_genai, sleep=_no_sleep)
    assert results["lean"]["calls"] == len(batches)
    assert results["lean"]["rows_already_written"] == 2
    probe_rows = [json.loads(x) for x in
                  ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
    usage_rows = [json.loads(x) for x in
                  usage_path.read_text(encoding="utf-8").splitlines()
                  if x.strip()]
    assert len(probe_rows) == len(usage_rows) == len(batches)
    uids = [r["ab_row_uid"] for r in probe_rows]
    assert len(set(uids)) == len(uids)                 # no duplicate identity
    assert set(uids) == {r["ab_row_uid"] for r in usage_rows}
    assert len(batch_genai.jobs) == 1                  # and no second spend


def test_the_ab_probe_rows_can_actually_feed_the_backfill_fits(
        cfg, registry, batch_genai, batch_stats):
    """S6. backfill_probe_stats reads the batch size as fp.get("n") -- the key
    the wave-1 probe harness wrote. The A/B wrote only "n_expected", so every
    A/B row, on BOTH surfaces, was silently skipped for EXPECTED_OUTPUT,
    PROMPT_TOKENS_fit and prompt_sha256_per_n. Criterion (e) is exactly
    `--declare-prompt-id rich-core-1 --rebase-measurement`, so the rich arm's
    own fits could never have been re-derived from the ledger the A/B writes."""
    tool = _ab_tool()
    batches, ledger = _ab_varied_workspace(cfg, tool)
    tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                      client=batch_genai, sleep=_no_sleep)

    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8")
            .splitlines() if x.strip()]
    assert len(rows) == len(batches)
    assert all(r["request_fingerprint"]["n"]
               == r["request_fingerprint"]["n_expected"] for r in rows)
    derived = _backfill_tool().rederive(rows)
    # THE PROPERTY: the fits are computed, not skipped
    assert derived["EXPECTED_OUTPUT"] is not None
    assert derived["PROMPT_TOKENS_fit"] is not None
    assert derived["EXPECTED_OUTPUT"]["points"] == len(batches)
    assert derived["PROMPT_TOKENS_fit"]["points"] == len(batches)


def test_no_cache_is_created_when_the_wave_split_refuses_the_run(
        cfg, registry, batch_genai, batch_stats, monkeypatch):
    """S3. A CachedContent is a billable object with a multi-hour TTL, and the
    one-job-per-arm refusal is a NEW one production does not have, reachable by
    an ordinary --entries mistake. Every refusal that can still fire has to fire
    before caches.create."""
    tool = _ab_tool()
    batches, ledger = _ab_setup(cfg, tool)
    # a token target so small that every entry needs its own job
    monkeypatch.setattr(BW, "job_token_target", lambda *a, **kw: 1)

    with pytest.raises(FatalError) as exc:
        tool.run_batch_ab(cfg, "German", ["lean"], batches, ledger,
                          client=batch_genai, sleep=_no_sleep)
    assert "one job per arm" in str(exc.value)
    assert batch_genai.caches_created == []      # nothing billable was made
    assert batch_genai.jobs == {} and batch_genai.uploads == {}
