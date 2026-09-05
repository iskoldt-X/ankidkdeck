"""The contract of one LLM request: what is sent, how it is sized, and what it
costs. All offline -- there is no key, no network and no SDK here.

These are the audit's Wave-0 assertions turned into tests. Each one stands for a
measured failure, so the docstrings name the measurement rather than the rule:

  * W0-1 FAILED when it was first run. The "constant" definition prompt
    interpolated the batch size into its last line, so 30 payloads produced 7
    distinct sha256 values, one per value of n, differing by one or two
    characters. An explicit cache is keyed on exact bytes, which made
    one-cache-per-language impossible while looking like a formatting detail --
    and the cache is what halves the definition wave.
  * The system/cache exclusion is a HARD 400, verbatim: "CachedContent can not
    be used with GenerateContent request setting system_instruction, tools or
    tool_config."
  * The output cap comes from a measured fit (a=35.964, b=23.07, R^2=0.985 over
    62 points) READ OFF DISK. The numbers do not appear in the package.
  * thinkingLevel=LOW measured 0 derived thinking over 38 observations,
    including on a 4.5k prompt; MEDIUM measured mean 578.7. So the level is sent
    as a literal on every request, and thinking is derived from the token
    identity rather than read from thoughtsTokenCount (protobuf omits the field
    exactly when it is zero).
"""

import json
import math

import pytest
from conftest import MEASURED_CONSTANTS

from ankidkdeck.config import VERIFIED_MODELS, Config, load_config
from ankidkdeck.stages import s42_translate as S42
from ankidkdeck.util import FatalError, sha256_str, write_json

LANGS = ("Chinese", "English", "German", "Spanish")


# --------------------------------------------------------- 1.7 / W0-1

def test_prompt_is_constant():
    """definition_prompt(lang) and expression_prompt(lang) depend on the
    LANGUAGE ONLY. One sha per language per prompt, or there is no cache."""
    for lang in LANGS:
        assert S42.definition_prompt(lang) == S42.definition_prompt(lang)
        assert S42.expression_prompt(lang) == S42.expression_prompt(lang)
    # the count is nowhere in the system prompt, in any language
    for lang in LANGS:
        for text in (S42.definition_prompt(lang), S42.expression_prompt(lang)):
            assert "{n_defs}" not in text and "{n_items}" not in text
            assert "MUST have a length of" not in text
            assert "as many objects as the user message states" in text
    # ...and the batch size cannot reach them: the functions take one argument
    with pytest.raises(TypeError):
        S42.definition_prompt("German", 20)          # the old signature
    with pytest.raises(TypeError):
        S42.expression_prompt("German", 20)


def test_prompt_sha_is_one_value_per_language_across_every_batch_size():
    """The W0-1 regression nail: the same 7-sha failure, expressed as the
    property that broke it."""
    for lang in LANGS:
        for kind, prompt in (("def", S42.definition_prompt),
                             ("expr", S42.expression_prompt)):
            shas = {S42.prompt_sha256(prompt(lang)) for _ in range(1, 21)}
            assert len(shas) == 1, (lang, kind)
    # and the count DOES reach the user message and the schema, which is where
    # it is enforced
    rows = [{"text": "d%d" % i, "grammar": "", "hint": ""} for i in range(3)]
    user = S42.definition_user_payload("hus", rows)
    assert "exactly 3 definition objects" in user
    assert S42.definition_schema(3)["properties"]["definitions"]["minItems"] == 3


def test_the_prompt_skeleton_is_the_same_text_in_every_language():
    """Masking the language slot, the three non-English prompts are byte
    identical. English is the documented special case: it has no critical_rule
    block at all, because there is no "do not answer in English" rule to state
    when the target language IS English."""
    def masked(lang):
        return S42.definition_prompt(lang).replace(lang, "[L]")

    non_english = {masked(lang) for lang in ("Chinese", "German", "Spanish")}
    assert len(non_english) == 1
    english = masked("English")
    assert english != non_english.pop()
    assert "CRITICAL RULE" not in english


# ------------------------------------------------------------------ 1.3

def test_generate_system_xor_cache():
    """Three states, and only the fourth is refused. The guard sits before the
    SDK import, so it fires with no google.genai installed at all -- and it is
    in _generate, not only in the batch builder, because stage 50 imports
    _generate directly."""
    def req(**kw):
        return S42.LlmRequest(kind="definition", label="l", user="u",
                              schema=None, n_expected=1, max_output_tokens=1024,
                              **kw)

    req(system="SYSTEM")                             # 1. system only: fine
    req(cache_name="cachedContents/abc")             # 2. cache only: fine
    req()                                            # 3. neither: fine (review)
    with pytest.raises(FatalError) as exc:           # 4. both: refused
        req(system="SYSTEM", cache_name="cachedContents/abc")
    assert "XOR" in str(exc.value)

    # ...and _generate refuses a hand-built request object too, without
    # importing anything: pool=None proves no client was constructed.
    class _Loose:
        kind = "definition"
        label = "loose"
        system = "S"
        cache_name = "cachedContents/abc"

    with pytest.raises(FatalError):
        S42._generate(None, "gemini-3.7-flash", _Loose())


def test_a_cached_request_carries_no_system_instruction():
    """With a cache attached the system prompt IS the cache. CallContext is the
    one place that decides it, so no transport can get it wrong."""
    cfg = Config()
    ctx = S42.CallContext(cfg=cfg, pool=None, fit=(35.964, 23.07), lang="German",
                          cache_name="cachedContents/abc")
    r = ctx.request("definition", "l", "u", S42.definition_schema(3), 3,
                    S42.definition_prompt("German"))
    assert r.system is None and r.cache_name == "cachedContents/abc"
    plain = S42.CallContext(cfg=cfg, pool=None, fit=(35.964, 23.07),
                            lang="German").request(
        "definition", "l", "u", S42.definition_schema(3), 3,
        S42.definition_prompt("German"))
    assert plain.system == S42.definition_prompt("German")
    assert plain.cache_name is None


# ------------------------------------------------------------------ 1.4

def test_max_output_formula(tmp_path):
    """ceil(a*n + b) * 1.5, no thinking term, with a and b READ OFF DISK."""
    cfg = Config(work_dir=tmp_path / "work")
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)
    a, b = S42.output_fit(cfg)
    assert (a, b) == (35.964, 23.07)
    for n in (1, 3, 8, 12, 20):
        raw = math.ceil(math.ceil(a * n + b) * 1.5)
        assert S42.max_output_tokens(n, (a, b), floor=0) == raw
        # the floor is the lowest cap that was actually measured end to end
        assert S42.max_output_tokens(n, (a, b)) == max(1024, raw)
    # n=20 stays inside the model's own output ceiling by a wide margin
    assert S42.max_output_tokens(20, (a, b)) < S42.MODEL_OUTPUT_CEILING
    assert S42.max_output_tokens(10 ** 6, (a, b)) == S42.MODEL_OUTPUT_CEILING
    # no thinking term: MEDIUM's measured 578.7 is not reserved for
    assert S42.max_output_tokens(8, (a, b), floor=0) == math.ceil(
        math.ceil(a * 8 + b) * 1.5)


def test_the_measured_constants_are_never_hard_coded(tmp_path):
    """No file, wrong model, or a missing key: all three refuse, because the
    alternative is sizing a paid request from a number measured on something
    else."""
    cfg = Config(work_dir=tmp_path / "work")
    with pytest.raises(FatalError) as exc:
        S42.output_fit(cfg)
    assert "no measured LLM constants" in str(exc.value)

    write_json(cfg.probe_stats_path, {**MEASURED_CONSTANTS,
                                      "model": "gemini-2.0-flash"})
    with pytest.raises(FatalError) as exc:
        S42.output_fit(cfg)
    assert "measured on model" in str(exc.value)

    stats = dict(MEASURED_CONSTANTS)
    stats.pop("EXPECTED_OUTPUT")
    write_json(cfg.probe_stats_path, stats)
    with pytest.raises(FatalError) as exc:
        S42.output_fit(cfg)
    assert "EXPECTED_OUTPUT" in str(exc.value)

    # the fit's numbers are not literals anywhere in the stage. They are quoted
    # in comments and docstrings on purpose -- a reader needs to know the order
    # of magnitude -- so the assertion is on the parsed constants, not on the
    # text: no number in this module IS the fit.
    import ast
    tree = ast.parse(open(S42.__file__, encoding="utf-8").read())
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, float)}
    assert 35.964 not in literals and 23.07 not in literals


def test_the_generation_config_is_the_measured_shape():
    """thinkingConfig and responseSchema inside generationConfig, no
    temperature, and the cache reference is NOT a generationConfig field (a
    misplaced cachedContent is rejected at submit with a 400 -- but only in
    batch; the shape is pinned here for every transport)."""
    r = S42.LlmRequest(kind="definition", label="l", user="u",
                       schema=S42.definition_schema(3), n_expected=3,
                       max_output_tokens=1115, thinking_level="LOW",
                       cache_name="cachedContents/abc")
    gen = r.generation_config()
    assert gen["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert gen["maxOutputTokens"] == 1115
    assert gen["responseMimeType"] == "application/json"
    assert "responseSchema" in gen
    assert "temperature" not in gen
    assert "cachedContent" not in gen
    assert "cachedContent" not in json.dumps(gen)


def test_derived_thinking_never_reads_the_thoughts_field():
    """thoughtsTokenCount is absent exactly when thinking is 0 and present when
    it is not, so "missing" and "zero" are indistinguishable at the one place
    the difference is the whole question."""
    class _Low:                      # LOW: no thoughts attribute at all
        prompt_token_count = 1350
        candidates_token_count = 307
        total_token_count = 1657

    class _Medium:                   # MEDIUM: the measured mean, present
        prompt_token_count = 1350
        candidates_token_count = 307
        total_token_count = 1657 + 579
        thoughts_token_count = 579

    assert S42.derived_thinking(_Low()) == 0
    assert S42.derived_thinking(_Medium()) == 579
    # the batch transport sees plain JSON, and must derive the same number
    assert S42.derived_thinking({"promptTokenCount": 1350,
                                 "candidatesTokenCount": 307,
                                 "totalTokenCount": 2236}) == 579


def test_cached_tokens_are_a_subset_of_prompt_tokens():
    """The single easiest way to get the bill wrong. Measured: cached is
    constant at 1,135 while prompt grows with n, so cached/prompt falls from
    0.935 at n=1 to 0.632 at n=20 -- adding them would inflate the input by the
    cache on every single row."""
    row = S42.normalize_usage({"promptTokenCount": 1795,
                               "cachedContentTokenCount": 1135,
                               "candidatesTokenCount": 742,
                               "totalTokenCount": 2537},
                              model="gemini-3.7-flash", label="l")
    assert row["prompt_tokens"] == 1795
    assert row["cached_tokens"] == 1135
    assert row["uncached_prompt_tokens"] == 660
    assert row["cached_tokens"] < row["prompt_tokens"]
    assert row["thinking_tokens"] == 0


# ------------------------------------------------------------ W0-6 / N-13

def test_count_lock_survives_serialization():
    """minItems == maxItems == n has to survive the SDK's own Schema model, not
    just our dict. This one PASSED when it was first run; it is here so it keeps
    passing, because a short array is not a wrong answer -- zip() shifts every
    gloss onto the wrong definition from the missing one onwards."""
    types = pytest.importorskip("google.genai.types",
                                reason="the real SDK is not installed here")
    for n in (1, 3, 20):
        for schema, key in ((S42.definition_schema(n), "definitions"),
                            (S42.expression_schema(n), "fixed_expressions")):
            wire = types.Schema.model_validate(schema).model_dump(
                mode="json", exclude_none=True, by_alias=True)
            assert wire["properties"][key]["minItems"] == str(n) or \
                wire["properties"][key]["minItems"] == n
            assert wire["properties"][key]["maxItems"] == str(n) or \
                wire["properties"][key]["maxItems"] == n
            item = wire["properties"][key]["items"]["properties"]
            assert item["lemma"]["minLength"] in (1, "1")
            # no description fields: they are output tokens on every request
            assert "description" not in wire["properties"][key]


def test_the_schemas_carry_no_description_fields():
    for schema in (S42.definition_schema(3), S42.expression_schema(3)):
        assert "description" not in json.dumps(schema)


# ------------------------------------------------------------------ 1.6

def test_api_errors_are_classified_not_all_retried():
    """One `except Exception` with a five-attempt ladder, wrapped in a
    five-attempt count-lock ladder, meant 25 paid attempts to discover a 400 --
    and made a dead cache reference look exactly like a transient."""
    def exc(text):
        return RuntimeError(text)

    assert S42.classify_api_error(exc(
        "400 INVALID_ARGUMENT: no such field: 'cachedContent'")) == S42.ERR_FATAL
    assert S42.classify_api_error(exc(
        "429 RESOURCE_EXHAUSTED: quota")) == S42.ERR_THROTTLE
    assert S42.classify_api_error(exc(
        "503 UNAVAILABLE: the model is currently experiencing high demand"
    )) == S42.ERR_UNAVAILABLE
    assert S42.classify_api_error(exc("500 INTERNAL")) == S42.ERR_RETRYABLE
    # the verbatim cache message, which is a 403 and not a 404
    assert S42.classify_api_error(exc(
        "403 PERMISSION_DENIED: CachedContent not found (or permission denied)"
    )) == S42.ERR_CACHE_MISSING
    # a 403 that is NOT about the cache is a credentials problem: not retryable
    assert S42.classify_api_error(exc(
        "403 PERMISSION_DENIED: The caller does not have permission"
    )) == S42.ERR_FATAL


# ----------------------------------------------------------------- 1.10

def test_provenance_is_a_closed_ascii_vocabulary():
    prov = S42._provenance("gemini-3.7-flash", "v4-frozen", "LOW",
                           date="2026-08-26")
    assert prov == "gemini:gemini-3.7-flash+v4-frozen+LOW@2026-08-26"
    assert prov.isascii()
    for bad in ("prompt pack v4", "v4/frozen", "v4+frozen"):
        with pytest.raises(FatalError):
            S42._provenance("gemini-3.7-flash", bad, "LOW", date="2026-08-26")


# ------------------------------------------- F3: caps per request KIND

def test_only_the_measured_kind_uses_the_measured_fit():
    """The 62 points behind EXPECTED_OUTPUT are ALL definition requests. An
    expression gloss is a whole explanatory sentence, and at n=20 the definition
    fit's own headroom is 1.42x (783 observed against a 1,115 cap) -- so
    extending it to a kind nobody measured is a guess with a measurement's
    face on."""
    cfg = Config()
    fit = (35.964, 23.07)
    assert S42.MEASURED_OUTPUT_KINDS == ("definition",)
    assert S42.resolve_max_output(cfg, 20, fit, "definition") == \
        S42.max_output_tokens(20, fit, floor=cfg.max_output_floor)
    for kind in ("expression", "pos", "rank", "review"):
        assert S42.resolve_max_output(cfg, 20, fit, kind) == \
            cfg.max_output_unmeasured
    # ...and an explicit pin still wins for everything
    pinned = Config(max_output_tokens=2048)
    assert S42.resolve_max_output(pinned, 20, fit, "expression") == 2048
    assert S42.resolve_max_output(pinned, 20, fit, "definition") == 2048


def test_a_truncation_raises_the_budget_once_before_it_gives_up(monkeypatch):
    """Spec 5.6 on the interactive surface. MAX_TOKENS is a cap error, so
    resending the identical request is useless -- but aborting a multi-hour paid
    wave over one under-sized cap is worse. One raise, x2 to a ceiling, both
    attempts in the ledger."""
    import sys
    import types as _types
    seen = []

    class _Resp:
        def __init__(self, finish, text):
            self.text = text
            self.candidates = [type("C", (), {"finish_reason": finish})()]
            self.usage_metadata = None

    class _Models:
        def __init__(self, plan):
            self.plan = plan

        def generate_content(self, model=None, contents=None, config=None):
            seen.append(config.kwargs["max_output_tokens"])
            return self.plan(len(seen))

    def install(plan):
        genai = _types.ModuleType("google.genai")
        gtypes = _types.ModuleType("google.genai.types")
        gtypes.GenerateContentConfig = type(
            "Cfg", (), {"__init__": lambda s, **kw: setattr(s, "kwargs", kw)})
        gtypes.ThinkingConfig = type(
            "TC", (), {"__init__": lambda s, **kw: setattr(s, "kwargs", kw)})
        genai.types = gtypes
        google = _types.ModuleType("google")
        google.genai = genai
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
        return type("Pool", (), {"client": lambda s: type(
            "Cli", (), {"models": _Models(plan)})(),
            "count": lambda s: None, "force_rotate": lambda s: None})()

    req = S42.LlmRequest(kind="expression", label="expr batch x", user="u",
                         schema=None, n_expected=20, max_output_tokens=1024,
                         system="S")
    # the raise WORKS: attempt 2 comes back complete
    usage = S42.UsageLog()
    pool = install(lambda i: _Resp("MAX_TOKENS", "{") if i == 1
                   else _Resp("STOP", '{"ok": 1}'))
    comp = S42._generate(pool, "gemini-3.7-flash", req, usage=usage)
    assert comp.parsed == {"ok": 1}
    assert seen == [1024, 2048]
    assert [r["max_output_tokens"] for r in usage.rows] == [1024, 2048]
    assert usage.rows[0]["finish_reason"] == "MAX_TOKENS"

    # ...and it happens ONCE. A second truncation is a real configuration error
    seen.clear()
    usage = S42.UsageLog()
    pool = install(lambda i: _Resp("MAX_TOKENS", "{"))
    with pytest.raises(FatalError) as exc:
        S42._generate(pool, "gemini-3.7-flash", req, usage=usage)
    assert seen == [1024, 2048]
    assert len(usage.rows) == 2
    assert "already retried once at the raised budget" in str(exc.value)
    assert "thinking" in str(exc.value)          # the error names the real risk


# --------------------------------------- F4/F6: what the spend gate requires

def test_spending_above_low_needs_the_measurement_and_the_acknowledgement():
    """thinkingLevel is pinned to LOW for this program, and the pin is
    ARITHMETIC: maxOutputTokens is ONE budget that thoughts and candidates
    SHARE, while the derived cap is ceil(a*n + b) * 1.5 with no thinking term.
    At MEDIUM the measured p95 is 1,042 thought tokens against an n=20 batch's
    entire 1,115-token cap, and both MAX_TOKENS finishes in the probe set came
    from MEDIUM.

    So "MEDIUM is measured" is not a licence to spend: a measured thinking cost
    that the cap formula never reads is a number nobody is using. Both halves
    are required, and either one missing is a refusal.
    """
    Config(thinking_level="LOW").validate(spending=True, stats={})
    assert Config().thinking_level_override_ack is False

    # MEDIUM *is* measured on the real artifact -- and is still refused
    medium = Config(thinking_level="MEDIUM")
    medium.validate()                          # a dry run may quote any level
    with pytest.raises(FatalError) as exc:
        medium.validate(spending=True, stats=MEASURED_CONSTANTS)
    assert "thinking_level_override_ack = true" in str(exc.value)
    assert "pinned to LOW" in str(exc.value)

    # ...and with BOTH the measurement and the acknowledgement it may spend
    acked = Config(thinking_level="MEDIUM", thinking_level_override_ack=True)
    acked.validate(spending=True, stats=MEASURED_CONSTANTS)

    # the acknowledgement alone is not enough either: HIGH was never measured
    with pytest.raises(FatalError) as exc:
        Config(thinking_level="HIGH",
               thinking_level_override_ack=True).validate(
                   spending=True, stats=MEASURED_CONSTANTS)
    assert "THINKING_PER_REQUEST_HIGH" in str(exc.value)
    with pytest.raises(FatalError):
        acked.validate(spending=True, stats={"thinking": {}})


def test_the_spend_gate_needs_every_constant_the_money_math_reads(tmp_path):
    """F6. Before this, probe_stats() checked the file, the model and
    EXPECTED_OUTPUT.a/b -- so a stats.json with no cache floor and no thinking
    constant went straight through `translate --confirm-spend`, and only
    `doctor` complained. translate does not call doctor."""
    cfg = Config(work_dir=tmp_path / "work")
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)
    assert S42.missing_stats_keys(MEASURED_CONSTANTS) == []
    S42.probe_stats(cfg)
    for key, _why in S42.REQUIRED_STATS_KEYS:
        stats = json.loads(json.dumps(MEASURED_CONSTANTS))
        node, parts = stats, key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node.pop(parts[-1])
        assert key in S42.missing_stats_keys(stats)
        write_json(cfg.probe_stats_path, stats)
        with pytest.raises(FatalError) as exc:
            S42.probe_stats(cfg)
        assert key in str(exc.value)


def test_doctor_blocks_a_thinking_level_above_low(tmp_path, capsys):
    """F4, all three directions. It used to print a one-line arrow and return 0.

    doctor's verdict has to match what the stage will actually do, and the stage
    now needs the measurement AND the acknowledgement -- so the BLOCK line says
    which half is missing and why LOW is pinned.
    """
    from ankidkdeck.cli import doctor
    cfg = Config(work_dir=tmp_path / "work", thinking_level="MEDIUM")
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)

    # measured, but not acknowledged
    assert doctor(cfg) == 1
    out = capsys.readouterr().out
    assert "BLOCKED: thinking_level = MEDIUM is measured" in out
    assert "thinking_level_override_ack is false" in out
    assert "ONE budget shared by thoughts and candidates" in out
    assert "pinned to LOW" in out

    # measured AND acknowledged
    assert doctor(Config(work_dir=cfg.work_dir, thinking_level="MEDIUM",
                         thinking_level_override_ack=True)) == 0
    assert "override ack: True" in capsys.readouterr().out

    # acknowledged but NOT measured: still blocked
    stats = json.loads(json.dumps(MEASURED_CONSTANTS))
    stats["thinking"].pop("THINKING_PER_REQUEST_MEDIUM")
    write_json(cfg.probe_stats_path, stats)
    assert doctor(Config(work_dir=cfg.work_dir, thinking_level="MEDIUM",
                         thinking_level_override_ack=True)) == 1
    assert "has no measured thinking cost" in capsys.readouterr().out
    # ...and LOW on the same artifact is fit to spend
    assert doctor(Config(work_dir=cfg.work_dir)) == 0


def test_doctor_refuses_a_language_with_no_system_prompt_token_count(
        tmp_path, capsys):
    """The Russian month. `doctor` checked five NON-per-language dotted keys and
    printed "fit to spend" while PROMPT_TOKENS_system_only.Russian was absent;
    the only hard stop was batch/transport.py at wave-split time, which arrives
    with the scope frozen and the operator already committed.

    doctor's whole reason to exist is that nothing printed the effective spend
    configuration before money was placed, so a per-language constant the bill
    reads has to be printed per language.
    """
    from ankidkdeck.cli import doctor
    cfg = Config(work_dir=tmp_path / "work",
                 langs=["German", "Russian"])
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)
    assert S42.system_prompt_tokens(MEASURED_CONSTANTS, "Russian") is None

    assert doctor(cfg) == 1
    out = capsys.readouterr().out
    assert "system tokens German    1135 (measured)" in out
    assert "system tokens Russian   MISSING" in out
    assert ("BLOCKED: PROMPT_TOKENS_system_only has no usable entry for Russian"
            in out)
    # the message names the way OUT, not just the problem
    assert "--declare-system-tokens Russian=N" in out
    assert "--declaration-basis" in out
    # every configured language is named, not just the first, and in the order
    # the operator configured them -- the same order the lines above print in
    cfg = Config(work_dir=cfg.work_dir, langs=["Russian", "Italian"])
    assert doctor(cfg) == 1
    out = capsys.readouterr().out
    assert "no usable entry for Russian, Italian" in out
    assert "--declare-system-tokens Russian=N" in out
    assert "--declare-system-tokens Italian=N" in out

    # a DECLARED value is fit to spend, and says so rather than passing itself
    # off as a measurement: the declaration mode writes no structured node, so
    # the change log is the only record of the difference.
    stats = json.loads(json.dumps(MEASURED_CONSTANTS))
    stats["PROMPT_TOKENS_system_only"]["Russian"] = 1135
    stats["backfilled"] = {"changes": [
        "PROMPT_TOKENS_system_only.Russian = 1135 (declared, not measured; "
        "2026-09-03; basis: byte-identical to the Spanish prompt)"]}
    write_json(cfg.probe_stats_path, stats)
    assert doctor(Config(work_dir=cfg.work_dir,
                         langs=["German", "Russian"])) == 0
    out = capsys.readouterr().out
    assert "system tokens Russian   1135 (declared, not measured)" in out
    assert "system tokens German    1135 (measured)" in out


def test_doctor_reads_the_declared_marker_as_an_exact_value_not_a_prefix(
        tmp_path, capsys):
    """"Declared" vs "measured" is an HONESTY label on the one output a human
    reads before --confirm-spend, and it was decided by a substring test.

    The real change-log line for a declared 1135 also contains "1", "11" and
    "113", so a language later RE-MEASURED at any decimal prefix of the declared
    number was reported as still declared.
    """
    from ankidkdeck.cli import doctor
    cfg = Config(work_dir=tmp_path / "work", langs=["Russian"])
    declared_1135 = ["PROMPT_TOKENS_system_only.Russian = 1135 (declared, not "
                     "measured; 2026-09-03; basis: the Spanish prompt)"]
    for value, expected in ((1135, "declared, not measured"),
                            (113, "measured"),
                            (11, "measured"),
                            (1, "measured")):
        stats = json.loads(json.dumps(MEASURED_CONSTANTS))
        stats["PROMPT_TOKENS_system_only"]["Russian"] = value
        stats["backfilled"] = {"changes": declared_1135}
        write_json(cfg.probe_stats_path, stats)
        assert doctor(cfg) == 0
        assert ("system tokens Russian   %d (%s)" % (value, expected)
                in capsys.readouterr().out), value


def test_doctor_treats_a_zero_or_a_bool_system_token_count_as_missing(
        tmp_path, capsys):
    """A present-but-unusable value is worse than an absent one: the wave
    splitter would subtract nothing and quote the WHOLE prompt as uncached
    payload, so the bill a human approves is the wrong bill.

    `True` is an int in Python and printed as a one-token system prompt.
    """
    from ankidkdeck.cli import doctor
    cfg = Config(work_dir=tmp_path / "work", langs=["Russian"])
    for value in (0, -50, True, False):
        stats = json.loads(json.dumps(MEASURED_CONSTANTS))
        stats["PROMPT_TOKENS_system_only"]["Russian"] = value
        write_json(cfg.probe_stats_path, stats)
        assert doctor(cfg) == 1, value
        out = capsys.readouterr().out
        # printed as what it IS -- saying MISSING about a value sitting in the
        # file is the same species of lie the round is removing -- and refused
        # with the same remedy
        assert "system tokens Russian   UNUSABLE (%r)" % (value,) in out, value
        assert "no usable entry for Russian" in out, value
        assert "--declare-system-tokens Russian=N" in out, value
    # a float IS accepted, because that is exactly what the wave splitter
    # accepts (s42_translate.system_prompt_tokens), and doctor must not refuse
    # a run the stage would have taken
    stats = json.loads(json.dumps(MEASURED_CONSTANTS))
    stats["PROMPT_TOKENS_system_only"]["Russian"] = 1135.7
    write_json(cfg.probe_stats_path, stats)
    assert doctor(cfg) == 0
    assert "system tokens Russian   1135 (measured)" in capsys.readouterr().out


def test_doctor_never_prints_an_invented_cache_floor(tmp_path, capsys):
    """MINOR-2. The real stats.json has no IMPLICIT_CACHE_FLOOR key -- the 4096
    was a source constant printed as if it had been measured, in the one output
    a human reads before pressing --confirm-spend. The two floors are different
    numbers and are never merged into one."""
    from ankidkdeck.cli import doctor
    cfg = Config(work_dir=tmp_path / "work")
    write_json(cfg.probe_stats_path, MEASURED_CONSTANTS)
    doctor(cfg)
    floor_line = next(l for l in capsys.readouterr().out.splitlines()
                      if "cache floor" in l)
    assert floor_line.split("implicit")[1].strip() == "n/a (not in this artifact)"


# ---------------------------------------- F5: one source for the prompt sha

def test_the_bill_sha_and_the_wire_sha_have_one_source(monkeypatch):
    """The bill file computed prompt_sha256(definition_prompt(lang)) at its own
    call site while the request built its system instruction at another. Replace
    the builder at one of them -- which is exactly what the prompt-pack work
    does -- and G-PROMPT compares a stale sha to itself and reports agreement."""
    assert S42.prompt_shas("German")["definition"] == S42.prompt_sha256(
        S42.definition_prompt("German"))
    monkeypatch.setitem(S42._SYSTEM_PROMPTS, "definition",
                        lambda lang: "A DIFFERENT PROMPT for %s" % lang)
    # both readings move together, because there is only one reading
    assert S42.prompt_shas("German")["definition"] == S42.prompt_sha256(
        "A DIFFERENT PROMPT for German")
    ctx = S42.CallContext(cfg=Config(), pool=None, fit=(35.964, 23.07),
                          lang="German")
    assert ctx.request("definition", "l", "u", None, 3,
                       S42.system_prompt("definition", "German")).system == \
        "A DIFFERENT PROMPT for German"
    # ...and a second, divergent source for it is refused rather than sent
    with pytest.raises(FatalError) as exc:
        ctx.request("definition", "l", "u", None, 3, "hand-rolled prompt")
    assert "system_prompt" in str(exc.value)


# ------------------------------------------------------------------ 1.1

def test_the_configured_model_must_be_one_we_measured_and_priced(tmp_path):
    """The near miss this list exists for: the run host's ankidkdeck.toml had no
    model line at all, so the effective model there was the source default --
    which was gemini-2.0-flash, a model none of the v3 constants and none of the
    checked rate card apply to."""
    assert "gemini-3.7-flash" in VERIFIED_MODELS
    assert "gemini-2.0-flash" not in VERIFIED_MODELS
    toml = tmp_path / "ankidkdeck.toml"
    toml.write_text('gemini_model = "gemini-2.0-flash"\n', encoding="utf-8")
    with pytest.raises(FatalError) as exc:
        load_config(toml)
    assert "verified model list" in str(exc.value)
    # ...and the experiment escape hatch is explicit
    toml.write_text('gemini_model = "gemini-2.0-flash"\n'
                    'allow_unverified_model = true\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.model_is_verified() is False


def test_the_new_settings_are_settable_from_toml(tmp_path):
    """_apply FatalErrors on an unknown key, so a field that does not exist here
    cannot be configured on the run host at all."""
    toml = tmp_path / "ankidkdeck.toml"
    toml.write_text("\n".join([
        'gemini_model = "gemini-3.7-flash"',
        'mode = "batch"',
        'thinking_level = "LOW"',
        'prompt_id = "v4-frozen"',
        'cache_enabled = true',
        'cache_ttl_factor = 1.5',
        'cache_key_index = 0',
        'spend_cap_usd = 10.0',
        'retranslate_all = true',
        'retranslate_reason = "clean_redo"',
        'max_output_floor = 1024',
        'def_request_interval = 2.1',
        'expr_request_interval = 5.0',
        'pos_request_interval = 1.1',
        'rank_request_interval = 1.6',
        'max_per_api_key = 5',
        'rpm_limit = 60',
        'rpd_limit = 10000',
        'rate_limits_measured_at = "2026-08-26"',
        'max_output_unmeasured = 4096',
        'thinking_level_override_ack = false',
    ]) + "\n", encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.mode == "batch"
    # a batch row must never carry serviceTier
    assert cfg.effective_service_tier is None
    assert cfg.retranslate_all is True
    assert cfg.max_per_api_key == 5


def test_a_configuration_that_cannot_be_priced_or_sent_is_refused(tmp_path):
    toml = tmp_path / "ankidkdeck.toml"
    for body, needle in (
            ('mode = "turbo"\n', "not one of"),
            ('thinking_level = "low"\n', "thinking_level"),
            ('mode = "batch"\nservice_tier = "flex"\n', "serviceTier"),
            ('service_tier = "priority"\n', "service_tier"),
            ('spend_cap_usd = 0\n', "spend_cap_usd")):
        toml.write_text(body, encoding="utf-8")
        with pytest.raises(FatalError) as exc:
            load_config(toml)
        assert needle in str(exc.value)


def test_flex_mode_is_standard_plus_the_service_tier():
    cfg = Config(mode="flex")
    cfg.validate()
    assert cfg.effective_service_tier == "flex"
    assert Config(mode="standard").effective_service_tier is None


# ------------------------------------------------------------------ N-04

def test_the_grammar_note_travels_in_the_user_message_only():
    """38.8% of the senses on file carry one. It must not enter the cached
    prefix (it is per sense), and an entry without one must produce the same
    bytes as before."""
    plain = [{"text": "bygning man bor i", "grammar": "", "hint": ""}]
    withg = [{"text": "bygning man bor i", "grammar": "NOGET er et hus",
              "hint": ""}]
    assert "Grammar notes" not in S42.definition_user_payload("hus", plain)
    body = S42.definition_user_payload("hus", withg)
    assert "Grammar notes" in body and "NOGET er et hus" in body
    # and nothing about it reaches the system prompt
    assert "Grammar notes" not in S42.definition_prompt("German")
    assert S42.prompt_sha256(S42.definition_prompt("German")) == sha256_str(
        S42.definition_prompt("German"))


# ---------------------------------------------------- the unwired transports

def test_spending_on_a_transport_that_does_not_exist_is_refused():
    """Believing a discount applies costs more than knowing it does not.

    UPDATED for the batch transport: the batch branch of this guard is GONE, because
    the batch transport exists now (ankidkdeck.batch), and so is the blanket
    cache_enabled branch, because the cache lifecycle exists
    (ankidkdeck.batch.caches). What is left is the one combination still in the
    state the guard describes: the cache lifecycle is DRIVEN by the batch wave,
    so cache_enabled on the interactive surface would pay the full uncached rate
    while the bill quoted cache_works.
    """
    S42.transport_guard(Config(mode="batch"))
    S42.transport_guard(Config(mode="batch", cache_enabled=True))
    S42.transport_guard(Config(mode="standard"))
    S42.transport_guard(Config(mode="flex"))
    for mode in ("standard", "flex"):
        with pytest.raises(FatalError) as exc:
            S42.transport_guard(Config(mode=mode, cache_enabled=True))
        assert "cache_enabled" in str(exc.value)
        assert "batch" in str(exc.value)
