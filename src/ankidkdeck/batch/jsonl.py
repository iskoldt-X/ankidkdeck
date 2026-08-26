"""The batch JSONL writer (patch plan N-07, 5.2).

Ported from the probe reference implementation
(home-vm:~/v3run/probes/batch_jsonl.py, 16/16 offline checks, and the 32-row
job whose 30 cacheable rows all came back cached == declared). The probe wrote
its rows from a hand-built request dict; production writes them from the SAME
LlmRequest the interactive path sends, because "the bill, the JSONL and the
interactive call agree" has to be checkable rather than hoped for.

The golden line shape, verbatim from the probe wave:

    {"key": "<unique>",
     "request": {
       "contents": [{"role": "user", "parts": [{"text": ...}]}],
       "cachedContent": "cachedContents/xxx",       <- TOP-LEVEL sibling
       "generationConfig": {                         <- NOT the cache's home
          "responseMimeType": "application/json",
          "responseSchema": {...},
          "thinkingConfig": {"thinkingLevel": "LOW"},
          "maxOutputTokens": N}}}

Every invariant below was measured, not reasoned about:

  * `cachedContent` inside `generationConfig` is rejected AT SUBMIT with
    `400 INVALID_ARGUMENT: Error on line 1: invalid JSON, near column 436: no
    such field: 'cachedContent'`. It is NOT silently billed at full price --
    the best possible outcome, and the reason this file exists.
  * a cache plus a systemInstruction is a hard 400 (`CachedContent can not be
    used with GenerateContent request setting system_instruction, tools or
    tool_config`), so the system prompt moves INTO the cache and the row drops
    the field entirely.
  * `request.model` is omitted: the model belongs to the job, and every
    official example omits it even though the REST schema marks it required.
  * a batch row never carries `serviceTier`. flex is a property of the
    INTERACTIVE surface; a batch job is already at batch rates.
  * `temperature` is never sent (deprecated on this model generation, and the
    A/B measured no difference on count-lock violations, MAX_TOKENS finishes or
    thinking).
"""

import json

from ..util import FatalError


def rest_schema(schema: dict) -> dict:
    """A production schema in its wire form, produced by the SDK's own model.

    Not by hand: the count lock (minItems == maxItems == n) is the one thing in
    the schema that must survive serialization, and hand-writing the camelCase
    conversion would be a second implementation of the SDK's alias table. The
    probe used exactly this call, so the rows this writer produces are
    byte-comparable with the rows that were actually accepted.
    """
    from google.genai import types             # noqa: PLC0415 - lazy by policy
    return types.Schema.model_validate(schema).model_dump(
        mode="json", exclude_none=True, by_alias=True)


def build_row(key: str, req, *, schema_serializer=None) -> dict:
    """One JSONL row from one LlmRequest.

    `req` is what CallContext.request() returned: the cap, the thinking level,
    the cache/system exclusion and the service tier were all decided there, and
    this function decides NOTHING except field placement. That is the point --
    a batch-only cap or a batch-only thinking level would be a second source
    for the numbers the bill quotes.

    `schema_serializer` exists for the offline field-placement tests, which have
    to run whether or not the SDK is installed. Production leaves it None and
    gets rest_schema.
    """
    serialize = schema_serializer or rest_schema
    system = getattr(req, "system", None)
    cache_name = getattr(req, "cache_name", None)
    if system and cache_name:
        # LlmRequest.__post_init__ already refuses this; asserted again here
        # because this writer is the last thing between the decision and the
        # wire, and the failure is a hard 400 on the whole file.
        raise FatalError(
            "%s: systemInstruction XOR cachedContent. Sending both is a hard "
            "400; the system prompt has to move INTO the cache."
            % getattr(req, "label", key))
    if getattr(req, "service_tier", None):
        raise FatalError(
            "%s: a batch JSONL row must not carry serviceTier (%r). flex is a "
            "property of the interactive surface; a batch job is already at "
            "batch rates, and the field is not part of the batch row schema."
            % (getattr(req, "label", key), req.service_tier))
    gen = req.generation_config()
    if "cachedContent" in gen:
        raise FatalError(
            "%s: cachedContent appeared inside generationConfig. That shape is "
            "rejected at submit with `no such field: 'cachedContent'`, and it "
            "is the placement error this writer exists to make impossible."
            % getattr(req, "label", key))
    if gen.get("responseSchema") is not None:
        gen["responseSchema"] = serialize(gen["responseSchema"])
    row_request = {
        "contents": [{"role": "user", "parts": [{"text": req.user}]}],
        "generationConfig": gen,
    }
    if system:
        row_request["systemInstruction"] = {"parts": [{"text": system}]}
    if cache_name:
        # TOP-LEVEL sibling of generationConfig. The whole file is here for
        # this line.
        row_request["cachedContent"] = cache_name
    return {"key": key, "request": row_request}


def write_jsonl(path, rows) -> str:
    """One row per line, UTF-8, no trailing whitespace, KEYS SORTED.

    `sort_keys` is the determinism, and it is a deliberate improvement on the
    probe's reference writer rather than a copy of it: that writer used a plain
    json.dumps, so its output depended on dict insertion order. The two writers
    therefore produce the same OBJECT and not the same BYTES -- semantically
    equivalent, different key order -- so any claim of byte-identity BETWEEN THEM
    has to say "after canonicalising".

    What byte-identity is claimed for is this writer against itself: the same
    wave written twice is byte-identical whatever order the row dicts were built
    in, which is what makes the wave fingerprint mean something and what patch
    plan 4.2(3) asks for. Writing the SAME dict object twice in one process
    cannot detect a missing sort_keys -- CPython preserves insertion order -- so
    the test that pins this builds an equivalent row with a DIFFERENT insertion
    order and compares raw bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def assert_batch_row_shape(row: dict) -> dict:
    """The W0-2 field-placement assertions, on a row that is about to be sent.

    Cheap enough to run on every row of every wave: the historical failure this
    checks for cost a full-price wave, and the check is six dictionary lookups.
    """
    req = row.get("request") or {}
    gen = req.get("generationConfig") or {}
    problems = []
    if "model" in req:
        problems.append("`model` inside `request` (the job carries the model)")
    if "cachedContent" in gen:
        problems.append("`cachedContent` inside generationConfig")
    if req.get("cachedContent") and "systemInstruction" in req:
        problems.append("both cachedContent and systemInstruction")
    if "responseMimeType" not in gen:
        problems.append("no responseMimeType in generationConfig")
    if "thinkingConfig" not in gen:
        problems.append("no thinkingConfig in generationConfig (unset means "
                        "MEDIUM, measured at mean 578.7 thought tokens)")
    if "temperature" in gen:
        problems.append("`temperature` (deprecated on this model generation)")
    if "serviceTier" in json.dumps(row):
        problems.append("`serviceTier` anywhere in a batch row")
    if problems:
        raise FatalError(
            "batch row %r has the wrong shape: %s"
            % (row.get("key"), "; ".join(problems)))
    return row
