"""The batch transport: submit -> poll -> download -> ingest -> retry.

One wave is TWO --confirm-spend invocations (`--phase submit`, then
`--phase ingest` when the jobs have drained), or one (`--phase all`). What is
NOT split is the request construction: every batch row comes out of
CallContext.request(), the same call the interactive path makes, so the cap, the
thinking level, the cache/system exclusion and the service tier are decided once
for all three transports. A batch-only copy of that logic is exactly how the
bill ends up describing a different request from the one on the wire.

The order of operations, and why each step is where it is:

  1. cache first, for the definition kind only. The expression prompt is ~640
     tokens against a measured 1,024-token floor and is left uncached on
     purpose (padding it would buy a discount on tokens that only exist because
     of the padding).
  2. build every request, then SPLIT by uncached tokens. Cached tokens do not
     count towards the enqueued limit (measured), which is what makes the
     definition wave 2 jobs per language instead of 6.
  3. per job, in this order: reserve the fingerprint, write the JSONL, upload,
     create. batches.create is NOT idempotent, so the record of intent is
     written BEFORE the call and a crashed submit is recovered by listing jobs
     and matching the uploaded input file's resource name -- never by calling
     create again.
  4. ONE job in flight. Poll to a terminal state, download IMMEDIATELY (the
     documented retention is self-contradictory: 6 weeks in one place, 48h in
     another), then submit the next.
  5. ingest by KEY. The output order is NOT the input order past ~1000 rows
     (measured 2026-08-27 on the first real wave: four ~1000-row shards
     concatenated out of order), so the echoed key is the attribution and the
     position is a reported cross-check. See reconcile.py.
  6. retry waves, bounded by MAX_RETRY_WAVES and by a per-cell attempt counter
     that survives the process. Retries stay on batch (owner decision D-06):
     nothing downgrades to standard or flex by itself, because an automatic
     downgrade doubles the rate silently.
"""

import dataclasses
import datetime
import json
import math
import time

from ..util import FatalError, read_json, write_json
from . import caches as cache_lifecycle
from . import jsonl, keys, reconcile, registry, waves

# How long one job may take before this run gives up waiting for it. The
# documented target is 24 hours and the HARD expiry is 48; a job that has not
# finished by then is EXPIRED, which is terminal and recorded rather than
# retried in place.
JOB_WAIT_SECONDS = 50 * 3600
JOB_POLL_SECONDS = 150

# The expression wave's canary. The bill books an unmeasured 275 thought tokens
# per expression request (the ranking prompt's measured maximum at the same LOW)
# because nobody has ever probed the expression prompt: at 1,922 requests per
# language that prior is $0.99/language, $3.97 across four. One small job
# measures it for real, and the number lands on disk where the bill reads it.
CANARY_REQUESTS = 20


@dataclasses.dataclass
class PlannedRequest:
    """One request, before and after it has been through a job.

    `rows` are the todo rows (they carry the Danish text and never leave the
    process); `cells` is the serializable half that goes into the job registry,
    so an ingest hours later knows which cells this position owns without
    trusting that entries.json is still what it was at submit time.
    """
    key: str
    kind: str
    entry_id: str
    label: str
    rows: list
    cap: int
    cached_tokens: int = 0
    uncached_tokens: int = 0
    correction: str = ""
    pos: int = 0

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def cells(self) -> list:
        return [{"key": r["key"], "src_sha": r["src_sha"]} for r in self.rows]

    def plan_record(self) -> dict:
        return {"pos": self.pos, "key": self.key, "kind": self.kind,
                "entry_id": self.entry_id, "n": self.n, "label": self.label,
                "cells": self.cells, "cap": self.cap,
                "correction": self.correction,
                "cached_tokens": self.cached_tokens,
                "uncached_tokens": self.uncached_tokens}


# --------------------------------------------------------------------------
# the SDK surfaces, one function each (lazy import, never on the bill path)
# --------------------------------------------------------------------------

def upload_jsonl(client, path, display_name: str) -> str:
    """Upload the JSONL and return its FILE RESOURCE NAME ("files/abc123").

    That string, not the local path, is what batches.create takes as `src`, and
    it is also the only identifier a crashed submit can be recovered by.
    """
    from google.genai import types                    # noqa: PLC0415
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(display_name=display_name,
                                      mime_type="application/jsonl"))
    return uploaded.name


def create_job(client, *, model: str, src: str, display_name: str):
    from google.genai import types                    # noqa: PLC0415
    return client.batches.create(
        model=model, src=src,
        config=types.CreateBatchJobConfig(display_name=display_name))


def find_job_by_input_file(client, src: str):
    """The recovery path for a create() that failed with a 5xx.

    NEVER retry create blindly: it is not idempotent, both submissions are
    accepted, both run and both are billed. List the jobs and match on the input
    file resource name, which is the one identifier that survives on both sides.
    """
    try:
        listed = client.batches.list()
    except Exception as exc:                          # noqa: BLE001
        raise FatalError(
            "the batch submit failed and the job list could not be read either "
            "(%s). Do NOT resubmit: batches.create is not idempotent, so a "
            "second attempt may be a second charge. Check the console for a "
            "job whose input file is %r before running again." % (exc, src)) \
            from exc
    for job in listed:
        for attr in ("src", "input_file", "inputFile"):
            value = getattr(job, attr, None)
            if value and str(value) == str(src):
                return job
    return None


def poll_until_terminal(client, job_name: str, *, timeout_s=JOB_WAIT_SECONDS,
                        interval_s=JOB_POLL_SECONDS, sleep=None, now=None,
                        on_state=None):
    """Poll one job to a terminal state and return it, or raise on timeout.

    `sleep` and `now` are injectable so the drain loop is testable without
    waiting: patching time.sleep alone would turn a deadline loop into a spin.
    """
    sleep = sleep or time.sleep
    now = now or time.time
    deadline = now() + float(timeout_s)
    last = None
    while True:
        job = client.batches.get(name=job_name)
        state = registry.job_state_name(getattr(job, "state", None))
        if state != last and on_state is not None:
            on_state(state)
        last = state
        if registry.is_terminal(state):
            return job
        if now() >= deadline:
            raise FatalError(
                "batch job %s is still %s after %.0f hour(s) of waiting. The "
                "documented target is 24h and the hard expiry is 48h, so this "
                "is either an EXPIRED job the service has not relabelled or a "
                "wait that should be resumed with --phase ingest later. The "
                "job registry has kept its place; nothing is resubmitted."
                % (job_name, state, float(timeout_s) / 3600.0))
        sleep(interval_s)


def download_results(client, job, dest_path) -> str | None:
    """Write the result file next to the wave and return its path.

    Immediately, and to disk: the retention documentation contradicts itself (6
    weeks in the batch guide, 48 hours in the API reference), and the file is
    the only copy of what was paid for.
    """
    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest is not None else None
    if file_name is None and dest is not None:
        file_name = getattr(dest, "fileName", None)
    if not file_name:
        return None
    raw = client.files.download(file=file_name)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(text, encoding="utf-8")
    return str(dest_path)


# --------------------------------------------------------------------------
# request construction (through CallContext, never beside it)
# --------------------------------------------------------------------------

def plan_requests(s42, ctx, todo: list, kind: str, lang: str, *,
                  system_tokens: int, prompt_fit, cacheable: bool) -> list:
    """Every request of one (lang, kind), in a deterministic order.

    The per-request token counts come from s42.request_input_tokens, the same
    function the bill uses, so the splitter and the quote cannot disagree about
    how big a request is.
    """
    max_batch = (s42.MAX_DEFS_PER_BATCH if kind == "definition"
                 else s42.MAX_EXPR_PER_BATCH)
    out = []
    chunk_of: dict = {}
    for eid, rows in s42._group_by_entry(todo, kind, max_batch):
        chunk = chunk_of.get(eid, 0)
        chunk_of[eid] = chunk + 1
        label = ("%s %s" % (rows[0]["lemma"], rows[0]["pos_text"])).strip()
        chars = sum(len(r["text"]) + len(r.get("grammar") or "")
                    + len(r["hint"]) for r in rows)
        tokens = s42.request_input_tokens(
            kind, len(rows), chars, system_tokens=system_tokens,
            measured_system_tokens=system_tokens, prompt_fit=prompt_fit,
            cached=cacheable)
        out.append(PlannedRequest(
            key=keys.make_key(kind, lang, eid, chunk), kind=kind, entry_id=eid,
            label=label, rows=rows,
            cap=s42.resolve_max_output(ctx.cfg, len(rows), ctx.fit, kind),
            cached_tokens=tokens["cached"], uncached_tokens=tokens["uncached"]))
    keys.validate_keys([r.key for r in out])
    return out


def llm_request_for(s42, ctx, planned: PlannedRequest):
    """The LlmRequest for one planned request, built through CallContext.

    Two things are deliberately taken from `planned` rather than from the
    context: the cap (a retry may carry a raised one) and the correction (which
    travels at the END of the user message, never in front of the system prompt,
    because the system prompt is the cached prefix).
    """
    if planned.kind == "definition":
        user = s42.definition_user_payload(planned.label, planned.rows,
                                           planned.correction)
        schema = s42.definition_schema(planned.n)
    elif planned.kind == "expression":
        user = s42.expression_user_payload(ctx.lang, planned.label,
                                           planned.rows, planned.correction)
        schema = s42.expression_schema(planned.n)
    else:
        raise FatalError("the batch transport does not carry %r requests"
                         % planned.kind)
    req = ctx.request(planned.kind, "%s batch %s" % (planned.kind, planned.label),
                      user, schema, planned.n,
                      s42.system_prompt(planned.kind, ctx.lang))
    if planned.cap != req.max_output_tokens:
        # A raised budget from a MAX_TOKENS retry. Rebuilt from the same
        # object so nothing else can differ.
        req = dataclasses.replace(req, max_output_tokens=planned.cap)
    return req


def write_job_jsonl(s42, ctx, planned_rows: list, path):
    """Serialize one job's rows, LAZILY -- just before the submit.

    Not in advance: a recreated cache has a new resource name (an expired cache
    cannot be updated, only recreated), and a JSONL written before that carries
    a dead reference on every line.
    """
    rows = []
    for planned in planned_rows:
        req = llm_request_for(s42, ctx, planned)
        rows.append(jsonl.assert_batch_row_shape(
            jsonl.build_row(planned.key, req)))
    jsonl.write_jsonl(path, rows)
    return path


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

def translate_wave(cfg, *, langs, state, report, usage, fit, pool, provs,
                   provs_expr, stats, bill, phase, retranslate_all,
                   done_prefixes, pos_wanted, violations_by_lang=None,
                   client=None, sleep=None, now=None) -> dict:
    """Replace stage 42's per-language interactive loop with the batch wave.

    Called from s42.run() inside the same try/except that persists the spend
    records, so a crash anywhere in here still leaves every paid call on disk.

    `provs` / `provs_expr` come from the CALLER rather than being rebuilt here:
    the provenance string is what a resumed clean redo compares live rows
    against, so a second construction of it -- even a correct one -- would be a
    second place for it to drift.
    """
    from ..stages import s42_translate as s42        # noqa: PLC0415
    reg = registry.JobRegistry(cfg)
    client = client if client is not None else pool.client()
    violations_by_lang = {} if violations_by_lang is None else violations_by_lang
    out: dict = {"phase": phase, "languages": {},
                 "registry": reg.summary(), "cache": {}, "canary": None}
    counts_cached, counting_why = waves.enqueued_counts_cached(stats)
    target = waves.job_token_target()
    out["splitting"] = waves.enqueued_note(counts_cached, counting_why, target)
    prompt_fit = s42.prompt_token_fit(stats)
    if prompt_fit is None:
        raise FatalError(
            "PROMPT_TOKENS_fit is not in the measured constants, so the wave "
            "splitter cannot size a request. The enqueued limit is a hard "
            "refusal at submit; guessing the size of 3,623 requests is not a "
            "way to meet it.")
    for lang in langs:
        out["languages"][lang] = _one_language(
            s42, cfg, lang, state[lang], reg, client, report=report,
            usage=usage, fit=fit, pool=pool, stats=stats,
            phase=phase, retranslate_all=retranslate_all,
            done_prefix=done_prefixes.get(lang, ""), pos_wanted=pos_wanted,
            violations_by_lang=violations_by_lang, prov=provs[lang],
            prov_expr=provs_expr[lang],
            prompt_fit=prompt_fit, counts_cached=counts_cached, target=target,
            wave_out=out, sleep=sleep, now=now)
    out["registry"] = reg.summary()
    out["cache_prompt_shas"] = reg.cache_prompt_shas()
    out["declared_cache_tokens_by_language"] = _declared_seen(
        reg, langs, out["languages"])
    # A job that ended terminal with no result file is money committed for
    # nothing. RECORDED as it happened (the wave kept going, because a
    # FatalError mid-wave discards the jobs that did work) and raised at the
    # very end by the caller, after the report is on disk. Unrecovered
    # REQUESTS are deliberately not fatal: those cells are simply still
    # `missing`, which is a state the next run handles.
    # THIS INVOCATION's failures, not every failure the registry ever recorded.
    # Reading them off reg.in_state(FAILED) meant a failure that had since been
    # resolved -- the wave released and successfully resubmitted -- kept raising
    # on every later run for good. A job that was PLANNED and never uploaded is
    # also not in this list: nothing was submitted, so nothing was committed.
    failed = []
    for lang in langs:
        summary = out["languages"].get(lang) or {}
        for job in summary.get("jobs") or []:
            if job.get("state") != "FAILED":
                continue
            record = reg.jobs().get(job["job_id"]) or {}
            failed.append({"job_id": job["job_id"],
                           "why": record.get("failure"),
                           "job_state": job.get("job_state"),
                           "resubmittable": record.get("resubmittable")})
        for row in summary.get("resumed") or []:
            if row.get("action") == "unclaimable":
                record = reg.jobs().get(row["job_id"]) or {}
                failed.append({"job_id": row["job_id"],
                               "why": record.get("failure"),
                               "job_state": record.get("job_state"),
                               "resubmittable": False})
    if failed:
        out["failed_jobs"] = failed
        out["failure"] = (
            "%d batch job(s) ended without a result file: %s. A submitted job "
            "that produced nothing is money committed for nothing, and it does "
            "not resolve itself -- check the job in the console before "
            "re-running." % (len(failed), "; ".join(
                "%s (%s)" % (f["job_id"], f["why"]) for f in failed)))
    # The gates that adjudicate a wave that has ALREADY been paid for. Evaluated
    # and recorded here, raised by the caller after the run's own report is on
    # disk: a failing gate that raises first leaves translate_report.json
    # unwritten and the PREVIOUS run's file on disk describing a different run.
    if phase in ("all", "ingest") and usage.rows:
        out["gates"] = _post_wave_gates(cfg, bill, usage, langs, reg, stats,
                                        out["languages"])
        out["gates_ok"] = out["gates"]["ok"]
        out["gates_error"] = out["gates"]["error"]
    return out


def _declared_for(reg, lang, summaries=None):
    """G-CACHE's denominator FOR ONE LANGUAGE: what this wave's cache declared.

    Three sources, in order, and none of them is a min() across history:

      1. this invocation's own summary for this language, recorded before the
         end-of-wave delete;
      2. the per-language declaration the registry persisted at cache creation,
         which is what carries the number across the process boundary between
         `--phase submit` and `--phase ingest`;
      3. nothing -- and then the gate reports it cannot check, which for a wave
         that expected a cache is a refusal.

    The old fallback took min() over EVERY job in the registry with no language
    filter, and because the cache record is deleted before the gates run, that
    fallback was the normal path rather than the exceptional one. The English
    definition prompt is 1,092 tokens against 1,135 for the other three, so a
    four-language run adjudicated all four against 1,092 and the `cached ==
    declared` criterion was dead for three of them. Worse, the direction is
    unsafe: the share is sum(cached)/(declared x requests), so a stale small
    declaration RAISES it -- measured, a stray declared=5 made the 1-hit /
    99-fallback wave score 2.27 and pass.
    """
    own = ((summaries or {}).get(lang) or {}).get("declared_cache_tokens")
    if own:
        return int(own)
    return reg.declared_for(lang)


def _declared_seen(reg, langs, summaries=None) -> dict:
    """{lang: declared tokens} for the report. Different per language, on purpose."""
    return {lang: _declared_for(reg, lang, summaries) for lang in langs}


def _post_wave_gates(cfg, bill, usage, langs, reg, stats, ranges):
    """G-BILL / G-THINK / G-PROMPT / G-CACHE, per language, on this wave's rows.

    Split by the row RANGE each language wrote rather than by a field added to
    the row: G-CACHE's denominator is "the requests of this wave that should
    have been cached", so mixing two languages' rows into one verdict would make
    a healthy wave look half-cached, and G-BILL would count one language's cost
    against another's quote. The range keeps the ledger row exactly the shape
    the money stack defined.
    """
    from ..gates import failure_message, post_wave_gates, run_gates
    results = []
    shas = reg.cache_prompt_shas()
    seen = _declared_seen(reg, langs, ranges)
    for lang in langs:
        start, end = (ranges.get(lang) or {}).get("usage_rows") or (0, 0)
        rows = usage.rows[start:end]
        if not rows:
            continue
        results += run_gates(
            post_wave_gates(cfg, bill, rows, lang=lang,
                            declared_cache_tokens=seen.get(lang),
                            cache_prompt_shas=shas, stats=stats),
            cfg, stage="42", raise_on_failure=False)
    message = failure_message(results)
    return {"ok": not message, "error": message,
            "rows": [{"id": r["id"], "ok": r["ok"], "extra": r["extra"]}
                     for r in results],
            # PER LANGUAGE, because they differ: the English definition prompt
            # is 1,092 tokens against 1,135 for the others.
            "declared_cache_tokens_by_language": seen,
            "cache_prompt_shas": sorted(shas)}


def _one_language(s42, cfg, lang, st, reg, client, *, report, usage, fit, pool,
                  stats, phase, retranslate_all, done_prefix, pos_wanted,
                  violations_by_lang, prov, prov_expr, prompt_fit,
                  counts_cached, target, wave_out, sleep, now) -> dict:
    from .. import prompts                            # noqa: PLC0415
    tdir, defs, exprs = st["dir"], st["defs"], st["exprs"]
    # The EFFECTIVE id: the pack version is part of the prompt text, so a wave
    # fingerprint under the bare cfg.prompt_id would call two different waves
    # the same wave.
    effective_id = prompts.effective_prompt_id(lang)
    # Where this language's ledger rows start. The money gates are adjudicated
    # per language and the range is how they are told apart.
    row_start = len(usage.rows)
    summary: dict = {"written": {"definitions": 0, "expressions": 0},
                     "jobs": [], "failed_requests": [], "retry_waves": 0,
                     "effective_prompt_id": effective_id}
    if retranslate_all and phase in ("all", "submit"):
        summary["archived_for_redo"] = s42.archive_for_redo(cfg, lang, st,
                                                            done_prefix)
    system_tokens = s42.system_prompt_tokens(stats, lang)
    if system_tokens is None:
        raise FatalError(
            "PROMPT_TOKENS_system_only.%s is not in the measured constants. The "
            "wave splitter subtracts the system half from the measured prompt "
            "fit to get the uncached payload, and the enqueued limit only "
            "counts that half." % lang)

    # ---- FIRST: finish what a previous invocation left in flight ----------
    #
    # Before anything else and on BOTH phases. On an ingest this is the recovery
    # of a submit that died or timed out mid-drain (the money is already
    # committed and the results are still fetchable). On a submit it is what
    # keeps "one job in flight" true ACROSS processes: the constant used to be a
    # number in the report, and a second `--phase submit` while the first job was
    # still SUBMITTED went ahead and opened another one.
    _resume_in_flight(cfg, reg, client, lang, summary=summary, sleep=sleep,
                      now=now)

    # ---- the cache, definition kind only --------------------------------
    #
    # On a submit the cache is created or verified against the service. On a
    # pure ingest it is read from the registry WITHOUT a call: an ingest places
    # no request, so checking a cache's life there would be an API call to
    # answer a question nobody asked. Every path that is ABOUT TO SUBMIT
    # verifies it again, per job.
    handle = _cache_for(cfg, lang, reg, client, stats, s42, todo=st["todo"],
                        report=wave_out, verify=phase in ("all", "submit"))
    contexts = {}
    for kind in ("definition", "expression"):
        contexts[kind] = s42.CallContext(
            cfg=cfg, pool=pool, fit=fit, lang=lang, usage=usage,
            prompt_id=cfg.prompt_id, mode="batch",
            cache_name=handle.name if (handle is not None
                                       and kind in s42.CACHEABLE_KINDS)
            else None,
            violations_path=cfg.review_dir
            / ("count_lock_violations_%s.json" % lang))
    violations_by_lang.setdefault(lang, contexts["definition"].violations)
    contexts["expression"].violations = contexts["definition"].violations

    # ---- submit ----------------------------------------------------------
    if phase in ("all", "submit"):
        for kind in ("definition", "expression"):
            ctx = contexts[kind]
            planned = plan_requests(
                s42, ctx, st["todo"], kind, lang, system_tokens=system_tokens,
                prompt_fit=prompt_fit,
                cacheable=bool(ctx.cache_name))
            if not planned:
                continue
            jobs = waves.split_into_jobs(
                [{"entry_id": p.entry_id, "cached_tokens": p.cached_tokens,
                  "uncached_tokens": p.uncached_tokens, "planned": p}
                 for p in planned],
                target_tokens=target, counts_cached=counts_cached)
            batches = [([row["planned"] for row in job], False) for job in jobs]
            if kind == "expression":
                batches = _with_canary(batches, stats, lang, summary)
            for index, (group, canary) in enumerate(batches):
                # BEFORE EACH SUBMIT, not once per language: one job in flight at
                # a time and a 50-hour poll deadline means job 2's submit can be
                # a day after job 1's, and a cache is only alive while its TTL
                # runs. caches.get asserts, caches.update extends, and an expired
                # one is recreated with a NEW resource name -- which is why the
                # JSONL is written inside _submit_and_drain and not in advance.
                handle = _cache_before_submit(cfg, lang, reg, client, stats, s42,
                                              todo=st["todo"], report=wave_out,
                                              handle=handle, kind=kind,
                                              contexts=contexts)
                _submit_and_drain(s42, cfg, ctx, reg, client, lang, kind, group,
                                  handle=_cache_of(s42, handle, kind), wave=0,
                                  index=index, effective_id=effective_id,
                                  summary=summary, sleep=sleep, now=now,
                                  canary=canary)

    # ---- ingest, then bounded retry waves --------------------------------
    if phase in ("all", "ingest"):
        todo_by_key = {r["key"]: r for r in st["todo"]}
        pending = _ingest_ready(s42, cfg, lang, reg, defs, exprs, tdir,
                                usage=usage, prov=prov, prov_expr=prov_expr,
                                contexts=contexts, summary=summary,
                                stats=stats, wave_out=wave_out,
                                todo_by_key=todo_by_key)
        wave = 1
        while pending and waves.within_retry_bound(wave):
            summary["retry_waves"] = wave
            # The cache is verified per JOB inside the loop below
            # (_cache_before_submit), which is the same rule the submit phase
            # follows: a retry wave may be hours after the submit and the cache
            # is resolved when each row EXECUTES.
            resubmitted = 0
            for kind in ("definition", "expression"):
                group = [p for p in pending if p.kind == kind]
                group = _drop_exhausted(reg, group, summary)
                if not group:
                    continue
                reg.bump_attempts([p.key for p in group])
                handle = _cache_before_submit(cfg, lang, reg, client, stats, s42,
                                              todo=st["todo"], report=wave_out,
                                              handle=handle, kind=kind,
                                              contexts=contexts)
                _submit_and_drain(s42, cfg, contexts[kind], reg, client, lang,
                                  kind, group,
                                  handle=_cache_of(s42, handle, kind),
                                  wave=wave, index=0,
                                  effective_id=effective_id,
                                  summary=summary, sleep=sleep, now=now)
                resubmitted += len(group)
            if not resubmitted:
                break
            pending = _ingest_ready(s42, cfg, lang, reg, defs, exprs, tdir,
                                    usage=usage, prov=prov, prov_expr=prov_expr,
                                    contexts=contexts, summary=summary,
                                    stats=stats, wave_out=wave_out,
                                    todo_by_key=todo_by_key)
            wave += 1
        if pending:
            # RECORD AND CONTINUE. A FatalError here would throw away a wave
            # that has already been paid for; the cells stay missing, they are
            # named in the report, and the next run's compute_todo picks them
            # up as `missing`.
            summary["unrecovered"] = [{"key": p.key, "kind": p.kind,
                                       "n": p.n, "attempts":
                                           reg.attempts_for(p.key)}
                                      for p in pending]
        # The POS table is ONE request per language and its prompt is 400-odd
        # tokens: it is not worth a job, a cache or a wave. It runs on the
        # interactive surface, in its own context, so its ledger row says
        # standard and is priced as standard.
        summary["pos_keys_written"] = _pos_on_the_interactive_surface(
            s42, cfg, lang, st, pool, fit, usage, pos_wanted)
        # BEFORE the delete. G-CACHE's denominator is what THIS language's cache
        # declared, and reading it after the cache object is gone is how it ended
        # up being min() over every job the workspace ever ran.
        summary["declared_cache_tokens"] = _declared_for(reg, lang)
        # From the REGISTRY, not from `handle`: a pure ingest with no retries
        # never touched the cache, and a cache left behind is billed by the
        # token-hour for as long as its TTL runs.
        _end_of_wave_cache(cfg, lang, reg, client, wave_out)

    report["languages"][lang] = {
        "written": summary["written"],
        "pos_keys_written": summary.get("pos_keys_written", 0),
        "definition_rows": len(defs), "expression_rows": len(exprs),
        "pos_rows": len(st["pos"]), "provenance": prov,
        "provenance_expressions": prov_expr,
        "count_lock_violations": len(contexts["definition"].violations),
        "transport": "batch", "phase": phase,
        "batch_jobs": len(summary["jobs"]),
        "retry_waves": summary["retry_waves"],
        "unrecovered_requests": len(summary.get("unrecovered") or []),
    }
    summary["usage_rows"] = [row_start, len(usage.rows)]
    return summary


# --------------------------------------------------------------------------
# submit / drain / ingest
# --------------------------------------------------------------------------

def _cache_of(s42, handle, kind):
    """The cache handle a job of this KIND may reference, or None.

    The wave has exactly one cache object and it holds the DEFINITION prompt, so
    handing it to an expression job would stamp the definition prompt's sha on
    every expression row and fold the definition cache's name into the expression
    job's fingerprint. Caught by G-PROMPT the first time this ran: the expression
    rows reported the definition prompt's sha against the bill's expression sha.
    """
    if handle is None or kind not in s42.CACHEABLE_KINDS:
        return None
    return handle


def _submit_and_drain(s42, cfg, ctx, reg, client, lang, kind, group, *, handle,
                      wave, index, effective_id, summary, sleep, now,
                      canary: bool = False) -> dict:
    """One job, all the way from fingerprint to a downloaded result file.

    ONE JOB IN FLIGHT: this function does not return until the job it submitted
    is terminal and its results are on disk. Two jobs in flight would share the
    enqueued ceiling and the failure mode is a submit the service refuses in the
    middle of a drain.
    """
    for pos, planned in enumerate(group):
        planned.pos = pos
    row_keys = [p.key for p in group]
    model = (cfg.gemini_model if kind == "definition"
             else cfg.expressions_model)
    fingerprint = registry.wave_fingerprint(
        model=model, prompt_id=effective_id, lang=lang,
        keys=row_keys, cache_name=handle.name if handle else None, wave=wave)
    existing = reg.find_by_fingerprint(fingerprint)
    if existing:
        print("  batch: %s %s wave %d job %d is already in the registry as %r "
              "(state %s); not resubmitting"
              % (lang, kind, wave, index, existing,
                 reg.jobs()[existing].get("state")))
        return reg.jobs()[existing]
    # The base id is deterministic, which is what makes a repeat of the SAME
    # command land on the same record. A resubmittable failure (the service said
    # the job produced nothing) gets the next free attempt suffix instead of
    # wedging the slot; a failure whose results we merely could not fetch refuses
    # here rather than reopening a job that may already have been billed.
    job_id = reg.next_job_id("%s-%s-w%d-%02d"
                             % (lang, keys.KIND_TAG[kind], wave, index))
    path = cfg.work_dir / "batch" / ("%s.jsonl" % job_id)
    # Reserve the fingerprint and the plan FIRST. If the process dies between
    # here and the create() below, the recovery is a batches.list match on the
    # uploaded file -- never a second create.
    record = reg.plan(
        job_id, fingerprint=fingerprint, lang=lang, kind=kind, model=model,
        prompt_id=effective_id,
        cache_name=handle.name if handle else None,
        declared_cache_tokens=handle.declared_tokens if handle else None,
        cache_prompt_sha256=handle.prompt_sha256 if handle else None,
        jsonl_path=str(path), plan=[p.plan_record() for p in group],
        enqueued_tokens=sum(p.uncached_tokens for p in group), wave=wave,
        canary=canary)
    write_job_jsonl(s42, ctx, group, path)
    display = "ankidkdeck-%s" % job_id
    src = upload_jsonl(client, path, display)
    reg.mark_uploaded(job_id, src)
    try:
        job = create_job(client, model=model, src=src, display_name=display)
    except Exception as exc:                          # noqa: BLE001
        kind_of = s42.classify_api_error(exc)
        if kind_of == s42.ERR_FATAL:
            # A 400/403 refusal is the service declining to create the job at
            # all: nothing runs and nothing is billed, so the corrected wave may
            # be submitted again.
            reg.mark_failed(job_id, "create refused: %s" % str(exc)[:300],
                            resubmittable=True)
            raise
        recovered = find_job_by_input_file(client, src)
        if recovered is None:
            # The create may have been accepted and the answer lost. This is the
            # one failure that must NOT be reopened automatically.
            reg.mark_failed(job_id, "create failed and no job matches input "
                                    "file %s: %s" % (src, str(exc)[:300]),
                            resubmittable=False)
            raise FatalError(
                "batches.create failed (%s) and no submitted job references the "
                "input file %s. batches.create is NOT idempotent, so this run "
                "will not try again -- a second create may be a second charge. "
                "The registry has recorded the attempt as FAILED; re-run once "
                "the console shows whether the job exists." % (exc, src)) \
                from exc
        print("  batch: create returned %s but job %s already exists for input "
              "file %s; adopting it instead of resubmitting"
              % (kind_of, recovered.name, src))
        job = recovered
    reg.mark_submitted(job_id, job.name, getattr(job, "state", None))
    print("  batch: submitted %s (%d rows, %d uncached tokens) as %s"
          % (job_id, len(group), record["enqueued_tokens"], job.name))
    return _drain_one(cfg, reg, client, job_id, job.name, rows=len(group),
                      wave=wave, kind=kind, summary=summary,
                      enqueued_tokens=record["enqueued_tokens"],
                      sleep=sleep, now=now)


def _drain_one(cfg, reg, client, job_id, job_name, *, rows, wave, kind, summary,
               enqueued_tokens=None, sleep=None, now=None) -> dict:
    """Poll one SUBMITTED job to terminal and get its result file on disk.

    Shared by the submit loop and by the crash recovery, which is the point: the
    recovery is not a second, simpler implementation of "wait for a job" -- it is
    the same one, so a job that is resumed hours later goes through the same
    terminal-state reading, the same immediate download and the same
    FAILED/DOWNLOADED bookkeeping as one that was never interrupted.
    """
    job = poll_until_terminal(
        client, job_name, sleep=sleep, now=now,
        on_state=lambda s: print("    %s %s" % (job_id, s)))
    final_state = registry.job_state_name(getattr(job, "state", None))
    reg.mark_job_state(job_id, final_state)
    results = download_results(client, job,
                              cfg.work_dir / "batch" / ("%s_results.jsonl"
                                                        % job_id))
    if results is None:
        # WHO failed decides whether this wave may be submitted again. The
        # service saying FAILED/CANCELLED/EXPIRED means nothing was produced and
        # nothing was billed; a SUCCEEDED job whose result file we could not read
        # is money already committed, and reopening it would be a second charge.
        service_produced_nothing = final_state in ("FAILED", "CANCELLED",
                                                   "EXPIRED")
        reg.mark_failed(job_id, "job ended %s with no result file" % final_state,
                        job_state=final_state,
                        resubmittable=service_produced_nothing)
        summary["jobs"].append({"job_id": job_id, "state": "FAILED",
                                "job_state": final_state, "rows": rows,
                                "resubmittable": service_produced_nothing})
        return reg.get(job_id)
    reg.mark_downloaded(job_id, results, job_state=final_state,
                        batch_stats=reconcile.batch_stats_dict(job))
    failed = reconcile.failed_request_count(job)
    summary["jobs"].append({"job_id": job_id, "job_name": job_name,
                            "job_state": final_state, "rows": rows,
                            "wave": wave, "kind": kind,
                            "enqueued_tokens": enqueued_tokens,
                            "failed_request_count_reported": failed,
                            "results": results})
    return reg.get(job_id)


def _resume_in_flight(cfg, reg, client, lang, *, summary, sleep=None,
                      now=None) -> list:
    """Finish every job this workspace left in flight for `lang`.

    THIS IS THE RECOVERY THE REST OF THIS MODULE ALREADY PROMISED. `--phase
    submit` is a foreground call that may run for a day or two; poll_until_terminal
    tells the operator the wait "should be resumed with --phase ingest later",
    and registry.py's docstring describes adopting a PLANNED job by listing the
    service's jobs and matching the uploaded input file. Neither was wired up:
    `--phase ingest` only looked at DOWNLOADED jobs, so a submitted-and-paid job
    stayed SUBMITTED for good and its cells stayed missing.

    Two windows, and the identifier that survives each one:

      SUBMITTED  the job name is on disk. Poll it to terminal and download --
                 the same drain the submit loop performs, never a new create.
      PLANNED    the JSONL is written and the fingerprint reserved. With an
                 uploaded_file the job may or may not exist on the service, and
                 the input file's resource name is the one identifier both sides
                 agree on, so it is ADOPTED by listing rather than created again
                 (batches.create is not idempotent: the same job submitted twice
                 is accepted twice, runs twice and is billed twice). Without an
                 uploaded_file nothing was ever sent, so the reservation is
                 released and the wave plans it again.
    """
    resumed = []
    for job in list(reg.in_flight()):
        if job.get("lang") != lang:
            continue
        job_id = job["job_id"]
        name = job.get("job_name")
        rows = int(job.get("rows") or 0)
        if not name:
            src = job.get("uploaded_file")
            if not src:
                reg.mark_failed(
                    job_id, "planned but never uploaded, so nothing was "
                            "submitted and nothing was billed",
                    resubmittable=True)
                resumed.append({"job_id": job_id, "action": "released",
                                "why": "no input file was ever uploaded"})
                print("  batch: %s was planned but never uploaded; releasing it "
                      "so this wave can plan it again" % job_id)
                continue
            found = find_job_by_input_file(client, src)
            if found is None:
                # The create may have landed and the answer been lost. NOT
                # resubmittable: this is the one state where guessing costs money.
                reg.mark_failed(
                    job_id, "planned and uploaded (%s) but no job on the service "
                            "references that input file" % src,
                    resubmittable=False)
                resumed.append({"job_id": job_id, "action": "unclaimable",
                                "uploaded_file": src})
                print("  batch: %s was uploaded as %s but no submitted job "
                      "references it. NOT resubmitting -- batches.create is not "
                      "idempotent. Check the console." % (job_id, src))
                continue
            name = found.name
            reg.mark_submitted(job_id, name, getattr(found, "state", None))
            print("  batch: adopted %s for %s (matched on input file %s); no "
                  "second create" % (name, job_id, src))
            resumed.append({"job_id": job_id, "action": "adopted",
                            "job_name": name})
        else:
            resumed.append({"job_id": job_id, "action": "resumed",
                            "job_name": name})
            print("  batch: %s was left %s as %s; resuming the drain instead of "
                  "resubmitting" % (job_id, job.get("state"), name))
        _drain_one(cfg, reg, client, job_id, name, rows=rows,
                   wave=int(job.get("wave") or 0), kind=job.get("kind"),
                   summary=summary, enqueued_tokens=job.get("enqueued_tokens"),
                   sleep=sleep, now=now)
    if resumed:
        summary["resumed"] = resumed
    return resumed


def _ingest_ready(s42, cfg, lang, reg, defs, exprs, tdir, *, usage, prov,
                  prov_expr, contexts, summary, stats, wave_out,
                  todo_by_key) -> list:
    """Ingest every DOWNLOADED job for this language; return what needs a retry.

    DOWNLOADED -> RECOVERED is what stops a second ingest from booking the same
    paid rows twice: the ledger dedupes on a per-call (ts, seq) that a second
    process cannot reproduce, so the guard has to be the job's state and not the
    row's identity.
    """
    pending: list = []
    for job in list(reg.in_state(registry.DOWNLOADED)):
        if job.get("lang") != lang:
            continue
        lines = registry.read_results(job["results_path"])
        # The join is by key; the order cross-check is recorded and printed but
        # never gates the write. A shuffled file with an intact bijection is a
        # correct file -- measured on the first real wave, where the service
        # returned four ~1000-row shards in a permuted order.
        order = {}
        outcomes = reconcile.reconcile(job["plan"], lines, report=order)
        if not order.get("in_input_order", True):
            print("  batch: %s %s"
                  % (job["job_id"], reconcile.order_note(order)))
        written, failures = _absorb(
            s42, cfg, lang, job, outcomes, defs, exprs, tdir, usage=usage,
            prov=prov, prov_expr=prov_expr, contexts=contexts, summary=summary)
        reg.mark_recovered(job["job_id"], {
            "written": written, "failed": len(failures),
            "keys_echoed": sum(1 for o in outcomes if o.key_echoed),
            "rows": len(outcomes), "order_cross_check": order})
        summary["written"]["definitions"] += written.get("definitions", 0)
        summary["written"]["expressions"] += written.get("expressions", 0)
        if job.get("kind") == "expression" and job.get("canary"):
            wave_out["canary"] = _record_canary(cfg, lang, job, outcomes, stats)
        for outcome, decision in failures:
            summary["failed_requests"].append(
                {"key": outcome.key, "kind": outcome.kind,
                 "why": outcome.why(), "retry": decision["retry"],
                 "decision": decision["why"], "job_id": job["job_id"]})
            if not decision["retry"]:
                continue
            rebuilt = _rebuild(job, outcome, decision, todo_by_key)
            if rebuilt is None:
                summary["failed_requests"][-1]["retry"] = False
                summary["failed_requests"][-1]["decision"] = \
                    "source_changed_since_submit"
                continue
            pending.append(rebuilt)
    return pending


def _absorb(s42, cfg, lang, job, outcomes, defs, exprs, tdir, *, usage, prov,
            prov_expr, contexts, summary) -> tuple:
    """Write the cells of every successful row; return (written, failures).

    The count lock is checked HERE, locally, on the client side: the schema
    already asked for minItems == maxItems == n, and a short array that got
    through anyway must not reach zip(rows, items) -- that shifts every gloss
    onto the wrong sense from the missing one onwards, which is catastrophic and
    invisible.
    """
    written = {"definitions": 0, "expressions": 0}
    failures = []
    ceiling = s42.MAX_OUTPUT_RETRY_CEILING
    for row, outcome in zip(job["plan"], outcomes):
        ledger = s42.normalize_usage(
            outcome.usage, model=job["model"],
            label="%s batch %s" % (outcome.kind, row.get("label") or row["key"]),
            kind=outcome.kind, mode="batch", cache_name=job.get("cache_name"),
            cache_prompt_sha256=job.get("cache_prompt_sha256"),
            prompt_id=cfg.prompt_id, finish_reason=outcome.finish_reason,
            n_expected=outcome.n_expected)
        ledger["attempt"] = int(job.get("wave") or 0) + 1
        ledger["max_output_tokens"] = row.get("cap")
        # None on the cached path BY CONSTRUCTION (systemInstruction XOR
        # cachedContent). That is what cache_prompt_sha256 above is for.
        ledger["prompt_sha256"] = (None if job.get("cache_name")
                                   else s42.prompt_sha256(
                                       s42.system_prompt(outcome.kind, lang)))
        ledger["job_name"] = job.get("job_name")
        ledger["row_key"] = outcome.key
        if outcome.error:
            ledger["error"] = "batch_row_error"
            ledger["error_text"] = json.dumps(outcome.error)[:400]
        usage.record(ledger)
        if outcome.ok:
            table, provenance, bucket = ((defs, prov, "definitions")
                                         if outcome.kind == "definition"
                                         else (exprs, prov_expr, "expressions"))
            cells = row["cells"]
            for cell, obj in zip(cells, outcome.items):
                table[cell["key"]] = {"lemma": obj.get("lemma"),
                                      "gloss": obj.get("gloss"),
                                      "src_sha": cell["src_sha"],
                                      "provenance": provenance}
                written[bucket] += 1
            continue
        decision = waves.retry_decision(outcome, cap=int(row.get("cap") or 0),
                                        ceiling=ceiling)
        if not outcome.error and not outcome.count_ok \
                and outcome.finish_reason == "STOP":
            # A count-lock violation, WITH the finishReason and the cap that
            # produced it: "the model dropped a sense" and "the cap truncated
            # the JSON" need opposite fixes and used to be the same log line.
            contexts[outcome.kind].note_violation(s42._count_lock_row(
                outcome.kind, row.get("label") or outcome.key,
                outcome.n_expected, outcome.got,
                s42.Completion(parsed={}, finish_reason=outcome.finish_reason,
                               usage=ledger),
                int(job.get("wave") or 0) + 1, row.get("cap")))
        failures.append((outcome, decision))
    # After EVERY job, not at the end of the wave: a job's rows are paid for and
    # a crash before the next one must not lose them.
    write_json(tdir / "definitions.json", defs)
    write_json(tdir / "expressions.json", exprs)
    return written, failures


def _rebuild(job, outcome, decision, todo_by_key):
    """A retry request for one failed row, or None if the source moved.

    The Danish text is NOT stored in the registry (it belongs in
    json/entries.json, not in a bookkeeping artifact), so a retry recomputes it
    from the live todo and checks the src_sha the submit recorded. A source that
    changed between submit and retry is NOT resent: the request would be a
    different request wearing the old one's key.
    """
    plan_row = None
    for row in job["plan"]:
        if row["key"] == outcome.key:
            plan_row = row
            break
    if plan_row is None:
        return None
    rows = []
    for cell in plan_row["cells"]:
        live = todo_by_key.get(cell["key"])
        if live is None or live.get("src_sha") != cell["src_sha"]:
            return None
        rows.append(live)
    return PlannedRequest(
        key=plan_row["key"], kind=plan_row["kind"],
        entry_id=plan_row["entry_id"], label=plan_row.get("label") or "",
        rows=rows, cap=int(decision["cap"] or plan_row.get("cap") or 0),
        cached_tokens=int(plan_row.get("cached_tokens") or 0),
        uncached_tokens=int(plan_row.get("uncached_tokens") or 0),
        correction=decision.get("correction") or "")


def _drop_exhausted(reg, group, summary) -> list:
    """Requests whose per-cell attempt counter has run out are not resubmitted.

    The bound is per CELL and it survives the process, because a cell that
    failed in three different jobs has had three attempts and "keep opening
    jobs until the wave is drained" has no other end.
    """
    keep = []
    for planned in group:
        if reg.attempts_for(planned.key) >= waves.MAX_RETRY_WAVES:
            summary.setdefault("exhausted", []).append(
                {"key": planned.key, "kind": planned.kind,
                 "attempts": reg.attempts_for(planned.key)})
            continue
        keep.append(planned)
    return keep


# --------------------------------------------------------------------------
# the cache, the canary, and the one interactive call left
# --------------------------------------------------------------------------

def cacheable_request_count(s42, todo, kind: str = "definition") -> int:
    """How many REQUESTS of `kind` this wave will place.

    Not how many cells: `todo` is one row per cell and the shipped corpus batches
    3.7 definition cells into one request (13,632 cells, 3,642 requests). The
    TTL is sized from this number and the variable that used to feed it was
    already called `requests`.
    """
    max_batch = (s42.MAX_DEFS_PER_BATCH if kind == "definition"
                 else s42.MAX_EXPR_PER_BATCH)
    return len(s42._group_by_entry(todo, kind, max_batch))


def _cache_for(cfg, lang, reg, client, stats, s42, *, todo, report,
               verify: bool):
    """The definition cache for this language: reused, extended or created.

    Returns None when caching is off, and also when this wave places NO
    definition request: a cache nobody references is billed for storage by the
    token-hour and it makes G-CACHE adjudicate a wave that had nothing to cache.
    An incremental wave with only expression cells left is exactly that -- and it
    is the state the documented recovery path leaves behind, so the run after an
    unrecovered definition request used to create a cache, reference it nowhere,
    and then fail the whole run on G-CACHE.

    `verify=False` reads the registry and places NO call: that is the ingest,
    which submits nothing. `verify=True` is every path that is about to submit,
    and it either asserts the cache's remaining life and extends it, or recreates
    it. It is called ONCE PER JOB, not once per language: with one job in flight
    at a time and a 50-hour poll deadline, the second job's submit can be a day
    after the first one's.
    """
    if not cfg.cache_enabled:
        return None
    key = "%s/definition" % lang
    record = reg.cache_record(key)
    if not verify:
        return (cache_lifecycle.CacheHandle.from_record(record)
                if record else None)
    requests = cacheable_request_count(s42, todo, "definition")
    if not requests:
        report.setdefault("cache", {}).setdefault("skipped", []).append(
            {"lang": lang, "why": ("this wave places no definition request, so "
                                   "a cache object would be storage nobody "
                                   "references")})
        return None
    plan = cache_lifecycle.cache_ttl_plan(
        requests, poll_deadline_s=JOB_WAIT_SECONDS, factor=cfg.cache_ttl_factor)
    ttl = plan["ttl_seconds"]
    if record is not None:
        handle = cache_lifecycle.CacheHandle.from_record(record)
        try:
            # caches.get, BEFORE EVERY SUBMIT. Never submitting against a cache
            # whose life nobody looked at is the whole rule: the rows resolve the
            # cache when they execute, and the failure is per row, hours later.
            left = cache_lifecycle.remaining_seconds(client, handle)
            report.setdefault("cache", {}).setdefault("verified", []).append(
                {"lang": lang, "name": handle.name, "seconds_left": left,
                 "ttl_wanted": ttl})
            if left is None or left < ttl:
                cache_lifecycle.extend(client, handle, ttl)
                report.setdefault("cache", {}).setdefault("extended", []).append(
                    {"lang": lang, "name": handle.name,
                     "seconds_left": left, "ttl": ttl})
            reg.remember_cache(key, handle.as_record())
            reg.remember_declared(lang, handle.declared_tokens)
            return handle
        except s42.CacheUnavailable as exc:
            # Expired. It cannot be updated, only recreated, and the recreate
            # changes the resource name -- which is why the JSONL is written per
            # job just before the submit.
            print("  batch: %s" % exc)
            reg.forget_cache(key)
    text = s42.system_prompt("definition", lang)
    handle = cache_lifecycle.create(
        client, model=cfg.gemini_model, system_text=text, lang=lang,
        kind="definition", ttl=ttl, stats=stats,
        display_name="ankidkdeck-def-%s" % lang)
    reg.remember_cache(key, handle.as_record())
    # PERSISTED per language, separately from the cache object, because the cache
    # object is deleted at the end of the wave and G-CACHE's denominator is read
    # after that -- and because --phase ingest is a different process from
    # --phase submit.
    reg.remember_declared(lang, handle.declared_tokens)
    report.setdefault("cache", {}).setdefault("created", []).append(
        {"lang": lang, "name": handle.name,
         "declared_tokens": handle.declared_tokens,
         "prompt_sha256": handle.prompt_sha256, "ttl": plan})
    print("  batch: cache %s for %s, %d declared tokens, ttl %ds (%s; %d "
          "definition request(s))"
          % (handle.name, lang, handle.declared_tokens, ttl,
             plan["decided_by"], requests))
    return handle


def _cache_before_submit(cfg, lang, reg, client, stats, s42, *, todo, report,
                         handle, kind, contexts):
    """Assert and extend the cache immediately before ONE job's submit.

    Patch plan 5.4 asks for `caches.get` before every submit and `caches.update`
    when the remaining life is short. It used to happen once per language per
    phase, which measured as a call sequence of exactly ['create', 'delete'] --
    zero gets, zero updates -- for an invocation that submitted two jobs. With
    one job in flight and a 50-hour poll deadline, "once per language" and "once
    per submit" are up to two days apart.

    Returns the handle to use, which may be a NEW object with a new resource name
    (an expired cache cannot be updated, only recreated). The definition context's
    cache_name is updated with it, because the fingerprint, the JSONL and the
    ledger row's cache identity all come from there.
    """
    if not cfg.cache_enabled or kind not in s42.CACHEABLE_KINDS:
        return handle
    fresh = _cache_for(cfg, lang, reg, client, stats, s42, todo=todo,
                       report=report, verify=True)
    for cached_kind in s42.CACHEABLE_KINDS:
        if cached_kind in contexts:
            contexts[cached_kind].cache_name = fresh.name if fresh else None
    return fresh


def _end_of_wave_cache(cfg, lang, reg, client, report) -> None:
    """Delete the cache once the language is drained. Storage is billed hourly.

    NOT while a job of this language is still PLANNED or SUBMITTED. Batch
    resolves the cache AT EXECUTION TIME: the probes measured that a cache
    deleted immediately after a successful submit made all 21 rows of that job
    fail with gRPC code 7. Deleting here to save $0.03 of storage while a job is
    in flight turns "stranded but recoverable" into "certain to be void", and the
    resume path above is what makes keeping the cache worth something.
    """
    record = reg.cache_record("%s/definition" % lang)
    if not record:
        return
    in_flight = [j["job_id"] for j in reg.in_flight()
                 if j.get("lang") == lang]
    if in_flight:
        report.setdefault("cache", {}).setdefault("kept", []).append(
            {"lang": lang, "name": record.get("name"),
             "jobs_in_flight": sorted(in_flight),
             "why": ("batch resolves the cache when each row EXECUTES, so "
                     "deleting it now would fail every remaining row of these "
                     "job(s) with gRPC code 7")})
        print("  batch: keeping cache %s -- %d job(s) of %s are still in flight "
              "(%s). Deleting it now would void their rows."
              % (record.get("name"), len(in_flight), lang,
                 ", ".join(sorted(in_flight))))
        return
    handle = cache_lifecycle.CacheHandle.from_record(record)
    hours = max(0.0, float(handle.ttl_seconds) / 3600.0)
    cost = cache_lifecycle.storage_cost(handle, hours, cfg.mode)
    result = cache_lifecycle.delete(client, handle)
    reg.forget_cache("%s/definition" % lang)
    report.setdefault("cache", {}).setdefault("deleted", []).append(
        dict(result, lang=lang, storage_cost=cost))


def _with_canary(batches: list, stats, lang, summary) -> list:
    """Split a small canary job off the FRONT of the expression wave.

    The bill books an unmeasured thinking prior on every expression request
    because the expression prompt has never been probed at any level. One small
    job measures it, and the measurement lands on disk where the bill, doctor
    and G-BUDGET all read it. Skipped when the artifact already carries a
    measurement for this prompt family.
    """
    if not batches:
        return batches
    if _measured_expression_family(stats, lang) is not None:
        summary["canary"] = "skipped: the artifact already measures this family"
        return batches
    first = batches[0][0]
    if len(first) <= CANARY_REQUESTS:
        summary["canary"] = ("the first job is already small (%d requests) and "
                             "is the canary" % len(first))
        return [(first, True)] + batches[1:]
    summary["canary"] = "the first %d expression requests" % CANARY_REQUESTS
    return ([(first[:CANARY_REQUESTS], True), (first[CANARY_REQUESTS:], False)]
            + batches[1:])


def _family_key(lang) -> str:
    """The key the canary's measurement is filed under.

    Delegated to s42.thinking_family_key rather than spelled out here: the BILL
    reads this key, and the bill path may not import ankidkdeck.batch (the dry
    path asserts that neither `google*` nor `ankidkdeck.batch*` is in
    sys.modules). One definition, and it lives on the side that cannot import
    this one.
    """
    from ..stages import s42_translate as s42         # noqa: PLC0415
    return s42.thinking_family_key("expression", lang)


def _measured_expression_family(stats, lang):
    """The measurement the bill would READ for this family, or None.

    Matched the way the bill matches it -- on (level, kind), not on the exact
    character count in the key -- so "the artifact already measures this family"
    means the same thing here and in unmeasured_thinking_prior. Matching the
    literal key would have skipped the canary only when the prompt was
    byte-identical to the probed one, while the bill was already reading a
    different entry.
    """
    from ..stages import s42_translate as s42         # noqa: PLC0415
    return s42._thinking_families(stats, "LOW").get("expression")


def _record_canary(cfg, lang, job, outcomes, stats) -> dict:
    """Write what the canary job measured: a report always, the artifact if clean.

    The artifact is the file that authorises spending, so it is only written
    when the measurement is unambiguous (every row STOP, no errors) and the
    value carries its own provenance. The previous value is recorded next to it
    rather than overwritten silently.
    """
    # DERIVED, never thoughtsTokenCount: protobuf omits the field when thinking
    # is zero, so "absent" and "zero" are indistinguishable at the one place
    # where the difference is the whole question.
    thinking = [_derived(o.usage) for o in outcomes]
    clean = bool(outcomes) and all(o.finish_reason == "STOP" and not o.error
                                  for o in outcomes)
    measurement = {
        "n_observations": len(thinking),
        "mean": (sum(thinking) / float(len(thinking))) if thinking else None,
        "max": max(thinking) if thinking else None,
        "p95": _p95(thinking),
        "distinct_values": sorted(set(thinking)),
        "measured_by": "batch expression canary job %s" % job.get("job_id"),
        "measured_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "job_name": job.get("job_name"),
        "language": lang,
    }
    out = {"family": _family_key(lang), "clean": clean,
           "measurement": measurement, "written_to_artifact": False}
    report_path = cfg.report_dir / ("expression_thinking_canary_%s.json" % lang)
    disk = read_json(cfg.probe_stats_path, default=None)
    if not clean:
        out["why"] = ("the canary job had errors or a non-STOP finish, so the "
                      "measurement is not written to the artifact that "
                      "authorises spending")
    elif not isinstance(disk, dict):
        out["why"] = "there is no measured-constants artifact to write into"
    else:
        node = disk.setdefault("thinking", {}) \
            .setdefault("THINKING_AT_LOW_BY_PROMPT_FAMILY", {})
        out["superseded"] = node.get(_family_key(lang))
        node[_family_key(lang)] = measurement
        disk["thinking"].setdefault(
            "THINKING_AT_LOW_BY_PROMPT_FAMILY_canaries", []).append(
            {"family": _family_key(lang), "at": measurement["measured_at"],
             "job_name": job.get("job_name"), "max": measurement["max"]})
        write_json(cfg.probe_stats_path, disk)
        out["written_to_artifact"] = True
    # The report is written LAST so its verdict is the one that happened: an
    # earlier write said written_to_artifact = false whatever followed it.
    write_json(report_path, out)
    return out


def _derived(usage) -> int:
    from ..stages.s42_translate import derived_thinking  # noqa: PLC0415
    return derived_thinking(usage or {})


def _p95(values) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                int(math.ceil(0.95 * len(ordered))) - 1)
    return float(ordered[max(0, index)])


def _pos_on_the_interactive_surface(s42, cfg, lang, st, pool, fit, usage,
                                    pos_wanted) -> int:
    """The POS table, on standard, in its own context.

    One request per language, and only for a language the checked-in registry
    does not cover -- which is what makes adding a language a no-registry-edit
    operation. Porting it to batch would be a job, a fingerprint and a
    reconciliation for one call; and once the 12 missing strings are in the
    registry it is dead code. Its ledger row says mode=standard because that is
    what it is: filing it as batch would price it at half.
    """
    if not st["pos_todo"]:
        return 0
    ctx = s42.CallContext(cfg=cfg, pool=pool, fit=fit, lang=lang, usage=usage,
                          prompt_id=cfg.prompt_id, mode="standard")
    mapping = s42._translate_pos(ctx, cfg.expressions_model, list(pos_wanted))
    merged = dict(st["pos"])
    for key in st["pos_todo"]:
        merged[key] = mapping[key]
    write_json(st["dir"] / "pos.json", merged)
    st["pos"] = merged
    return len(st["pos_todo"])
