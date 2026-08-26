"""The job registry and the wave fingerprint (patch plan 5.1).

`batches.create` IS NOT IDEMPOTENT. Measured: the same job submitted back to
back was accepted twice, both jobs ran, both hit the cache, and both were
billed. There is no request id, no client token and no dedupe on the service
side, so the only thing standing between a retried submit and a double charge is
a record on our own disk -- written BEFORE the call, not after it.

Hence the state machine, ported from the GPCR reference implementation with the
two dead parts skipped (its key naming collides on this corpus, and
PARTIALLY_SUCCEEDED does not exist on the Developer API):

    PLANNED -> SUBMITTED -> DOWNLOADED -> RECOVERED
                                       -> FAILED

PLANNED is the crash window and the reason the machine has five states rather
than four: the JSONL is on disk and the fingerprint is reserved, but no job name
exists yet. A crash there must NOT be resolved by submitting again -- the first
call may have been accepted and the answer lost. It is resolved by listing the
service's jobs and matching on the uploaded input file's resource name, which is
the one identifier both sides agree on.

DOWNLOADED -> RECOVERED is the other crash window: results on disk that no
translation row has absorbed yet. Those are re-ingested on the next invocation
(the results are a file, ingesting them twice writes the same rows), which is
what makes "download IMMEDIATELY" safe advice -- the download is cheap and the
retention policy is self-contradictory (6 weeks in one place, 48h in another).
"""

import datetime
import json

from ..util import FatalError, read_json, sha256_str, write_json

PLANNED = "PLANNED"
SUBMITTED = "SUBMITTED"
DOWNLOADED = "DOWNLOADED"
RECOVERED = "RECOVERED"
FAILED = "FAILED"
STATES = (PLANNED, SUBMITTED, DOWNLOADED, RECOVERED, FAILED)

# The service-side job states that are terminal. Both spellings exist in the
# documentation (JOB_STATE_* in the batch guide, BATCH_STATE_* in the API
# reference) and neither list contains PARTIALLY_SUCCEEDED, which is a Vertex
# enum: the GPCR reference implementation's `succeeded_states` includes it and
# that part must not be ported. The prefix is stripped before comparison rather
# than either spelling being hard-coded.
TERMINAL_JOB_STATES = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")
STATE_PREFIXES = ("JOB_STATE_", "BATCH_STATE_")


def job_state_name(state) -> str:
    """A service job state as a bare string: SUCCEEDED, not JOB_STATE_SUCCEEDED.

    Reads the SDK enum's own name when there is one, so a new state the SDK
    knows about and this module does not still arrives as text instead of as a
    repr.
    """
    name = getattr(state, "name", None) or str(state)
    for prefix in STATE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_terminal(state) -> bool:
    return job_state_name(state) in TERMINAL_JOB_STATES


def wave_fingerprint(*, model: str, prompt_id: str, lang: str, keys,
                     cache_name=None, wave: int = 0) -> str:
    """sha256(model | prompt_id | lang | sorted(keys) | cache_name | wave).

    `prompt_id` must be the EFFECTIVE prompt id (prompts.effective_prompt_id),
    not cfg.prompt_id: the pack version is part of the prompt text, so two waves
    that differ only in the pack are different waves, and under the bare id they
    would share a fingerprint and the second one would be refused as a
    duplicate.

    The cache name is in the identity because a recreated cache (the only way
    back from an expiry -- update refuses with 403) has a new resource name, and
    the rows that reference it are genuinely different rows.

    `wave` IS A DELIBERATE ADDITION to the five fields the patch plan lists, and
    without it the retry loop cannot exist: a retry wave resubmits the SAME keys
    with a different payload (a correction appended to the user message, or a
    raised output cap), so under the five-field material it is indistinguishable
    from an accidental repeat and is refused. The failure the fingerprint exists
    for is still caught, because an accidental repeat happens at the same wave
    index: a crashed `--phase submit` that is run again, or the same invocation
    twice.
    """
    material = "|".join([
        str(model), str(prompt_id), str(lang),
        ",".join(sorted(str(k) for k in keys)),
        str(cache_name or ""), "wave=%d" % int(wave),
    ])
    return sha256_str(material)


class JobRegistry:
    """work/batch/jobs.json: every job this workspace ever submitted.

    Append-mostly and rewritten atomically. Small (one record per job, a few
    hundred bytes plus the plan) and read on every invocation, so a crash leaves
    a readable file rather than a half-written one.
    """

    FILE = "jobs.json"

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cfg.work_dir / "batch"
        self.path = self.dir / self.FILE
        self.data = read_json(self.path, default={"jobs": {}, "schema": 1})
        self.data.setdefault("jobs", {})

    # ---- persistence ----------------------------------------------------

    def save(self) -> None:
        self.data["written_at"] = _now()
        write_json(self.path, self.data)

    def jobs(self) -> dict:
        return self.data["jobs"]

    def get(self, job_id: str) -> dict:
        job = self.jobs().get(job_id)
        if job is None:
            raise FatalError("no job %r in %s" % (job_id, self.path))
        return job

    # ---- the state machine ----------------------------------------------

    def plan(self, job_id: str, *, fingerprint: str, lang: str, kind: str,
             model: str, prompt_id: str, cache_name, declared_cache_tokens,
             cache_prompt_sha256, jsonl_path: str, plan: list,
             enqueued_tokens: int, wave: int, retry_of=None,
             canary: bool = False) -> dict:
        """Reserve the fingerprint and record the intent, BEFORE any API call."""
        if job_id in self.jobs():
            raise FatalError("job %r is already in the registry" % job_id)
        clash = self.find_by_fingerprint(fingerprint)
        if clash:
            raise FatalError(
                "wave fingerprint %s was already submitted as job %r (state "
                "%s). batches.create is NOT idempotent -- the same job "
                "submitted twice is accepted twice, runs twice and is billed "
                "twice -- so a repeat submission is refused here. Ingest the "
                "existing job, or change the wave (its keys, its cache or its "
                "prompt) if it really is a different one."
                % (fingerprint[:16], clash, self.jobs()[clash].get("state")))
        row = {
            "job_id": job_id, "state": PLANNED, "fingerprint": fingerprint,
            "lang": lang, "kind": kind, "model": model,
            "effective_prompt_id": prompt_id, "cache_name": cache_name,
            "declared_cache_tokens": declared_cache_tokens,
            "cache_prompt_sha256": cache_prompt_sha256,
            "jsonl_path": jsonl_path, "rows": len(plan), "plan": plan,
            "enqueued_tokens": enqueued_tokens, "wave": wave,
            "canary": bool(canary),
            "retry_of": retry_of, "uploaded_file": None, "job_name": None,
            "job_state": None, "results_path": None,
            "history": [{"state": PLANNED, "at": _now()}],
        }
        self.jobs()[job_id] = row
        self.save()
        return row

    def _advance(self, job_id: str, state: str, **fields) -> dict:
        if state not in STATES:
            raise FatalError("unknown job registry state %r" % state)
        job = self.get(job_id)
        job.update(fields)
        job["state"] = state
        job.setdefault("history", []).append({"state": state, "at": _now()})
        self.save()
        return job

    def mark_uploaded(self, job_id: str, uploaded_file: str) -> dict:
        """The input file exists on the service. This is what a crashed submit
        is recovered BY: batches.list matched on this resource name."""
        return self._advance(job_id, PLANNED, uploaded_file=uploaded_file)

    def mark_submitted(self, job_id: str, job_name: str, job_state=None) -> dict:
        return self._advance(job_id, SUBMITTED, job_name=job_name,
                             job_state=job_state_name(job_state)
                             if job_state is not None else None,
                             submitted_at=_now())

    def mark_job_state(self, job_id: str, job_state) -> dict:
        job = self.get(job_id)
        job["job_state"] = job_state_name(job_state)
        job["job_state_at"] = _now()
        self.save()
        return job

    def mark_downloaded(self, job_id: str, results_path: str,
                        job_state=None, batch_stats=None) -> dict:
        return self._advance(job_id, DOWNLOADED, results_path=results_path,
                             job_state=job_state_name(job_state)
                             if job_state is not None else None,
                             batch_stats=batch_stats)

    def mark_recovered(self, job_id: str, outcome: dict) -> dict:
        return self._advance(job_id, RECOVERED, outcome=outcome)

    def mark_failed(self, job_id: str, why: str, job_state=None,
                    resubmittable: bool = False) -> dict:
        """Terminal, and RECORDED rather than raised.

        A FatalError in the middle of a wave throws away a job that has already
        been paid for. The wave records the failure, keeps its place, and the
        caller decides at the end of the wave whether the run as a whole is a
        failure.

        `resubmittable` IS THE WHOLE DECISION and the caller has to make it
        explicitly, because two very different events used to arrive here as one:

          resubmittable=True   the SERVICE says this job produced nothing --
                               FAILED, CANCELLED or EXPIRED, or a create that
                               was refused outright. Nothing was billed, so the
                               corrected wave may be submitted again.
          resubmittable=False  WE did not get the results of a job that may well
                               have run: a create whose outcome is unknown, or a
                               SUCCEEDED job whose result file could not be
                               read. batches.create is not idempotent, so
                               resubmitting is a second charge. The fingerprint
                               stays reserved and a human looks at it.

        Defaults to False: the safe answer for a caller that has not thought
        about it is "do not spend this again".
        """
        return self._advance(job_id, FAILED, failure=why,
                             resubmittable=bool(resubmittable),
                             job_state=job_state_name(job_state)
                             if job_state is not None else None)

    # ---- queries --------------------------------------------------------

    def is_resubmittable(self, job_id: str) -> bool:
        """Whether a FAILED job's wave may be submitted again. See mark_failed."""
        job = self.jobs().get(job_id) or {}
        return job.get("state") == FAILED and bool(job.get("resubmittable"))

    def find_by_fingerprint(self, fingerprint: str):
        """The job that already owns this fingerprint, or None.

        A FAILED job releases its fingerprint ONLY when it was recorded as
        resubmittable. A FAILED job whose results we merely could not fetch keeps
        it: that is the one case where letting the wave through would be a second
        charge for the same rows.
        """
        for job_id, job in self.jobs().items():
            if job.get("fingerprint") != fingerprint:
                continue
            if job.get("state") == FAILED and job.get("resubmittable"):
                continue
            return job_id
        return None

    def next_job_id(self, base: str) -> str:
        """`base`, or `base` plus an attempt suffix, or a refusal.

        The job id is deterministic (lang-kind-wave-index), so a FAILED job used
        to wedge its (lang, kind, wave, index) slot for good: find_by_fingerprint
        let the wave through and then plan() refused on the id, with an error
        message that named the collision and no way out of it. A resubmittable
        failure now yields the next free suffix; a non-resubmittable one refuses
        HERE, where the message can say what to do.
        """
        if base not in self.jobs():
            return base
        if not self.is_resubmittable(base):
            job = self.jobs()[base]
            raise FatalError(
                "job %r is already in the registry (state %s%s). It is not "
                "marked resubmittable, which means this workspace committed "
                "money for it and did not get the results back -- "
                "batches.create is NOT idempotent, so opening it again may be a "
                "second charge for the same rows. Look the job up in the console "
                "(job_name %r, input file %r); if it truly produced nothing, set "
                '"resubmittable": true on its record in %s and re-run.'
                % (base, job.get("state"),
                   ": %s" % job.get("failure") if job.get("failure") else "",
                   job.get("job_name"), job.get("uploaded_file"), self.path))
        for attempt in range(2, 100):
            candidate = "%s-a%d" % (base, attempt)
            if candidate not in self.jobs():
                return candidate
        raise FatalError(
            "job %r has been reopened 98 times; that is not a retry, it is a "
            "loop. Look at %s before submitting anything else."
            % (base, self.path))

    def in_state(self, *states) -> list:
        return [j for j in self.jobs().values() if j.get("state") in states]

    def unfinished(self) -> list:
        """Jobs that still owe this workspace something, oldest first."""
        return [j for j in self.jobs().values()
                if j.get("state") in (PLANNED, SUBMITTED, DOWNLOADED)]

    def in_flight(self) -> list:
        return self.in_state(PLANNED, SUBMITTED)

    def attempts(self) -> dict:
        """{cell key: attempts so far} across every job in this registry.

        Persisted per KEY rather than per job, because the bound patch plan 5.6
        asks for is per key: a cell that has failed three times must stop being
        resubmitted even though each attempt lived in a different job.
        """
        return self.data.setdefault("attempts", {})

    def bump_attempts(self, keys) -> dict:
        counts = self.attempts()
        for key in keys:
            counts[key] = int(counts.get(key) or 0) + 1
        self.save()
        return counts

    def attempts_for(self, key: str) -> int:
        return int(self.attempts().get(key) or 0)

    # ---- the cache objects this workspace owns --------------------------
    #
    # A wave is TWO invocations (submit, then ingest hours later), so the cache
    # the submit created has to be findable by the ingest: its resource name, its
    # declared token count (G-CACHE's denominator) and the sha of the prompt
    # inside it (G-PROMPT's only handle on a cached row) all outlive the process
    # that made them.

    def caches(self) -> dict:
        return self.data.setdefault("caches", {})

    def remember_cache(self, key: str, record: dict) -> dict:
        self.caches()[key] = record
        self.save()
        return record

    def cache_record(self, key: str):
        return self.caches().get(key)

    def forget_cache(self, key: str) -> None:
        self.caches().pop(key, None)
        self.save()

    def declared(self) -> dict:
        return self.data.setdefault("declared_cache_tokens", {})

    def remember_declared(self, lang: str, tokens, *, wave_id: str = "") -> dict:
        """Record what THIS language's cache declared, for G-CACHE's denominator.

        Separate from the cache record on purpose: the cache object is deleted at
        the end of the wave (storage is billed by the hour) and the gates run
        AFTER that, across every language, so reading the denominator off the live
        cache record meant reading it off an empty dict and falling back to
        min() over every job the workspace ever ran. A denominator that is too
        small does not fail a healthy wave -- it PASSES a broken one, because the
        share is sum(cached)/(declared x requests) and shrinking the divisor
        raises the share. Measured: at declared=5 the original failure mode (1
        hit, 99 full-price fallbacks) scored 2.27 and passed.

        The record is per LANGUAGE, overwritten by each wave, and it survives
        both forget_cache and the process boundary between --phase submit and
        --phase ingest.
        """
        row = {"declared_tokens": int(tokens) if tokens else None,
               "at": _now(), "wave_id": wave_id}
        self.declared()[lang] = row
        self.save()
        return row

    def declared_for(self, lang: str):
        """The declared cache size of this language's most recent wave, or None."""
        row = self.declared().get(lang) or {}
        value = row.get("declared_tokens")
        return int(value) if value else None

    def cache_prompt_shas(self) -> dict:
        """{cache resource name: sha of the prompt inside it}.

        What post_wave_gates takes as `cache_prompt_shas`. On the cached path a
        usage row's own prompt_sha256 is None by construction (systemInstruction
        XOR cachedContent), so without this map G-PROMPT has nothing to check
        and used to report a green verdict on a wave in which it checked
        nothing.
        """
        out = {}
        for record in self.caches().values():
            if record.get("name") and record.get("prompt_sha256"):
                out[record["name"]] = record["prompt_sha256"]
        for job in self.jobs().values():
            if job.get("cache_name") and job.get("cache_prompt_sha256"):
                out.setdefault(job["cache_name"], job["cache_prompt_sha256"])
        return out

    def summary(self) -> dict:
        by_state: dict = {}
        for job in self.jobs().values():
            state = job.get("state") or "?"
            by_state[state] = by_state.get(state, 0) + 1
        return {"jobs": len(self.jobs()), "by_state": by_state,
                "path": str(self.path)}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc) \
        .isoformat(timespec="seconds")


def read_results(path) -> list:
    """The downloaded result file, one JSON object per line.

    Blank lines are skipped; a line that does not parse is FATAL rather than
    skipped, because the reconciliation is positional and a dropped line shifts
    every row after it onto the wrong request.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise FatalError(
                    "%s line %d is not JSON (%s). The reconciliation is "
                    "positional, so a line that cannot be read cannot be "
                    "skipped: every row after it would be attributed to the "
                    "wrong request." % (path, i, exc)) from exc
    return out
