"""Wave splitting and the retry bound (patch plan 5.5, 5.6).

THE ENQUEUED LIMIT DOES NOT COUNT CACHED TOKENS. Measured: a 21-row job whose
nominal enqueued total was 3,160,563 tokens (150,503 cached tokens x 21) was
accepted against a 3,000,000-token limit. So the splitter counts the UNCACHED
payload only, and the definition wave is 2 jobs per language rather than 6.

That measurement is read off the probe artifact, never assumed: without
`wave3.W3_5_ENQUEUED.ENQUEUED_COUNTS_CACHED` on disk the splitter counts every
token including the cached prefix. Absence of the measurement makes it
CONSERVATIVE (more jobs than necessary, no rejected submit) rather than
optimistic.

The retry bound exists because the wave loop as originally specified ("keep
opening jobs until the wave is drained") had none. MAX_RETRY_WAVES is that
bound, and the per-KEY attempt counter in the job registry is what makes it a
bound on a cell rather than on a job: a cell that failed in three different jobs
has had three attempts.
"""

from ..util import FatalError

# Tier 1's enqueued-token ceiling, summed across every ACTIVE job (not per
# job). Read off the rate-limit documentation on 2026-08-13, not measured, and
# recorded here with its date for the same reason prices.py records the pricing
# page's date: a number copied off a page is only as good as the day it was
# copied.
ENQUEUED_TOKEN_LIMIT = 3_000_000
ENQUEUED_LIMIT_READ_AT = "2026-08-13"
ENQUEUED_LIMIT_TIER = "paid Tier 1"

# The target per job: 80% of the ceiling. A design choice, not a measurement --
# the token estimate is a fit with an R2 of 0.93, and a wave that is refused at
# submit costs a round trip and a human.
JOB_TOKEN_TARGET_SHARE = 0.8

# One job in flight at a time (patch plan 5.5). The limit is per active job SET,
# so two jobs in flight would have to share the ceiling, and the failure mode of
# getting that wrong is a rejected submit in the middle of a drain.
MAX_JOBS_IN_FLIGHT = 1

# The bound on the retry loop.
MAX_RETRY_WAVES = 3

# The DOCUMENTED batch window, which is what the cache TTL has to cover. The
# target is 24 hours and the hard expiry is 48; a job that has not finished by
# then is EXPIRED. These are documentation numbers, recorded with that label,
# and they are deliberately NOT an extrapolation of probe throughput -- see
# drain_window_seconds.
DOCUMENTED_JOB_TARGET_SECONDS = 24 * 3600
DOCUMENTED_JOB_HARD_EXPIRY_SECONDS = 48 * 3600

# Added on top of the drain window. The cache is created BEFORE the JSONL is
# written, the file is uploaded and the job is created, and it has to still be
# alive when the LAST row of that job executes. Two hours covers the submit
# overhead at a cost of 1,135 tokens x 2h x $0.50/M = $0.001 per language.
TTL_MARGIN_SECONDS = 2 * 3600

# Error codes seen on a batch result row, from the probe wave. Bare gRPC status
# codes with no detail: 7 is what a dead cache reference produces on every row
# of a job, 3 is a request the service refused.
GRPC_PERMISSION_DENIED = 7
GRPC_INVALID_ARGUMENT = 3


def job_token_target(limit: int = ENQUEUED_TOKEN_LIMIT,
                     share: float = JOB_TOKEN_TARGET_SHARE) -> int:
    return int(limit * share)


def enqueued_counts_cached(stats: dict | None) -> tuple:
    """(counts_cached, why). Reads the artifact; missing means conservative.

    The measured answer is False -- cached tokens do not count towards the
    enqueued limit -- and that is what turns the definition wave from 6 jobs per
    language into 2. It is worth exactly as much as the artifact that carries
    it, so when the artifact does not carry it the splitter counts everything.
    """
    node = ((stats or {}).get("wave3") or {}).get("W3_5_ENQUEUED") or {}
    value = node.get("ENQUEUED_COUNTS_CACHED")
    if value is None:
        return True, ("wave3.W3_5_ENQUEUED.ENQUEUED_COUNTS_CACHED is not on "
                      "disk, so the cached prefix is counted too. This makes "
                      "more jobs than necessary; the alternative is a submit "
                      "the service refuses.")
    if value:
        return True, ("wave3.W3_5_ENQUEUED.ENQUEUED_COUNTS_CACHED is true on "
                      "disk: the cached prefix counts towards the enqueued "
                      "limit.")
    return False, ("wave3.W3_5_ENQUEUED.ENQUEUED_COUNTS_CACHED = false "
                   "(measured: 3,160,563 nominal enqueued tokens accepted "
                   "against a 3,000,000 limit), so only uncached payload is "
                   "counted.")


def split_into_jobs(requests: list, *, target_tokens: int,
                    counts_cached: bool = False) -> list:
    """[[request, ...]] -- consecutive requests packed up to target_tokens.

    Requests belonging to one ENTRY are never split across jobs (the GPCR
    reference implementation's rule): an entry's cells then either all come back
    in one job or all retry together, which is what keeps a partially failed
    entry from being half-written under two different provenances.

    A single entry that exceeds the target on its own gets its own job and says
    so, rather than being silently split: at the measured sizes it cannot happen
    (the largest entry is 71 senses = 4 requests) and if it ever does, the
    honest answer is a job that may be refused, not a rule quietly broken.
    """
    target = int(target_tokens)
    if target <= 0:
        raise FatalError("job token target must be positive, got %r"
                         % (target_tokens,))
    jobs: list = []
    current: list = []
    current_tokens = 0
    for group in _by_entry(requests):
        weight = sum(_weight(r, counts_cached) for r in group)
        if current and current_tokens + weight > target:
            jobs.append(current)
            current, current_tokens = [], 0
        current.extend(group)
        current_tokens += weight
    if current:
        jobs.append(current)
    return jobs


def _by_entry(requests: list) -> list:
    """Consecutive runs of requests that share an entry id, order preserved."""
    out: list = []
    for req in requests:
        eid = req.get("entry_id")
        if out and out[-1][0].get("entry_id") == eid:
            out[-1].append(req)
        else:
            out.append([req])
    return out


def _weight(request: dict, counts_cached: bool) -> int:
    if counts_cached:
        return int(request.get("cached_tokens") or 0) \
            + int(request.get("uncached_tokens") or 0)
    return int(request.get("uncached_tokens") or 0)


def wall_clock_estimate_seconds(total_requests: int, *, per_request: float = 6.0,
                                floor: float = 3600.0) -> float:
    """A throughput EXTRAPOLATION for one wave. Not the drain window.

    The 32-row probe job finished in about 3 minutes (~6s/row). Batch is a queue
    service, so where a 3,642-request job sits in that queue is not 3,642 x 6s,
    and this number must never be the only term in a cache TTL: see
    drain_window_seconds, which is what the TTL is actually measured against.

    Kept because it is the honest upper term for a wave so large that even the
    documented window is optimistic -- `total_requests` is a count of REQUESTS,
    never of cells.
    """
    return max(float(floor), float(total_requests) * float(per_request))


def drain_window_seconds(total_requests: int, *, poll_deadline_s: float,
                         per_request: float = 6.0) -> float:
    """How long the cache must stay alive for ONE job of `total_requests`.

    `total_requests` is a count of REQUESTS. Counting cells here (there are 3.7
    cells per definition request on the shipped corpus) produced a TTL that was
    long enough for the wrong reason, and the variable that fed it was already
    named `requests`: the next person to make the name true would have shortened
    the TTL by 3.7x.

    The window is the DOCUMENTED one, not a throughput extrapolation:

      * the transport's own poll deadline (`poll_deadline_s`) is how long this
        program is willing to wait for one job. A cache that dies before the
        deadline turns a job that was still running into a job whose every
        remaining row fails with gRPC code 7 (measured, probe W3-4: a cache
        deleted after a successful submit failed all 21 rows). The TTL has to
        cover the wait this program actually performs.
      * the documented hard expiry (48h) is the floor under that, so shortening
        the poll deadline cannot silently shorten the TTL below the window the
        service itself documents.
      * the throughput extrapolation is kept as an upper term for a wave large
        enough to exceed both.

    The artifact's own TTL_POLICY is the criterion: "the cache must be alive
    when each ROW EXECUTES, so TTL must cover the whole drain window".
    """
    return max(float(poll_deadline_s),
               float(DOCUMENTED_JOB_HARD_EXPIRY_SECONDS),
               float(total_requests) * float(per_request))


def retry_decision(outcome, *, cap: int, ceiling: int) -> dict:
    """Whether one failed row may be retried, and as WHAT.

    Four different failures reach this function and three of them need a
    different request rather than the same one again:

      count lock      the model dropped an item. Retried with a CORRECTION in
                      the user message -- never prepended to the system prompt,
                      which is the cached prefix, because moving it would
                      forfeit the discount on precisely the requests being
                      redone.
      MAX_TOKENS      a cap error. Retried ONCE at double the cap. The
                      interactive path raises the budget inside _generate; the
                      batch path never goes through _generate, so this is the
                      only raise on this path and the two do not stack.
      code 7          the cache reference is dead (403 / PERMISSION_DENIED,
                      billed $0). Recoverable: recreate the cache and resubmit.
      code 3          the service refused the request. NOT retried unchanged --
                      that is the 400 class, and 25 paid attempts to discover
                      one is the failure mode the error classifier exists for.
    """
    if outcome.error:
        code = outcome.error.get("code")
        if code == GRPC_PERMISSION_DENIED:
            return {"retry": True, "why": "cache_missing", "cap": cap,
                    "correction": "",
                    "note": ("the cache reference was dead when the row "
                             "executed; the cache is recreated before the "
                             "retry wave")}
        if code == GRPC_INVALID_ARGUMENT:
            return {"retry": False, "why": "invalid_argument", "cap": cap,
                    "correction": "",
                    "note": ("the service refused the request itself, so "
                             "resending it unchanged cannot help")}
        return {"retry": True, "why": "transport_error", "cap": cap,
                "correction": "", "note": ""}
    if outcome.finish_reason == "MAX_TOKENS":
        if cap >= ceiling:
            return {"retry": False, "why": "max_tokens_at_ceiling", "cap": cap,
                    "correction": "",
                    "note": ("already at the %d-token retry ceiling; raising "
                             "further is a configuration decision, not a "
                             "retry" % ceiling)}
        return {"retry": True, "why": "max_tokens", "cap": min(ceiling, cap * 2),
                "correction": "",
                "note": "the response was truncated; the budget is raised once"}
    if outcome.parse_error:
        return {"retry": True, "why": "unparseable_body", "cap": cap,
                "correction": "", "note": ""}
    if outcome.finish_reason and outcome.finish_reason != "STOP":
        return {"retry": False, "why": "finish_reason_%s" % outcome.finish_reason,
                "cap": cap, "correction": "",
                "note": ("not STOP and not a truncation: the same request "
                         "produces the same verdict")}
    if not outcome.count_ok:
        return {"retry": True, "why": "count_lock", "cap": cap,
                "correction": ("the previous answer contained %s objects; "
                               "return exactly %d, one per numbered input, in "
                               "the same order."
                               % (outcome.got, outcome.n_expected)),
                "note": ""}
    return {"retry": False, "why": "ok", "cap": cap, "correction": "", "note": ""}


def within_retry_bound(wave: int) -> bool:
    return int(wave) <= MAX_RETRY_WAVES


def enqueued_note(counts_cached: bool, why: str, target: int) -> dict:  # noqa: D401
    """What the report says about how the wave was split."""
    return {"enqueued_token_limit": ENQUEUED_TOKEN_LIMIT,
            "enqueued_limit_read_at": ENQUEUED_LIMIT_READ_AT,
            "enqueued_limit_tier": ENQUEUED_LIMIT_TIER,
            "job_token_target": target,
            "counted_cached_tokens": bool(counts_cached),
            "counting_basis": why,
            "max_jobs_in_flight": MAX_JOBS_IN_FLIGHT,
            "max_retry_waves": MAX_RETRY_WAVES}
