"""The explicit-cache lifecycle (patch plan 5.4).

The explicit cache is the ONLY discount path. There is no implicit caching
inside batch (measured: 8 batch rows with byte-identical 1,135-token inline
system prompts all came back with cachedContentTokenCount = 0) and the
interactive implicit fallback measured 0/0/0 as well. So this module is where
half the definition wave's bill comes from, and every rule in it was measured:

  floor 1024        `400 INVALID_ARGUMENT: Cached content is too small.
                    total_token_count=23, min_total_token_count=1024`. Read off
                    the artifact, never hard-coded: the number is a property of
                    the service on the day it was measured.
  XOR               a request that carries a cache may not also carry a system
                    instruction: `CachedContent can not be used with
                    GenerateContent request setting system_instruction, tools or
                    tool_config`. The system prompt moves INTO the cache.
  resolved late     batch resolves the cache AT EXECUTION time, not at submit:
                    a cache deleted immediately after a successful submit made
                    all 21 rows of that job fail. The TTL therefore has to cover
                    the whole drain window, not just the submit.
  expiry is loud    an expired cache is `403 PERMISSION_DENIED: CachedContent
                    not found (or permission denied)` on generate AND on update,
                    and every row of a batch job fails with gRPC code 7 at
                    prompt=0, billed $0. Nothing was ever silently charged at
                    the full rate, so an expiry costs a whole drain window
                    rather than money -- which is why the TTL is sized against
                    the DOCUMENTED window (see cache_ttl_plan) rather than
                    against the 3x insurance the audit wanted for silent
                    billing that does not happen.
  one key           a cache is bound to the key/project that created it; another
                    key referencing it gets the same 403. _pool_from_env already
                    pins one key when cache_enabled is set.
  free tier never   TotalCachedContentStorageTokensPerModelFreeTier limit=0.
                    A free key cannot create a cache at all, and the error says
                    so once you know to look for it.
"""

import dataclasses
import datetime
import math

from ..util import FatalError
from . import waves

# The error text a free-tier key gets when it tries to create a cache. Turned
# into a sentence rather than passed through, because the raw message names a
# quota id and not the fact that this tier can never do this.
FREE_TIER_MARKERS = ("cachedcontentstoragetokens", "freetier")

# The 403 that means "this cache is gone". Same string on generate and on
# update, which is why an expired cache cannot be extended, only recreated.
CACHE_GONE_MARKERS = ("cachedcontent not found", "permission_denied",
                      "permission denied")


@dataclasses.dataclass
class CacheHandle:
    """One live CachedContent, and everything a gate or a ledger row needs.

    `declared_tokens` is the size the SERVICE reported at creation
    (usageMetadata.totalTokenCount). It is the denominator G-CACHE compares
    every row's cachedContentTokenCount against -- `cached == declared`, per
    row. The ratio the audit proposed (cached/prompt >= 0.90) is measurably
    wrong: on a fully cached wave it slides from 0.935 at n=1 to 0.632 at n=20
    while the cache is hitting 1.00 every time.
    """
    name: str
    declared_tokens: int
    prompt_sha256: str
    model: str
    lang: str
    kind: str
    ttl_seconds: int
    created_at: str
    display_name: str = ""
    recreated: int = 0

    def as_record(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record: dict):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in fields})


def cache_floor(stats: dict | None) -> int:
    """The measured explicit-cache minimum, from the artifact.

    Refuses rather than defaults: a cache created under the floor is a 400, and
    guessing the floor from a source constant is how a number nobody measured
    ends up authorising a spend. The IMPLICIT floor (a documented 4,096) is a
    DIFFERENT number and is deliberately not consulted here.
    """
    value = ((stats or {}).get("wave2") or {}).get("EXPLICIT_CACHE_FLOOR")
    if value is None:
        raise FatalError(
            "wave2.EXPLICIT_CACHE_FLOOR is not in the measured constants, so "
            "the explicit-cache floor is unknown. It is a server-side number "
            "(the refusal is `Cached content is too small. total_token_count=%s"
            ", min_total_token_count=%s`) and it is not the implicit floor. "
            "Re-run the probe artifact backfill before enabling the cache."
            % ("N", "N"))
    return int(value)


def qualifies_for_cache(text: str, stats: dict | None) -> tuple:
    """(ok, estimated_tokens, floor). The pre-flight for a cache creation.

    Uses the money stack's offline token estimator, so the number checked here
    is the number the bill used. An over-estimate here would create a cache the
    service refuses; an under-estimate would skip a discount we qualify for.
    """
    from .. import billing                    # noqa: PLC0415 - import cycle
    floor = cache_floor(stats)
    tokens = billing.estimated_prompt_tokens(text, stats)
    if tokens is None:
        raise FatalError(
            "the prompt size cannot be estimated offline (no CHARS_PER_TOKEN "
            "in the measured constants), so whether it clears the %d-token "
            "explicit-cache floor is unknown." % floor)
    return int(tokens) >= floor, int(tokens), floor


def ttl_seconds(estimated_wall_clock_s: float, factor: float) -> int:
    """ttl = factor x an estimated wall clock. One term of the TTL, not the TTL.

    See cache_ttl_plan: this term is the throughput extrapolation, and it only
    decides the TTL for a wave large enough to exceed the documented drain
    window. The window itself is not a multiple of anything.
    """
    return int(math.ceil(float(estimated_wall_clock_s) * float(factor)))


def cache_ttl_plan(total_requests: int, *, poll_deadline_s: float,
                   factor: float) -> dict:
    """The cache TTL for one wave, and the arithmetic that produced it.

    `total_requests` is a count of REQUESTS. It used to be handed a count of
    CELLS, which on the shipped corpus is 3.7x larger (13,632 definition cells
    against 3,642 definition requests): the TTL was long enough, but for a
    reason nobody had written down, and the variable was already named
    `requests`. Making the name true would have cut the TTL to 9.1h.

    Two terms, and the larger wins:

      window + margin   the DOCUMENTED drain window (the transport's own poll
                        deadline, floored at the documented 48h hard expiry)
                        plus a submit-overhead margin. This is the term that
                        decides the TTL at every size this program will ever
                        run, and it is the one the artifact's TTL_POLICY asks
                        for: "TTL must cover the whole drain window".
      factor x estimate the throughput extrapolation (6s/request from the 32-row
                        probe job) scaled by cfg.cache_ttl_factor. Only reaches
                        the answer above ~31,000 requests in one job, which the
                        splitter's 2.4M-token target makes impossible today --
                        kept so that a corpus that outgrows the documented
                        window is not silently under-covered.

    Cost of the window term at the shipped size: a 1,135-token definition prompt
    held for 52 hours is 1,135 x 52 / 1e6 x $0.50 = $0.0295 per language, or
    $0.118 across four. The alternative is the failure the probes paid for: a
    cache that dies mid-drain makes every remaining row of the job fail with
    gRPC code 7 at prompt=0, which costs $0 and an entire 48-hour window.
    """
    window = waves.drain_window_seconds(total_requests,
                                        poll_deadline_s=poll_deadline_s)
    documented = int(math.ceil(window + float(waves.TTL_MARGIN_SECONDS)))
    throughput = ttl_seconds(
        waves.wall_clock_estimate_seconds(max(1, int(total_requests))), factor)
    ttl = max(documented, throughput)
    return {"ttl_seconds": int(ttl),
            "requests": int(total_requests),
            "drain_window_seconds": int(window),
            "poll_deadline_seconds": int(poll_deadline_s),
            "margin_seconds": int(waves.TTL_MARGIN_SECONDS),
            "documented_window_term": documented,
            "throughput_term": int(throughput),
            "ttl_factor": float(factor),
            "decided_by": ("documented drain window + margin" if ttl == documented
                           else "%.2fx the throughput extrapolation" % factor),
            "basis": ("the cache is resolved when each ROW EXECUTES, so the TTL "
                      "covers the whole drain window (poll deadline %.1fh, "
                      "documented hard expiry %.1fh) plus a %.1fh submit margin "
                      "-- not a multiple of a 6s/request extrapolation from a "
                      "32-row probe job"
                      % (float(poll_deadline_s) / 3600.0,
                         waves.DOCUMENTED_JOB_HARD_EXPIRY_SECONDS / 3600.0,
                         waves.TTL_MARGIN_SECONDS / 3600.0))}


def classify_cache_error(exc: Exception) -> str:
    """"free_tier" | "gone" | "other" for an exception from a caches call."""
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    squashed = text.replace(" ", "").replace("_", "")
    if all(marker in squashed for marker in FREE_TIER_MARKERS):
        return "free_tier"
    if any(marker in text for marker in CACHE_GONE_MARKERS):
        return "gone"
    return "other"


def declared_tokens_of(cached) -> int | None:
    """usageMetadata.totalTokenCount from a CachedContent, both spellings."""
    usage = getattr(cached, "usage_metadata", None)
    if usage is None:
        usage = getattr(cached, "usageMetadata", None)
    if usage is None:
        return None
    for name in ("total_token_count", "totalTokenCount"):
        value = (usage.get(name) if isinstance(usage, dict)
                 else getattr(usage, name, None))
        if value is not None:
            return int(value)
    return None


def create(client, *, model: str, system_text: str, lang: str, kind: str,
           ttl: int, stats: dict | None, display_name: str = "") -> CacheHandle:
    """Create one CachedContent holding the system prompt for (kind, lang).

    The system prompt goes in WHOLE and leaves the request entirely: sending
    both is a hard 400, so there is no half-way state where the cache exists and
    the prompt is also inlined.

    The floor is checked BEFORE the call, because a refusal here is free and a
    refusal from the service costs a round trip on the money path.
    """
    from ..stages.s42_translate import prompt_sha256   # noqa: PLC0415
    ok, tokens, floor = qualifies_for_cache(system_text, stats)
    if not ok:
        raise FatalError(
            "the %s prompt for %s is about %d tokens, under the measured "
            "%d-token explicit-cache floor, so a cache object for it would be "
            "refused (`Cached content is too small`). Padding it to qualify "
            "would buy a discount on tokens that only exist because of the "
            "padding -- the expression prompt is deliberately left uncached "
            "for this reason." % (kind, lang, tokens, floor))
    from google.genai import types                     # noqa: PLC0415
    try:
        cached = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_text,
                ttl="%ds" % int(ttl),
                display_name=display_name or ("ankidkdeck-%s-%s" % (kind, lang))))
    except Exception as exc:                           # noqa: BLE001
        if classify_cache_error(exc) == "free_tier":
            raise FatalError(
                "this key cannot create an explicit cache: the free tier's "
                "TotalCachedContentStorageTokensPerModelFreeTier limit is 0, so "
                "explicit caching is impossible there and it is the only "
                "discount path there is (batch has no implicit caching). Use a "
                "paid Tier 1 key, or set cache_enabled = false and accept the "
                "full uncached rate. Original error: %s" % exc) from exc
        raise
    declared = declared_tokens_of(cached)
    if not declared:
        raise FatalError(
            "the service did not report a token count for the cache it just "
            "created (%s). That number is G-CACHE's denominator -- every row's "
            "cachedContentTokenCount is compared against it -- so a wave "
            "without it cannot be checked for whether the cache was used."
            % getattr(cached, "name", "?"))
    return CacheHandle(
        name=cached.name, declared_tokens=int(declared),
        prompt_sha256=prompt_sha256(system_text), model=model, lang=lang,
        kind=kind, ttl_seconds=int(ttl), created_at=_now(),
        display_name=display_name)


def remaining_seconds(client, handle: CacheHandle, *, now=None):
    """Seconds of life left on the cache, or None if the service will not say.

    None is not "fine": it is reported and the caller extends anyway. The one
    thing that must not happen is a job submitted against a cache whose life
    nobody looked at, because the rows resolve the cache when they execute and
    the failure is per row.

    A cache that is already gone answers this question with the SAME 403 that
    generate and update give, so it is raised as CacheUnavailable here too --
    the caller's recovery is identical (recreate, new resource name) and having
    one of the three paths raise a bare transport error would strand it.
    """
    from ..stages.s42_translate import CacheUnavailable  # noqa: PLC0415
    try:
        got = client.caches.get(name=handle.name)
    except Exception as exc:                             # noqa: BLE001
        if classify_cache_error(exc) == "gone":
            raise CacheUnavailable(
                "the explicit cache %s is gone (%s). It cannot be updated, only "
                "recreated, and the recreate changes the resource name."
                % (handle.name, exc)) from exc
        raise
    expire = getattr(got, "expire_time", None) or getattr(got, "expireTime", None)
    if expire is None:
        return None
    if isinstance(expire, str):
        try:
            expire = datetime.datetime.fromisoformat(expire.replace("Z", "+00:00"))
        except ValueError:
            return None
    reference = now or datetime.datetime.now(datetime.timezone.utc)
    if expire.tzinfo is None:
        expire = expire.replace(tzinfo=datetime.timezone.utc)
    return (expire - reference).total_seconds()


def extend(client, handle: CacheHandle, ttl: int) -> CacheHandle:
    """caches.update the TTL. Raises CacheUnavailable if the cache is gone.

    An expired cache CANNOT be updated -- update returns the same 403 as
    generate -- so "extend" is only ever a live-cache operation and the caller's
    recovery is a recreate, which changes the resource name. That is why the
    JSONL is written per job just before the submit and never in advance.
    """
    from ..stages.s42_translate import CacheUnavailable  # noqa: PLC0415
    from google.genai import types                       # noqa: PLC0415
    try:
        client.caches.update(name=handle.name,
                             config=types.UpdateCachedContentConfig(
                                 ttl="%ds" % int(ttl)))
    except Exception as exc:                             # noqa: BLE001
        if classify_cache_error(exc) == "gone":
            raise CacheUnavailable(
                "the explicit cache %s is gone (%s). An expired cache cannot be "
                "updated, only recreated, and a recreate changes the resource "
                "name." % (handle.name, exc)) from exc
        raise
    handle.ttl_seconds = int(ttl)
    return handle


def delete(client, handle: CacheHandle) -> dict:
    """End of the wave. A cache left behind is billed for storage by the hour.

    Never fatal: the wave is over, the rows are written, and an undeleted cache
    costs $0.50 per million token-hours. It is reported so the number is
    visible.
    """
    try:
        client.caches.delete(name=handle.name)
    except Exception as exc:                             # noqa: BLE001
        return {"deleted": False, "name": handle.name, "why": str(exc)[:200]}
    return {"deleted": True, "name": handle.name}


def storage_cost(handle: CacheHandle, hours: float, mode: str) -> dict:
    """What the cache object itself cost, separately from the discount.

    Storage is $0.50 per million token-hours and is NOT part of the input rate,
    so it is quoted on its own line: a 1,135-token definition prompt held for a
    24-hour drain is $0.0136, and a RICH 2,815-token one is $0.034. Small
    against a $4.09 wave, not small against the $0.22 of headroom four
    languages leave under a $10 cap -- which is the argument for computing it
    rather than calling it negligible.
    """
    from .. import prices                               # noqa: PLC0415
    try:
        usd = prices.cache_storage_usd(handle.declared_tokens, hours,
                                       handle.model, mode)
    except Exception as exc:                            # noqa: BLE001
        return {"usd": None, "why": str(exc)[:200]}
    return {"usd": usd, "tokens": handle.declared_tokens, "hours": hours}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc) \
        .isoformat(timespec="seconds")
