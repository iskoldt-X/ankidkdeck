"""Polite HTTP layer for ordnet.dk.

Rules (all measured, see the vault's final guide):
- Serial requests, 2-4s jitter, honest desktop UA.
- A WAF challenge (HTTP 202 or the x-amzn-waf-action header) is FATAL, never
  retried: the 2025-era 5x60s retry loop would hammer the WAF. Run from a
  residential IP; datacenter IPs are challenged.
- 403/429 fatal; one retry on 5xx; circuit breaker on degraded runs.
- static.ordnet.dk (audio) is a separate plain nginx: no WAF, shorter sleep.
- A 200 that carries no audio (empty body, or a content-type that is not audio)
  is a FAILURE, retried once and then raised as AudioUnavailable. It never
  becomes a file on disk.
"""

import random
import time

import requests

from .util import AudioUnavailable, FatalError

# What static.ordnet.dk answers with for a real file: `audio/mpeg` on every one
# of the three health probes (2026-08-27, nginx/1.10.3). A body whose declared
# type is neither audio/* nor the generic octet-stream is not an mp3, whatever
# its status line says -- the four upstream-dead slots answer `text/html`, and a
# WAF challenge page served as 200 would too (get_audio has no 202 check because
# this host has no WAF, so the content-type is the only signal there is).
# octet-stream is allowed because a static file server that loses its mime table
# still serves the right bytes, and bricking 5,889 downloads over a header is the
# worse failure.
AUDIO_CONTENT_TYPES_OK = ("audio/", "application/octet-stream")
# Extra wait before the one retry of a no-audio 200. Added to get_audio's own
# 1.0 s, so the retry lands >= 2.0 s after the first attempt -- the spacing the
# 2026-08-27 verification probes used, and looser than the stage's own pacing,
# never tighter.
AUDIO_RETRY_SLEEP = 1.0


class Circuit:
    def __init__(self):
        self.consecutive_failures = 0
        self.results: list[bool] = []

    def record(self, ok: bool) -> None:
        self.results.append(ok)
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
        if self.consecutive_failures >= 3:
            raise FatalError("circuit breaker: 3 consecutive failed requests")
        if len(self.results) > 50:
            rate = sum(self.results[-50:]) / 50
            if rate < 0.95:
                raise FatalError(f"circuit breaker: rolling success rate {rate:.0%} < 95%")


class Net:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.ua
        self.circuit = Circuit()
        self.request_count = 0

    def _sleep(self):
        time.sleep(random.uniform(self.cfg.sleep_min, self.cfg.sleep_max))

    def get(self, url: str, retried: bool = False) -> requests.Response:
        self._sleep()
        self.request_count += 1
        r = self.session.get(url, timeout=30, allow_redirects=True)
        if r.status_code == 202 or "x-amzn-waf-action" in r.headers:
            raise FatalError(
                "WAF challenge received. Run from a residential IP; never retry."
            )
        if r.status_code in (403, 429):
            raise FatalError(f"blocked/throttled: HTTP {r.status_code} for {url}")
        if r.status_code >= 500:
            if not retried:
                # Record the failed FIRST attempt: without it a run that 5xx'd
                # on every other request and recovered each time showed a 100%
                # rolling success rate, and the circuit breaker -- whose whole
                # job is to notice a degraded run -- never tripped.
                self.circuit.record(False)
                time.sleep(30)
                return self.get(url, retried=True)
            raise FatalError(f"persistent 5xx for {url}")
        self.circuit.record(r.status_code == 200)
        return r

    def get_audio(self, url: str, retried: bool = False,
                  expected_missing: bool = False) -> requests.Response:
        """static.ordnet.dk: plain nginx, no WAF, 1s sleep -- but the SAME
        accounting as every other request.

        It used to bypass both request_count and the circuit breaker, so a stage
        60 run that fetched ~4,600 files reported 0 requests, and a degraded
        audio host produced 4,600 individual FatalErrors on resume instead of one
        breaker trip. The failure is recorded BEFORE it is raised, so three
        consecutive failures trip the breaker with its own message rather than
        the third file's.

        THE NO-AUDIO RUNG. A 200 whose body is empty, or whose content-type is
        not audio, had no rung on this ladder at all: a 5xx was retried once, a
        404 was fatal, and a 200-with-nothing was simply RETURNED -- so the one
        failure that actually happened (4 declared slots answering 200 /
        content-length 0 / text/html, one shared zero-byte placeholder) reached
        the caller as a successful response and only stage 60's own `if not
        r.content` kept it off the disk. A content-type check never existed, so a
        challenge page served as 200 would have been written as an mp3 and
        G-MEDIA, which tests for zero bytes, would have passed it. Now: one
        retry, then AudioUnavailable. The response is never returned, so nothing
        that is not audio can be written to a file.

        `expected_missing` is for a slot already recorded in
        registry/known_missing_audio.json. Stage 60 re-probes those on every run
        so a future DDO repair surfaces by itself, and two things must not happen
        to that probe: it must not spend a retry hammering a URL we have already
        proved dead over three attempts, and its failure must not reach the
        circuit breaker -- four known-dead probes in a row are four consecutive
        failures, which would trip a breaker whose job is to notice a degraded
        HOST and would abort an otherwise fully cached rerun. An UNKNOWN no-audio
        200 IS recorded, once, so a host that starts serving empty bodies
        wholesale still trips at three in a row -- which is the right place for a
        human to look at it rather than for the stage to collect 5,893 gaps.
        """
        time.sleep(1.0)
        self.request_count += 1
        r = self.session.get(url, timeout=30)
        if r.status_code != 200:
            self.circuit.record(False)
            raise FatalError(f"audio fetch failed: {url} -> HTTP {r.status_code}")
        ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        why = ""
        if not r.content:
            why = "empty body"
        elif ctype and not ctype.startswith(AUDIO_CONTENT_TYPES_OK):
            why = "content-type is not audio"
        if why:
            # Recorded on the FIRST attempt and not again on the raise, exactly
            # like the 5xx rung above: one URL that could not be fetched is ONE
            # host failure. Recording both attempts would make two dead slots
            # look like four, and the breaker's rolling window (3 failures in 50
            # requests) is tight enough that the difference decides whether a
            # handful of upstream-dead slots aborts a 1,276-file download.
            if not retried and not expected_missing:
                self.circuit.record(False)
                time.sleep(AUDIO_RETRY_SLEEP)
                return self.get_audio(url, retried=True)
            raise AudioUnavailable(url, status=r.status_code, content_type=ctype,
                                   n_bytes=len(r.content), retried=retried,
                                   why=why)
        self.circuit.record(True)
        return r
