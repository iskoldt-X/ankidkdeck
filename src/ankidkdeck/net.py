"""Polite HTTP layer for ordnet.dk.

Rules (all measured, see the vault's final guide):
- Serial requests, 2-4s jitter, honest desktop UA.
- A WAF challenge (HTTP 202 or the x-amzn-waf-action header) is FATAL, never
  retried: the 2025-era 5x60s retry loop would hammer the WAF. Run from a
  residential IP; datacenter IPs are challenged.
- 403/429 fatal; one retry on 5xx; circuit breaker on degraded runs.
- static.ordnet.dk (audio) is a separate plain nginx: no WAF, shorter sleep.
"""

import random
import time

import requests

from .util import FatalError


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
                time.sleep(30)
                return self.get(url, retried=True)
            raise FatalError(f"persistent 5xx for {url}")
        self.circuit.record(r.status_code == 200)
        return r

    def get_audio(self, url: str) -> requests.Response:
        time.sleep(1.0)
        r = self.session.get(url, timeout=30)
        if r.status_code != 200:
            raise FatalError(f"audio fetch failed: {url} -> HTTP {r.status_code}")
        return r
