"""Stage 12: the crawl. Pilot -> phase A (one GET per wordlist word) ->
phase B (lemma completion) -> phase C (overrides / entry_id recovery).

Design constraints, all measured:
- The article set depends on WHICH form you query (/har -> 1 article,
  /have -> 3), so phase B is a correctness requirement, not an optimisation.
- DDO never answers 404; a miss is detected from page content. #results-label
  reconciles against the article count and separates nohit / ok / fetch-error.
- The ledger records ATTEMPTS, not successes, so a no-hit word is not re-queried
  on every resume; raw HTML is stored under sha1(word) -- a word is never a
  filename (case-insensitive filesystems, spaces, dots).
"""

import datetime
import re

from bs4 import BeautifulSoup

from ..config import Config
from ..gates import sitemap_shortfall
from ..net import Net
from ..urls import word_url
from ..util import (NFC, FatalError, atomic_write_text, nk, read_json, sha1_hex,
                    sha256_str, write_json)

NOHIT_MARKER = "matcher ingen opslag i ordbogen"
RESULTS_RE = re.compile(r"^(\d+) resultater$")


def _now() -> str:
    """Wall clock is allowed HERE and nowhere else in the pipeline: the ledger is
    crawl provenance, not a build output, and a 5-hour crawl with no timeline is
    an attempts ledger that cannot answer the one question it exists for."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def raw_path(cfg: Config, word: str):
    return cfg.raw_dir / f"{sha1_hex(NFC(word))}.html"


def verdict_of(body: str) -> tuple[str, str | None, int]:
    """Three-state verdict from #results-label vs the article count."""
    soup = BeautifulSoup(body, "html.parser")
    label_el = soup.select_one("#results-label")
    label = label_el.get_text(strip=True) if label_el is not None else None
    n = len(soup.select("div.artikel"))
    if label is None and n == 0 and NOHIT_MARKER in body:
        return "nohit", label, n
    if label == "" and n == 1:
        return "ok", label, n
    m = RESULTS_RE.fullmatch(label or "")
    if m and int(m.group(1)) == n:
        return "ok", label, n
    return "error", label, n


class Ledger:
    def __init__(self, cfg: Config):
        self.path = cfg.json_dir / "fetch_ledger.json"
        self.data = read_json(self.path, default={})

    def save(self):
        write_json(self.path, self.data)

    def get(self, word):
        return self.data.get(word)


def fetch(cfg: Config, net: Net, ledger: Ledger, word: str, phase: str,
          url: str | None = None) -> dict:
    if NFC(word) != word:
        raise FatalError(f"non-NFC word reached fetch(): {word!r}")
    led = ledger.get(word)
    if led and led.get("status") in ("ok", "nohit"):
        return led
    rec = ledger.data.setdefault(word, {"attempts": 0, "phase": phase})
    rec.setdefault("first_at", _now())
    rec["attempts"] += 1
    rec["last_at"] = _now()
    rec.setdefault("status", "attempted")
    u = url or word_url(word)
    rec["url"] = u
    # Checkpoint BEFORE the request as well as after. The WAF stop is fatal and
    # never retried, so without this the word that triggered it left no record
    # that it had been attempted at all -- and `attempts` was a counter that
    # could only ever be incremented by a success.
    ledger.save()
    try:
        r = net.get(u)
    except FatalError as exc:
        if "WAF" in str(exc):
            rec.update(status="waf", waf=True, last_at=_now())
            ledger.save()
        raise
    body = r.text
    atomic_write_text(raw_path(cfg, word), body)
    status, label, n = verdict_of(body)
    rec.update(status=status, url=u, http=r.status_code, bytes=len(body),
               redirects=len(r.history), body_sha256=sha256_str(body),
               results_label=label, article_count=n, last_at=_now())
    ledger.save()  # checkpoint after EVERY request
    return rec


def run_pilot(cfg: Config, net: Net, gates: dict) -> dict:
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    step = max(1, len(wordlist) // cfg.pilot_size)
    sample = wordlist[::step][: cfg.pilot_size]
    ledger = Ledger(cfg)
    for w in sample:
        fetch(cfg, net, ledger, w["word"], "pilot")
    stats = _stats(ledger, {w["word"] for w in sample})
    stats["pilot_ok"] = stats["error_rate"] <= gates.get("pilot_max_error_rate", 0.01)
    write_json(cfg.report_dir / "pilot_report.json", stats)
    if not stats["pilot_ok"]:
        raise FatalError(f"pilot failed: error rate {stats['error_rate']:.2%}")
    return stats


def run_phase_a(cfg: Config, net: Net) -> dict:
    wordlist = read_json(cfg.json_dir / "wordlist.json")["words"]
    pilot = cfg.report_dir / "pilot_report.json"
    if not pilot.exists():
        raise FatalError("run the pilot first (ankidkdeck crawl --pilot)")
    ledger = Ledger(cfg)
    for w in wordlist:
        fetch(cfg, net, ledger, w["word"], "A")
    return _stats(ledger, {w["word"] for w in wordlist})


def run_phase_b(cfg: Config, net: Net, lemmas_needed: set[str]) -> dict:
    """Fetch each kept lemma that was never queried directly: the lemma's own
    page is the one that carries the full homograph set and flex tables."""
    ledger = Ledger(cfg)
    wordset = {w["word"] for w in read_json(cfg.json_dir / "wordlist.json")["words"]}
    nk_wordset = {nk(w) for w in wordset}
    fetched = []
    for lemma in sorted(lemmas_needed):
        lemma = NFC(lemma)
        # nk(), as the guide specifies: an exact string match re-fetches a
        # case-differing lemma we already have.
        if nk(lemma) in nk_wordset or ledger.get(lemma):
            continue
        fetch(cfg, net, ledger, lemma, "B")
        fetched.append(lemma)
    return {"phase_b_fetched": len(fetched)}


def run_phase_c(cfg: Config, net: Net, registry) -> dict:
    ledger = Ledger(cfg)
    fetched = []
    wordset = {w["word"] for w in read_json(cfg.json_dir / "wordlist.json",
                                            default={"words": []})["words"]}
    nk_wordset = {nk(w) for w in wordset}
    for form, lemma in registry.form_to_lemma.items():
        lemma = NFC(lemma)
        # nk() comparison, as the guide specifies: an exact string match costs an
        # extra request for a case-differing lemma.
        if nk(lemma) in nk_wordset or ledger.get(lemma):
            continue
        fetch(cfg, net, ledger, lemma, "C")
        fetched.append(lemma)
    out = {"phase_c_fetched": len(fetched)}
    # Report-only here: this is the last stage that COULD remedy a shortfall with
    # a fetch, so it is where the number is worth printing. The gate itself is
    # enforced at export (guide 4.12); stage 30 only measures it.
    sf = read_json(cfg.json_dir / "sitemap_shortfall.json",
                   default={"rows": [], "n_families": 0})
    ok, detail = sitemap_shortfall(
        sf.get("rows") or [], sf.get("n_families") or 0,
        float(registry.gates.get("sitemap_shortfall_max_rate", 0.01)))
    out["sitemap_shortfall"] = {"within_max_rate": ok,
                                "families_short": detail["families_short"],
                                "rate": detail["rate"],
                                "max_rate": detail["max_rate"]}
    return out


def _stats(ledger: Ledger, words: set[str]) -> dict:
    rows = [v for k, v in ledger.data.items() if k in words]
    n = len(rows)
    err = sum(1 for r in rows if r.get("status") == "error")
    return {
        "n": n,
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "nohit": sum(1 for r in rows if r.get("status") == "nohit"),
        "attempted": sum(1 for r in rows if r.get("status") == "attempted"),
        "errors": err,
        "error_rate": err / n if n else 0.0,
        # From the records. A WAF stop is fatal, so a clean run legitimately
        # reports 0 -- but it is measured, not asserted.
        "waf": sum(1 for r in rows if r.get("waf")),
    }
