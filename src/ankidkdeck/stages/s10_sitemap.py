"""Stage 10: sitemap inventory (9 requests, once per release).

The sitemap is an INVENTORY and an assertion source, never a fetch plan: it
holds ~90k URLs, 20x the wordlist. Its per-URL <lastmod> is a generation stamp
(one distinct value per shard) and carries no per-entry change information --
refresh is driven by our own content hashes, never by lastmod.
"""

import gzip
import io
import re
from urllib.parse import unquote

from ..config import Config
from ..gates import G_SITEMAP_INV, Gate, run_gates, sitemap_inventory
from ..net import Net
from ..util import NFC, FatalError, nk, write_json

SITEMAP_INDEX = "https://ordnet.dk/sitemaps/ddo/index.xml"
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
LASTMOD_RE = re.compile(r"<lastmod>([^<]+)</lastmod>")
TRAILING_N = re.compile(r"^(.*)_(\d+)$")


def _shard_urls(xml: str) -> list[tuple[str, str | None]]:
    locs = LOC_RE.findall(xml)
    mods = LASTMOD_RE.findall(xml)
    return list(zip(locs, mods + [None] * (len(locs) - len(mods))))


def robots_forbids_ddo(robots: str) -> str | None:
    """The offending directive, or None.

    A blanket `Disallow: /` under `User-agent: *` is the same governance stop as
    an explicit /ddo rule -- checking only for /ddo let the strictest possible
    robots.txt pass the gate.
    """
    if re.search(r"^Disallow:\s*/ddo\b", robots, re.M):
        return "Disallow: /ddo"
    agent = None
    for raw in robots.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "user-agent":
            agent = val
        elif key == "disallow" and agent == "*" and val == "/":
            return "User-agent: * + Disallow: /"
    return None


def _shard_xml(su: str, r) -> str:
    """Gunzip on the MAGIC BYTES only, never on the .gz suffix.

    CloudFront serves these objects with `content-encoding: br` and
    `vary: Accept-Encoding`, so what requests hands back depends on whether
    brotli is installed, and any decoding proxy produces plain XML at a .gz URL.
    Trusting the suffix raised a bare BadGzipFile traceback.
    """
    if r.content[:2] != b"\x1f\x8b":
        return r.text
    try:
        return gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise FatalError(
            "sitemap shard %s claims gzip but could not be decompressed (%s); "
            "first bytes %r" % (su, exc, r.content[:16])) from exc


def run(cfg: Config, net: Net, gates: dict) -> dict:
    robots = net.get("https://ordnet.dk/robots.txt").text
    forbidden = robots_forbids_ddo(robots)
    if forbidden:
        raise FatalError(
            "robots.txt now forbids the crawl (%s) -- governance stop" % forbidden)
    if "/sitemaps/ddo/index.xml" not in robots:
        raise FatalError("robots.txt no longer advertises the DDO sitemap index")
    (cfg.report_dir).mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / "robots_snapshot.txt").write_text(robots, encoding="utf-8")

    idx = net.get(SITEMAP_INDEX).text
    shard_urls = [loc for loc, _ in _shard_urls(idx)]
    if not 5 <= len(shard_urls) <= 12:
        raise FatalError(f"unexpected sitemap shard count: {len(shard_urls)}")

    lemmas: dict[str, dict] = {}
    # A SET: a lemma with homograph URLs (-hed_1, -hed_2) is one affix slug, and
    # counting it twice made the gate compare a duplicate-inflated number
    # against a range derived from unique slugs.
    affix_slugs: set[str] = set()
    lastmods_per_shard: dict[str, set] = {}
    total = 0
    for su in shard_urls:
        r = net.get(su)
        xml = _shard_xml(su, r)
        mods = set()
        for loc, mod in _shard_urls(xml):
            total += 1
            mods.add(mod)
            slug = NFC(unquote(loc.rsplit("/", 1)[-1]))
            m = TRAILING_N.match(slug)
            base, n = (m.group(1), int(m.group(2))) if m else (slug, None)
            row = lemmas.setdefault(nk(base), {"display": base, "homographs": [], "urls": []})
            row["urls"].append(loc)
            if n is not None:
                row["homographs"].append(n)
            # Affix detection is by SHAPE, never by shard: the 'other' shard is
            # 82% ordinary ae/oe/digit-initial words, not an affix inventory.
            if base.startswith("-") or base.endswith("-"):
                affix_slugs.add(base)
        lastmods_per_shard[su] = mods

    # Both inventory bounds are DATA and both are RECORDED. The URL total used
    # to be `if total < 80_000: raise FatalError(...)` -- a source constant
    # extrapolated from a partial measurement, and a stop that never reached
    # gates_report.json. It now lives in registry/gates.json as
    # sitemap_total_range, shipped null = report-only until a human copies the
    # first real 9-request run's total in as a band. The affix range is already
    # baselined from a real 4-shard measurement (285 unique slugs over
    # a_d/e_h/other/u_z, scaled to the three unmeasured shards; the old
    # [150, 400] ceiling came from TWO shards and would have hard-stopped the
    # first real run), so it stays enforced -- as a gate row, not a bare raise.
    total_range = gates.get("sitemap_total_range")
    affix_range = gates.get("affix_count_range", [150, 600])
    run_gates([
        Gate(G_SITEMAP_INV, "the sitemap inventory's URL total and unique affix "
                            "slug count are inside their declared ranges",
             lambda: sitemap_inventory(total, total_range, len(affix_slugs),
                                       affix_range),
             stage="10"),
    ], cfg, stage="10")

    out = {
        "total_urls": total,
        "n_lemmas": len(lemmas),
        "lemmas": lemmas,
        "affix_slugs": sorted(affix_slugs),
        "lastmod_note": "uniform per shard; provenance only, never a skip condition",
        "lastmod_distinct_per_shard": {k: sorted(x for x in v if x) for k, v in lastmods_per_shard.items()},
    }
    write_json(cfg.json_dir / "sitemap.json", out)
    report = {"total_urls": total, "n_lemmas": len(lemmas),
              "n_affix": len(affix_slugs), "requests": net.request_count,
              "sitemap_total_range": total_range,
              "baseline_hint": (
                  None if total_range else
                  "sitemap_total_range is null (report-only). Copy "
                  "sitemap_total_range: [%d, %d] into registry/gates.json to "
                  "baseline this inventory." % (int(total * 0.75),
                                                int(total * 1.25)))}
    write_json(cfg.report_dir / "sitemap_report.json", report)
    return report
