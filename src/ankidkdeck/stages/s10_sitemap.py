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


def run(cfg: Config, net: Net, gates: dict) -> dict:
    robots = net.get("https://ordnet.dk/robots.txt").text
    if re.search(r"^Disallow:\s*/ddo\b", robots, re.M):
        raise FatalError("robots.txt now disallows /ddo -- governance stop")
    if "/sitemaps/ddo/index.xml" not in robots:
        raise FatalError("robots.txt no longer advertises the DDO sitemap index")
    (cfg.report_dir).mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / "robots_snapshot.txt").write_text(robots, encoding="utf-8")

    idx = net.get(SITEMAP_INDEX).text
    shard_urls = [loc for loc, _ in _shard_urls(idx)]
    if not 5 <= len(shard_urls) <= 12:
        raise FatalError(f"unexpected sitemap shard count: {len(shard_urls)}")

    lemmas: dict[str, dict] = {}
    affix_slugs: list[str] = []
    lastmods_per_shard: dict[str, set] = {}
    total = 0
    for su in shard_urls:
        r = net.get(su)
        xml = r.text
        if su.endswith(".gz") or r.content[:2] == b"\x1f\x8b":
            xml = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8")
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
                affix_slugs.append(base)
        lastmods_per_shard[su] = mods

    if total < 80_000:
        raise FatalError(f"sitemap total {total} URLs; expected > 80k -- site changed?")
    lo, hi = gates.get("affix_count_range", [150, 400])
    if not lo <= len(affix_slugs) <= hi:
        raise FatalError(f"affix slug count {len(affix_slugs)} outside [{lo},{hi}]")

    out = {
        "total_urls": total,
        "n_lemmas": len(lemmas),
        "lemmas": lemmas,
        "affix_slugs": sorted(set(affix_slugs)),
        "lastmod_note": "uniform per shard; provenance only, never a skip condition",
        "lastmod_distinct_per_shard": {k: sorted(x for x in v if x) for k, v in lastmods_per_shard.items()},
    }
    write_json(cfg.json_dir / "sitemap.json", out)
    return {"total_urls": total, "n_lemmas": len(lemmas), "n_affix": len(set(affix_slugs)),
            "requests": net.request_count}
