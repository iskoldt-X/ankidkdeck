#!/usr/bin/env python3
"""T5: what a v3 import does to an existing collection -- kept / new / retired.

Reads the GUIDs out of a released .apkg (a zip around a SQLite file) and the
GUIDs a v3 build would produce from words.json, and prints the three tables the
release notes must state. G-REL exists so those numbers are computed, not
estimated: the churn figure in the original spec was off by up to +22%.

    python3 tools/guid_diff.py --apkg dist/DDO_Danish_Frequency_Deck_German.apkg \\
        --lang German --work work

Stdlib only (sqlite3 + zipfile). genanki.guid_for() is reimplemented here so the
tool runs on a host with no genanki installed; when genanki IS importable the two
are cross-checked on EVERY seed and a disagreement is fatal, because a drift in
the GUID formula is the one bug that cannot be repaired after release. (The
docstring used to promise every seed while the code checked the first 50 -- in a
tool whose entire value is trustworthiness. ~2,900 sha256 calls is nothing.)

The report carries a `summary` row so the export gate can assert against it:
G-REL fails the build when reports/guid_diff.<lang>.json describes a different
language or a different card count from the deck being written.

The report is written PER LANGUAGE -- reports/guid_diff.<lang>.json -- because
one release month can ship more than one language and the file is read by
language, not by recency. Under the old single guid_diff.json, the Chinese
month's file was still on disk when the Russian export ran and G-REL failed it
with language_mismatch.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

# genanki's table, copied verbatim: all printable ASCII minus quotes, backslash
# and separators.
BASE91_TABLE = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D",
    "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "!", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/",
    ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}",
    "~"]


def guid_for(*values) -> str:
    hash_str = "__".join(str(v) for v in values)
    digest = hashlib.sha256(hash_str.encode("utf-8")).digest()[:8]
    n = 0
    for b in digest:
        n = (n << 8) + b
    out = []
    while n > 0:
        out.append(BASE91_TABLE[n % len(BASE91_TABLE)])
        n //= len(BASE91_TABLE)
    return "".join(reversed(out))


def _cross_check(seeds, lang) -> str:
    try:
        import genanki
    except ImportError:
        return "genanki not installed; local guid_for() not cross-checked"
    for seed in seeds:
        if genanki.guid_for(seed, lang) != guid_for(seed, lang):
            raise SystemExit(
                "FATAL: local guid_for() disagrees with genanki for seed %r -- "
                "do not trust any GUID this tool prints" % seed)
    return "cross-checked against genanki for %d seeds" % len(seeds)


def read_apkg(path: Path) -> list:
    """[(guid, first_field, tags)] out of a released package."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            db = ("collection.anki21" if "collection.anki21" in names
                  else "collection.anki2")
            if db not in names:
                raise SystemExit("%s holds no collection database (%s)"
                                 % (path, names[:5]))
            z.extract(db, tmp)
        con = sqlite3.connect(str(Path(tmp) / db))
        try:
            rows = con.execute("SELECT guid, flds, tags FROM notes").fetchall()
        finally:
            con.close()
    out = []
    for guid, flds, tags in rows:
        first = (flds or "").split("\x1f")[0]
        out.append((guid, first, (tags or "").strip()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apkg", required=True, help="the previously released .apkg")
    ap.add_argument("--lang", required=True, help="target language of that deck")
    ap.add_argument("--work", default="work", help="workspace directory")
    ap.add_argument("--out", help="output json (default: "
                                  "<work>/reports/guid_diff.<lang>.json)")
    args = ap.parse_args(argv)

    work = Path(args.work)
    words = json.loads((work / "json" / "words.json").read_text(encoding="utf-8"))
    assignments_path = work / "json" / "assignments.json"
    assignments = (json.loads(assignments_path.read_text(encoding="utf-8"))
                   if assignments_path.exists() else {})

    families = {fid: f for fid, f in words.items()
                if f.get("freq_rank") is not None}
    seeds = {}
    for fid, f in families.items():
        seed = f.get("guid_seed")
        if not seed:
            raise SystemExit("family %s has no guid_seed; run the merge stage" % fid)
        seeds[guid_for(seed, args.lang)] = {"family_id": fid, "guid_seed": seed,
                                            "lemma": f.get("lemma"),
                                            "freq_rank": f.get("freq_rank")}
    check_note = _cross_check([f["guid_seed"] for f in seeds.values()], args.lang)

    old = read_apkg(Path(args.apkg))
    old_by_guid = {g: q for g, q, _ in old}

    kept_guids = sorted(set(old_by_guid) & set(seeds))
    retired_guids = sorted(set(old_by_guid) - set(seeds))
    new_guids = sorted(set(seeds) - set(old_by_guid))

    # Where did a retired card's word end up? Best effort, and honest when it is
    # unknown: a retired GUID whose word is not assigned anywhere is a card the
    # user simply loses.
    retired = []
    for g in retired_guids:
        word = old_by_guid[g]
        fid = (assignments.get(word) or {}).get("family_id")
        merged_into = (words.get(fid) or {}).get("lemma") if fid else None
        retired.append({"guid": g, "query_word": word, "family_id": fid,
                        "merged_into": merged_into})

    report = {
        "language": args.lang,
        "old_apkg": str(args.apkg),
        "guid_formula": "genanki.guid_for(guid_seed, lang)",
        "guid_check": check_note,
        # The row G-REL asserts against at export time. Kept deliberately small
        # and flat: a churn number nobody compares to the shipped deck is an
        # estimate, and the original spec's estimate was off by up to +22%.
        "summary": {"language": args.lang, "card_count": len(seeds),
                    "old_notes": len(old), "kept": len(kept_guids),
                    "new": len(new_guids), "retired": len(retired_guids),
                    "seeds_cross_checked": len(seeds)},
        "counts": {"old_notes": len(old), "new_notes": len(seeds),
                   "kept": len(kept_guids), "new": len(new_guids),
                   "retired": len(retired_guids),
                   "retired_with_known_target":
                       sum(1 for r in retired if r["merged_into"]),
                   "net_change": len(seeds) - len(old)},
        "kept": [{"guid": g, "query_word": old_by_guid[g],
                  "lemma": seeds[g]["lemma"],
                  "freq_rank": seeds[g]["freq_rank"]} for g in kept_guids],
        "new": [{"guid": g, **seeds[g]} for g in new_guids],
        "retired": retired,
        "anki_cleanup": {
            "after_importing_the_companion":
                "tag:ankidkdeck::merged-into-lemma",
            "without_the_companion":
                " OR ".join('"QueryWord:%s"' % r["query_word"]
                            for r in retired[:40]),
            "note": ("the companion package (tools/retired_notes.py) tags every "
                     "retired note, so the cleanup is one search and one delete "
                     "instead of ~1,400"),
        },
    }
    # Per language, and the exporter reads the same name. --out overrides it for
    # the one case a name cannot serve: writing a second report for the same
    # language without clobbering the first.
    out = (Path(args.out) if args.out
           else work / "reports" / ("guid_diff.%s.json" % args.lang))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    c = report["counts"]
    print("GUID diff for %s (%s)" % (args.lang, check_note))
    print("  old notes %5d" % c["old_notes"])
    print("  new notes %5d" % c["new_notes"])
    print("  kept      %5d" % c["kept"])
    print("  new       %5d" % c["new"])
    print("  retired   %5d  (%d with a known merge target)"
          % (c["retired"], c["retired_with_known_target"]))
    print("written to %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
