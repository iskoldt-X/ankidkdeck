#!/usr/bin/env python3
"""T6: the companion package that tells a user which cards were merged away.

A v3 import upgrades every carried GUID in place and adds the new ones, but the
notes whose GUID no longer exists just sit there forever -- the user cannot tell
a retired duplicate from a card they still need. This writes a second .apkg that
re-uses the SAME notetype and the SAME GUIDs, so importing it OVERWRITES each
retired note in place with one sentence and one tag:

    Content       = "This card was merged into <lemma>."
    tag           = ankidkdeck::merged-into-lemma
    FrequencyRank = "0"   the sort field, identical on every retired note, so
                          they arrive as one contiguous block in the browser
                          instead of scattering through the live frequency order

Release notes then say: import the companion, search
tag:ankidkdeck::merged-into-lemma, select all, delete. One click instead of
~1,400.

    python3 tools/guid_diff.py --apkg <old>.apkg --lang German --work work
    python3 tools/retired_notes.py --lang German --work work

Requires genanki (the notetype must be byte-identical to the main package, so it
is built by importing the exporter, never re-declared here).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

TAG = "ankidkdeck::merged-into-lemma"
UNKNOWN_TARGET = "This card was merged into another card."
# The sort-field value every retired note carries. See the comment at the write
# site: one constant, ahead of every live rank, so the retired block is
# contiguous in the browser.
RETIRED_RANK = "0"


def main(argv=None) -> int:
    import genanki

    from ankidkdeck.config import load_config
    from ankidkdeck.stages.s70_export import (FIELD_NAMES, build_model,
                                              deck_meta, guid_diff_language,
                                              lang_hash)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--diff", help="guid_diff.<lang>.json (default: "
                                   "<work>/reports/)")
    ap.add_argument("--out", help="output .apkg path")
    ap.add_argument("--config", help="ankidkdeck.toml")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None,
                      work_dir=Path(args.work))
    reports = Path(args.work) / "reports"
    per_lang = reports / ("guid_diff.%s.json" % args.lang)
    # The per-language name tools/guid_diff.py writes, then the unsuffixed name
    # it used to write. The language check below is what makes reading the
    # legacy file safe: a report for another language is refused, not used --
    # and it is s70_export's check, imported rather than restated, because the
    # two tools reading one file used to disagree about which language it was
    # about (this one read the top-level key, the exporter read summary.language).
    diff_path = (Path(args.diff) if args.diff
                 else (per_lang if per_lang.exists()
                       else reports / "guid_diff.json"))
    remedy = ("run tools/guid_diff.py --apkg <released>.apkg --lang %s --work "
              "%s, which writes %s" % (args.lang, args.work, per_lang))
    if not diff_path.exists():
        raise SystemExit("no %s -- %s" % (diff_path, remedy))
    diff = json.loads(diff_path.read_text(encoding="utf-8"))
    says = guid_diff_language(diff)
    if says != args.lang:
        raise SystemExit("%s describes %r, not %r -- %s"
                         % (diff_path.name, says, args.lang, remedy))
    retired = diff.get("retired") or []
    if not retired:
        print("nothing retired for %s; no companion package needed" % args.lang)
        return 0

    meta = deck_meta(args.lang, cfg.copyright_year)
    model = build_model(args.lang, cfg.copyright_year)
    # A subdeck of the user's own deck: on a fresh profile the notes land
    # somewhere obvious, and on an existing profile Anki updates the notes in
    # place (an import never moves the cards of a note it already has).
    deck = genanki.Deck(0x30000000 + lang_hash(args.lang),
                        "%s::Retired (merged into other cards)"
                        % meta["deck_name"],
                        description="Retired duplicates from the v3 dedup. "
                                    "Search tag:%s, select all, delete." % TAG)
    n_known = 0
    for row in sorted(retired, key=lambda r: r["guid"]):
        lemma = row.get("merged_into")
        if lemma:
            content = "This card was merged into %s." % lemma
            n_known += 1
        else:
            content = UNKNOWN_TARGET
        fields = [""] * len(FIELD_NAMES)
        fields[0] = row.get("query_word") or ""
        fields[2] = content
        # FrequencyRank is the SORT FIELD (sort_field_index = 7), and Anki stores
        # it with SQLite integer affinity. Leaving it blank gave every retired
        # note an empty sort column, so they scattered through the browser's
        # frequency ordering instead of collecting in one block -- which is the
        # opposite of what a package whose entire purpose is "select all, delete"
        # wants. RETIRED_RANK is a constant, not a real rank: 0 sorts ahead of
        # every live card (1..N) and is identical on every retired note, so they
        # arrive as one contiguous run. It is also visibly not a rank, so nobody
        # mistakes a retired card for card number 0.
        fields[FIELD_NAMES.index("FrequencyRank")] = RETIRED_RANK
        deck.add_note(genanki.Note(model=model, fields=fields,
                                   guid=row["guid"], tags=[TAG]))

    out = Path(args.out) if args.out else Path(
        "dist") / ("DDO_Danish_v3_Retired_%s.apkg" % args.lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    pkg = genanki.Package(deck)
    pkg.media_files = []          # a retired note never carries audio
    pkg.write_to_file(str(out))
    print("wrote %s" % out)
    print("  retired notes %d (%d name their merge target)"
          % (len(retired), n_known))
    print("  sort field    FrequencyRank = %r on every note, so they sort "
          "together" % RETIRED_RANK)
    print("  notetype id   %d (identical to the main package)" % model.model_id)
    print("  user cleanup  search 'tag:%s' -> select all -> delete" % TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
