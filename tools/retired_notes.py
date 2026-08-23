#!/usr/bin/env python3
"""T6: the companion package that tells a user which cards were merged away.

A v3 import upgrades every carried GUID in place and adds the new ones, but the
notes whose GUID no longer exists just sit there forever -- the user cannot tell
a retired duplicate from a card they still need. This writes a second .apkg that
re-uses the SAME notetype and the SAME GUIDs, so importing it OVERWRITES each
retired note in place with one sentence and one tag:

    Content = "This card was merged into <lemma>."
    tag     = ankidkdeck::merged-into-lemma

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


def main(argv=None) -> int:
    import genanki

    from ankidkdeck.config import load_config
    from ankidkdeck.stages.s70_export import (FIELD_NAMES, build_model,
                                              deck_meta, lang_hash)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--diff", help="guid_diff.json (default: <work>/reports/)")
    ap.add_argument("--out", help="output .apkg path")
    ap.add_argument("--config", help="ankidkdeck.toml")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None,
                      work_dir=Path(args.work))
    diff_path = (Path(args.diff) if args.diff
                 else Path(args.work) / "reports" / "guid_diff.json")
    if not diff_path.exists():
        raise SystemExit("no %s -- run tools/guid_diff.py first" % diff_path)
    diff = json.loads(diff_path.read_text(encoding="utf-8"))
    if diff.get("language") != args.lang:
        raise SystemExit("guid_diff.json is for %r, not %r"
                         % (diff.get("language"), args.lang))
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
    print("  notetype id   %d (identical to the main package)" % model.model_id)
    print("  user cleanup  search 'tag:%s' -> select all -> delete" % TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
