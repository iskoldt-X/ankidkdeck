#!/usr/bin/env python3
"""T8: deep content comparison of two .apkg files (rebuilt vs release).

    python3 tools/compare_apkg.py <rebuilt.apkg> <released.apkg>

Parity dimensions:
  1. notes: count, guid set, per-field byte equality (8 named fields)
  2. model: id, name, field names, templates (qfmt/afmt), css, sortf
  3. deck:  id, name, description
  4. media: filename set from media manifest + per-file SHA256

Exit 0 if parity holds everywhere except explicitly whitelisted diffs
(deck description (c) year), else exit 1.

WHAT THIS IS AND IS NOT A PROOF OF. Ported verbatim from the sibling
danish_pipelines repo, where it is the standing proof that the v2.1 exporter did
not drift: run it against the `v2.1-pipeline` branch's output and full byte
parity is the expected result. It is NOT a gate v3 can satisfy, and nobody should
read "byte-parity harness" as something v3 passes -- v3 changed the card UNIT
from one query word to one word family, so the note count, the GUID set and every
field differ by design (guide D10). What v3 asserts instead is IDENTITY: deck id,
model id, the 8 field names, sortf and guid_for(seed, lang), pinned by
tests/test_export.py and, when the released decks are on the host, by
tests/test_export_parity.py.
"""
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

FIELD_NAMES = ["QueryWord", "FrontSideSummary", "Content", "Collocations",
               "Variants", "Derivatives", "Etymology", "FrequencyRank"]


def load_apkg(path):
    tmp = tempfile.mkdtemp(prefix="apkgcmp_")
    z = zipfile.ZipFile(path)
    # genanki writes collection.anki2; newer Anki exports carry anki21 as well,
    # and a hardcoded name made this tool unusable against those.
    names = z.namelist()
    db = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
    z.extract(db, tmp)
    media_manifest = json.loads(z.read("media").decode("utf-8"))
    # media hash by real filename
    media_hash = {}
    for num, fname in media_manifest.items():
        h = hashlib.sha256(z.read(num)).hexdigest()
        media_hash[fname] = h
    con = sqlite3.connect(str(Path(tmp) / db))
    cur = con.cursor()
    notes = {}
    for guid, flds in cur.execute("SELECT guid, flds FROM notes"):
        notes[guid] = flds.split("\x1f")
    models_json, decks_json = cur.execute(
        "SELECT models, decks FROM col").fetchone()
    con.close()
    models = json.loads(models_json)
    decks = json.loads(decks_json)
    # drop the default deck (id 1)
    decks = {k: v for k, v in decks.items() if k != "1"}
    return {"notes": notes, "models": models, "decks": decks,
            "media": media_hash}


def main(rebuilt_path, release_path):
    A = load_apkg(rebuilt_path)   # rebuilt
    B = load_apkg(release_path)   # release ground truth
    fail = False

    # --- notes ---
    print(f"notes: rebuilt={len(A['notes'])} release={len(B['notes'])}")
    ga, gb = set(A["notes"]), set(B["notes"])
    if ga != gb:
        fail = True
        print(f"  GUID MISMATCH: only-rebuilt={len(ga-gb)} only-release={len(gb-ga)}")
        for g in list(ga - gb)[:5]:
            print(f"    only-rebuilt guid {g}: {A['notes'][g][0]}")
        for g in list(gb - ga)[:5]:
            print(f"    only-release guid {g}: {B['notes'][g][0]}")
    else:
        print(f"  guid sets identical ({len(ga)})")

    per_field_diff = {f: 0 for f in FIELD_NAMES}
    diff_examples = []
    for g in ga & gb:
        fa, fb = A["notes"][g], B["notes"][g]
        for i, fname in enumerate(FIELD_NAMES):
            if fa[i] != fb[i]:
                per_field_diff[fname] += 1
                if len(diff_examples) < 5:
                    diff_examples.append((fa[0], fname, fa[i][:120], fb[i][:120]))
    total_diff_fields = sum(per_field_diff.values())
    if total_diff_fields == 0:
        print(f"  ALL {len(ga & gb)} notes x 8 fields BYTE-IDENTICAL")
    else:
        fail = True
        print(f"  field diffs: {per_field_diff}")
        for word, fname, va, vb in diff_examples:
            print(f"    [{word}].{fname}\n      rebuilt: {va!r}\n      release: {vb!r}")

    # --- model ---
    ma = list(A["models"].values())
    mb = list(B["models"].values())
    print(f"models: rebuilt={len(ma)} release={len(mb)}")
    if len(ma) == 1 and len(mb) == 1:
        m1, m2 = ma[0], mb[0]
        for key in ["id", "name", "sortf", "css"]:
            same = m1.get(key) == m2.get(key)
            print(f"  model.{key}: {'==' if same else 'DIFF'}"
                  + ("" if same else f"  rebuilt={str(m1.get(key))[:80]!r} release={str(m2.get(key))[:80]!r}"))
            if not same:
                fail = True
        fa = [f["name"] for f in m1["flds"]]
        fb = [f["name"] for f in m2["flds"]]
        print(f"  model.fields: {'==' if fa == fb else 'DIFF ' + str((fa, fb))}")
        if fa != fb:
            fail = True
        for part in ["qfmt", "afmt"]:
            t1 = m1["tmpls"][0][part]
            t2 = m2["tmpls"][0][part]
            same = t1 == t2
            print(f"  template.{part}: {'==' if same else 'DIFF'}")
            if not same:
                fail = True
                # show first differing region
                for i, (c1, c2) in enumerate(zip(t1, t2)):
                    if c1 != c2:
                        print(f"    first diff at char {i}: rebuilt={t1[i:i+60]!r} release={t2[i:i+60]!r}")
                        break
    else:
        fail = True

    # --- deck ---
    da = list(A["decks"].values())
    db = list(B["decks"].values())
    if len(da) == 1 and len(db) == 1:
        d1, d2 = da[0], db[0]
        for key in ["id", "name"]:
            same = d1.get(key) == d2.get(key)
            print(f"deck.{key}: {'==' if same else 'DIFF rebuilt=%r release=%r' % (d1.get(key), d2.get(key))}")
            if not same:
                fail = True
        desc1, desc2 = d1.get("desc", ""), d2.get("desc", "")
        if desc1 == desc2:
            print("deck.desc: ==")
        else:
            # whitelist: only difference should be the (c) year
            n1 = re.sub(r"20\d\d", "YYYY", desc1)
            n2 = re.sub(r"20\d\d", "YYYY", desc2)
            if n1 == n2:
                print("deck.desc: == modulo (c) year "
                      f"(rebuilt {re.findall(r'20..', desc1)} vs release {re.findall(r'20..', desc2)}) [whitelisted]")
            else:
                fail = True
                print("deck.desc: DIFF beyond year")
                for i, (c1, c2) in enumerate(zip(desc1, desc2)):
                    if c1 != c2:
                        print(f"  first diff at char {i}: rebuilt={desc1[i:i+80]!r} release={desc2[i:i+80]!r}")
                        break
    else:
        fail = True
        print(f"deck count mismatch: rebuilt={len(da)} release={len(db)}")

    # --- media ---
    fa_set, fb_set = set(A["media"]), set(B["media"])
    print(f"media: rebuilt={len(fa_set)} release={len(fb_set)}")
    if fa_set != fb_set:
        fail = True
        print(f"  filename sets DIFF: only-rebuilt={sorted(fa_set-fb_set)[:5]} only-release={sorted(fb_set-fa_set)[:5]}")
    else:
        bad = [f for f in fa_set if A["media"][f] != B["media"][f]]
        if bad:
            fail = True
            print(f"  {len(bad)} media files differ in content: {bad[:5]}")
        else:
            print(f"  all {len(fa_set)} media filenames AND sha256 identical")

    print("\nRESULT:", "PARITY FAILED" if fail else "FULL CONTENT PARITY (modulo whitelisted year)")
    return 1 if fail else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
