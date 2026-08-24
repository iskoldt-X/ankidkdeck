# G-IMPORT: the Anki import smoke test (human, required)

This is the highest-risk unverified assumption in the whole v3 plan. genanki only
*writes* a package -- nobody in the design or review chain has observed what
Anki's notetype merge actually does with a v3 package on top of a v2.1
collection. Until a human runs this checklist and signs it, the words
**"study progress is preserved"** must not appear in any release note, README,
or forum post.

Run it once per release, on at least one language (German or English are the
best cases: they have the largest carried set).

## Inputs

| what | where |
|---|---|
| the previously released deck | `DDO_Danish_Frequency_Deck_<LANG>.apkg` (v2.0/v2.1 release asset) |
| the new deck | `dist/DDO_Danish_Frequency_Deck_<LANG>.apkg` |
| the companion | `dist/DDO_Danish_v3_Retired_<LANG>.apkg` (`tools/retired_notes.py`) |
| the numbers to check against | `work/reports/registry_freeze_report.json`, `work/reports/guid_diff.json` |

Use a **fresh Anki profile** (File > Switch Profile > Add). Never run this on the
profile you study with.

## Procedure

1. **Baseline.** Fresh profile. Import the OLD deck.
   - Record: note count, deck name, notetype name, notetype id
     (Tools > Manage Note Types > the id is in the URL-ish footer, or
     Browse > right-click a note > Info).
   - Record the number of fields (must be 8) and the sort field (must be
     `FrequencyRank`).

2. **Study something.** Pick 5 cards whose `QueryWord` appears in the `kept`
   table of `guid_diff.json`. Prefer at least one case-sensitive one (`er`, `at`,
   `de`, `var`) because 55 pairs differ only by case and that is where a GUID
   change hides. Answer each card (Good / Hard, does not matter).
   - Record for each: QueryWord, due date, interval, reps.

3. **Import the new deck.** File > Import > the v3 `.apkg`. Read the import
   summary before clicking through it.

4. **Check, in this order.** Every line is pass/fail, no interpretation:

   - [ ] **Notetype id unchanged.** Same id as step 1. If it changed, every card
         was recreated and all progress is gone.
   - [ ] **No cloned notetype.** Manage Note Types shows exactly ONE
         `Danish Frequency Deck V2.1 (<LANG>)`. A second entry with a suffix
         (`... -1a2b3`) means Anki refused the merge and split the collection.
   - [ ] **The 5 studied cards kept their scheduling.** Same due date, same
         interval, same reps as recorded in step 2, and their revlog still shows
         the review (Browse > select the card > Card Info).
   - [ ] **Carried note count matches.** Notes that existed before and still
         exist == `kept` in `guid_diff.json` == `carried` in
         `registry_freeze_report.json`. Off-by-anything means the seed registry
         and the released deck disagree.
   - [ ] **Total note count matches.** `old_notes + new - retired` from
         `guid_diff.json` (retired notes are *updated*, not removed, so they are
         still counted).
   - [ ] **The new template is live.** Open a merged family card (a lemma with
         several homographs): the front shows the POS-grouped IPA with audio, the
         back shows numbered senses with translations, the `Variants` line shows
         the paradigm rows, and `Show more definitions...` folds/unfolds.
   - [ ] **Search finds a card by an inflected form.** e.g. search `huset` and
         land on the `hus` card (the hidden searchable-forms span).
   - [ ] **Audio plays** on at least three cards.
   - [ ] **No duplicate cards** for a word you studied in step 2 (Browse, search
         the QueryWord: exactly one note).

5. **Import the companion.** File > Import > `DDO_Danish_v3_Retired_<LANG>.apkg`.

   - [ ] Search `tag:ankidkdeck::merged-into-lemma` returns exactly the
         `retired` count from `guid_diff.json`.
   - [ ] One of those notes reads `This card was merged into <lemma>.` and the
         lemma is a real card in the deck.
   - [ ] Select all > Notes > Delete leaves the studied cards untouched
         (re-check one card from step 2 afterwards).

6. **Second import (idempotence).** Import the v3 deck again.
   - [ ] Note count unchanged, notetype count unchanged, the studied cards'
         due dates unchanged.

## Sign-off

Copy this block into the release notes / the build record; an unsigned gate is a
failed gate.

```
G-IMPORT
  build            : <git sha / build date>
  language         : <LANG>
  anki version     : <e.g. 25.02.4>
  old apkg         : <file + release tag>
  notetype id      : before <id> / after <id>
  cloned notetype  : no
  kept notes       : <n> (guid_diff: <n>)
  studied 5 cards  : due/ivl/reps preserved  yes
  retired tag count: <n> (guid_diff: <n>)
  verdict          : PASS / FAIL
  signed           : <name> <date>
```

## If it fails

Do **not** patch the deck and re-release. The failure modes and what they mean:

| symptom | cause | action |
|---|---|---|
| notetype cloned | field count/order or notetype id changed | stop; the 8-field schema and `MODEL_ID` are frozen for a reason -- find what moved |
| studied cards reset | GUID changed for a carried family | `guid_seed` was recomputed instead of read from `registry/card_keys.json`; G-SEED should have caught it |
| duplicate cards | GUID changed AND the old note kept | same as above |
| kept count too low | the registry overlay in `work/registry/card_keys.json` was never copied into the package registry | copy it, re-run merge, diff again |
| audio silent | media list not written, or a `[sound:]` with no file | G-MEDIA; check `work/audio/manifest.json` |
