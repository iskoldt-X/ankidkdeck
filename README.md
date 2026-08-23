# ankidkdeck

Build a Danish frequency Anki deck from [Den Danske Ordbog](https://ordnet.dk)
(DDO): one card per word family, with IPA, audio, numbered senses, example
sentences, fixed expressions, inflection tables, derivatives, etymology and
AI-assisted translations into your target language.

This is **v3**. The published v2.0 decks (Chinese, English, German, Spanish) were
built with the v2.1 script pipeline, which is still on the `v2.1-pipeline`
branch. v3 is a proper Python package with one CLI, and it is designed so that
re-importing a rebuilt deck **upgrades your existing cards instead of duplicating
them** -- see [Study progress](#study-progress).

## Install

```bash
uvx ankidkdeck --help                      # run without installing
pip install "ankidkdeck[llm]"              # or install it, with the Gemini extra
git clone https://github.com/iskoldt-X/ankidkdeck && pip install -e ".[llm,dev]"
```

Python 3.11+. The `llm` extra is only needed for the translation stages; crawling
and exporting work without it.

## The four commands

Everything is written under `./work` (add `--work PATH` to move it).

```bash
ankidkdeck crawl --pilot                     # 300 pages first, then:
ankidkdeck crawl --full                      # ~5,000 pages, 4-5 h, resumable
ankidkdeck crawl --phase-b                   # lemma pages the inflections need

ankidkdeck build                             # parse -> classify -> resolve -> merge -> bind

export GEMINI_API_KEYS="key1,key2"
ankidkdeck translate --lang German           # prints a BILL and stops
ankidkdeck translate --lang German --confirm-spend

ankidkdeck audio --seed-legacy               # reuse a previous run's mp3s
ankidkdeck export --lang German              # dist/DDO_Danish_Frequency_Deck_German.apkg
```

`ankidkdeck status` prints where you are; `ankidkdeck gates` prints every quality
gate and whether it passed. `ankidkdeck --help` lists every stage individually
(`wordlist`, `sitemap`, `parse`, `classify`, `resolve`, `merge`, `bind`,
`migrate`, `priority`).

Options worth knowing:

| option | why |
|---|---|
| `--work PATH` | where the corpus, JSON and reports live (default `./work`) |
| `--config PATH` | TOML config (default `./ankidkdeck.toml`): languages, model, sleep range, copyright year |
| `--legacy-workspace PATH` | a previous run's workspace, for reusing its translations, rankings and audio |
| `--confirm-spend` | the only way `translate` and `priority` place a paid API call |
| `--check-determinism` | build twice and compare, before writing the `.apkg` |

## You need a residential IP

DDO sits behind a WAF that challenges datacenter addresses. From a VPS or a
cloud runner the crawl gets challenged, and the crawler treats a challenge as
fatal on purpose -- it never retries into a block. Run it from a home
connection.

The crawl is polite by design: one request at a time, 2-4 s apart, an honest
User-Agent, no parallelism, ~5,000 requests once per release. Every request is
checkpointed, so a stopped run resumes where it left off instead of starting
over.

## Translations

Translation is the only step that costs money, so it is the only step that asks:

```
$ ankidkdeck translate --lang German
--- translation bill (model: gemini-2.0-flash) ---
  German     571 cells  (definitions 571: 571 new / 0 changed | expressions 0: ...)
             412 entries, 43-43 API requests, ~18k source tokens
  TOTAL 571 cells across 1 language(s)
  nothing has been sent. Re-run with --confirm-spend to place calls.
```

Without `--confirm-spend` nothing is imported from the Gemini SDK and no request
is made. Keys come from `GEMINI_API_KEYS` (comma-separated; the pool rotates
every `MAX_PER_API` requests, default 5). Only missing cells are translated: if
you point `--legacy-workspace` at an earlier run, its translations are re-keyed
and reused, and you pay for the gap only.

The card's target-language text is machine-translated and marked as such.

## Study progress

Anki matches an imported note to an existing one by its GUID. v3 keeps every
card's GUID stable by never deriving it from data at build time: the seed lives
in a checked-in, append-only registry (`src/ankidkdeck/registry/card_keys.json`),
is reviewed once per major version, and is never recomputed. That matters
because `guid_for()` hashes the exact bytes, and 55 word pairs in this deck
differ only by case (`Er` the erbium symbol vs `er` the verb form, at rank 1).

v3 also merges inflections into one card per family (~2,900 cards instead of
~4,400 notes), so some old cards become duplicates. `tools/guid_diff.py` prints
exactly which GUIDs are kept, new and retired, and `tools/retired_notes.py`
writes a companion package that tags the retired ones so you can delete them
with one search.

**Not yet verified:** the deck's behaviour on import is checked by a human
against a real Anki, following `tools/import_smoke_test.md`. Until that
checklist is signed for a build, treat "progress is preserved" as untested.

## Data policy

This repository contains **code only**.

- DDO content -- pages, definitions, examples, audio -- stays on your machine.
  Everything the pipeline downloads or derives lands in `work/` and `dist/`,
  which are gitignored, and so are the test fixtures built from it.
- The tracked exception is `src/ankidkdeck/registry/*.json`: word identities and
  hand-curated rules (which lemma a form belongs to, which parts of speech are
  demoted, gate baselines). No DDO text.
- The decks you build are for your own study. Do not redistribute them. You are
  responsible for complying with DDO's
  [terms of use](https://ordnet.dk/copyright), and the frequency list comes from
  [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist)
  under CC BY-SA 3.0.

## Quality gates

Every stage ends in named gates, and a failed gate means no output -- there is no
warning-and-continue path. They exist because each one is a defect that shipped
once: bare untranslated cards (`G-COV`), an unreproducible media order
(`G-MEDIA`), a silently re-anchored GUID (`G-SEED`), empty card bodies
(`G-EMPTY-C`), a build that differs from itself (`G-DET`). `ankidkdeck gates`
prints the standing record. Two gates are human: reading
`work/review/rejected.json` and the Anki import smoke test.

Gate thresholds are data, not code: they live in
`src/ankidkdeck/registry/gates.json`. A run may override any of them by dropping
a partial `work/registry/gates.json` with just the keys it wants changed -- the
two are merged, packaged defaults first. That is what makes a **partial build**
possible without editing the checked-in baselines: building 200 words instead of
5,000 legitimately fails `G-NOTE` (`note_count_range`) and `G-MEDIA`
(`media_floor`), so a dev overlay such as

```json
{ "note_count_range": [1, 100000], "media_floor": 0 }
```

lets the rest of the gates do their job. The overlay lives under `work/`, which
is gitignored, so it can never be mistaken for a released baseline.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests               # pure-logic tests, no data needed
ANKIDKDECK_FIXTURES=work/fixtures python -m pytest tests   # + golden tests
```

The golden tests need saved DDO pages, which cannot be committed; build them on
the machine that holds the corpus with `tools/build_fixtures.py`. Without them
those modules skip and say so.

## Upgrading from a previous run

If you have a workspace from an earlier version, `ankidkdeck migrate
--legacy-workspace <path>` re-keys its translations onto the current sense ids
offline (no network, no API calls), records every dropped row with a reason code,
and lets `priority` and `audio` reuse its homograph rankings and mp3s. Run
`migrate` before `merge`, so the GUID registry can be frozen from the words that
actually shipped.

## License

MIT -- see [LICENSE](LICENSE). Code only; the data is not ours to license.

## Acknowledgments

Den Danske Ordbog (definitions, IPA, audio -- personal use), Wiktionary
(frequency list, CC BY-SA 3.0), Google Gemini (translations).
