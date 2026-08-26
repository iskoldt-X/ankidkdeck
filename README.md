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
ankidkdeck priority                          # homograph display order (no calls)

cp -a work work.before-translate             # see "the dry path is not read-only"
ankidkdeck doctor                            # what a spend would actually use

export GEMINI_API_KEYS="key1,key2"
ankidkdeck translate --lang German           # prints a BILL and stops
ankidkdeck translate --lang German --confirm-spend

ankidkdeck audio --seed-legacy               # reuse a previous run's mp3s
ankidkdeck export --lang German              # dist/DDO_Danish_Frequency_Deck_German.apkg
```

The order is `build -> priority -> translate -> audio -> export`, and `priority`
is not optional bookkeeping: `merge` and `priority` write the same `entry_ids`
field of `words.json`, so a `build` after a `priority` run silently restores
every homograph display order -- and `G-ORDER`, which only runs inside
`priority`, never says a word about it.

### The dry path is not read-only

`translate` and `priority` without `--confirm-spend` place no API call and import
nothing from the Gemini SDK. They are not, however, read-only:

* `translate` runs the archive GC and writes `definitions.json`,
  `expressions.json`, `archive.json`, `reports/translate_bill_<lang>.json` and
  `reports/translate_report.json`, and it runs `G-ORPH` -- which can fail the run
  before a human ever sees the bill;
* `priority` rewrites `words.json`, `priority_orders.json`,
  `priority_conflicts.json` and `ranking_queue.json`.

"Nothing was sent" is true. "Nothing changed" is not. Snapshot `work/` before the
first dry run of either. A confirmed `--mode batch` run adds `work/batch/` on top
of that -- the job registry, the uploaded JSONL and the downloaded results -- and
those files are the only record of what has been paid for, so they belong in the
same snapshot.

Two gates refuse before the first paid call rather than after it: `G-BUDGET`
(month-to-date plus this run's forecast against `spend_cap_usd` -- the only place
the languages are **summed**, since the bill's own ceiling check is per language)
and `G-SCOPE-FROZEN`. The second one refuses today by design: until the release
refreeze has happened and `registry/refreeze_stamp.json` signs for the scope and
the `card_keys.json` row count, paying to translate a scope that is about to
change is paying twice.

`ankidkdeck status` prints where you are; `ankidkdeck gates` prints every quality
gate and whether it passed; `ankidkdeck doctor` prints the configuration a paid
run would actually use. `ankidkdeck --help` lists every stage individually
(`wordlist`, `sitemap`, `parse`, `classify`, `resolve`, `merge`, `bind`,
`migrate`, `priority`, `review`).

Options worth knowing:

| option | why |
|---|---|
| `--work PATH` | where the corpus, JSON and reports live (default `./work`) |
| `--config PATH` | TOML config (default `./ankidkdeck.toml`): languages, model, sleep range, copyright year |
| `--legacy-workspace PATH` | a previous run's workspace, for reusing its translations, rankings and audio |
| `--confirm-spend` | the only way `translate`, `priority` and `review` place a paid API call |
| `--mode {standard,batch,flex}` | transport for one `translate` run. `batch` is half price and asynchronous; nothing downgrades by itself |
| `--retranslate-all` | bill (and with `--confirm-spend`, redo) every cell in scope, not only the missing and changed ones |
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
--- translation bill (model: gemini-3.7-flash) ---
  mode standard | thinking LOW | prompt v4-frozen
  German     571 cells  (definitions 571: 571 new / 0 changed / 0 redo | expressions 0: ...)
             412 entries, 43-1075 API requests, ~18k source tokens
  TOTAL 571 cells across 1 language(s)
  request ceiling includes transport retries; source tokens are the Danish payload only, not a bill
  nothing has been sent. Re-run with --confirm-spend to place calls.
```

`ankidkdeck doctor` prints the configuration a spend would actually use -- the
effective model, the transport, the thinking level, the prompt hashes, the spend
cap, and the state of the measured constants under `work/probes/` -- and exits
non-zero when the run is not fit to spend on. The two numbers in the request
range are "one request per batch" and every retry ladder at full stretch -- and
the second one is **per transport**, because the transports do not share a
ceiling. On the interactive surface it is the count-lock ladder times the
transport ladder inside `_generate` (25x, transport retries included). A batch
wave never goes through `_generate`, so its bound is the first submit plus the
retry waves (4x), enforced twice over: by the wave loop and by a per-cell attempt
counter that survives the process.

The bill also carries three dollar figures per language -- the cache working, the
cache failing with the current prompt inlined on every request, and the forbidden
case of an enriched prompt inlined -- next to the accepted ceiling. The tokens
behind them come from `work/probes/stats.json`; the prices come from the rate
card, and while that module is absent every figure reads `null` with the reason
stated rather than a plausible number.

`--retranslate-all` bills every cell in scope instead of the missing and changed
ones. With `--confirm-spend` it moves the existing rows into `archive.json` with
`reason=clean_redo` and does not read them back; without it, it only quotes the
price and touches nothing.

A clean retranslation is **resumable**, which matters because one language is
~5,565 requests over several hours:

* rows that already carry the running redo's provenance (same model, same prompt
  pack, same thinking level -- any date) are kept, are not billed again and are
  not re-archived, so re-running the same command after a crash pays for the
  remainder only, and the bill prints the arithmetic;
* a row the redo retired is **never** restored from `archive.json`. Before that,
  a plain `translate` after a crashed redo -- the natural "carry on" -- silently
  brought every old row home, reported 0 cells, and left two translators mixed
  into one file.

### The batch wave

`--mode batch` is half price and asynchronous. It is a human choice: nothing
downgrades a batch wave to the interactive surface by itself, because an
automatic downgrade doubles the rate silently.

One wave is one or two `--confirm-spend` invocations:

```
ankidkdeck translate --lang German --mode batch --phase submit --confirm-spend
ankidkdeck translate --lang German --mode batch --phase ingest --confirm-spend
ankidkdeck translate --lang German --mode batch --confirm-spend     # both at once
```

The submit creates the explicit cache, writes one JSONL file per job, uploads it,
opens the job and waits for it to drain -- **one job in flight at a time** --
downloading each result file the moment its job reaches a terminal state. It
writes no translation row. The ingest reconciles those files **by position**,
checks the count lock locally, writes the cells, and opens bounded retry waves
(at most three, counted per cell) for whatever failed. Retries stay on batch. The
drift ledger is consumed on the ingest only.

`--mode batch` is also what `cache_enabled` needs: the explicit cache is the only
discount path there is (batch has no implicit caching at all), and its lifecycle
-- create, assert and extend the TTL before **each** submit, recreate when it has
expired, delete once the language is drained -- is driven by the wave. On the
interactive surface `cache_enabled` is refused rather than quietly ignored.

The TTL is sized against the drain window this program actually waits for (the
poll deadline, floored at the documented 48-hour hard expiry, plus a submit
margin), not against a throughput extrapolation: batch resolves the cache when
each row *executes*, so a cache that dies mid-job fails every remaining row with
a permission error at zero prompt tokens. Holding a 1,135-token prompt for that
whole window costs about $0.03 per language.

Everything the wave knows lives under `work/batch/`:

| file | what |
|---|---|
| `work/batch/jobs.json` | one record per job: its state, its fingerprint, its cache, and the per-request plan the ingest reconciles against |
| `work/batch/<job>.jsonl` | exactly what was uploaded |
| `work/batch/<job>_results.jsonl` | exactly what came back |

A job is submitted once. Its fingerprint (model, prompt id, language, keys,
cache, wave) is written **before** the API call, because `batches.create` is not
idempotent: the same job submitted twice is accepted twice, runs twice and is
billed twice. If a create fails with a 5xx the wave does not retry it -- it lists
the jobs, matches the uploaded input file, and adopts the one that already
exists.

**A drain that dies is resumed, never resubmitted.** `--phase submit` can be a
foreground call for a day or more, so run it under `tmux` or `nohup`. If it dies
anyway -- a dropped connection, a reboot, the 50-hour wait running out -- the
next invocation of either phase picks up where it stopped, before it plans
anything new:

| left as | what happens next run |
|---|---|
| `SUBMITTED` | polled to a terminal state and downloaded. The money was already committed; this is the results being collected, not a second order |
| `PLANNED` with an uploaded file | the jobs are listed and the one matching that input file is **adopted**. If none matches, the run stops and says so: the create may have been accepted and the answer lost, and guessing there is a duplicate charge |
| `PLANNED` with nothing uploaded | released. Nothing was sent, so nothing was billed, and the wave plans it again |

Because that sweep runs first, **one job in flight at a time** holds across
invocations too, not only inside one. The cache is kept alive while any job of
that language is still in flight: batch resolves the cache when each row
*executes*, so deleting it early fails every remaining row of a job that was
otherwise recoverable.

A job that ends terminal with no result file is recorded as `FAILED`, and the
record says whether the wave may be submitted again. The service reporting
`FAILED`, `CANCELLED` or `EXPIRED` means nothing ran and nothing was billed, so
it may. "We could not read the results of a job that succeeded" means the money
is spent, so it may not -- that one needs a human and the console.

### What a paid run leaves behind

Every spend record is on disk before anything can fail, because a crash is the
only time it matters:

| file | when |
|---|---|
| `reports/translate_usage.jsonl` | one line per call, flushed and fsync'd as it happens |
| `reports/translate_usage.json` | the same rows as an array, on the way out (crash or not) |
| `review/count_lock_violations_<lang>.json` | as each violation happens |
| `reports/translate_report.json` | always, with a `crashed` block on a fatal path |
| `reports/bind_report_pre_translate.json` | once, before the first paid call: the pre-LLM migration audit |
| `work/batch/jobs.json` | batch only: every job's state, written before each API call |

Five paid calls followed by a crash leave five rows. `priority` writes the same
kind of per-call ledger to `reports/priority_usage.jsonl`.

`ankidkdeck review --lang German --fix <keys>` is the hand-run correction pass.
Nothing triggers it automatically: its work list is a file under `work/review/`,
and one invocation is one pass.

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
  demoted, gate baselines, the part-of-speech label table). No DDO text.
  `pos_translations.json` is what makes an `.apkg` buildable with no LLM call at
  all: the ~20 `data-pos-key` values in four languages are hand-written
  grammatical vocabulary, not machine output. A `pos_key` no source covers fails
  `G-COV` loudly instead of shipping an unlabelled group, and
  `work/json/translations/<lang>/pos.json` still overrides any row.
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
is gitignored, so it can never be mistaken for a released baseline. Nested
objects are merged one level deeper, so an overlay naming a single
`empty_rate_baseline_pct` field keeps the other four.

Gate rows are keyed on `(id, stage, scope)`, and the export-time gates carry the
language as their scope. A passing German export therefore cannot overwrite a
failing Chinese row, and `ankidkdeck gates` exits non-zero if ANY recorded row
fails.

**`export` needs the test fixtures.** `G-SEP` re-runs the golden separator
comparison at export time -- a one-character drift in the extraction table
silently invalidates every existing translation cell, and its symptom is a big
translation bill rather than a blocked build. "Fixtures unavailable" is recorded
as a FAILURE, not skipped: point `ANKIDKDECK_FIXTURES` at them, or keep them in
`work/fixtures`.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests               # pure-logic tests, no data needed
ANKIDKDECK_FIXTURES=work/fixtures python -m pytest tests   # + golden tests
```

The golden tests need saved DDO pages, which cannot be committed; build them on
the machine that holds the corpus with `tools/build_fixtures.py`. Without them
those modules skip and say so. The builder takes either directories of
human-named pages (`--pages`) or the pipeline's own crawl corpus
(`--work <workspace>`, reading `raw/<sha1>.html` plus `json/fetch_ledger.json`),
which is how the fixture set grows past a handful of hand-saved pages.

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
