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

Not on PyPI, and not going there. The wheel is `src/ankidkdeck` and nothing else
(`pyproject.toml`), while much of what this README tells you to run lives outside
it: `tools/` holds the GUID diff, the retired-notes companion, the fixture
builder and the import checklist, and the golden tests need `tests/`. A clone is
the install.

```bash
git clone https://github.com/iskoldt-X/ankidkdeck
cd ankidkdeck && pip install -e ".[llm,dev]"
```

Python 3.11+. There are two extras: `llm` (`google-genai`) and `dev` (`pytest`).
`llm` is only needed for the translation stages -- the crawl and the build must
work with no Gemini dependency present at all, which is why it is optional in the
first place.

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
writes no translation row. The ingest reconciles those files **by key** -- the
`key` echoed on each result line, never the line's position -- checks the count
lock locally, writes the cells, and opens bounded retry waves (at most three,
counted per cell) for whatever failed. Retries stay on batch. The drift ledger is
consumed on the ingest only.

The documented "responses will be written in the same order as the input
requests" is untrue, and untrue in the one way no probe could catch: the service
completes a job in ~1,000-row shards and concatenates the shards out of order,
each shard internally in input order. Every probe wave was 32 rows or fewer, i.e.
a single shard, where the order does hold -- so the guarantee was unfalsified
rather than confirmed. The first real wave broke it. The Chinese definition job
agreed on positions 0-999 and diverged at exactly row 1000; the Russian
definition wave agreed on 0 of 3,644 positions across 3 shards, its expression
wave on 0 of 1,923 across 2. Where the permutation happens to start is
incidental, which is the point: it is not a prefix you can rely on. So the key
bijection is the guard, and it is absolute -- a duplicate key, a key that is not
in the plan, or a planned key with no result is FATAL, with no partial-credit
reading -- while the position is still computed and reported as a cross-check
nothing is gated on. A positional write under a shard permutation shifts glosses
onto the wrong senses, which is both catastrophic and invisible.

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

## Adding a target language

A target language is one word in `langs` in `ankidkdeck.toml`. The packaged
default is the four v2.0 languages; the run host that shipped Russian reads
`langs = ["Chinese", "English", "German", "Spanish", "Russian"]`. `--lang` must
then name a configured language **exactly**, case included, and `check_lang`
refuses anything else before a stage runs: none of the migrated cells are visible
under a lowercase key, so `translate --lang german --confirm-spend` would quietly
pay the full from-scratch price and write to `translations/german/`. Export is
protected by `G-COV`; the money is not.

Nothing else is a hand-prepared file, for a Latin-script target. A language with
no prompt pack under `src/ankidkdeck/registry/prompt_packs/` still runs: the
builder serves the frozen LEAN prompt, and the POS labels
`registry/pos_translations.json` does not cover are translated by one paid call
(see [Data policy](#data-policy)). Caching, the JSONL and the wave splitting are
unaffected. A missing pack is a normal state, never an error. A non-Latin target
is the one exception, and it needs its script known to the gate -- see below.

That LEAN prompt is `v4-frozen`, the default for every language, and it is the
default on purpose rather than because RICH is unfinished: a blind pairwise
comparison of 233 Chinese senses, judged twice and independently (2026-08-30),
went to LEAN, and every fatal mistranslation it found was on the RICH side.
Changing `prompt_id` also refuses every `--confirm-spend` on its own, until the
thinking constant has been re-measured on the prompt you mean to spend on: the
constant that prices a wave is only valid for the prompt it was measured on.

**The one number you have to supply is the prompt size.**
`PROMPT_TOKENS_system_only.<lang>` in `work/probes/stats.json` is what the wave
splitter subtracts to size the uncached payload, and what the cached half of the
bill is priced from. Nothing in the package has a default for it and the call
ledger cannot recompute it, so `ankidkdeck doctor` exits non-zero when a
configured language has no usable entry. It used to print "fit to spend" and
leave the refusal to the wave splitter, which arrives with the scope frozen and
the operator already committed. Measure it with the probe tooling, or DECLARE it:

```bash
python tools/backfill_probe_stats.py --probes work/probes \
    --declare-system-tokens Russian=1135 \
    --declaration-basis "why this number is believed" --write
```

A declaration may only fill a gap -- an existing value is refused, measured or
declared, whoever wrote it -- it needs a basis in your own words, it changes
nothing without `--write`, and it lands in the artifact's change log with the
words "declared, not measured". That was acceptable for Russian on two grounds:
its definition prompt is byte-identical to the Spanish one once the language name
is substituted (both 5,160 characters), and Spanish, Chinese and German each
measure 1,135; and the number is not what the cache is checked against. `G-CACHE`
compares every row's `cachedContentTokenCount` to the size the live
`caches.create` response reported, per row, so a declared figure only sizes the
wave and quotes the bill. Do not generalise that to "the language name is the
only variable": English measures 1,092, because its core carries no CRITICAL RULE
block.

**A non-Latin target needs its own script known to the gate.** `G-SCRIPT` derives
its forbidden set from the pack plus `_SCRIPTS_BY_LANGUAGE` in
`src/ankidkdeck/gates.py` (today: `russian -> cyrillic`). With neither an entry
there nor a pack naming the script, Cyrillic is simply the first block in the
table and every cell of the first wave is a BLOCK-tier `forbidden_script` at
ingest, with the money already spent. Above that the classes split by tier: a
Latin letter welded inside a target-script word (`script_weld_in_lemma`) is
BLOCK, because that is the homoglyph and dropped-stem family and two such lemmas
came back from the first Russian wave; separated Latin -- cross-references,
acronyms (`latin_in_lemma`) -- is REVIEW, and so is the one weld shape the rule
exempts, a Latin acronym or brand carrying the target's own ending (SMS plus an
instrumental ending, PDF plus a diminutive, iPhone, USB: two groups, the Latin
one first, two characters or more, at least one capital). Glosses get the weld
check at REVIEW (`script_weld_in_gloss`), because the alternative -- any Latin
letter in a gloss -- flagged 199 cells that are Danish headwords, Latin binomials
and letter names, against the one that is a real defect.

For a first release there is nothing to migrate and `G-REL` records `N/A` (see
[Study progress](#study-progress)). Note that the deck id and the notetype id are
derived from the language name -- `0x10000000 + adler32(lang)` and
`0x20000000 + adler32(lang)`, the Chinese deck id being a legacy override -- so a
new name mints a brand-new deck rather than upgrading anything, and a typo mints
one too. The deck name, description and footer use the English template for every
language except Chinese.

What it cost in practice, measured on one run: Russian, 22,288 cells, 5,568
requests quoted and 5,578 placed (the extra ten were one retry wave for the
definition rows that finished on the output cap), about 54 minutes from ignition
to drained, and $3.09 against a $3.45 quote at the conservative cached-input rate
-- that figure being the month's whole ledger, two sub-cent correction calls
included. One run is not a rate card.

## Study progress

Anki matches an imported note to an existing one by its GUID. v3 keeps every
card's GUID stable by never deriving it from data at build time: the seed lives
in a checked-in, append-only registry (`src/ankidkdeck/registry/card_keys.json`),
is reviewed once per major version, and is never recomputed. That matters
because `guid_for()` hashes the exact bytes, and 55 word pairs in this deck
differ only by case (`Er` the erbium symbol vs `er` the verb form, at rank 1).

v3 also merges inflections into one card per family (~2,900 cards instead of
~4,400 notes), so some old cards become duplicates. `tools/guid_diff.py` prints
exactly which GUIDs are kept, new and retired, and writes
`reports/guid_diff.<lang>.json`; `tools/retired_notes.py` reads that same
per-language file and writes a companion package that tags the retired ones so
you can delete them with one search. The report is per language because one
release month ships more than one, and a single `guid_diff.json` is then a file
about whichever language was diffed last: the Chinese month's copy was still on
disk when the Russian export ran, and `G-REL` failed the Russian deck with
`language_mismatch`.

`G-REL` reads that file at export time and asserts it describes this language and
the same card count the exporter is about to write. When no usable report for
this language was read -- none on disk, or only one that describes another
language -- the row is recorded as NOT APPLICABLE, printed `N/A` by `ankidkdeck
gates`, carrying "nothing was checked" in its own detail and never printed as a
pass. It says only that, and not "there is no previous release", because a
function that has skipped a candidate file cannot make a claim about the disk.
Recorded rather than skipped, because gate rows merge on `(id, stage, scope)` and
are never pruned: a row that is not written cannot replace a stale failure for
the same scope, and that is exactly how a `G-REL[lang=Russian]` failure sat on
file for a language the gate can never run against.

**Verified for the Chinese v3 build, 2026-08-29.** The deck's import behaviour is
checked by a human against a real Anki, following `tools/import_smoke_test.md`,
and that run signed it: 2,921 notes found = 2 new (the two families `guid_diff`
lists as new) + 2,919 updated in place, with a 300-card review queue still due
afterwards. The checklist is what a human signs per build, not once for the
project. A brand-new language does not raise the question at all -- Russian's
first release, 2026-09-03, had no existing cards to preserve, every note was new,
and `G-REL` recorded `N/A`.

## Data policy

This repository contains **code only**.

- DDO content -- pages, definitions, examples, audio -- stays on your machine.
  Everything the pipeline downloads or derives lands in `work/` and `dist/`,
  which are gitignored, and so are the test fixtures built from it.
- The tracked exception is `src/ankidkdeck/registry/*.json`: word identities and
  hand-curated rules (which lemma a form belongs to, which parts of speech are
  demoted, gate baselines, the part-of-speech label table). No DDO text.
  `pos_translations.json` is what makes an `.apkg` buildable with no LLM call at
  all: the 25 `data-pos-key` values in four languages are hand-written
  grammatical vocabulary, not machine output. A `pos_key` no source covers fails
  `G-COV` loudly instead of shipping an unlabelled group, and
  `work/json/translations/<lang>/pos.json` still overrides any row. A language
  the table does not cover is not blocked by that: stage 42 asks for its labels
  in one paid call, writes them to `work/json/translations/<lang>/pos.json`, and
  `G-COV` accepts them from there -- which is what keeps adding a language a
  no-registry-edit operation. A `pos_key` the checked-in table already covers is
  never billed.
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
