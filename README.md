# ankidkdeck

A pipeline of Python scripts that builds high-quality Danish frequency Anki decks
(~4,400 words from the top 5,000 of the Wiktionary Danish frequency list) with
IPA, audio, definitions, example sentences, fixed expressions, derivatives,
etymology, and AI-generated translations into a target language.

This is the **V2.1 pipeline** -- the exact code that built the decks published in
the [v2.0 release](https://github.com/iskoldt-X/ankidkdeck/releases/tag/v2.0)
(Chinese, English, German, Spanish). The earlier v1 scripts have been removed
from HEAD and remain available in the git history.

## Pipeline overview

| Stage | Script | Input -> Output |
|---|---|---|
| 1 | `01_download_all_ddo_versions.py` | Wiktionary frequency list -> `ddo_html_all_versions/` + `download_map.json` (discovers and downloads every homograph page per word from Den Danske Ordbog) |
| 2 | `02_generate_entries.py` | HTML corpus -> `ddo_entries.json` (structured entries: definitions, examples, IPA/audio links, wordforms, etymology, fixed expressions, derivatives) |
| 3 | `03_rank_homographs.py` | `ddo_entries.json` -> `priority_map.json` (Gemini ranks homograph meanings from most to least common) |
| 4 | `04_translate_pos.py` | POS tags -> `pos_translations_<LANG>_gemini.json` |
| 5 | `05_translate_definitions.py` | definitions -> `definition_translations_<LANG>_gemini-2.0-flash.json` (lemma + gloss per definition, resumable) |
| 6 | `06_translate_expressions.py` | fixed expressions -> `expression_translations<LANG>_gemini-2.0-flash.json` (batched, with a generate-review-correct loop against language contamination) |
| 7 | `07_review_translations.py` | any translation file -> `review_issues_*.json` (post-hoc AI review report for manual inspection) |
| 8 | `08_download_audio.py` | `ddo_entries.json` -> `audio/` + `audio_map.json` |
| 9 | `09_export_apkg.py` | everything above -> `DDO_Danish_Frequency_Deck_<LANG>.apkg` |

Stages 1, 5, 6, and 8 are resumable: they checkpoint progress to their output
files and skip already-processed items on restart.

## Prerequisites

- **Python 3.12+**
- Install required packages:
  ```bash
  pip install requests beautifulsoup4 tqdm genanki google-genai google-api-core tiktoken
  ```
- Gemini API key(s) for stages 3-7, provided via environment variable
  (comma-separated to rotate across a pool of keys):
  ```bash
  export GEMINI_API_KEYS="key1,key2,..."
  export MAX_PER_API=5   # optional: requests per key before rotating
  ```

## Usage

Before downloading content from Den Danske Ordbog, please read and respect
their [terms of use](https://ordnet.dk/copyright).

Set `TARGET_LANG` at the top of `04_translate_pos.py`,
`05_translate_definitions.py`, `06_translate_expressions.py`, and
`09_export_apkg.py` to your target language (e.g. `"English"`, `"Chinese"`),
then run the stages in order:

```bash
python 01_download_all_ddo_versions.py
python 02_generate_entries.py
python 03_rank_homographs.py
python 04_translate_pos.py
python 05_translate_definitions.py
python 06_translate_expressions.py
python 07_review_translations.py   # optional quality check
python 08_download_audio.py
python 09_export_apkg.py
```

Stage 9 produces an `.apkg` file ready to import into Anki.

## Deck design notes

- One card per word: homograph entries are merged into a single note, ordered
  by the commonness ranking from stage 3.
- Stable identities: deck and model IDs are derived from `adler32(TARGET_LANG)`,
  and note GUIDs from `(query_word, TARGET_LANG)`, so re-imports of rebuilt
  decks preserve study progress.
- Known issue: the GUID scheme changed between v1 and v2 decks, so importing a
  v2 deck alongside an old v1 deck creates duplicate cards instead of upgrading
  them in place. A migration path is being worked on.

## Disclaimer

This repository contains code only. Content downloaded from Den Danske Ordbog
(definitions, examples, audio) stays on your machine, is ignored by git, and
must not be redistributed -- the generated decks are for your personal study
use only. If you use data from Den Danske Ordbog, you are responsible for
complying with their official terms of use: https://ordnet.dk/copyright

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for
details.

## Acknowledgments

- **Data sources**: Den Danske Ordbog (definitions, IPA, audio -- personal use),
  Wiktionary (frequency list, CC BY-SA).
- **Translations**: Google Gemini (2.0 Flash / 2.5 Flash).
