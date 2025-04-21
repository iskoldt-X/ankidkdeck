# ankidkdeck

A collection of Python scripts and data to build a high‑quality Danish Anki deck (core ~3 000 words) with audio, definitions, example sentences, fixed expressions, and translations. Data is sourced from Wiktionary (frequency lists) and Den Danske Ordbog, and translations are powered by a local LLM via Ollama.

## Prerequisites

- **Python 3.12+**
- Install required packages:
  ```bash
  pip install requests beautifulsoup4 genanki ollama
  ```
- A local LLM compatible with Ollama (e.g. `gemma3:12b`).


## Usage

1. **Download & update raw HTML**  
   ```bash
   python download_ddo_pages.py
   python update_ddo_html_versions.py
   ```

2. **Parse & dedupe entries**  
   ```bash
   python generate_ddo_entries.py
   python dedupe_ddo_entries.py
   ```

3. **Download audio**  
   ```bash
   python download_audio_and_map.py
   ```

*Joke:* Actually, I’ve already done steps 1–3 for you. To avoid hammering DDO’s servers, feel free to skip directly to step 4.

4. **Translate metadata**  

    Please edit the TARGET_LANG = "" in the translate scripts to your desired target language. Fx. TARGET_LANG = "English" or TARGET_LANG = "Chinese".

   ```bash
   python translate_pos_llm.py
   python translate_definitions_batch.py
   python translate_fixed_expressions_batch.py
   ```

5. **Export Anki deck**

    Please note to edit the Configuration section in the `export_danish_target_lang_apkg.py` file to your desired target language. 

    ```python
    # Configuration
    json_path = "ddo_entries_unique.json"
    audio_map_path = "audio_map.json"
    TARGET_LANG = ""
    definition_translation_path = f"definition_translations_lemma_gloss_{TARGET_LANG}.json"
    expr_translation_path = f"expr_translations_{TARGET_LANG}.json"
    pos_translation_path = f"pos_translations_{TARGET_LANG}.json"
    output_apkg = f"danish_{TARGET_LANG}.apkg"
    limit = None
    ENABLE_DEBUG_PRINTING = True

    ```


   ```bash
   python export_danish_target_lang_apkg.py
   ```

After running step 5, you’ll have an `.apkg` file ready to import into Anki, along with all audio and mapping files.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Data sources**: Den Danske Ordbog (for audio & IPA), Wiktionary (frequency lists, CC BY‑SA).  
- **Translations**: Powered by a local LLM via Ollama.
