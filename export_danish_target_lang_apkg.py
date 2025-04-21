import json
from pathlib import Path
from random import randrange
import genanki
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


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

# --- Wiktionary URL ---
WIKTIONARY_URL = (
    "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist"
)


# --- Anki Model Configuration ---
DECK_ID = randrange(1 << 30, 1 << 31)
MODEL_ID = randrange(1 << 30, 1 << 31)

FIELDS = [
    {"name": "Headword"},
    {"name": "POS"},
    {"name": "IPA"},
    {"name": "Definition"},
    {"name": "Grammar"},
    {"name": "Example"},
    {"name": "Variants"},
    {"name": "Collocations"},
    {"name": "Derivatives"},
    {"name": "Etymology"},
    {"name": "FrequencyRank"},  # New field for frequency ranking
]

CSS = """
/* Custom card styling */
.card { font-family: sans-serif; line-height:1.35; color:#222; }
.front-container { text-align:center; }
.front-headword { font-size:2.2em; font-weight:600; }
.front-pos { font-size:0.9em; color:#555; margin-top:4px; line-height: 1.2; }
.front-ipa { margin-top:8px; }
.ipa-row { margin:4px 0; display:flex; align-items:center; justify-content:center; }
.ipa-text { font-size:1.1em; color:#555; margin-right:6px; }
:root { --replay-bg: #a8d5ba; --replay-icon: #2f5d62; }
.replay-button svg { width: 20px; height: 20px; }
.replay-button svg circle { fill: var(--replay-bg); }
.replay-button svg path { stroke: var(--replay-icon); fill: var(--replay-icon); }
.core-section { font-size:1.25em; margin-bottom:4px; }
.small-section { font-size:0.9em; color:#444; margin-top:6px; }
hr { margin:10px 0; }
.translation { color: #006400; font-size: 0.95em; margin-top: 3px; }
.translation b { color: #008000; }
.freq-rank { font-size: 0.7em; color: #888; text-align: right; margin-top: 15px; }
"""

QFMT = """
<div class="front-container">
  <div class="front-headword">{{Headword}}</div>
  {{#POS}}<div class="front-pos">{{POS}}</div>{{/POS}}
  <div class="front-ipa">{{IPA}}</div>
</div>
"""

# Answer format includes frequency rank at bottom
AFMT = """
{{FrontSide}}
<hr>
<div class="core-section">{{Definition}}</div>
{{#Grammar}}<div class="small-section"><b>Grammar:</b> {{Grammar}}</div>{{/Grammar}}
{{#Example}}<div class="small-section"><b>Example:</b> {{Example}}</div>{{/Example}}
{{#Variants}}<div class="small-section"><b>Variants:</b> {{Variants}}</div>{{/Variants}}
{{#Collocations}}<div class="small-section"><b>Collocations:</b><br>{{Collocations}}</div>{{/Collocations}}
{{#Derivatives}}<div class="small-section"><b>Derivatives:</b> {{Derivatives}}</div>{{/Derivatives}}
{{#Etymology}}<div class="small-section"><b>Etymology:</b> {{Etymology}}</div>{{/Etymology}}
{{#FrequencyRank}}<div class="freq-rank">Freq Rank: {{FrequencyRank}}</div>{{/FrequencyRank}}
"""

MODEL = genanki.Model(
    MODEL_ID,
    "DDO Danish 3000",
    fields=FIELDS,
    templates=[{"name": "Card 1", "qfmt": QFMT, "afmt": AFMT}],
    css=CSS,
    sort_field_index=10,  # Sort by FrequencyRank field (index 10)
)


def fetch_danish_wordlist(url):
    """
    Request the specified Wiktionary page and parse the list of Danish words (in page order).
    Returns a list in the format [word1, word2, ...].
    """
    # Get the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page: HTTP {response.status_code}")
        return []

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the <h3> tag marking the Danish section (id="Danish")
    danish_heading = soup.find("h3", id="Danish")
    if not danish_heading:
        print("Could not find the Danish section heading.")
        return []

    # Get the first ordered list <ol> after the Danish section heading
    word_list_tag = danish_heading.find_next("ol")
    if not word_list_tag:
        print("No word list found in the Danish section.")
        return []

    words = []
    # Iterate through each <li> and extract the text of the <a> tag (the word)
    for li in word_list_tag.find_all("li"):
        a_tag = li.find("a")
        if a_tag:
            word = a_tag.get_text(strip=True)
            words.append(word)

    return words


def sanitize(text: str) -> str:
    """
    Remove braces and trim whitespace; convert non-string types to string.
    """
    if not isinstance(text, str):
        if ENABLE_DEBUG_PRINTING:
            print(
                f"[DEBUG SANITIZE WARNING] sanitize received non-string type: {type(text)}. Value: {text}"
            )
        text = str(text)
    return text.replace("{", "").replace("}", "").strip()


def build_ipa_html(entry, audio_map):

    rows = []
    for u in entry.get("udtale", []):
        ipa_raw = sanitize(u.get("ipa", ""))
        if not ipa_raw:
            continue
        core = ipa_raw.strip("[]").strip()
        label = u.get("label")
        label_html = (
            f'<span class="ipa-label">({sanitize(label)})</span>' if label else ""
        )
        url = u.get("audio", "")
        sound_tag = f"[sound:{Path(audio_map[url]).name}]" if url in audio_map else ""
        rows.append(
            f'<div class="ipa-row">'
            f'<span class="ipa-text">[{core}]</span>'
            f"{label_html}"
            f"{sound_tag}"
            f"</div>"
        )
    return "".join(rows).rstrip("} \n")


def extract_definitions_with_translation(entry, headword_key, definition_translations):

    parts = []
    entry_translations = definition_translations.get(headword_key)

    if entry_translations is None:
        if ENABLE_DEBUG_PRINTING and headword_key:
            print(
                f"[DEBUG DEF] Headword key not found in definition translations: '{headword_key}'"
            )
        entry_translations = {}

    for d in entry.get("definitions", []):
        num = sanitize(d.get("number", ""))
        txt = sanitize(d.get("definition", ""))
        if txt:
            formatted_definition = (num + " " + txt).strip()
            translation_data = entry_translations.get(txt)

            if translation_data:
                lemma_raw = translation_data.get("lemma", "")
                gloss_raw = translation_data.get("gloss", "")
                lemma = str(lemma_raw) if not isinstance(lemma_raw, str) else lemma_raw
                gloss = str(gloss_raw) if not isinstance(gloss_raw, str) else gloss_raw
                if lemma is not lemma_raw and ENABLE_DEBUG_PRINTING:
                    print(
                        f"[DEBUG TYPE WARNING - DEF] Lemma type mismatch. Headword: '{headword_key}', Def: '{txt}'."
                    )
                if gloss is not gloss_raw and ENABLE_DEBUG_PRINTING:
                    print(
                        f"[DEBUG TYPE WARNING - DEF] Gloss type mismatch. Headword: '{headword_key}', Def: '{txt}'."
                    )

                translation_html = f'<div class="translation">{sanitize(lemma)} ({sanitize(gloss)})</div>'
                formatted_definition += translation_html
            elif ENABLE_DEBUG_PRINTING and headword_key in definition_translations:
                print(
                    f"[DEBUG DEF] Definition text not found for headword '{headword_key}': '{txt}'"
                )

            parts.append(formatted_definition)

    return "<br>".join(parts) if parts else "(no definition)"


def extract_first_example(entry):
    for d in entry.get("definitions", []):
        for ex in d.get("examples") or []:
            txt = sanitize(ex.get("text", ""))
            if txt:
                return txt
    return ""


def extract_collocations_with_translation(entry, headword_key, expr_translations):
    collocation_parts = []
    entry_expr_data = expr_translations.get(headword_key)
    entry_expr_translations = {}

    if entry_expr_data and isinstance(entry_expr_data.get("fixed_expressions"), dict):
        entry_expr_translations = entry_expr_data["fixed_expressions"]
    # elif ENABLE_DEBUG_PRINTING and headword_key: #
    #     if headword_key not in expr_translations:
    #          print(f"[DEBUG EXPR] Headword key not found in expression translations: '{headword_key}'")
    #     #

    for fx in entry.get("fixed_expressions", []):
        expression_text = fx.get("expression", "")
        if expression_text:
            sanitized_expr = sanitize(expression_text)
            formatted_collocation = sanitized_expr
            translation_data = entry_expr_translations.get(expression_text)

            if translation_data:
                lemma_raw = translation_data.get("lemma", "")
                gloss_raw = translation_data.get("gloss", "")
                lemma = str(lemma_raw) if not isinstance(lemma_raw, str) else lemma_raw
                gloss = str(gloss_raw) if not isinstance(gloss_raw, str) else gloss_raw
                if lemma is not lemma_raw and ENABLE_DEBUG_PRINTING:
                    print(
                        f"[DEBUG TYPE WARNING - EXPR] Lemma type mismatch. Headword: '{headword_key}', Expr: '{expression_text}'."
                    )
                if gloss is not gloss_raw and ENABLE_DEBUG_PRINTING:
                    print(
                        f"[DEBUG TYPE WARNING - EXPR] Gloss type mismatch. Headword: '{headword_key}', Expr: '{expression_text}'."
                    )

                translation_html = f'<div class="translation" style="margin-left: 10px;">{sanitize(lemma)} ({sanitize(gloss)})</div>'
                formatted_collocation += translation_html
            elif (
                ENABLE_DEBUG_PRINTING
                and headword_key in expr_translations
                and isinstance(
                    expr_translations.get(headword_key, {}).get("fixed_expressions"),
                    dict,
                )
            ):
                print(
                    f"[DEBUG EXPR] Fixed expression text not found for headword '{headword_key}': '{expression_text}'"
                )

            collocation_parts.append(formatted_collocation)

    return "<br>".join(collocation_parts)


def join_safe(lst, sep="; "):
    return sep.join(filter(None, map(sanitize, lst))) if lst else ""


# --- Script Execution ---
if __name__ == "__main__":

    print("Loading data files...")
    try:
        entries_raw = json.load(open(json_path, encoding="utf-8"))
        audio_map = json.load(open(audio_map_path, encoding="utf-8"))
        definition_translations = json.load(
            open(definition_translation_path, encoding="utf-8")
        )
        expr_translations = json.load(open(expr_translation_path, encoding="utf-8"))
        pos_translations = json.load(open(pos_translation_path, encoding="utf-8"))
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure all JSON files exist.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}. Please check the file format.")
        exit(1)

    print(f"Loaded {len(entries_raw)} raw entries.")
    if limit:
        print(f"Applying limit: processing first {limit} entries.")
        entries_raw = entries_raw[:limit]

    print("Fetching and processing frequency wordlist...")
    wordlist = fetch_danish_wordlist(WIKTIONARY_URL)

    final_sorted_entries = []
    if wordlist:

        word_to_rank = {word.lower(): i + 1 for i, word in enumerate(wordlist)}
        print(f"Created rank map for {len(word_to_rank)} unique words.")

        ranked_entries_data = []
        unranked_entries = []
        processed_files = set()

        for entry in entries_raw:
            if "file" not in entry:
                if ENABLE_DEBUG_PRINTING:
                    hw_for_debug = sanitize(entry.get("headword", "N/A"))
                    print(
                        f"[DEBUG SORT] Entry missing 'file' key (Headword: '{hw_for_debug}'). Treating as unranked."
                    )
                unranked_entries.append(entry)
                continue

            try:
                file_key = Path(entry["file"]).stem.lower()
            except Exception as e:
                if ENABLE_DEBUG_PRINTING:
                    hw_for_debug = sanitize(entry.get("headword", "N/A"))
                    print(
                        f"[DEBUG SORT] Error processing file key for '{entry.get('file', 'N/A')}' (Headword: '{hw_for_debug}'): {e}. Treating as unranked."
                    )
                unranked_entries.append(entry)
                continue

            if file_key in processed_files:
                continue
            processed_files.add(file_key)

            rank = word_to_rank.get(file_key)
            if rank is not None:

                entry["frequency_rank"] = rank
                ranked_entries_data.append((rank, entry))
            else:
                entry["frequency_rank"] = None
                unranked_entries.append(entry)

        ranked_entries_data.sort(key=lambda x: x[0])

        sorted_ranked_entries = [item[1] for item in ranked_entries_data]

        final_sorted_entries = sorted_ranked_entries + unranked_entries
        print(
            f"Sorting complete: {len(sorted_ranked_entries)} ranked entries, {len(unranked_entries)} unranked entries."
        )

    else:
        print(
            "Warning: Could not fetch or process frequency wordlist. Using original entry order."
        )
        final_sorted_entries = entries_raw

        for entry in final_sorted_entries:
            entry["frequency_rank"] = None

    deck = genanki.Deck(DECK_ID, f"丹麦要你命3000词")
    media_files = set()

    print(f"Generating Anki notes for {len(final_sorted_entries)} entries...")

    for i, entry in enumerate(final_sorted_entries):

        hw_raw = entry.get("headword", "")
        hw = sanitize(hw_raw)
        headword_key = hw_raw.strip()

        freq_rank = entry.get("frequency_rank")

        freq_rank_str = f"{freq_rank:05d}" if freq_rank is not None else ""

        if not headword_key and ENABLE_DEBUG_PRINTING:
            print(f"[DEBUG MAIN] Entry {i} has empty headword key.")

        pos_original = sanitize(entry.get("pos", ""))
        pos_zh = pos_translations.get(pos_original)
        pos_full = (
            f"{pos_original} ({sanitize(pos_zh)})"
            if pos_original and pos_zh
            else pos_original
        )

        ipa_html = build_ipa_html(entry, audio_map)

        for u in entry.get("udtale", []):
            url = u.get("audio", "")
            if url in audio_map:
                media_files.add(audio_map[url])

        definition = ""
        collocs = ""
        if headword_key:
            definition = extract_definitions_with_translation(
                entry, headword_key, definition_translations
            )
            collocs = extract_collocations_with_translation(
                entry, headword_key, expr_translations
            )
        else:

            definition = extract_definitions_with_translation(entry, "", {})
            collocs = extract_collocations_with_translation(entry, "", {})
            definition = definition if definition != "(no definition)" else ""

        grammar = sanitize(
            next(
                (
                    d.get("grammar")
                    for d in entry.get("definitions", [])
                    if d.get("grammar")
                ),
                "",
            )
        )
        example = extract_first_example(entry)
        variants = join_safe(entry.get("wordforms", []))
        od = entry.get("orddannelser", {})
        derivs = (
            od.get("Afledninger", [])
            + od.get("Sammensætninger", [])
            + od.get("Øvrige", [])
        )
        derivatives = join_safe(derivs, ", ")
        raw_ety = (entry.get("etymology") or {}).get("raw", "")
        etymology = (
            sanitize(raw_ety)[:120] + "…"
            if raw_ety and len(raw_ety) > 120
            else sanitize(raw_ety)
        )

        guid_base_hw = sanitize(entry.get("headword", ""))
        guid_for_note = genanki.guid_for(guid_base_hw, pos_original)

        note = genanki.Note(
            model=MODEL,
            fields=[
                hw,
                pos_full,
                ipa_html,
                definition,
                grammar,
                example,
                variants,
                collocs,
                derivatives,
                etymology,
                freq_rank_str,
            ],
            guid=guid_for_note,
        )
        deck.add_note(note)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(final_sorted_entries)} entries...")

    print("Packaging Anki deck...")
    pkg = genanki.Package(deck)
    pkg.media_files = list(media_files)
    pkg.write_to_file(output_apkg)
    print(
        f"✓ Generated {output_apkg}: {len(deck.notes)} cards, {len(pkg.media_files)} unique media files"
    )
    if ENABLE_DEBUG_PRINTING:
        print(f"\n--- Debug Summary ---")
        print(f"Entries processed: {len(final_sorted_entries)}")
