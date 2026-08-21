# 03_rank_homographs.py
#
# Acts as an Editor-in-Chief for a Danish dictionary. It reads ddo_entries.json,
# groups entries by their query_word, and for each group with multiple entries,
# it asks Gemini to provide a ranked list of the meanings from most to least common.
# The result is a JSON file mapping each query_word to its sorted list of filenames.

import json
import os
import time
import random
import logging
from collections import defaultdict
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "ddo_entries.json"
OUTPUT_FILE = "priority_map.json"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5
SAVE_EVERY = 5
# --- Precise Delay Control ---
REQUEST_INTERVAL_SECONDS = 1.6

# --- Gemini API Pool (reused and stable) ---
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
]
if not GEMINI_API_KEYS:
    raise RuntimeError(
        "Please set GEMINI_API_KEYS in your environment (comma-separated)."
    )
MAX_PER_API = int(
    os.getenv("MAX_PER_API", "9")
)  # With 30RPM, a key can last about a minute.
_current_key_idx, _request_count = 0, 0
_client = genai.Client(api_key=GEMINI_API_KEYS[_current_key_idx])


def _get_client():
    global _current_key_idx, _request_count, _client
    if _request_count >= MAX_PER_API:
        _current_key_idx = (_current_key_idx + 1) % len(GEMINI_API_KEYS)
        _client = genai.Client(api_key=GEMINI_API_KEYS[_current_key_idx])
        _request_count = 0
        logging.info(
            f"🔄 Switched to API key #{_current_key_idx+1}/{len(GEMINI_API_KEYS)}"
        )
    return _client


def _count_request():
    global _request_count
    _request_count += 1


# --- Logging & Helpers ---
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


def get_response_schema(num_ids):
    return {
        "type": "object",
        "properties": {
            "sorted_ids": {
                "type": "array",
                "minItems": num_ids,
                "maxItems": num_ids,
                "items": {"type": "string"},
            }
        },
        "required": ["sorted_ids"],
    }


# --- Core Logic ---
def rank_meanings(headword: str, entries_to_rank: list[dict]) -> list[str] | None:
    num_ids = len(entries_to_rank)

    system_prompt = """
You are the Editor-in-Chief for a new Danish-English dictionary aimed at foreign language learners. Your current task is to rank the different meanings of Danish homographs based on their frequency and relevance in contemporary Danish.
### CONTEXT
I will provide a Danish headword and a JSON array of its various entries. Each entry has:
- "id": A unique identifier (the filename).
- "pos": The part of speech.
- "definitions": One or more Danish definitions, separated by "; ".
### YOUR TASK
Analyze the entries and sort them in descending order of common usage. The first item in your list should be the most fundamental and frequently used meaning. The last item should be the most specialized, archaic, or least common meaning.
### OUTPUT FORMAT
You MUST return a JSON object with a single key, "sorted_ids". The value must be an array of strings, containing all the provided "id"s sorted by priority. The array must contain the exact same IDs as the input, with no extras or omissions.
"""
    example_prompt = """
**Example:**
Input headword: "en"
Entries:
[
  {"id": "en__1.html", "pos": "talord", "definitions": "symboliserer tallet 1"},
  {"id": "en__2.html", "pos": "adverbium", "definitions": "blot, kun"},
  {"id": "en__3.html", "pos": "artikel", "definitions": "bruges for at angive ubestemt form"},
  {"id": "en__4.html", "pos": "pronomen", "definitions": "en uspecificeret person"}
]
Your output MUST be a sorted list, for example:
{
  "sorted_ids": [
    "en__3.html",
    "en__4.html",
    "en__1.html",
    "en__2.html"
  ]
}
"""

    user_msg = (
        f"{system_prompt.strip()}\n\n{example_prompt.strip()}\n\n"
        f"--- YOUR TURN ---\n"
        f'Input headword: "{headword}"\n'
        f"Entries:\n{json.dumps(entries_to_rank, ensure_ascii=False, indent=2)}"
    )

    last_feedback = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_feedback:
            logging.warning(
                f"[{headword}] Retrying (Attempt {attempt}/{MAX_RETRIES}). Reason: {last_feedback}"
            )

        cli = _get_client()
        try:
            resp = cli.models.generate_content(
                model=MODEL_NAME,
                contents=[user_msg],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=get_response_schema(num_ids),
                ),
            )
            raw = resp.text
            parsed = json.loads(raw)
            sorted_ids = parsed.get("sorted_ids")
            original_ids = {e["id"] for e in entries_to_rank}
            if (
                sorted_ids
                and isinstance(sorted_ids, list)
                and set(sorted_ids) == original_ids
            ):
                _count_request()
                return sorted_ids
            else:
                last_feedback = f"Returned list is invalid. Got: {sorted_ids}"
        except (ResourceExhausted, TooManyRequests) as e:
            _request_count = MAX_PER_API
            last_feedback = f"Rate limit: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)
        except Exception as e:
            last_feedback = f"API Error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)

    logging.error(
        f"[{headword}] Failed to get a valid sorted list after {MAX_RETRIES} attempts."
    )
    return None


def main():
    logging.info(
        "--- Dictionary Editor-in-Chief starting the ranking process (v5). ---"
    )

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
    except FileNotFoundError:
        logging.critical(
            f"Input file not found: {INPUT_FILE}. Please run 02_generate_entries.py first."
        )
        return

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            priority_map = json.load(f)
        logging.info(
            f"Loaded {len(priority_map)} existing rankings from {OUTPUT_FILE}."
        )
    else:
        priority_map = {}

    grouped_entries = defaultdict(list)
    for entry in all_entries:
        grouped_entries[entry["query_word"]].append(entry)

    words_to_rank = {
        word: entries for word, entries in grouped_entries.items() if len(entries) > 1
    }
    logging.info(
        f"Found {len(words_to_rank)} words with multiple meanings to be ranked."
    )

    pending_ranking = [
        word for word in sorted(words_to_rank.keys()) if word not in priority_map
    ]
    if not pending_ranking:
        logging.info("All rankings are complete.")
        return

    logging.info(f"Starting ranking for {len(pending_ranking)} words.")

    ranked_count = 0
    # Use tqdm for a progress bar
    for word in tqdm(pending_ranking, desc="Ranking Words"):
        # --- Add a precise delay before each new word is processed ---
        time.sleep(REQUEST_INTERVAL_SECONDS)

        entries = words_to_rank[word]

        # --- MODIFICATION: Include all definitions ---
        entries_for_prompt = []
        for entry in entries:
            all_defs = [
                d.get("definition", "")
                for d in entry.get("definitions", [])
                if d.get("definition")
            ]
            full_definition_str = "; ".join(all_defs)

            entries_for_prompt.append(
                {
                    "id": entry["filename"],
                    "pos": entry.get("pos", "N/A"),
                    "definitions": full_definition_str,  # Changed from "definition" to "definitions" for clarity in prompt
                }
            )

        # Update system prompt to reflect the change
        # (This is done inside rank_meanings, let's update it there)

        sorted_ids = rank_meanings(word, entries_for_prompt)

        if sorted_ids:
            priority_map[word] = sorted_ids
            tqdm.write(
                f"[{word}] Ranking complete. Order: {', '.join(sorted_ids)}"
            )  # Use tqdm.write to not mess up the bar
            ranked_count += 1
        else:
            tqdm.write(f"[{word}] Ranking failed.")  # Use tqdm.write

        if ranked_count > 0 and ranked_count % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(priority_map, f, indent=2, ensure_ascii=False)
            tqdm.write(f"Progress saved. {ranked_count} new rankings recorded.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(priority_map, f, indent=2, ensure_ascii=False)
    logging.info(
        "--- Editor-in-Chief's work is done. All pending rankings are complete. ---"
    )


if __name__ == "__main__":
    main()
