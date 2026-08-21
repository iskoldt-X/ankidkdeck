# 04_translate_pos.py
#
# Translates the unique Danish part-of-speech (POS) tags found in
# ddo_entries.json into the target language using the Gemini API,
# producing pos_translations_<LANG>_gemini.json for the exporter.
import json
import os
import time
import logging
from collections import Counter

from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

# --- Configuration ---
# Target language (e.g. "English", "Chinese", "German")
TARGET_LANG = "Spanish"  # Change to your target language
INPUT_FILE = "ddo_entries.json"
OUTPUT_FILE = f"pos_translations_{TARGET_LANG}_gemini.json"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# Workflow & API settings
MAX_RETRIES = 5  # Max retries for a single API call
BASE_RETRY_DELAY = 5  # Base retry delay in seconds
REQUEST_INTERVAL_SECONDS = 1.1  # Respect the free-tier 60 RPM limit (60/60 = 1s, with a small margin)

# --- Gemini API Pool & Helpers (shared pattern across the translation scripts) ---
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
]
if not GEMINI_API_KEYS:
    raise RuntimeError("Please set GEMINI_API_KEYS in your environment (comma-separated).")

MAX_PER_API = int(os.getenv("MAX_PER_API", "50"))  # Conservative value for the free tier's 60 requests/min
_current_key_idx, _request_count = 0, 0
_client = genai.Client(api_key=GEMINI_API_KEYS[_current_key_idx])

def _get_client():
    """Rotate API keys to avoid rate limits."""
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
    """Count one API request."""
    global _request_count
    _request_count += 1

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)

# --- Schema Definition ---
def _build_pos_schema(tags: list[str]) -> dict:
    """Build the schema for the expected JSON object dynamically."""
    properties = {tag: {"type": "string", "description": f"The {TARGET_LANG} translation for the Danish POS tag '{tag}'."} for tag in tags}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys())
    }

# --- Core Translation Function ---
def translate_all_pos(tags: list[str]) -> dict | None:
    """
    Translate a list of POS tags with Gemini using a strict JSON schema.
    """
    schema = _build_pos_schema(tags)

    system_prompt = f"""
You are a linguistic assistant specializing in lexicography. Your task is to translate a list of Danish part-of-speech (POS) abbreviations into their full, clear equivalents in {TARGET_LANG}.

### INSTRUCTIONS
- For each key in the input JSON, provide its translation as the value.
- The translation should be the common term for that part-of-speech in {TARGET_LANG}.
- Adhere strictly to the JSON output format. The output MUST be a single JSON object with the exact same keys as the input, and all values must be strings.
- If the target language is Chinese, DO NOT include any pinyin or other phonetic transcriptions.
"""

    # Build a dict with empty values as an example payload to guide the model
    payload = {tag: "" for tag in tags}
    user_prompt = f"Please translate the following Danish POS tags into {TARGET_LANG}:\n" \
                  f"{json.dumps(payload, indent=2, ensure_ascii=False)}"

    last_feedback = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_feedback:
            logging.warning(
                f"Retrying (Attempt {attempt}/{MAX_RETRIES}). Reason: {last_feedback}"
            )

        # Sleep at the start of each attempt to respect rate limits
        time.sleep(REQUEST_INTERVAL_SECONDS)

        cli = _get_client()
        try:
            resp = cli.models.generate_content(
                model=MODEL_NAME,
                contents=[user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,  # Deterministic task, so temperature 0
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            _count_request()

            # Parse the returned JSON
            parsed_json = json.loads(resp.text)

            # Verify the returned keys exactly match the requested keys
            if set(parsed_json.keys()) != set(tags):
                missing = set(tags) - set(parsed_json.keys())
                extra = set(parsed_json.keys()) - set(tags)
                last_feedback = f"Key mismatch. Missing: {missing}, Extra: {extra}"
                continue  # Retry on key mismatch

            logging.info("Translation successful and keys validated!")
            return parsed_json

        except (ResourceExhausted, TooManyRequests) as e:
            _request_count = MAX_PER_API  # Force a key switch
            last_feedback = f"Rate limit error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)
        except Exception as e:
            last_feedback = f"API Error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)

    logging.error(f"Translation failed after {MAX_RETRIES} attempts.")
    return None

# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Load ddo_entries.json
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        logging.critical(f"Input file not found: {INPUT_FILE}. Please run 02_generate_entries.py first.")
        exit(1)

    # 2. Collect all unique POS tags with a Counter
    pos_counter = Counter(e.get("pos", "") for e in entries if e.get("pos"))
    tags = list(pos_counter.keys())

    if not tags:
        logging.info("No POS tags found in the input file.")
        exit(0)

    logging.info(f"Found {len(tags)} unique POS tags, translating to {TARGET_LANG}.")

    # 3. Translate
    try:
        mapping = translate_all_pos(tags)
    except Exception as e:
        logging.critical(f"Unrecoverable error during translation: {e}")
        mapping = None

    # 4. Save the result
    if mapping:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
        logging.info(f"Done! Translations saved to {OUTPUT_FILE}")
    else:
        logging.error("Failed to produce a translation mapping; no output file was created.")
