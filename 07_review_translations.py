# 07_review_translations.py
#
# A script to review previously generated translation files for language contamination.
# It reads a large translation JSON, batches entries to maximize context window usage,
# and uses a fast AI model to flag entries containing non-target language characters.
# The output is a concise JSON report of potential issues for manual review.

import json
import os
import time
import logging
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tqdm import tqdm
import tiktoken  # Using tiktoken for accurate token counting

# --- Configuration ---
# --- WHAT TO REVIEW ---
# Point this to the translation file you want to check.
# Can be definition translations or expression translations.

TARGET_LANG = "Spanish"  # Must match the language of the translation file
INPUT_TRANSLATION_FILE = f"definition_translations_{TARGET_LANG}_gemini-2.0-flash.json"
# --- WHERE TO SAVE THE REPORT ---
OUTPUT_ISSUE_REPORT_FILE = f"review_issues_{os.path.basename(INPUT_TRANSLATION_FILE)}"

# --- AI & BATCHING CONFIGURATION ---
REVIEWER_MODEL_NAME = "gemini-2.0-flash"
MAX_TOKENS_PER_BATCH = 30000  # Safe limit, well below model's max (e.g., 1M)
MAX_RETRIES = 5
BASE_RETRY_DELAY = 10
REQUEST_INTERVAL_SECONDS = 5  # To respect 30 RPM limit for free tier

# --- Gemini API Pool & Helpers ---
# (This section is identical to your other scripts, for brevity it's assumed)
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
]
if not GEMINI_API_KEYS:
    raise RuntimeError("Please set GEMINI_API_KEYS in your environment.")
MAX_PER_API = int(os.getenv("MAX_PER_API", "5"))
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


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)

# Initialize tokenizer for counting
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    logging.warning(
        "tiktoken not found, using a less accurate len() for token counting."
    )
    tokenizer = None


def count_tokens(text: str) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text) // 3  # Rough approximation if tiktoken is not available


# --- Core Review Logic ---


def get_review_schema():
    """Returns the JSON schema for the reviewer's response."""
    return {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry_key": {"type": "string"},
                        "detected_words": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["entry_key", "detected_words"],
                },
            }
        },
        "required": ["issues"],
    }


def review_batch(batch_of_entries: dict) -> dict | None:
    """Sends a batch of entries to the reviewer model."""
    if TARGET_LANG.lower() != "english":
        user_prompt = f"""
    You are a meticulous language quality inspector. Your task is to review a batch of dictionary translations from Danish to {TARGET_LANG} (the purpose of the translations is to provide Danish learning materials for {TARGET_LANG} speakers, so the translations should be in {TARGET_LANG} or English only), and identify any entries that contain **CHARACTERS** that are NOT part of the {TARGET_LANG} or English languages.

    **Your Task:**
    Review the provided JSON object. For each entry (keyed by a filename), check if its translated text (`lemma` and `gloss` values) contains any language character that is NOT {TARGET_LANG}.
    Pay special attention to Cyrillic, Greek, or other non-standard symbols.

    **Output Format:**
    You MUST respond in pure JSON format matching the specified schema.
    The `issues` array should ONLY contain entries where you found non-compliant characters. If an entry is clean, do not include it in the `issues` array. If the entire batch is clean, return an empty `issues` array.

    **Input Batch to Inspect:**
    ---
    {json.dumps(batch_of_entries, ensure_ascii=False, indent=2)}
    ---
    """
    else:
        user_prompt = f"""
    You are a meticulous language quality inspector. Your task is to review a batch of dictionary translations from Danish to {TARGET_LANG} (the purpose of the translations is to provide Danish learning materials for {TARGET_LANG} speakers, so the translations should be in {TARGET_LANG} only), and identify any entries that contain **CHARACTERS** that are NOT part of the {TARGET_LANG} language.

    **Your Task:**
    Review the provided JSON object. For each entry (keyed by a filename), check if its translated text (`lemma` and `gloss` values) contains any language character that is NOT {TARGET_LANG}.
    Pay special attention to Cyrillic, Greek, or other non-standard symbols.

    **Output Format:**
    You MUST respond in pure JSON format matching the specified schema.
    The `issues` array should ONLY contain entries where you found non-compliant characters. If an entry is clean, do not include it in the `issues` array. If the entire batch is clean, return an empty `issues` array.

    **Input Batch to Inspect:**
    ---
    {json.dumps(batch_of_entries, ensure_ascii=False, indent=2)}
    ---
    """

    last_feedback = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_feedback:
            logging.warning(
                f"Retrying review batch (Attempt {attempt}/{MAX_RETRIES}). Reason: {last_feedback}"
            )

        cli = _get_client()
        try:
            resp = cli.models.generate_content(
                model=REVIEWER_MODEL_NAME,
                contents=[user_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=get_review_schema(),
                ),
            )
            _count_request()
            return json.loads(resp.text)
        except (ResourceExhausted, TooManyRequests) as e:
            _request_count = MAX_PER_API
            last_feedback = f"Rate limit: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)
        except Exception as e:
            last_feedback = f"API Error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)

    logging.error(f"Failed to review batch after {MAX_RETRIES} attempts.")
    return None


# --- Main Execution ---
def main():
    logging.info(f"--- Starting Post-hoc Review for '{INPUT_TRANSLATION_FILE}' ---")

    try:
        with open(INPUT_TRANSLATION_FILE, "r", encoding="utf-8") as f:
            all_translations = json.load(f)
    except FileNotFoundError:
        logging.critical(f"Input file not found: {INPUT_TRANSLATION_FILE}")
        return
    except json.JSONDecodeError:
        logging.critical(
            f"Could not parse JSON from {INPUT_TRANSLATION_FILE}. Please check the file."
        )
        return

    all_issues = []
    current_batch = {}
    current_token_count = 0

    # Using tqdm to show progress iterating through the large translation file
    for entry_key, translations in tqdm(
        all_translations.items(), desc="Processing entries"
    ):
        # We represent the entry as a string to count its tokens
        entry_str = json.dumps({entry_key: translations})
        entry_token_count = count_tokens(entry_str)

        # If adding the next entry exceeds the batch limit, process the current batch first
        if current_batch and (
            current_token_count + entry_token_count > MAX_TOKENS_PER_BATCH
        ):
            tqdm.write(
                f"Processing a batch of {len(current_batch)} entries ({current_token_count} tokens)..."
            )
            time.sleep(REQUEST_INTERVAL_SECONDS)  # Respect rate limits between batches

            review_result = review_batch(current_batch)
            if review_result and review_result.get("issues"):
                # Add original translation data to the issue for context
                for issue in review_result["issues"]:
                    issue_key = issue["entry_key"]
                    if issue_key in current_batch:
                        issue["original_content"] = current_batch[issue_key]
                all_issues.extend(review_result["issues"])

            # Reset for the next batch
            current_batch = {}
            current_token_count = 0

            tqdm.write(f"Batch processed. Total issues found so far: {len(all_issues)}")

        # Add the current entry to the batch
        current_batch[entry_key] = translations
        current_token_count += entry_token_count

    # Process the final remaining batch
    if current_batch:
        tqdm.write(
            f"Processing the final batch of {len(current_batch)} entries ({current_token_count} tokens)..."
        )
        review_result = review_batch(current_batch)
        if review_result and review_result.get("issues"):
            for issue in review_result["issues"]:
                issue_key = issue["entry_key"]
                if issue_key in current_batch:
                    issue["original_content"] = current_batch[issue_key]
            all_issues.extend(review_result["issues"])

    # Save the final report
    logging.info(
        f"Review complete. Found a total of {len(all_issues)} potential issues."
    )

    with open(OUTPUT_ISSUE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2, ensure_ascii=False)

    logging.info(f"✔ Issue report saved to '{OUTPUT_ISSUE_REPORT_FILE}'.")
    if not all_issues:
        logging.info("Great news! No issues were found.")


if __name__ == "__main__":
    # You might need to install tiktoken: pip install tiktoken
    main()
