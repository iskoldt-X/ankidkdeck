# 06_translate_expressions.py
#
# Reads ddo_entries.json, and for each entry, translates all its fixed expressions
# using the Gemini API. It leverages batching and a two-stage review-and-correct
# workflow for maximum quality and robustness.

import json
import os
import time
import logging
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tqdm import tqdm

# --- Configuration ---
TARGET_LANG = "Spanish"  # Change to your target language
INPUT_FILE = "ddo_entries.json"

# Stage 1: Generation Model (High Quality)
GENERATOR_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
# Stage 2: Reviewer Model (Fast & Cheap)
REVIEWER_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# The filename keeps the historical "gemini-2.0-flash" suffix regardless of the
# generator model actually used: 09_export_apkg.py expects this exact pattern,
# and the released decks were shipped with files named this way.
OUTPUT_FILE = f"expression_translations{TARGET_LANG}_gemini-2.0-flash.json"

# Workflow & API settings
MAX_RETRIES = 5  # Retries for a single API call
MAX_CORRECTION_ATTEMPTS = 3  # How many times to try correcting a bad translation
BASE_RETRY_DELAY = 10
SAVE_EVERY = 1
REQUEST_INTERVAL_SECONDS = 5  # To respect 30 RPM limit for free tier
MAX_EXPR_PER_BATCH = 20  # How many expressions to send in one API call

# --- Gemini API Pool & Helpers (Identical to other scripts) ---
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


# --- Schema Definitions ---
def _build_translation_schema(n_items: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "fixed_expressions": {
                "type": "array",
                "minItems": n_items,
                "maxItems": n_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "lemma": {"type": "string"},
                        "gloss": {"type": "string"},
                    },
                    "required": ["lemma", "gloss"],
                },
            }
        },
        "required": ["fixed_expressions"],
    }


def _build_review_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "contains_other_languages": {"type": "boolean"},
            "detected_words": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["contains_other_languages", "detected_words"],
    }


# --- Stage 2: Review Function ---
def review_translation(json_string: str) -> dict | None:
    """
    Uses a fast, cheap model to check for language contamination.
    Returns: A dict like {"contains_other_languages": bool, "detected_words": list} or None on failure.
    """
    if TARGET_LANG.lower() == "english":
        user_prompt = f"""
You are a meticulous language quality inspector. Your only task is to detect if the lemma part of the following text contains any **CHARACTERS** that are NOT part of the English language.

- You MUST ignore JSON structure characters like `{{`, `}}`, `"` and keys like "lemma", "gloss".
- If no other languages are found, "contains_other_languages" must be false and "detected_words" must be an empty list.

Text to inspect:
---
{json_string}
---
"""
    else:
        user_prompt = f"""
You are a meticulous language quality inspector. Your only task is to detect if the lemma part of the following text contains any **CHARACTERS** that are NOT part of the {TARGET_LANG} or English languages.

- You MUST ignore JSON structure characters like `{{`, `}}`, `"` and keys like "lemma", "gloss".
- If no other languages are found, "contains_other_languages" must be false and "detected_words" must be an empty list.

Text to inspect:
---
{json_string}
---
"""
    last_feedback = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_feedback:
            logging.warning(
                f"[Reviewer] Retrying (Attempt {attempt}/{MAX_RETRIES}). Reason: {last_feedback}"
            )

        cli = _get_client()
        try:
            resp = cli.models.generate_content(
                model=REVIEWER_MODEL_NAME,
                contents=[user_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_build_review_schema(),
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

    logging.error(f"[Reviewer] Failed to review content after {MAX_RETRIES} attempts.")
    return None


# --- Stage 1: Generation Function ---
def translate_expressions_batch(
    headword: str, expr_objs: list[dict], correction_instruction: str = ""
) -> str | None:
    """
    Translates a batch of expressions.
    Returns: The raw JSON string of the translation, or None on failure.
    """
    payload = {str(i): obj for i, obj in enumerate(expr_objs)}
    n_items = len(payload)
    response_schema = _build_translation_schema(n_items)

    system_prompt_base = f"""
You are a senior lexicographer translating Danish fixed expressions and idioms into {TARGET_LANG} for a dictionary aimed at **language learners**.
### CORE TASK
For each Danish expression in the input, provide a `lemma` and a `gloss` in {TARGET_LANG}.
### CONTEXT PROVIDED
- `expr`: The Danish expression to translate.
- `hint`: An optional Danish definition to clarify the expression's meaning. Use this hint to find the best translation, but do not translate the hint itself.
### RULES FOR "lemma":
- It must be the most natural and common equivalent idiom or phrase in {TARGET_LANG}.
- If a direct idiom exists, prefer it. If not, use a concise, descriptive phrase.
### RULES FOR "gloss":
- It must be a full, explanatory sentence in {TARGET_LANG} that clarifies the meaning and usage of the Danish expression for a learner.
### OUTPUT FORMAT
Return pure JSON matching the schema. The `fixed_expressions` array MUST have a length of {n_items}.
"""
    critical_rules = ""
    if TARGET_LANG.lower() != "english":
        critical_rules = f"""
**CRITICAL RULES: These are non-negotiable.**
1.  The primary language for both "lemma" and "gloss" MUST be **{TARGET_LANG}**.
2.  You must avoid all other languages. **DO NOT USE RUSSIAN under any circumstances.**
3.  **AS A LAST RESORT, ONLY IF** a concept is truly untranslatable into {TARGET_LANG}, you are permitted to use a concise English word or phrase in the `gloss`. This should be extremely rare. Never use English in the `lemma`.
"""
    # Dynamically add the correction instruction if provided
    if correction_instruction:
        final_system_prompt = f"**IMPORTANT CORRECTION:** {correction_instruction}\n\n{critical_rules.strip()}\n\n{system_prompt_base.strip()}"
    else:
        final_system_prompt = (
            f"{critical_rules.strip()}\n\n{system_prompt_base.strip()}"
        )

    user_msg = (
        f'Headword: "{headword}"\n'
        f"Please translate the following {n_items} fixed expressions into {TARGET_LANG}:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    last_feedback = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_feedback:
            logging.warning(
                f"[{headword}] Retrying generation (Attempt {attempt}/{MAX_RETRIES}). Reason: {last_feedback}"
            )

        cli = _get_client()
        try:
            resp = cli.models.generate_content(
                model=GENERATOR_MODEL_NAME,
                contents=[user_msg],
                config=types.GenerateContentConfig(
                    system_instruction=final_system_prompt,
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            raw_text = resp.text
            # Basic validation before returning
            parsed = json.loads(raw_text)
            if (
                parsed.get("fixed_expressions")
                and len(parsed["fixed_expressions"]) == n_items
            ):
                _count_request()
                return raw_text
            else:
                last_feedback = (
                    "Generated JSON is malformed or has incorrect item count."
                )

        except (ResourceExhausted, TooManyRequests) as e:
            _request_count = MAX_PER_API
            last_feedback = f"Rate limit: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)
        except Exception as e:
            last_feedback = f"API Error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)

    logging.error(
        f"[{headword}] Failed to generate translation after {MAX_RETRIES} attempts."
    )
    return None


# --- Main Logic with Review-and-Correct Workflow ---
def main():
    logging.info(
        f"--- Starting Expression Translator v5 for {TARGET_LANG} (Review-and-Correct) ---"
    )

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
    except FileNotFoundError:
        logging.critical(
            f"Input file not found: {INPUT_FILE}. Please run this script in the correct directory."
        )
        return

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            translations = json.load(f)
        logging.info(
            f"Loaded {len(translations)} existing translations from {OUTPUT_FILE}."
        )
    else:
        translations = {}

    pending_entries = [
        entry for entry in all_entries if entry.get("filename") not in translations
    ]
    if not pending_entries:
        logging.info("All expressions are already translated. Nothing to do.")
        return

    logging.info(
        f"Found {len(pending_entries)} entries needing expression translation."
    )

    translated_count = 0
    for entry in tqdm(pending_entries, desc="Translating Expressions"):
        entry_key = entry.get("filename")
        if not entry_key:
            continue

        headword = entry.get("headword")

        expr_objs = []
        for fx in entry.get("fixed_expressions", []):
            if expr_txt := fx.get("expression"):
                hint = ""
                for detail in fx.get("details", []):
                    if detail.get("type") == "definition" and detail.get("text"):
                        hint = detail["text"]
                        break
                expr_objs.append({"expr": expr_txt, "hint": hint})

        if not expr_objs:
            translations[entry_key] = {}
            continue

        all_results = {}
        has_failed = False
        for i in range(0, len(expr_objs), MAX_EXPR_PER_BATCH):
            batch = expr_objs[i : i + MAX_EXPR_PER_BATCH]
            batch_id = f"Batch {i//MAX_EXPR_PER_BATCH + 1}"

            correction_attempts = 0
            final_batch_result = None
            last_detected_words = []

            while correction_attempts < MAX_CORRECTION_ATTEMPTS:
                time.sleep(REQUEST_INTERVAL_SECONDS)

                # --- 1. Generation Stage ---
                correction_instruction = ""
                if last_detected_words:
                    correction_instruction = (
                        f"Your previous attempt for this batch included forbidden words: {last_detected_words}. "
                        f"Please regenerate the entire batch, ensuring these words are completely removed and replaced with pure {TARGET_LANG} equivalents."
                    )

                raw_json_output = translate_expressions_batch(
                    headword, batch, correction_instruction
                )

                if not raw_json_output:
                    tqdm.write(
                        f"[{headword} | {entry_key}] {batch_id} FAILED generation."
                    )
                    has_failed = True
                    break

                # --- 2. Review Stage ---
                time.sleep(REQUEST_INTERVAL_SECONDS)  # Interval for reviewer model
                review_result = review_translation(raw_json_output)

                if review_result and not review_result["contains_other_languages"]:
                    tqdm.write(f"[{headword} | {entry_key}] {batch_id} PASSED review.")
                    try:
                        parsed_result = json.loads(raw_json_output)
                        # Reconstruct the dict format {expr: {lemma:..., gloss:...}}
                        final_batch_result = {}
                        for j, original_expr_obj in enumerate(batch):
                            final_batch_result[original_expr_obj["expr"]] = (
                                parsed_result["fixed_expressions"][j]
                            )
                    except json.JSONDecodeError:
                        tqdm.write(
                            f"[{headword} | {entry_key}] {batch_id} FAILED to parse the passed JSON."
                        )
                        has_failed = True
                    break  # Exit correction loop
                else:
                    correction_attempts += 1
                    detected_words = (
                        review_result.get("detected_words", ["Unknown"])
                        if review_result
                        else ["Reviewer Failed"]
                    )
                    last_detected_words = detected_words
                    tqdm.write(
                        f"[{headword} | {entry_key}] {batch_id} FAILED review. Detected: {detected_words}. "
                        f"Retrying ({correction_attempts}/{MAX_CORRECTION_ATTEMPTS})..."
                    )

            if final_batch_result:
                all_results.update(final_batch_result)
            else:
                has_failed = True
                tqdm.write(
                    f"[{headword} | {entry_key}] {batch_id} FAILED to correct after {MAX_CORRECTION_ATTEMPTS} attempts. Skipping batch."
                )
                break  # Exit batch loop for this entry

        if not has_failed:
            translations[entry_key] = all_results
            tqdm.write(
                f"[{headword} | {entry_key}] Successfully translated {len(all_results)} expressions."
            )
            translated_count += 1

        if translated_count > 0 and translated_count % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
            tqdm.write("--- Progress Saved ---")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    logging.info("--- All pending expressions have been processed. ---")


if __name__ == "__main__":
    main()
