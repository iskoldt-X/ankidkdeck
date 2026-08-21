# 05_translate_definitions.py
#
# Reads ddo_entries.json, and for each entry, translates all its definitions
# using the Gemini API. It produces a lemma and a gloss for each definition.
# The script is resumable and uses the entry's filename as a unique key.

import json
import os
import time
import logging
from collections import OrderedDict
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tqdm import tqdm

# --- Configuration ---
TARGET_LANG = "Spanish"  # Change this to your target language
INPUT_FILE = "ddo_entries.json"
MODEL_NAME = "gemini-2.0-flash"  # Using the model you prefer
OUTPUT_FILE = f"definition_translations_{TARGET_LANG}_{MODEL_NAME}.json"
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5
SAVE_EVERY = 10
REQUEST_INTERVAL_SECONDS = 2.1  # To respect 10 RPM limit

# --- Gemini API Pool ---
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


# --- Logging & Helpers ---
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


def _build_response_schema(n_defs: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "headword": {"type": "string"},
            "definitions": {
                "type": "array",
                "minItems": n_defs,
                "maxItems": n_defs,
                "items": {
                    "type": "object",
                    "properties": {
                        "lemma": {"type": "string"},
                        "gloss": {"type": "string"},
                    },
                    "required": ["lemma", "gloss"],
                },
            },
        },
        "required": ["headword", "definitions"],
    }


# --- Core API Call ---
def translate_definitions_for_entry(headword: str, defs: list[str]) -> dict | None:
    payload = {str(i): d for i, d in enumerate(defs)}
    n_defs = len(payload)
    response_schema = _build_response_schema(n_defs)

    # === START OF PROMPT V4 LOGIC ===

    critical_rule = ""
    if TARGET_LANG.lower() != "english":
        critical_rule = f"""
    **CRITICAL RULE: Both "lemma" and "gloss" MUST be in {TARGET_LANG}. Do NOT use English in your output, unless there is absolutely no way to express a concept in {TARGET_LANG}.**
    """

    system_prompt_base = f"""
    You are a senior lexicographer creating a new Danish-{TARGET_LANG} dictionary specifically for **language learners**. Your translations must be clear, practical, and intuitive.

    ### CORE TASK
    For each Danish definition provided, you must generate a corresponding `lemma` and `gloss` in **{TARGET_LANG}**.

    ### RULES FOR "lemma" (The {TARGET_LANG} headword):
    1.  **Be a real word/phrase:** It must be a concise, common {TARGET_LANG} word or a widely used, natural-sounding fixed phrase.
    2.  **Be practical, not overly technical:** AVOID purely grammatical descriptions. A conceptual hint is better than a niche technical term. If a grammatical term is widely understood by learners in the context of {TARGET_LANG} (e.g., a {TARGET_LANG} term for "noun suffix" when dealing with a suffix that forms nouns), it can be acceptable if truly the best and most concise fit.
    3.  **Balance specificity, naturalness, and informativeness. Focus on DEFINITION'S CORE SEMANTICS:**
        a.  **Capture Key Nuances:** The `lemma` should accurately reflect the specific and defining characteristics of THAT particular Danish definition.
        b.  **Prioritize Natural & Common Usage:** Choose {TARGET_LANG} words/phrases that are natural and commonly used by native speakers.
        c.  **Handling Multiple Meanings of Same Headword:** If multiple Danish definitions of the *same Danish headword* naturally map to the *same core {TARGET_LANG} lemma* (because it genuinely covers those nuances), you MAY reuse that lemma. Rely on distinct `glosses` for precise differentiation.
        d.  **When to Be More Specific (Informativeness):** However, if a Danish definition highlights a *distinct, crucial aspect* (e.g., a specific skill, a combined action, or a specialized context) that a very general {TARGET_LANG} lemma would obscure, prefer a *slightly more specific (yet still natural and common)* {TARGET_LANG} lemma that better conveys this key information. For example, if a definition describes "kicking a ball with light force AND high control", a lemma reflecting both aspects is better than one reflecting only "light kicking" if a natural {TARGET_LANG} term for the combined concept exists.
        e.  **Avoid Over-Simplification & Literal Traps:** Do NOT over-simplify the lemma to the point of losing essential semantic information from the Danish definition. Also, be cautious of being overly influenced by the literal translation of the *Danish headword itself* if the *definition* points to a more nuanced or specific meaning in {TARGET_LANG}. The definition is paramount.
        f.   **Clarity for Learners:** Ultimately, the lemma (in conjunction with the gloss) should provide maximum clarity for a language learner.
    4.  **Conciseness preferred:** Aim for brevity. Single words or short, common phrases in {TARGET_LANG} are ideal, but not at the cost of Rule 3's requirements.

    ### RULES FOR "gloss" (The explanatory translation):
    1.  **Be a clear, complete thought:** It must be a grammatically correct and natural-sounding sentence or a very clear, self-contained explanatory phrase in {TARGET_LANG}. Full sentences are preferred, but clarity and conciseness for the learner are paramount.
    2.  **Be explanatory, focused, AND faithful:** It should clearly explain the meaning and usage context relevant to THAT specific definition, being faithful to the scope and nuances of the Danish original. AVOID over-explaining with information not directly implied by the Danish definition, but ensure all key semantic components of the Danish definition are represented in the {TARGET_LANG} gloss. Focus on direct translation and essential clarification in {TARGET_LANG}.
    3.  **Match formality:** The tone of the gloss in {TARGET_LANG} should generally match the formality of the Danish definition.

    ### OUTPUT FORMAT
    Return pure JSON that **exactly matches the response schema**. The `definitions` array MUST have a length of {n_defs}.
    """

    unified_example = f"""
    ### EXAMPLE OF STRUCTURE (target language is {TARGET_LANG}):
    Input Headword: "spille" (example headword, not necessarily from your data)
    Input Definitions: {{
        "0": "udføre musik på et instrument", 
        "1": "deltage i et spil for fornøjelses skyld",
        "2": "i fodbold, aflevere bolden til en medspiller med præcis kontrol og ofte for at skabe en scoringsmulighed" 
    }}

    Expected JSON Output Structure:
    {{
    "headword":"spille", 
    "definitions":[
    {{
        "lemma":"[A {TARGET_LANG} lemma for 'to play music']",
        "gloss":"[A {TARGET_LANG} gloss explaining playing music on an instrument]."
    }},
    {{
        "lemma":"[A {TARGET_LANG} lemma for 'to play a game']",
        "gloss":"[A {TARGET_LANG} gloss explaining participating in a game for fun]."
    }},
    {{
        "lemma":"[A {TARGET_LANG} lemma for 'to pass with control (football)', informative and natural phrase]",
        "gloss":"[A {TARGET_LANG} gloss explaining the football-specific action of passing with control to create an opportunity, ensuring all key elements are covered]."
    }}
    ]}}
    """
    final_system_prompt = (
        f"{critical_rule}\n{system_prompt_base.strip()}\n\n{unified_example.strip()}"
    )

    # === END OF PROMPT V4 LOGIC ===

    user_msg = (
        f'Headword: "{headword}"\n'
        f"Expecting exactly {n_defs} definition objects.\n"
        f"Input Definitions JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
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
                    system_instruction=final_system_prompt,
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            raw = resp.text
            parsed = json.loads(raw)
            out_defs = parsed.get("definitions")
            if out_defs and isinstance(out_defs, list) and len(out_defs) == n_defs:
                result = OrderedDict()
                for i, dk_def in enumerate(defs):
                    result[dk_def] = out_defs[i]
                _count_request()
                return result
            else:
                last_feedback = f"Invalid 'definitions' array in response: {out_defs}"

        except (ResourceExhausted, TooManyRequests) as e:
            _request_count = MAX_PER_API
            last_feedback = f"Rate limit: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)
        except Exception as e:
            last_feedback = f"API Error: {e}"
            time.sleep(BASE_RETRY_DELAY * attempt)

    logging.error(
        f"[{headword}] Failed after {MAX_RETRIES} attempts. Last feedback: {last_feedback}"
    )
    return None


# --- Main Logic ---
def main():
    logging.info(f"--- Starting Definition Translator for {TARGET_LANG} ---")

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
            translations = json.load(f)
        logging.info(
            f"Loaded {len(translations)} existing translations from {OUTPUT_FILE}."
        )
    else:
        translations = {}

    # Filter for entries that haven't been translated yet
    pending_entries = [
        entry for entry in all_entries if entry.get("filename") not in translations
    ]

    if not pending_entries:
        logging.info("All definitions are already translated. Nothing to do.")
        return

    logging.info(f"Found {len(pending_entries)} entries needing translation.")

    translated_count = 0
    for entry in tqdm(pending_entries, desc="Translating Definitions"):
        # --- CRITICAL CHANGE: Use filename as the unique key ---
        entry_key = entry.get("filename")
        if not entry_key:
            continue

        headword = entry.get("headword")
        definitions = [
            d.get("definition")
            for d in entry.get("definitions", [])
            if d.get("definition")
        ]

        if not definitions:
            translations[entry_key] = {}  # Mark as processed even if no defs
            continue

        # --- Add precise delay before each API call ---
        time.sleep(REQUEST_INTERVAL_SECONDS)

        result = translate_definitions_for_entry(headword, definitions)

        if result:
            translations[entry_key] = result
            tqdm.write(
                f"[{headword} | {entry_key}] Successfully translated {len(result)} definitions."
            )
            translated_count += 1
        else:
            tqdm.write(f"[{headword} | {entry_key}] Failed to translate.")

        if translated_count > 0 and translated_count % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
            tqdm.write("Progress saved.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    logging.info("--- All pending definitions have been processed. ---")


if __name__ == "__main__":
    main()
