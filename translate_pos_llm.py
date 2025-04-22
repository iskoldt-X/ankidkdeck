# Use a local LLM to translate part-of-speech labels

import json
import re
import time
import random
import unicodedata
import logging
from collections import defaultdict, Counter
import ollama

# Configuration
TARGET_LANG = ""
MODEL_NAME = "gemma3:12b-it-q8_0"
MAX_RETRIES = 5
BASE_DELAY = 2
INPUT_JSON = "ddo_entries_unique.json"
OUTPUT_JSON = f"pos_translations_{TARGET_LANG}.json"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def try_parse_json(raw: str) -> dict | None:
    raw = unicodedata.normalize("NFC", raw)
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.debug(f"Failed to parse JSON: {raw}")
        return None


def translate_all(tags: list[str]) -> dict[str, str]:
    """
    Translate all POS tags in one request, returning a mapping of {original_tag: translated_tag}.
    Prompts are adjusted dynamically based on the target language.
    """
    payload = {tag: "" for tag in tags}

    system_prompt = (
        "You are a translation assistant specialized in glossary labels."
        f"\nTranslate each Danish part-of-speech term into {TARGET_LANG}, preserving punctuation."
    )

    if TARGET_LANG.lower() == "chinese":
        system_prompt += "\nDo NOT include any phonetic transcription or romanization."
    system_prompt += (
        "\nReturn only a JSON object with the same keys and translated values."
    )

    # Base system prompt for all languages
    user_prompt = (
        "Translate the following JSON dictionary of POS tags into "
        + TARGET_LANG
        + ":\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        delay = BASE_DELAY * attempt + random.random()
        logging.info(
            f"Translation attempt {attempt}/{MAX_RETRIES}, delay {delay:.1f}s..."
        )
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp["message"]["content"]
        parsed = try_parse_json(raw)

        if not parsed:
            logging.warning(f"Attempt {attempt}: no valid JSON parsed, retrying...")
        else:
            if set(parsed.keys()) != set(payload.keys()):
                missing = set(payload) - set(parsed)
                extra = set(parsed) - set(payload)
                logging.warning(f"Key mismatch: missing {missing}, extra {extra}")
            else:
                # Check for suspicious duplicates
                freq = defaultdict(list)
                for k, v in parsed.items():
                    freq[v].append(k)
                suspicious = {t: ks for t, ks in freq.items() if len(ks) > 4}
                if suspicious:
                    logging.warning(f"Suspicious duplicate translations: {suspicious}")
                else:
                    logging.info("Translation successful!")
                    return parsed

        time.sleep(delay)

    raise RuntimeError(f"Translation failed after {MAX_RETRIES} attempts.")


if __name__ == "__main__":
    entries = json.load(open(INPUT_JSON, encoding="utf-8"))
    pos_counter = Counter(e.get("pos", "") for e in entries if e.get("pos"))
    tags = [t for t, _ in pos_counter.most_common()]
    logging.info(f"Found {len(tags)} unique POS tags, translating → {TARGET_LANG}.")

    mapping = translate_all(tags)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logging.info(f"Done! Saved translations to {OUTPUT_JSON}")
