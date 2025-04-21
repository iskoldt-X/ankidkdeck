# Translate definitions in batch mode using a local LLM
import json, os, time, random, re, unicodedata, logging
from collections import OrderedDict
import ollama

# Configuration
MODEL_NAME = "gemma3:12b"
TARGET_LANG = ""
INPUT = "ddo_entries_unique.json"
OUTPUT = f"definition_translations_lemma_gloss_{TARGET_LANG}.json"
MAX_RETRIES = 10
BASE_DELAY = 2
SAVE_EVERY = 1
MAX_DEFS_PER_BATCH = 30  #  Maximum number of definitions to send per batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def try_parse_json(raw: str):
    """
    Strip code fences (```json) and attempt to parse the raw text as JSON.
    Returns the parsed object or None on failure.
    """
    raw = unicodedata.normalize("NFC", raw)
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def translate_definitions_for_entry(headword: str, defs: list[str]) -> dict:
    """
    Call the LLM to translate all definitions for a given headword.
    Each output entry should include 'lemma' and 'gloss'.
    Accepts multiple JSON schemas and returns a mapping:
      original definition -> {lemma, gloss}
    """
    system = (
        f"You are a professional dictionary editor translating Danish definitions into {TARGET_LANG} for this Danish studying program.\n"
        "Do NOT translate the headword; keep it exactly as provided.\n"
        "For each definition, output a JSON object with two fields:\n"
        f"  lemma — the most appropriate {TARGET_LANG} equivalent of the Danish headword for this particular definition (one word or fixed phrase), preserving part of speech;\n"
        f"    • if a natural, commonly‑used word or fixed phrase exists, prefer it over a literal description;\n"
        f"    • gloss — a brief explanatory translation of the definition in {TARGET_LANG}.\n"
        "Return ONLY valid JSON in one of these schemas:\n"
        '  { "headword": ..., "definitions": { "0": {lemma,gloss}, … } }\n'
        'or { "<headword>": { "definitions": { ... } } }\n'
        'or { "<headword>": { "0": {lemma,gloss}, … } }.'
    )

    payload = {str(i): d for i, d in enumerate(defs)}
    last_feedback = None

    for attempt in range(1, MAX_RETRIES + 1):
        logging.info(f"[{headword}] translate attempt {attempt}/{MAX_RETRIES}")

        user_lines = [
            f'Headword: "{headword}"',
            "Input definitions JSON:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
        if last_feedback:
            user_lines.append(f"# NOTE: last attempt error → {last_feedback}")
        user_lines.append("Output format ONLY JSON in one of the accepted schemas")
        user = "\n\n".join(user_lines)

        try:
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            last_feedback = f"ollama error: {e}"
            logging.warning(f"[{headword}] {last_feedback}, retrying…")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        raw = resp["message"]["content"]
        parsed = try_parse_json(raw)
        if not parsed:
            last_feedback = "invalid JSON"
            logging.error(f"[{headword}] {last_feedback}:\n  raw: {raw!r}")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        out_defs = None
        if parsed.get("headword") == headword and "definitions" in parsed:
            out_defs = parsed["definitions"]
        elif headword in parsed and isinstance(parsed[headword], dict):
            cand = parsed[headword]
            if "definitions" in cand and isinstance(cand["definitions"], dict):
                out_defs = cand["definitions"]
            elif all(k.isdigit() for k in cand.keys()):
                out_defs = cand

        if out_defs is None:
            last_feedback = "no definitions in accepted schema"
            logging.error(f"[{headword}] {last_feedback}:\n  raw: {raw!r}")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        if isinstance(out_defs, list):
            if len(out_defs) == len(payload):
                out_defs = {str(i): out_defs[i] for i in range(len(out_defs))}
                logging.info(f"[{headword}] auto-converted list→dict")
            else:
                last_feedback = (
                    f"list length {len(out_defs)} != expected {len(payload)}"
                )
                logging.error(f"[{headword}] {last_feedback}")
                time.sleep(BASE_DELAY * attempt + random.random())
                continue

        if not isinstance(out_defs, dict) or set(out_defs.keys()) != set(
            payload.keys()
        ):
            last_feedback = f"keys mismatch, got {set(out_defs.keys())}"
            logging.error(f"[{headword}] {last_feedback}")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        result = {}
        for idx, val in out_defs.items():
            if isinstance(val, dict) and "lemma" in val and "gloss" in val:
                entry = val
            elif isinstance(val, str):

                entry = {"lemma": val, "gloss": val}
            else:
                last_feedback = f"invalid entry type for {idx}: {type(val).__name__}"
                logging.error(f"[{headword}] {last_feedback}")
                entry = {"lemma": "", "gloss": ""}

            result[payload[idx]] = entry

        return result

    logging.error(
        f"[{headword}] failed after {MAX_RETRIES}, last feedback: {last_feedback}"
    )
    raise RuntimeError(f"[{headword}] failed after {MAX_RETRIES}: {last_feedback}")


def save_progress(done: dict):
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)


def main():
    entries = json.load(open(INPUT, encoding="utf-8"))
    done = json.load(open(OUTPUT, encoding="utf-8")) if os.path.exists(OUTPUT) else {}

    completed = 0
    for entry in entries:
        hw = entry["headword"]
        if hw in done:
            continue

        defs = [
            d["definition"] for d in entry.get("definitions", []) if d.get("definition")
        ]
        if not defs:
            done[hw] = {}
            continue

        if len(defs) > MAX_DEFS_PER_BATCH:
            logging.info(
                f"[{hw}] {len(defs)} definitions exceed batch size, splitting into batches of {MAX_DEFS_PER_BATCH}"
            )
            merged = {}
            for idx in range(0, len(defs), MAX_DEFS_PER_BATCH):
                batch = defs[idx : idx + MAX_DEFS_PER_BATCH]
                try:
                    part = translate_definitions_for_entry(hw, batch)
                    merged.update(part)
                    logging.info(
                        f"[{hw}] batch {idx//MAX_DEFS_PER_BATCH + 1} complete, {len(part)} entries"
                    )
                except Exception as e:
                    logging.error(
                        f"[{hw}] batch {idx//MAX_DEFS_PER_BATCH + 1} failed: {e}"
                    )

                    merged = {}
                    break
            done[hw] = merged
            logging.info(f"[{hw}] merged total {len(merged)} definitions.")
        else:
            try:
                done[hw] = translate_definitions_for_entry(hw, defs)
                logging.info(f"[{hw}] saved {len(done[hw])} defs with lemma/gloss.")
            except Exception as e:
                logging.error(str(e))
                continue

        completed += 1
        if completed % SAVE_EVERY == 0:
            save_progress(done)
    logging.info("All done! ✓")


if __name__ == "__main__":
    main()
