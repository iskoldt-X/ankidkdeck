# Batch translation of fixed expressions using a local LLM

import json, os, time, random, re, unicodedata, logging
import ollama

# Configuration
MODEL_NAME = "gemma3:12b"
TARGET_LANG = ""
INPUT = "ddo_entries_unique.json"
OUTPUT = f"expr_translations_{TARGET_LANG}.json"
MAX_RETRIES = 10
BASE_DELAY = 2
SAVE_EVERY = 1
MAX_EXPR_PER_BATCH = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def try_parse_json(raw: str):
    raw = unicodedata.normalize("NFC", raw)
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def translate_fixed_expressions_for_entry(headword: str, exprs: list[str]) -> dict:
    """
    Call the LLM to translate all fixed expressions for a given headword.
    Each output entry should include 'lemma' and 'gloss'.
    Returns a mapping of original expression to {'lemma', 'gloss'}.
    """
    system = (
        f"You are a professional dictionary editor translating Danish fixed expressions into {TARGET_LANG}.\n"
        "Do NOT translate the headword; keep it exactly as provided.\n"
        "For each Danish expression, output a JSON object with:\n"
        f"  lemma — the most natural {TARGET_LANG} equivalent (one word or fixed phrase),\n"
        f"  gloss — a brief explanatory translation in {TARGET_LANG}.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        '  "headword": ..., \n'
        '  "fixed_expressions": {\n'
        '    "0": {"lemma":..., "gloss":...},\n'
        "    …\n"
        "  }\n"
        "}"
    )

    payload = {str(i): expr for i, expr in enumerate(exprs)}
    last_feedback = None

    for attempt in range(1, MAX_RETRIES + 1):
        logging.info(f"[{headword}] expr translate attempt {attempt}/{MAX_RETRIES}")
        user = "\n\n".join(
            [
                f'Headword: "{headword}"',
                "Input fixed_expressions JSON:",
                json.dumps(payload, ensure_ascii=False, indent=2),
                *([f"# NOTE: last error → {last_feedback}"] if last_feedback else []),
                "Output format ONLY JSON as per schema.",
            ]
        )

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
            logging.error(f"[{headword}] {last_feedback}:\n{raw!r}")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        out = None
        if parsed.get("headword") == headword and "fixed_expressions" in parsed:
            out = parsed["fixed_expressions"]
        elif headword in parsed and isinstance(parsed[headword], dict):
            cand = parsed[headword]
            if "fixed_expressions" in cand:
                out = cand["fixed_expressions"]

        if not isinstance(out, dict) or set(out.keys()) != set(payload.keys()):
            last_feedback = (
                f"keys mismatch: got {set(out.keys()) if isinstance(out,dict) else out}"
            )
            logging.error(f"[{headword}] {last_feedback}")
            time.sleep(BASE_DELAY * attempt + random.random())
            continue

        result = {}
        for idx, val in out.items():
            if isinstance(val, dict) and "lemma" in val and "gloss" in val:
                result[payload[idx]] = val
            else:
                last_feedback = f"invalid entry at {idx}: {val}"
                logging.error(f"[{headword}] {last_feedback}")
                result[payload[idx]] = {"lemma": "", "gloss": ""}

        return result

    raise RuntimeError(
        f"[{headword}] failed after {MAX_RETRIES}, last: {last_feedback}"
    )


def save_progress(done: dict):
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)


def main():
    entries = json.load(open(INPUT, encoding="utf-8"))
    done = json.load(open(OUTPUT, encoding="utf-8")) if os.path.exists(OUTPUT) else {}

    for entry in entries:
        hw = entry["headword"]

        if hw in done and "fixed_expressions" in done[hw]:
            continue

        exprs = [e["expression"] for e in entry.get("fixed_expressions", [])]
        if not exprs:
            done.setdefault(hw, {})["fixed_expressions"] = {}
        else:

            merged = {}
            for i in range(0, len(exprs), MAX_EXPR_PER_BATCH):
                batch = exprs[i : i + MAX_EXPR_PER_BATCH]
                try:
                    part = translate_fixed_expressions_for_entry(hw, batch)
                    merged.update(part)
                    logging.info(
                        f"[{hw}] batch {i//MAX_EXPR_PER_BATCH+1} done: {len(part)} exprs"
                    )
                except Exception as e:
                    logging.error(f"[{hw}] batch {i//MAX_EXPR_PER_BATCH+1} failed: {e}")
                    merged = {}
                    break
            done.setdefault(hw, {})["fixed_expressions"] = merged

        if entry["headword"] in done and SAVE_EVERY and len(done) % SAVE_EVERY == 0:
            save_progress(done)

    save_progress(done)
    logging.info("All fixed_expressions translated! ✓")


if __name__ == "__main__":
    main()
