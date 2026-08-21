# 02_generate_entries.py
#
# Parses all downloaded HTML files based on the download_map.json,
# extracts detailed lexical information, and compiles it into a single
# structured JSON file (ddo_entries.json).

import os
import json
import re
import logging
from bs4 import BeautifulSoup, Tag
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
HTML_DIR = "ddo_html_all_versions"
MAP_FILE = "download_map.json"
OUTPUT_FILE = "ddo_entries.json"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Regular Expressions and Constants from old script ---
SUFFIX_RE = re.compile(r"^-[A-Za-zæøåÆØÅ]{1,6}$")
UNWANTED_TEXTS = {
    "...vis mere",
    "...vis mindre",
    "Læs mere om Den Danske Begrebsordbog",
}
DIGITS_TRAIL_RE = re.compile(r"\d+$")

# --- Parsing Functions (largely from your previous script, with minor tweaks) ---


def transform_wordforms(headword, forms):
    out = []
    for f in forms:
        if SUFFIX_RE.match(f):
            out.append(headword + f[1:])
        else:
            out.append(f)
    return out


def clean_links(a_tags):
    out = []
    for a in a_tags:
        txt = a.get_text(strip=True)
        if txt in UNWANTED_TEXTS:
            continue
        txt = DIGITS_TRAIL_RE.sub("", txt)
        if txt:
            out.append(txt)
    return out


def parse_orddannelser(soup):
    cont = soup.select_one("#content-orddannelser")
    if not cont:
        return {}
    result = {}
    for box in cont.find_all("div", class_="definitionBox"):
        if cat_tag := box.select_one("span.stempel"):
            cat = cat_tag.get_text(strip=True)
            items = []
            if inline := box.select_one("span.inlineList"):
                for child in inline.children:
                    if isinstance(child, Tag) and child.name == "a":
                        form = child.get_text(strip=True)
                        tail = (
                            child.next_sibling.strip()
                            if child.next_sibling
                            and isinstance(child.next_sibling, str)
                            else ""
                        )
                        items.append(f"{form} {tail}".strip())
            result[cat] = items
    return result


def parse_udtale(soup):
    out = []
    container = soup.select_one("div#id-udt")
    if not container or not (tekstmedium := container.select_one("span.tekstmedium")):
        return out
    children = list(tekstmedium.children)
    for idx, node in enumerate(children):
        if not (isinstance(node, Tag) and "lydskrift" in node.get("class", [])):
            continue
        ipa = node.get_text(" ", strip=True)
        audio_url = (
            node.select_one('a[href$=".mp3"]')["href"]
            if node.select_one('a[href$=".mp3"]')
            else None
        )
        label = None
        for prev in reversed(children[:idx]):
            if isinstance(prev, Tag):
                cls = prev.get("class", [])
                if "diskret" in cls:
                    label = prev.get_text(" ", strip=True)
                    break
                if "lydskrift" in cls:
                    break
        out.append({"ipa": ipa, "audio": audio_url, "label": label})
    return out


def parse_wordforms(soup):
    box = soup.find("span", class_="stempel", string="Bøjning")
    if not box or not (sib := box.find_next_sibling("span")):
        return []
    text = sib.get_text(strip=True)
    return [f.strip() for f in text.split(",") if f.strip()]


def parse_etymology(soup):
    box = soup.find("span", class_="stempel", string="Oprindelse")
    if not box or not (span := box.find_next_sibling("span", class_="tekstmedium")):
        return None
    raw = span.get_text(" ", strip=True)
    segments, desc = [], ""
    for node in span.contents:
        if isinstance(node, str):
            desc += node
        elif isinstance(node, Tag) and (
            node.name in ("span", "a")
            and ("ordform" in node.get("class", []) or node.name == "a")
        ):
            form = node.get_text(strip=True)
            segments.append({"form": form, "description": desc.strip(" ,")})
            desc = ""
        elif isinstance(node, Tag) and "dividerDot" in node.get("class", []):
            continue
        else:
            desc += node.get_text("", strip=True)
    # Add the last segment if any
    if desc.strip():
        if segments:
            segments[-1]["description"] += " " + desc.strip()
        else:
            # This case handles etymologies with no linked forms
            segments.append({"form": None, "description": desc.strip()})

    return {"raw": raw, "segments": segments}


def parse_definitions(soup):
    out = []
    container = soup.find(id="content-betydninger")
    if not container:
        return out
    for num_tag in container.select("div.definitionNumber"):
        num = num_tag.get_text(strip=True)
        indent = num_tag.find_next_sibling("div", class_="definitionIndent")
        if not indent:
            continue
        entry = {"number": num}
        if def_box := indent.select_one(
            'div.definitionBox[id^="betydning-"] span.definition'
        ):
            entry["definition"] = def_box.get_text(" ", strip=True)
        if gramm := indent.select_one("div.definitionBox.grammatik"):
            entry["grammar"] = (gramm.select_one("span.inlineList") or gramm).get_text(
                " ", strip=True
            )
        if sa := indent.select_one("div.definitionBox.onym"):
            entry["see_also"] = clean_links(sa.select("a"))
        if rel := indent.select_one("div.definitionBox.rel-begreber"):
            entry["related"] = clean_links(rel.select("a"))
        entry["examples"] = []
        for cite in indent.select("div.citat-box"):
            txt_tag = cite.select_one("span.citat")
            src_tag = cite.select_one("span.kilde")
            if txt_tag:
                entry["examples"].append(
                    {
                        "text": txt_tag.get_text(" ", strip=True),
                        "source": src_tag.get_text(strip=True) if src_tag else None,
                    }
                )
        out.append(entry)
    return out


def parse_fixed_expressions(soup):
    out = []
    art = soup.select_one("div.artikel")
    if not art or not (sec := art.select_one("#content-faste-udtryk")):
        return out
    for expr_div in sec.find_all("div", id=re.compile(r"^udtryk-\d+")):
        if not (match := expr_div.select_one("span.match")):
            continue
        expr = match.get_text(strip=True)
        details_div = expr_div.find_next_sibling("div", class_="definitionIndent")
        details = []
        if details_div:
            # This logic can be simplified if we just grab all text/examples under the expression
            # For now, keeping a simplified version
            for box in details_div.find_all(
                "div", class_="definitionBox", recursive=False
            ):
                if sub_def := box.select_one("span.definition"):
                    details.append(
                        {
                            "type": "definition",
                            "text": sub_def.get_text(" ", strip=True),
                        }
                    )
                if cite := box.select_one("div.citat-box"):
                    details.append(
                        {"type": "example", "text": cite.get_text(" ", strip=True)}
                    )
        out.append({"expression": expr, "details": details})
    return out


def parse_entry(html_path: Path, metadata: dict) -> dict | None:
    """
    Parses a single HTML file and combines its content with the provided metadata.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except FileNotFoundError:
        logging.error(f"HTML file not found: {html_path}")
        return None

    art = soup.select_one("div.artikel")
    if not art:
        logging.warning(f"No 'div.artikel' found in {html_path.name}. Skipping.")
        return None

    raw_head_tag = art.select_one("div.definitionBoxTop span.match")
    if not raw_head_tag:
        logging.warning(f"No headword found in {html_path.name}. Skipping.")
        return None

    raw_head = raw_head_tag.get_text(strip=True)
    headword = DIGITS_TRAIL_RE.sub("", raw_head).strip()

    pos_tag = art.select_one("div.definitionBoxTop span.tekstmedium")
    pos = pos_tag.get_text(strip=True) if pos_tag else None

    raw_forms = [f for f in parse_wordforms(soup) if f != raw_head]
    forms = transform_wordforms(headword, raw_forms)

    entry_data = {
        "headword": headword,
        "pos": pos,
        "udtale": parse_udtale(soup),
        "wordforms": forms,
        "etymology": parse_etymology(soup),
        "definitions": parse_definitions(soup),
        "fixed_expressions": parse_fixed_expressions(soup),
        "orddannelser": parse_orddannelser(soup),
    }

    # Combine parsed data with metadata from download_map.json
    # The filename is already part of the key, so we'll add it to the entry itself.
    full_entry = {"filename": html_path.name, **metadata, **entry_data}

    return full_entry


def main():
    """
    Main function to drive the parsing process.
    """
    if not os.path.exists(MAP_FILE):
        logging.critical(
            f"Map file '{MAP_FILE}' not found. Please run 01_download_all_ddo_versions.py first."
        )
        return

    with open(MAP_FILE, "r", encoding="utf-8") as f:
        download_map = json.load(f)

    logging.info(
        f"Loaded {len(download_map)} entries from map file. Starting parsing..."
    )

    all_entries = []
    skipped_files = []

    # Use tqdm for a progress bar
    for filename, metadata in tqdm(download_map.items(), desc="Parsing HTML files"):
        html_path = Path(HTML_DIR) / filename
        if not html_path.exists():
            skipped_files.append(filename)
            continue

        entry = parse_entry(html_path, metadata)
        if entry:
            all_entries.append(entry)
        else:
            skipped_files.append(filename)

    logging.info(f"Successfully parsed {len(all_entries)} entries.")
    if skipped_files:
        logging.warning(
            f"Skipped {len(skipped_files)} files (not found or failed to parse):"
        )
        for name in skipped_files[:10]:  # Log first 10 skipped files
            logging.warning(f"  - {name}")

    logging.info(f"Saving all entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    logging.info("✔ Parsing complete!")


if __name__ == "__main__":
    main()
