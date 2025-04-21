# Generate ddo_entries.json from downloaded HTML files
import os
import json
import re
from bs4 import BeautifulSoup, Tag

SUFFIX_RE = re.compile(r"^-[A-Za-zæøåÆØÅ]{1,6}$")
UNWANTED_TEXTS = {
    "...vis mere",
    "...vis mindre",
    "Læs mere om Den Danske Begrebsordbog",
}
DIGITS_TRAIL_RE = re.compile(r"\d+$")

HTML_DIR = "./ddo_html/"
OUTPUT = "ddo_entries.json"


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
        cat = box.select_one("span.stempel").get_text(strip=True)
        inline = box.select_one("span.inlineList")
        items = []
        if inline:
            for child in inline.children:
                if hasattr(child, "name") and child.name == "a":
                    form = child.get_text(strip=True)
                    # 捕获紧随的文本节点
                    tail = ""
                    sib = child.next_sibling
                    while sib and isinstance(sib, str):
                        tail += sib
                        sib = sib.next_sibling
                    tail = tail.strip()
                    if tail:
                        items.append(f"{form} {tail}")
                    else:
                        items.append(form)
        result[cat] = items
    return result


def parse_udtale(soup, fn):
    """
    Retrieve pronunciation blocks in page order and associate the nearest preceding 'diskret' as label.
    Returns a list of dictionaries with ipa, audio URL, and optional label.

    """
    from bs4 import Tag

    out = []
    container = soup.select_one("div#id-udt")
    if not container:
        print(f"Note {fn}: no pronunciation block")
        return out

    # The entire block of text is within <span class="tekstmedium">
    tekstmedium = container.select_one("span.tekstmedium")
    if not tekstmedium:
        return out

    children = list(tekstmedium.children)
    for idx, node in enumerate(children):
        if not (isinstance(node, Tag) and "lydskrift" in node.get("class", [])):
            continue

        ipa = node.get_text(" ", strip=True)
        a_mp3 = node.select_one('a[href$=".mp3"]')
        audio = a_mp3["href"] if a_mp3 else None

        # Search backwards for a label: find the nearest preceding 'diskret'
        # before the current 'lydskrift' node; stop if another 'lydskrift' is encountered.
        label = None
        for prev in reversed(children[:idx]):
            if not isinstance(prev, Tag):
                continue
            cls = prev.get("class", [])
            if "diskret" in cls:
                label = prev.get_text(" ", strip=True)
                break
            if "lydskrift" in cls:
                break

        out.append({"ipa": ipa, "audio": audio, "label": label})

    return out


def parse_wordforms(soup):
    box = soup.find("span", class_="stempel", string="Bøjning")
    if not box:
        return []
    sib = box.find_next_sibling("span")
    text = sib.get_text(strip=True) if sib else ""
    return [f.strip() for f in text.split(",") if f.strip()]


def parse_etymology(soup):
    box = soup.find("span", class_="stempel", string="Oprindelse")
    if not box:
        return None
    span = box.find_next_sibling("span", class_="tekstmedium")
    if not span:
        return None
    raw = span.get_text(" ", strip=True)
    segments, desc = [], ""
    for node in span.contents:
        if isinstance(node, str):
            desc += node
        elif node.name in ("span", "a") and (
            "ordform" in node.get("class", []) or node.name == "a"
        ):
            form = node.get_text(strip=True)
            segments.append({"form": form, "description": desc.strip(" ,")})
            desc = ""
        elif "dividerDot" in node.get("class", []):
            continue
        else:
            desc += node.get_text("", strip=True)
    return {"raw": raw, "segments": segments}


def parse_definitions(soup, fn):
    out = []
    container = soup.find(id="content-betydninger")
    if not container:
        return out
    for num_tag in container.select("div.definitionNumber"):
        num = num_tag.get_text(strip=True)
        indent = num_tag.find_next_sibling()
        while indent and not (
            isinstance(indent, Tag) and "definitionIndent" in indent.get("class", [])
        ):
            indent = indent.find_next_sibling()
        if not indent:
            print(f"Warning {fn}: no definitionIndent for sense {num}")
            continue
        entry = {"number": num}
        def_box = indent.select_one(
            'div.definitionBox[id^="betydning-"] span.definition'
        )
        entry["definition"] = def_box.get_text(" ", strip=True) if def_box else None
        gramm = indent.select_one("div.definitionBox.grammatik")
        entry["grammar"] = (
            (gramm.select_one("span.inlineList") or gramm).get_text(" ", strip=True)
            if gramm
            else None
        )
        if sa := indent.select_one("div.definitionBox.onym"):
            entry["see_also"] = clean_links(sa.select("a"))
        if rel := indent.select_one("div.definitionBox.rel-begreber"):
            entry["related"] = clean_links(rel.select("a"))
        entry["examples"] = []
        for cite in indent.select("div.citat-box"):
            txt = cite.select_one("span.citat").get_text(" ", strip=True)
            src = cite.select_one("span.kilde")
            entry["examples"].append(
                {
                    "text": txt,
                    "source": src.get_text(strip=True) if src else None,
                }
            )
        out.append(entry)
    return out


def parse_fixed_expressions(soup):
    out = []
    art = soup.select_one("div.artikel")
    sec = art.select_one("#content-faste-udtryk") if art else None
    if not sec:
        return out
    for ud in sec.find_all("div", id=re.compile(r"^udtryk-\d+")):
        match = ud.select_one("span.match")
        if not match:
            continue
        expr = match.get_text(strip=True)
        ud_id = ud.get("id", "")
        details = []
        node = ud.next_sibling
        while node:
            if isinstance(node, Tag):
                if node.name == "div" and node.get("id", "").startswith("udtryk-"):
                    break
                if "definitionIndent" in node.get("class", []):
                    for box in node.find_all(
                        "div", class_="definitionBox", recursive=False
                    ):
                        cls = box.get("class", [])
                        box_id = box.get("id", "")
                        if ud_id and box_id.startswith(f"{ud_id}-betydning"):
                            details.append(
                                {
                                    "type": "definition",
                                    "text": box.get_text(" ", strip=True),
                                }
                            )
                        elif "onym" in cls:
                            items = clean_links(box.select("a"))
                            details.append({"type": "see_also", "items": items})
                        elif "rel-begreber" in cls:
                            items = clean_links(box.select("a"))
                            details.append({"type": "related", "items": items})
                        elif "grammatik" in cls:
                            inline = box.select_one("span.inlineList")
                            text = (inline or box).get_text(" ", strip=True)
                            details.append({"type": "grammar", "text": text})
                        else:
                            st = box.select_one("span.stempel")
                            if st and st.get_text(strip=True) == "SPROGBRUG":
                                usage = box.select_one("span.tekstnormal")
                                if usage:
                                    details.append(
                                        {
                                            "type": "usage",
                                            "text": usage.get_text(strip=True),
                                        }
                                    )
                                continue
                            if cite := box.select_one("div.citat-box"):
                                txt = cite.select_one("span.citat").get_text(
                                    " ", strip=True
                                )
                                src = cite.select_one("span.kilde")
                                details.append(
                                    {
                                        "type": "example",
                                        "text": txt,
                                        "source": (
                                            src.get_text(strip=True) if src else None
                                        ),
                                    }
                                )
            node = node.next_sibling
        out.append({"expression": expr, "details": details})
    return out


def parse_entry(path):
    fn = os.path.basename(path)
    soup = BeautifulSoup(open(path, encoding="utf-8"), "html.parser")
    art = soup.select_one("div.artikel")
    if not art:
        return None

    raw_head = art.select_one("div.definitionBoxTop span.match").get_text(strip=True)
    headword = DIGITS_TRAIL_RE.sub("", raw_head.strip()).strip()

    pos_tag = art.select_one("div.definitionBoxTop span.tekstmedium")
    pos = pos_tag.get_text(strip=True) if pos_tag else None

    raw_forms = [f for f in parse_wordforms(soup) if f != raw_head]
    forms = transform_wordforms(headword, raw_forms)

    return {
        "file": fn,
        "headword": headword,
        "pos": pos,
        "udtale": parse_udtale(soup, fn),
        "wordforms": forms,
        "etymology": parse_etymology(soup),
        "definitions": parse_definitions(soup, fn),
        "fixed_expressions": parse_fixed_expressions(soup),
        "orddannelser": parse_orddannelser(soup),
    }


def main():
    entries = []
    skipped_files = []

    for fn in sorted(os.listdir(HTML_DIR)):
        if not fn.endswith(".html"):
            continue
        entry = parse_entry(os.path.join(HTML_DIR, fn))
        if entry:
            entries.append(entry)
        else:
            skipped_files.append(fn)

    print(f"Parsed {len(entries)} entries.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files:")
        for name in skipped_files:
            print("  -", name)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
