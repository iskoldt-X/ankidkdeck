# Output the Danish only Anki deck
json_path = "ddo_entries_unique.json"
audio_map_path = "audio_map.json"
output_apkg = "danish.apkg"
limit = None

import json
from pathlib import Path
from random import randrange
import genanki


DECK_ID = randrange(1 << 30, 1 << 31)
MODEL_ID = randrange(1 << 30, 1 << 31)

FIELDS = [
    {"name": "Headword"},
    {"name": "POS"},
    {"name": "IPA"},
    {"name": "Definition"},
    {"name": "Grammar"},
    {"name": "Example"},
    {"name": "Variants"},
    {"name": "Collocations"},
    {"name": "Derivatives"},
    {"name": "Etymology"},
]

CSS = """
.card { font-family: sans-serif; line-height:1.35; color:#222; }
.front-container { text-align:center; }
.front-headword { font-size:2.2em; font-weight:600; }
.front-pos { font-size:0.9em; color:#555; margin-top:4px; }
.front-ipa { margin-top:8px; }
.ipa-row { margin:4px 0; display:flex; align-items:center; justify-content:center; }
.ipa-text { font-size:1.1em; color:#555; margin-right:6px; }

/* Anki replay button */

:root {
  --replay-bg: #a8d5ba;
  --replay-icon: #2f5d62;
}

.replay-button svg {
  width: 20px;
  height: 20px;
}
.replay-button svg circle {
  fill: var(--replay-bg);
}
.replay-button svg path {
  stroke: var(--replay-icon);
  fill: var(--replay-icon);
}

.core-section { font-size:1.25em; margin-bottom:4px; }
.small-section { font-size:0.9em; color:#444; margin-top:6px; }
hr { margin:10px 0; }
"""

QFMT = """
<div class="front-container">
  <div class="front-headword">{{Headword}}</div>
  {{#POS}}<div class="front-pos">{{POS}}</div>{{/POS}}
  <div class="front-ipa">{{IPA}}</div>
</div>
"""

AFMT = """
{{FrontSide}}
<hr>
<div class="core-section">{{Definition}}</div>
{{#Grammar}}<div class="small-section"><b>Grammar:</b> {{Grammar}}</div>{{/Grammar}}
{{#Example}}<div class="small-section"><b>Example:</b> {{Example}}</div>{{/Example}}
{{#Variants}}<div class="small-section"><b>Variants:</b> {{Variants}}</div>{{/Variants}}
{{#Collocations}}<div class="small-section"><b>Collocations:</b> {{Collocations}}</div>{{/Collocations}}
{{#Derivatives}}<div class="small-section"><b>Derivatives:</b> {{Derivatives}}</div>{{/Derivatives}}
{{#Etymology}}<div class="small-section"><b>Etymology:</b> {{Etymology}}</div>{{/Etymology}}
"""

MODEL = genanki.Model(
    MODEL_ID,
    "DDO Danish Lexeme v19",
    fields=FIELDS,
    templates=[{"name": "Card 1", "qfmt": QFMT, "afmt": AFMT}],
    css=CSS,
    sort_field_index=0,
)


def sanitize(text: str) -> str:
    """
    Remove all braces and trim whitespace.
    """
    return text.replace("{", "").replace("}", "").strip() if text else ""


def build_ipa_html(entry, audio_map):
    """
    - For each pronunciation, generate a [sound:filename] tag to enable Anki replay button
    - Then display the IPA text; the replay button appears automatically after the text
    """
    rows = []
    for u in entry.get("udtale", []):
        ipa_raw = sanitize(u.get("ipa", ""))
        if not ipa_raw:
            continue
        core = ipa_raw.strip("[]").strip()
        label = u.get("label")
        label_html = (
            f'<span class="ipa-label">({sanitize(label)})</span>' if label else ""
        )
        url = u.get("audio", "")
        sound_tag = f"[sound:{Path(audio_map[url]).name}]" if url in audio_map else ""
        rows.append(
            f'<div class="ipa-row">'
            f'<span class="ipa-text">[{core}]</span>'
            f"{label_html}"
            f"{sound_tag}"
            f"</div>"
        )
    return "".join(rows).rstrip("} \n")


def extract_definitions(entry):
    parts = []
    for d in entry.get("definitions", []):
        num = sanitize(d.get("number", ""))
        txt = sanitize(d.get("definition", ""))
        if txt:
            parts.append((num + " " + txt).strip())
    return "<br>".join(parts)


def extract_first_example(entry):
    for d in entry.get("definitions", []):
        for ex in d.get("examples") or []:
            txt = sanitize(ex.get("text", ""))
            if txt:
                return txt
    return ""


def join_safe(lst, sep="; "):
    return sep.join(lst) if lst else ""


entries = json.load(open(json_path, encoding="utf-8"))
audio_map = json.load(open(audio_map_path, encoding="utf-8"))
if limit:
    entries = entries[:limit]


entries_by_name = {Path(e["file"]).stem.lower(): e for e in entries}
ordered = []
seen = set()
for w in wordlist:
    if w in entries_by_name:
        ordered.append(entries_by_name[w])
        seen.add(w)
# append the rest in original order
for e in entries:
    name = Path(e["file"]).stem.lower()
    if name not in seen:
        ordered.append(e)
entries = ordered


deck = genanki.Deck(DECK_ID, "Danish • DDO Core Vocabulary v19")
media_files = []

for entry in entries:
    hw = sanitize(entry.get("headword", ""))
    pos_full = sanitize(entry.get("pos", ""))
    ipa_html = build_ipa_html(entry, audio_map)

    for u in entry.get("udtale", []):
        url = u.get("audio", "")
        if url in audio_map:
            media_files.append(audio_map[url])

    definition = extract_definitions(entry) or "(no definition)"
    grammar = sanitize(
        next(
            (
                d.get("grammar")
                for d in entry.get("definitions", [])
                if d.get("grammar")
            ),
            "",
        )
    )
    example = extract_first_example(entry)
    variants = join_safe(entry.get("wordforms", []))
    collocs = join_safe(
        [fx.get("expression", "") for fx in entry.get("fixed_expressions", [])]
    )
    od = entry.get("orddannelser", {})
    derivs = (
        od.get("Afledninger", []) + od.get("Sammensætninger", []) + od.get("Øvrige", [])
    )
    derivatives = join_safe(derivs, ", ")
    raw_ety = (entry.get("etymology") or {}).get("raw", "")
    etymology = sanitize(raw_ety)[:120] + "…" if raw_ety else ""

    note = genanki.Note(
        model=MODEL,
        fields=[
            hw,
            pos_full,
            ipa_html,
            definition,
            grammar,
            example,
            variants,
            collocs,
            derivatives,
            etymology,
        ],
        guid=genanki.guid_for(hw, pos_full),
    )
    deck.add_note(note)

pkg = genanki.Package(deck)
pkg.media_files = media_files
pkg.write_to_file(output_apkg)
print(
    f"✓ Generated {output_apkg}: {len(deck.notes)} cards, {len(media_files)} media files"
)
