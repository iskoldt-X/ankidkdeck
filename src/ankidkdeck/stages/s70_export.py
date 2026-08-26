"""Stage 70: render one card per family and write the .apkg -- behind every gate.

IDENTITY IS BYTE-FROZEN. Deck id, model id, the 8 field names, the sort field
index, the Chinese deck-id override and the GUID formula are all reproduced from
the v2.1 exporter exactly. They are what makes a re-import an upgrade instead of
4,442 duplicate cards, and none of them may be "improved":

    LANG_HASH = zlib.adler32(lang) & 0x7FFFFFFF
    DECK_ID   = 0x10000000 + LANG_HASH        (Chinese: 1745409457359, legacy)
    MODEL_ID  = 0x20000000 + LANG_HASH
    FIELDS    = 8 names, sort_field_index = 7
    GUID      = genanki.guid_for(guid_seed, lang)

What DID change from v2.1 (guide D10, deliberate and gate-visible):

  * The unit is a FAMILY, not a query word: the anchor article heads the card and
    every inflection/variant of the family points at the same GUID.
  * The GUID seed comes from words.json (which copied it from the append-only
    registry). It is never derived from data at render time -- guid_for() hashes
    the bytes and 55 (headword, wordlist word) pairs differ only by case, with
    ('Er','er') at rank 1.
  * The card front groups by data-pos-key and shows the DDO display string;
    2025's 41 mangled POS strings are not translation keys any more.
  * Variants carries the paradigm block and the alternative spellings, and is
    honestly EMPTY when there are neither -- AFMT wraps it in a {{#Variants}}
    section, so a never-empty field printed a bare "Variants:" label. The hidden
    searchable-forms span that makes Anki search find the card by any member form
    rides at the end of Content instead, which is rendered unconditionally.
  * A missing translation for a renderable sense is a GATE FAILURE (G-COV), not
    a silently bare definition. That defect shipped 2,007 bare English cards.
  * COPYRIGHT_YEAR comes from config, never datetime.now().year -- that single
    environment dependency is what broke byte-parity in the v2.1 rebuild.

Oevrige (`#id-oevr`) is still EXCLUDED from Derivatives: the final guide argues
for including it, the owner deferred that change to a later version (2026-08-24).
"""

import json
import re
import sqlite3
import tempfile
import zipfile
import zlib
from pathlib import Path

from ..config import Config
from ..gates import (G_COV, G_DET, G_EMPTY_C, G_GUID, G_MEDIA, G_NOTE, G_RANK,
                     G_RATE, G_REL, G_SEED, G_SEP, G_SITEMAP, Gate,
                     dense_unique_ranks, guid_diff_reconciles,
                     registry_seed_bytes, run_gates, separator_golden,
                     sitemap_shortfall)
from ..util import NFC, FatalError, canonical_json, read_json, write_json

FIELD_NAMES = ["QueryWord", "FrontSideSummary", "Content", "Collocations",
               "Variants", "Derivatives", "Etymology", "FrequencyRank"]
SORT_FIELD_INDEX = 7
CONTENT_INDEX = FIELD_NAMES.index("Content")

# The hidden searchable-forms span is display:none, so it is not content a human
# can see -- and both empty-field gates measure what a human sees. Without this,
# putting the span in a field makes that field permanently non-empty and turns
# its gate row into a tautology, which is exactly what it did to Variants.
HIDDEN_SPAN_RE = re.compile(r'<span class="searchable-forms">.*?</span>', re.S)


def visible(field_text: str) -> str:
    """A field's text with the display:none searchable-forms span removed."""
    return HIDDEN_SPAN_RE.sub("", field_text or "")


DEFAULT_POS_KEY = "Uncategorized"       # v2.1's default_pos_key, unchanged
# The stage-20 schema keys, not raw DDO display text. Oevrige is parsed but not
# rendered (owner deferred that change, 2026-08-24).
DERIVATIVE_LABELS = ("Afledninger", "Sammensætninger")


# --------------------------------------------------------------------------
# byte-frozen identity
# --------------------------------------------------------------------------

def lang_hash(lang: str) -> int:
    return zlib.adler32(lang.encode("utf-8")) & 0x7FFFFFFF


def deck_id(lang: str) -> int:
    if lang.lower() == "chinese":
        return 1745409457359          # legacy override, UNCHANGED
    return 0x10000000 + lang_hash(lang)


def model_id(lang: str) -> int:
    return 0x20000000 + lang_hash(lang)


def model_name(lang: str) -> str:
    # Ported verbatim. The notetype is matched by MODEL_ID on import, so the
    # name is cosmetic -- but it is the name already sitting in every existing
    # user's collection, and renaming someone's notetype during the migration
    # that T7 is meant to verify buys nothing.
    return "Danish Frequency Deck V2.1 (%s)" % lang


def guid_for(seed: str, lang: str) -> str:
    import genanki
    return genanki.guid_for(seed, lang)


# --------------------------------------------------------------------------
# deck name / description / copyright -- ported verbatim as data
# --------------------------------------------------------------------------

def deck_meta(lang: str, year: int) -> dict:
    """Deck name, description and footer, ported verbatim as data.

    The Chinese branch's description is the final wording (its
    "AI-assisted translation" sentence is the one that shipped); the copyright
    year comes from config and is never datetime.now().year.
    """
    if lang.lower() == "chinese":
        name = "丹麦语要你命3000词"
        description = f"""<p>本牌组是基于丹麦语词频列表生成的 Anki 卡片。</p><p><b>数据来源说明:</b></p><ul><li>词频列表主要参考: <a href="https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist">Wiktionary Danish Frequency List</a></li><li>如果您从Den Danske Ordbog 获取单词定义、发音、例句等数据部分，请注意遵守 DDO 的使用条款，仅供您个人使用，切勿分享。</li><li>部分翻译由人工智能辅助生成，仅供参考。</li></ul><p><b>版权与许可:</b></p><ul><li>Wiktionary 内容遵循 CC BY-SA 3.0 许可。</li><li>本 Anki 牌组生成脚本及牌组结构由 iskoldt-X 创建 (© {year})。</li></ul><p><b>项目代码仓库:</b></p><p><a href="https://github.com/iskoldt-X/ankidkdeck">https://github.com/iskoldt-X/ankidkdeck</a></p><p>如果您觉得这个牌组有用，欢迎⭐️项目或捐赠，欢迎提出改进建议！</p>"""
        copyright_html = f"""
    <div class="copyright-info" style="font-size: 0.8em; color: #777; margin-top: 20px; text-align: center; border-top: 1px solid #eee; padding-top: 10px;">
        <small>
            使用 <a href="https://github.com/iskoldt-X/ankidkdeck" style="color: #55a;">ankidkdeck</a> 生成<br>
            Copyright © {year} iskoldt-X<br>
            如果您喜欢这个牌组，请考虑⭐本项目或捐赠
        </small>
    </div>
    """
    else:
        name = f"Avadanskedavra: 3000 Words to Slay {lang}"
        description = f"""<p>This deck is generated from a Danish frequency list as Anki cards.</p><p><b>Data Sources:</b></p><ul><li>Frequency list primarily based on: <a href="https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist">Wiktionary Danish Frequency List</a></li><li>If you get the word definitions, pronunciation, examples, etc. from Den Danske Ordbog (DDO). Please respect DDO’s terms of use—for personal study only, do not redistribute.</li><li>Translations are generated by AI and provided for reference only.</li></ul><p><b>Copyright & License:</b></p><ul><li>Wiktionary content is available under CC BY‑SA 3.0.</li><li>This Anki deck generation script and deck structure were created by iskoldt‑X (© {year}).</li></ul><p><b>Project Repository:</b></p><p><a href="https://github.com/iskoldt-X/ankidkdeck">https://github.com/iskoldt-X/ankidkdeck</a></p><p>If you find this deck useful, please feel free to ⭐ the project or open issues/suggestions!</p>"""
        copyright_html = f"""
    <div class="copyright-info" style="font-size: 0.8em; color: #777; margin-top: 20px; text-align: center; border-top: 1px solid #eee; padding-top: 10px;">
        <small>
            Generated using <a href="https://github.com/iskoldt-X/ankidkdeck" style="color: #55a;">ankidkdeck</a><br>
            Copyright © {year} iskoldt-X<br>
            If you like this deck, please consider ⭐ the project or donating
        </small>
    </div>
    """
    return {"deck_name": name, "deck_description": description,
            "copyright_html": copyright_html}


# --------------------------------------------------------------------------
# templates -- v2.1 CSS/QFMT/AFMT verbatim, plus a clearly separated v3 block
#
# VERBATIM MEANS BYTE-FOR-BYTE, trailing whitespace included: two CSS lines
# (`details > summary`) and three AFMT <script> lines carry a trailing space in
# 09_export_apkg.py, and an editor had stripped them. Inert on the rendered card,
# but the docstring claimed "verbatim" and it was not true; tests/
# test_export_parity.py now pins the sha256 of each ported block against
# `git show v2.1-pipeline:09_export_apkg.py`, so a stripping editor fails CI
# instead of quietly rewriting the notetype.
# --------------------------------------------------------------------------

CSS = """
.card { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; text-align: left; color: #333; }
.front-container { text-align: center; }
.query-word { font-size: 3em; font-weight: 600; }
.ipa-row { display: block; margin: 0 10px; padding: 3px 0; } /* Changed from inline-flex, made display: block */
.replay-button svg { width:22px; height:22px; vertical-align:middle; margin-left:5px; }
.meaning-block { margin-bottom: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fcfcfc; }
.meaning-block:last-child { margin-bottom: 0; }
.meaning-header { font-size: 1.2em; font-weight: 600; color: #005a9c; margin-bottom: 10px; }
.pos { font-style: italic; color: #666; margin-left: 8px; font-weight: normal; }
.definition-item { margin-bottom: 12px; }
.def-text { display: block; }
.translation { color: #2e7d32; font-size: 0.95em; margin-left: 1.2em; }
.example { color: #757575; font-size: 0.9em; margin-left: 1.2em; border-left: 2px solid #d0d0d0; padding-left: 8px; margin-top: 4px; }
details > summary { list-style: none; } 
details > summary::-webkit-details-marker { display: none; } 
.details-toggle { cursor: pointer; color: #0066cc; font-size: 0.9em; margin-top: 10px; }
.hr-divider { border: 0; height: 1px; background: #ddd; margin: 20px 0; }
.bottom-info { font-size: 0.9em; color: #666; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; }
.freq-rank { font-size: 0.8em; color: #aaa; text-align: right; float: right; }

/* --- NEW CSS FOR V2.1 --- */
.summary-divider { border: 0; height: 1px; background: #ddd; margin: 15px 0; }
.summary-divider-inner { border: 0; height: 1px; background: #eee; margin: 10px 0 10px 10px; } /* Added left margin */
.summary-content { font-size: 1.0em; color: #444; text-align: left; max-width: 600px; margin: 0 auto; padding: 0 10px; } /* Added padding */
.pos-group { margin-bottom: 10px; }
.pos-group:last-child { margin-bottom: 0; }
.pos-group-header { font-size: 1.1em; font-weight: 600; color: #005a9c; margin-bottom: 5px; }
.collocations-container { margin-top: 20px; margin-bottom: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fcfcfc; }
.collocations-header { font-size: 1.2em; font-weight: 600; color: #005a9c; margin-bottom: 10px; }


/* --- NIGHT MODE OVERRIDES --- */
.night_mode .card { color: #f0f0f0; }
.night_mode .query-word { color: #ffffff; }
.night_mode .ipa-row { color: #cccccc; }
.night_mode .meaning-block { background-color: #2e2e2e; border: 1px solid #444; }
.night_mode .meaning-header { color: #8ab4f8; }
.night_mode .pos { color: #aaa; }
.night_mode .translation { color: #81c784; }
.night_mode .example { color: #9e9e9e; border-left: 2px solid #555; }
.night_mode .details-toggle { color: #82aaff; }
.night_mode .hr-divider { background: #444; }
.night_mode .bottom-info { color: #b0b0b0; border-top: 1px solid #444; }

/* --- NEW NIGHT MODE CSS FOR V2.1 --- */
.night_mode .summary-divider { background: #444; }
.night_mode .summary-divider-inner { background: #383838; }
.night_mode .summary-content { color: #ccc; }
.night_mode .pos-group-header { color: #8ab4f8; }
.night_mode .collocations-container { background-color: #2e2e2e; border: 1px solid #444; }
.night_mode .collocations-header { color: #8ab4f8; }

/* --- V3 ADDITIONS: the paradigm block moved into Variants (guide D10) --- */
.boejning-row { font-size: 0.9em; color: #555; margin-top: 6px; }
.paradigm-block { margin-bottom: 6px; }
.paradigm-head { font-style: italic; color: #666; }
.paradigm-row { display: block; }
.paradigm-label { color: #888; }
.alt-forms { margin-top: 4px; }
.searchable-forms { display: none; }
.super { font-size: 0.65em; vertical-align: super; color: #666; margin-left: 1px; }
.night_mode .boejning-row { color: #bbb; }
.night_mode .paradigm-head { color: #aaa; }
.night_mode .paradigm-label { color: #999; }
.night_mode .super { color: #aaa; }
"""

QFMT = """
<div class="front-container">
  <div class="query-word">{{QueryWord}}</div>
  <hr class="summary-divider">
  <div class="summary-content">{{FrontSideSummary}}</div>
</div>
"""

AFMT_HEAD = """
{{FrontSide}}
<hr class="hr-divider">
<div>{{Content}}</div>
{{#Collocations}}
<div class="collocations-container">
    <div class="collocations-header">Fixed Expressions</div>
    {{Collocations}}
</div>
{{/Collocations}}
<div class="bottom-info">
  <div class="freq-rank">Rank: {{FrequencyRank}}</div>
  {{#Variants}}<b>Variants:</b> {{Variants}}<br>{{/Variants}}
  {{#Derivatives}}<b>Derivatives:</b> {{Derivatives}}<br>{{/Derivatives}}
  {{#Etymology}}<b>Etymology:</b> {{Etymology}}{{/Etymology}}
</div>
"""

AFMT_SCRIPT = """
    <script>
    (function () {
      const MORE = "Show more definitions...";
      const LESS = "Show less...";

      // Ensure cards initially start with <details> closed.
      // Using the more general selector as provided by ChatGPT's tested solution.
      document.querySelectorAll("details[open]").forEach(d => {
        // We only want to close the <details> elements that use our specific toggle class.
        // This avoids accidentally closing other <details> elements if they exist.
        if (d.querySelector("summary.details-toggle")) {
            d.removeAttribute("open");
        }
      });

      // Initialize all relevant summary texts to the "MORE" state.
      // Using the more general selector as provided by ChatGPT's tested solution.
      document.querySelectorAll("summary.details-toggle").forEach(s => { 
        s.textContent = MORE; 
      });

      // Use event delegation on the document for click events
      document.addEventListener("click", e => {
        // Find the closest summary.details-toggle that was clicked or contains the click target
        const summary = e.target.closest("summary.details-toggle");
        
        // If the click was not on or inside a relevant summary, do nothing
        if (!summary) return;

        const details = summary.parentElement; // The <details> element

        // requestAnimationFrame waits for the browser to complete its native open/close action
        // and update the 'open' attribute before we read it.
        requestAnimationFrame(() => {
          summary.textContent = details.open ? LESS : MORE;
        });
      }, true); // Using capture phase for the listener on document
    })();
    </script>
"""


def afmt(copyright_html: str) -> str:
    return AFMT_HEAD + copyright_html + AFMT_SCRIPT


def build_model(lang: str, year: int):
    import genanki
    return genanki.Model(
        model_id(lang),
        model_name(lang),
        fields=[{"name": n} for n in FIELD_NAMES],
        templates=[{"name": "Card 1", "qfmt": QFMT,
                    "afmt": afmt(deck_meta(lang, year)["copyright_html"])}],
        css=CSS,
        sort_field_index=SORT_FIELD_INDEX,
    )


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sanitize(text) -> str:
    """v2.1's sanitize(), kept: it is what decides whether a field is 'empty'."""
    return str(text).strip() if text is not None else ""


def headword_html(entry: dict) -> str:
    """The lemma, with DDO's homograph index rendered as a SUPERSCRIPT.

    Stage 20 used to store the glued form (`al2`, `udenfor1`, `i5` -- 45 of 134
    entries on the fixture set) as display_headword, which is not what DDO shows
    and which also became an Anki search token. lemma and super are separate
    fields; the markup is assembled here, where it belongs.
    """
    hw = sanitize(entry.get("lemma") or entry.get("display_headword"))
    sup = sanitize(entry.get("super"))
    if sup:
        return '%s<span class="super">%s</span>' % (hw, sup)
    return hw


class Media:
    """The audio cache as the exporter sees it: url -> ([sound:] tag, path).

    An entry that declares audio we do not have on disk is recorded as a miss
    and fails G-MEDIA; the tag is never emitted for a file that is absent,
    because a dead [sound:] tag is a silent card in Anki.

    The one exception is a slot registry/known_missing_audio.json declares dead
    UPSTREAM -- DDO's own article links it and DDO's own host answers 200 with a
    zero-byte text/html placeholder. Those misses are split off into
    `known_missing_upstream` so G-MEDIA can pass honestly on a defect that is not
    ours, and a slot the registry names which now HAS a file is recorded in
    `recovered_upstream` so the repair surfaces instead of passing unnoticed.
    Every other miss stays in `unexpected_missing`, which is what the gate reads.
    """

    def __init__(self, cfg: Config, known_missing=None):
        self.dir = Path(cfg.audio_dir)
        self.manifest = read_json(self.dir / "manifest.json", default={})
        self.used: dict[str, Path] = {}
        # Every miss, unsplit: the evidence, kept whole. The two lists below are
        # the verdicts drawn from it.
        self.missing: list = []
        self.unexpected_missing: list = []
        self.known_missing_upstream: list = []
        self.recovered_upstream: list = []
        if known_missing is None:
            # A Media that does not know the registry would fail G-MEDIA on the
            # four upstream-dead slots, so the default reads it rather than
            # defaulting to "no excuses exist".
            from ..registry import Registry              # noqa: PLC0415 - cycle
            known_missing = Registry(cfg).known_missing_audio
        self.known_missing = dict(known_missing or {})

    def _filename(self, url: str, entry_id: str, slot_n) -> str:
        row = self.manifest.get(url)
        if row and row.get("file"):
            return row["file"]
        if slot_n is not None:
            return "%s_%s.mp3" % (entry_id, slot_n)
        return url.rsplit("/", 1)[-1]

    def sound_tag(self, url: str, entry_id: str, slot_n) -> str:
        name = self._filename(url, entry_id, slot_n)
        path = self.dir / name
        if not path.exists() or path.stat().st_size == 0:
            row = {"url": url, "entry_id": entry_id, "expected_file": name}
            self.missing.append(row)
            if url in self.known_missing:
                self.known_missing_upstream.append(row)
            else:
                self.unexpected_missing.append(row)
            return ""
        if url in self.known_missing:
            self.recovered_upstream.append({"url": url, "entry_id": entry_id,
                                            "file": name})
        self.used[name] = path
        return "[sound:%s]" % name

    def sorted_paths(self) -> list:
        # A SORTED LIST, never a set: set iteration order is what made
        # "byte-identical rebuild" untrue in v2.1.
        return [str(self.used[n]) for n in sorted(self.used)]


# --------------------------------------------------------------------------
# FrontSideSummary
# --------------------------------------------------------------------------

def pron_html_for_group(ents: list, media: Media) -> str:
    """v2.1 build_pronunciations_html_for_group, unchanged except that the audio
    URL is resolved through the manifest instead of audio_map.json.

    IPA dedup is per group. The key is the IPA TEXT when there is one, and the
    audio slot otherwise: 21 udtale rows (all multiword compounds --
    `alle sammen`, `dag til dag-service`) carry an empty ipa with a valid
    audio_url, and deduping on the empty string collapsed them into one row,
    losing every [sound:] tag but the first. A row with neither IPA nor audio is
    still nothing to render.
    """
    seen, rows = set(), []
    for e in ents:
        for u in e.get("udtale", []):
            ipa_raw = sanitize(u.get("ipa")).strip("[]").strip()
            audio_url = u.get("audio_url")
            if not ipa_raw and not audio_url:
                continue
            slot = u.get("slot_n")
            key = ipa_raw or ("\x00audio\x00%s\x00%s"
                             % (e["entry_id"], slot if slot is not None else audio_url))
            if key in seen:
                continue
            seen.add(key)
            label_html = (f'<span class="ipa-label">({sanitize(u.get("label"))})</span>'
                          if u.get("label") else "")
            sound_tag = ""
            if audio_url:
                sound_tag = media.sound_tag(audio_url, e["entry_id"], slot)
            ipa_html = f"[{ipa_raw}] " if ipa_raw else ""
            rows.append(f'<div class="ipa-row">{ipa_html}{label_html} {sound_tag}</div>')
    return "".join(rows)


def front_summary_html(ents: list, pos_trans: dict, media: Media,
                       misses: list, report: dict) -> str:
    """v2.1 build_front_summary_html, re-keyed on data-pos-key.

    2025 grouped by the mangled display string; 2026 renders that string
    differently, so the group key is now pos_key and the display text is the
    first member's pos_text. When one pos_key covers two different display
    strings inside one family the group merges -- recorded, not hidden.
    """
    if not ents:
        return ""
    groups: dict[str, list] = {}
    order: list = []
    display: dict[str, str] = {}
    for e in ents:
        key = e.get("pos_key") or DEFAULT_POS_KEY
        if key not in groups:
            groups[key] = []
            order.append(key)
            display[key] = sanitize(e.get("pos_text"))
        elif sanitize(e.get("pos_text")) and sanitize(e.get("pos_text")) != display[key]:
            report.setdefault("pos_text_variants_in_group", []).append(
                {"pos_key": key, "kept": display[key],
                 "also_seen": sanitize(e.get("pos_text")), "entry_id": e["entry_id"]})
        groups[key].append(e)

    single = len(order) == 1
    parts = []
    for key in order:
        if key == DEFAULT_POS_KEY:
            # v2.1: an entry with no POS shows nothing when it is alone and
            # "Other" when it shares the card. Kept.
            pos_display = "" if single else "Other"
            translated = ""
        else:
            pos_display = display[key] or key
            translated = sanitize(pos_trans.get(key))
            if not translated:
                misses.append({"kind": "pos", "key": key})
        header_text = pos_display
        if translated:
            header_text += f" ({translated})"
        header = (f'<div class="pos-group-header">{header_text}</div>'
                  if header_text.strip() else "")
        prons = pron_html_for_group(groups[key], media)
        if header or prons:
            parts.append(f'<div class="pos-group">{header}{prons}</div>')
    return '<hr class="summary-divider-inner">'.join([p for p in parts if p])


def boejning_line(entry: dict) -> str:
    """The compact short notation from button.kilde ('-et, -e, -ene').

    Suppressed when absent: adjectives render 'Se boejning i skema' there, which
    stage 20 already refuses to store as a short form.
    """
    short = sanitize((entry.get("paradigm") or {}).get("short"))
    if not short:
        return ""
    return f'<div class="boejning-row">Bøjning: {short}</div>'


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

def renderable_senses(entry: dict) -> list:
    return [s for s in entry.get("senses", []) if sanitize(s.get("definition"))]


def meaning_block(entry: dict, defs_trans: dict, misses: list,
                  header_only: bool = False) -> str:
    """One block per article. Senses are renumbered 1..N (DDO's 1.a flattened);
    the first example only; the 3rd definition onward folded into <details> with
    v2.1's summary text."""
    items = []
    for i, s in enumerate(renderable_senses(entry)):
        def_text = sanitize(s.get("definition"))
        key = "%s:%s" % (entry["entry_id"], s.get("dannetid"))
        tr = defs_trans.get(key)
        if tr is None:
            # Never a silently bare definition: G-COV blocks the package.
            misses.append({"kind": "definition", "key": key,
                           "entry_id": entry["entry_id"],
                           "dannetid": s.get("dannetid"), "text": def_text})
            translation_html = ""
        else:
            translation_html = ('<div class="translation"><b>%s</b>: %s</div>'
                                % (sanitize(tr.get("lemma")), sanitize(tr.get("gloss"))))
        example_html = ""
        for ex in (s.get("examples") or []):
            first = sanitize(ex.get("text"))
            if first:
                example_html = f'<div class="example">E.g., "{first}"</div>'
            break
        items.append('<div class="definition-item"><span class="def-text">'
                     f'{i + 1}. {def_text}</span>{translation_html}{example_html}</div>')
    if not items and not header_only:
        return ""
    pos_html = (f'<span class="pos">{sanitize(entry.get("pos_text"))}</span>'
                if entry.get("pos_text") else "")
    header = f'<div class="meaning-header">{headword_html(entry)} {pos_html}</div>'
    visible = "".join(items[:2])
    folded = "".join(items[2:])
    details = ""
    if folded:
        details = f"""
            <details>
              <summary class="details-toggle">Show more definitions...</summary>
              <div style="margin-top: 10px;">{folded}</div>
            </details>
            """
    return f'<div class="meaning-block">{header}{visible}{details}</div>'


def content_html(ents: list, defs_trans: dict, misses: list,
                 stats: dict) -> str:
    """Blocks joined by v2.1's <hr class="hr-divider">.

    The 0-sense rule (guide M11.2) means an article with no renderable sense
    contributes no block. A family whose articles ALL have only expressions
    would then produce an empty Content, which G-EMPTY-C forbids and which the
    stage-30 renderability test (senses OR expressions) allows -- so exactly in
    that case one header-only block is emitted, and counted here.
    """
    blocks = [meaning_block(e, defs_trans, misses) for e in ents]
    blocks = [b for b in blocks if b]
    if not blocks:
        for e in ents:
            if e.get("expressions"):
                blocks.append(meaning_block(e, defs_trans, misses, header_only=True))
                stats["header_only_blocks"] = stats.get("header_only_blocks", 0) + 1
                break
    return '<hr class="hr-divider">'.join(blocks)


# --------------------------------------------------------------------------
# Collocations
# --------------------------------------------------------------------------

def collocations_html(ents: list, expr_trans: dict, misses: list) -> str:
    """Deduped on dannetid, not on text: 10 idioms are shared across two
    entry_ids and therefore share one translation."""
    seen, out = set(), []
    for e in ents:
        for x in e.get("expressions", []):
            key = x.get("dannetid")
            text = sanitize(x.get("expression"))
            if not text or not key or key in seen:
                continue
            seen.add(key)
            tr = expr_trans.get(key)
            if tr is None:
                misses.append({"kind": "expression", "key": key,
                               "entry_id": e["entry_id"], "text": text})
                translation_html = ""
            else:
                lemma, gloss = sanitize(tr.get("lemma")), sanitize(tr.get("gloss"))
                translation_html = ('<div class="translation"><b>%s</b>: %s</div>'
                                    % (lemma, gloss)) if (lemma or gloss) else ""
            out.append(f'<div class="definition-item">{text}{translation_html}</div>')
    return "".join(out)


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

def paradigm_block_html(ents: list) -> str:
    """One line per flex-table ROW, cells joined ' / '.

    A row holds orthographic alternatives of ONE slot (331 of 539 rows have
    several cells), so ' / ' is the honest join. Labels come from the checked-in
    registry table via stage 20 and are absent whenever the shape is
    unrecognised -- never invented. When several articles in the family carry a
    paradigm, each block is headed by its own headword.
    """
    with_rows = [e for e in ents if (e.get("paradigm") or {}).get("rows")]
    blocks = []
    for e in with_rows:
        lines = []
        for row in e["paradigm"]["rows"]:
            cells = " / ".join(sanitize(c) for c in row.get("cells", []) if sanitize(c))
            if not cells:
                continue
            label = sanitize(row.get("slot_label"))
            if label:
                lines.append(f'<div class="paradigm-row">'
                             f'<span class="paradigm-label">{label}:</span> {cells}</div>')
            else:
                lines.append(f'<div class="paradigm-row">{cells}</div>')
        if not lines:
            continue
        head = ""
        if len(with_rows) > 1:
            pos = sanitize(e.get("pos_text"))
            label = ("%s %s" % (headword_html(e), pos)).strip()
            head = f'<div class="paradigm-head">{label}</div>'
        blocks.append(f'<div class="paradigm-block">{head}{"".join(lines)}</div>')
    return "".join(blocks)


def alt_forms_html(fam: dict, ents: list) -> str:
    """Official alternative spellings plus the family's alias/variant members.

    No label is emitted: the card template already says 'Variants:', and adding
    an English word here would put English on the Chinese deck.
    """
    lemma_forms = {sanitize(e.get("lemma")) for e in ents}
    lemma_forms |= {sanitize(fam.get("lemma")), sanitize(fam.get("display_headword"))}
    forms = set()
    for e in ents:
        for a in e.get("alt_spellings", []):
            if a.get("official") and sanitize(a.get("form")):
                forms.add(sanitize(a["form"]))
    for m in fam.get("members", []):
        if m.get("relation") in ("variant", "alias") and sanitize(m.get("word")):
            forms.add(sanitize(m["word"]))
    forms -= lemma_forms
    forms.discard("")
    if not forms:
        return ""
    return '<div class="alt-forms">%s</div>' % ", ".join(sorted(forms))


def searchable_forms_hidden(fam: dict, ents: list) -> str:
    """Every member word, paradigm cell and alternative spelling, hidden.

    Anki searches the FIELD TEXT, not the rendered card, so this is what makes
    'hjaelp' find the 'hjaelpe' card after the merge.

    It is appended to CONTENT, not to Variants. AFMT wraps Variants in
    `{{#Variants}}<b>Variants:</b> {{Variants}}<br>{{/Variants}}` and Mustache's
    section test is "the field is non-empty", so a span that is always present
    printed a bare "Variants:" label with nothing visible after it on every card
    with no paradigm and no alternative spelling (measured: 5 of 34 on the German
    build; ~300 of 4,442 at full scale). It also pinned the measured Variants
    empty rate at 0.00% against a 10.96% baseline, so G-RATE could never fire for
    that field again. Content is rendered unconditionally in a bare
    `<div>{{Content}}</div>`, so it carries the span with no visible effect.
    """
    forms = list(fam.get("searchable_forms") or [])
    if not forms:
        forms = [fam.get("lemma"), fam.get("display_headword")]
        forms += [m.get("word") for m in fam.get("members", [])]
        for e in ents:
            forms += [c for r in (e.get("paradigm") or {}).get("rows", [])
                      for c in r.get("cells", [])]
            forms += [a.get("form") for a in e.get("alt_spellings", [])]
    clean = []
    seen = set()
    for f in forms:
        f = sanitize(f)
        if f and f not in seen:
            seen.add(f)
            clean.append(f)
    if not clean:
        return ""
    return '<span class="searchable-forms">%s</span>' % " ".join(clean)


def variants_html(fam: dict, ents: list) -> str:
    """The paradigm block and the alternative spellings -- and NOTHING ELSE.

    Empty is a legitimate and common answer here (a family with no flex table
    and no alias), and it has to be an EMPTY STRING so that AFMT's
    `{{#Variants}}` section drops the label with it. The hidden
    searchable-forms span moved to the end of Content.
    """
    return paradigm_block_html(ents) + alt_forms_html(fam, ents)


# --------------------------------------------------------------------------
# Derivatives / Etymology
# --------------------------------------------------------------------------

def derivatives_text(ents: list) -> str:
    items = set()
    for e in ents:
        od = e.get("orddannelser") or {}
        for label in DERIVATIVE_LABELS:
            # Oevrige is parsed but NOT rendered: owner deferred that change.
            for d in od.get(label, []) or []:
                if sanitize(d):
                    items.add(sanitize(d))
    return ", ".join(sorted(items))


def etymology_text(anchor: dict) -> str:
    ety = anchor.get("etymology") or {}
    return sanitize(ety.get("raw"))


# --------------------------------------------------------------------------
# note assembly
# --------------------------------------------------------------------------

def family_entries(fam: dict, entries: dict) -> list:
    """The family's articles, anchor first, 0-sense-and-0-expression articles
    dropped (they are real: godte2, vinge2, the engros* family)."""
    anchor = fam["anchor_entry_id"]
    ordered = [anchor] + [e for e in fam.get("entry_ids", []) if e != anchor]
    out = []
    for eid in ordered:
        e = entries.get(eid)
        if e is None:
            continue
        if e.get("senses") or e.get("expressions"):
            out.append(e)
    return out


def build_note(fam: dict, entries: dict, tr: dict, media: Media, misses: list,
               lang: str, stats: dict, report: dict) -> dict | None:
    ents = family_entries(fam, entries)
    if not ents:
        return None
    anchor = ents[0]
    seed = fam.get("guid_seed")
    if not seed:
        raise FatalError("family %s has no guid_seed; run the merge stage so the "
                         "registry can freeze it" % fam.get("family_id"))
    if NFC(seed) != seed:
        raise FatalError("guid_seed for family %s is not NFC: %r"
                         % (fam.get("family_id"), seed))
    front = front_summary_html(ents, tr["pos"], media, misses, report)
    front += boejning_line(anchor)
    stats["senses_rendered"] = stats.get("senses_rendered", 0) + sum(
        len(renderable_senses(e)) for e in ents)
    # A SHARED idiom is one rendered expression, not one per family: 10 idioms
    # sit on two entry_ids and collocations_html() deliberately collapses them to
    # one translation. Summing per-family distinct dannetids counted those twice
    # (measured 481 reported vs 477 distinct), and the number lands in G-COV's
    # `rendered` detail and in export_report_<lang>.json.
    seen_expr = stats.setdefault("_expression_dannetids", set())
    seen_expr.update(x["dannetid"] for e in ents
                     for x in e.get("expressions", []) if x.get("dannetid"))
    content = content_html(ents, tr["definitions"], misses, stats)
    fields = [
        seed,
        front,
        # The hidden searchable-forms span rides at the END of Content.
        content + searchable_forms_hidden(fam, ents),
        collocations_html(ents, tr["expressions"], misses),
        variants_html(fam, ents),
        derivatives_text(ents),
        etymology_text(anchor),
        str(fam.get("freq_rank")),
    ]
    return {"family_id": fam.get("family_id"), "guid_seed": seed,
            "guid": guid_for(seed, lang), "fields": fields,
            "freq_rank": fam.get("freq_rank")}


def pos_table(cfg: Config, registry, lang: str) -> tuple[dict, dict]:
    """The POS label table: the checked-in registry FIRST, the work-dir
    translations/<lang>/pos.json as an OVERRIDE on top.

    Stage 40 re-keys definitions and expressions only, and guide 1.11f rules that
    2025's `pos_translations_*_gemini.json` (41 mangled display strings) must not
    be re-used -- so before the registry table shipped, the only way to satisfy
    G-COV for `pos` was a paid Gemini call, and there was no offline path to an
    .apkg at all. The registry table is 22 keys x 4 languages of hand-written
    grammatical vocabulary; an LLM-written row still overrides it, and a pos_key
    neither source knows fails G-COV loudly.
    """
    reg = getattr(registry, "pos_for", None)
    from_registry = reg(lang) if callable(reg) else {}
    override = read_json(cfg.json_dir / "translations" / lang / "pos.json",
                         default={})
    return {**from_registry, **override}, {"from_registry": len(from_registry),
                                           "override_rows": len(override)}


def build_all(cfg: Config, registry, lang: str) -> dict:
    """Everything except genanki: notes, media, coverage misses, statistics."""
    entries = read_json(cfg.json_dir / "entries.json")
    families = read_json(cfg.json_dir / "words.json")
    tdir = cfg.json_dir / "translations" / lang
    pos, pos_note = pos_table(cfg, registry, lang)
    tr = {"definitions": read_json(tdir / "definitions.json", default={}),
          "expressions": read_json(tdir / "expressions.json", default={}),
          "pos": pos}
    media = Media(cfg, known_missing=registry.known_missing_audio)
    misses: list = []
    stats: dict = {}
    report: dict = {}

    renderable = [f for f in families.values()
                  if f.get("freq_rank") is not None
                  and any((entries.get(e) or {}).get("senses")
                          or (entries.get(e) or {}).get("expressions")
                          for e in f.get("entry_ids", []))]
    renderable.sort(key=lambda f: (f["freq_rank"], str(f["family_id"])))
    notes = []
    for fam in renderable:
        note = build_note(fam, entries, tr, media, misses, lang, stats, report)
        if note is not None:
            notes.append(note)
    # A missing pos_key is one gap, not one gap per card that shows it.
    seen, unique_misses = set(), []
    for m in misses:
        ident = (m["kind"], m.get("key"), m.get("entry_id"))
        if ident in seen:
            continue
        seen.add(ident)
        unique_misses.append(m)
    stats["pos_keys_rendered"] = len({e.get("pos_key")
                                      for f in renderable
                                      for e in family_entries(f, entries)
                                      if e.get("pos_key")})
    stats["expressions_rendered"] = len(stats.pop("_expression_dannetids", set()))
    return {"notes": notes, "media": media, "misses": unique_misses, "stats": stats,
            "report": {**report, "pos_table": pos_note},
            "families": families, "entries": entries,
            "translations": {k: len(v) for k, v in tr.items()},
            "pos_table": pos_note}


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def empty_content_gate(notes: list):
    """G-EMPTY-C. Reachable only together with the 0-sense rule.

    Measured on the VISIBLE content: Content now also carries the display:none
    searchable-forms span, and testing the raw field would make this gate
    unfalsifiable (the span is never empty).
    """
    bad = [n["family_id"] for n in notes
           if not visible(n["fields"][CONTENT_INDEX]).strip()]
    return not bad, {"notes": len(notes), "empty_content": len(bad),
                     "sample": bad[:20]}


def empty_rate_gate(notes: list, baseline_pct: dict, tolerance_pp: float):
    """G-RATE. Rates, not counts: at ~2,875 notes every absolute count from the
    4,442-note baseline drops ~35% for free, so the old gate was vacuous. An
    improvement (a LOWER rate) always passes.

    Emptiness is VISIBLE emptiness: a hidden span is not something a learner
    sees, and counting it made the field carrying it permanently 0.00% empty.
    """
    n = len(notes)
    rows, bad = {}, {}
    for i, name in enumerate(FIELD_NAMES):
        empties = sum(1 for note in notes if not visible(note["fields"][i]).strip())
        rate = (100.0 * empties / n) if n else 0.0
        base = float(baseline_pct.get(name, 0.0))
        limit = base + tolerance_pp
        rows[name] = {"empty": empties, "rate_pct": round(rate, 2),
                      "baseline_pct": base, "limit_pct": round(limit, 2)}
        if rate > limit:
            bad[name] = rows[name]
    return not bad, {"notes": n, "fields": rows, "violations": bad}


def coverage_gate(misses: list, counts: dict):
    """G-COV. 100% of renderable senses, expressions and pos_keys. Without it,
    2025 shipped 2,007 English cards with bare Danish definitions."""
    by_kind: dict[str, int] = {}
    for m in misses:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
    return not misses, {"missing_by_kind": by_kind, "rendered": counts,
                        "sample": misses[:20]}


def guid_gate(notes: list):
    """G-GUID. One anchor collision was measured under the naive rule, so this
    is not theoretical."""
    guids = [n["guid"] for n in notes]
    seen, dupes = set(), []
    for g in guids:
        if g in seen:
            dupes.append(g)
        seen.add(g)
    return len(seen) == len(guids), {"notes": len(guids), "unique": len(seen),
                                     "duplicates": dupes[:20]}


def media_gate(media: Media, floor: int, sound_names: list,
               known_missing_max: int | None = None):
    """G-MEDIA. Every [sound:] resolves, the list is SORTED, and the count
    clears the floor (4,629 files existed in 2025).

    BASELINED against registry/known_missing_audio.json, the same way stage 60's
    half is: a declared slot DDO answers with a zero-byte text/html placeholder
    is an upstream defect that no export can fix, so it is reported rather than
    counted absent -- while every other absent file still fails, which is the
    population this gate exists for. The registry's size is checked against
    gates.json:known_missing_audio_max so a row cannot be added without a human
    editing that number in the same commit, and it is fail-closed: no baseline
    means every row is over it.

    `recovered_upstream` and `no_longer_declared` are REPORTED and never fail --
    a repaired file is the mechanism working, and the report is how the release
    notes learn the row should be deleted. Note that the slots in this registry
    emit no [sound:] tag either way, so they can never appear in `dangling`:
    Media.sound_tag returns "" for an absent file.
    """
    paths = media.sorted_paths()
    basenames = {Path(p).name for p in paths}
    dangling = sorted({s for s in sound_names if s not in basenames})
    seen = {m["url"] for m in media.known_missing_upstream}
    seen |= {r["url"] for r in media.recovered_upstream}
    not_declared = sorted(u for u in media.known_missing if u not in seen)
    over = (known_missing_max is not None
            and len(media.known_missing) > known_missing_max)
    ok = (not media.unexpected_missing and not dangling and not over
          and paths == sorted(paths) and len(paths) >= floor)
    return ok, {"media_files": len(paths), "floor": floor,
                "declared_but_absent": len(media.unexpected_missing),
                "declared_but_absent_sample": media.unexpected_missing[:20],
                # Excused rows included, so the report still says how many
                # declared slots this build could not render at all.
                "declared_but_absent_total": len(media.missing),
                "dangling_sound_tags": dangling[:20],
                "sorted": paths == sorted(paths),
                "known_missing_upstream": {
                    "registry_rows": len(media.known_missing),
                    "max": known_missing_max, "over_baseline": bool(over),
                    "n_still_missing": len(media.known_missing_upstream),
                    "still_missing": [m["url"]
                                      for m in media.known_missing_upstream][:20],
                    "recovered": [r["url"]
                                  for r in media.recovered_upstream][:20],
                    "no_longer_declared": not_declared[:20]}}


def note_count_gate(notes: list, rng):
    """G-NOTE. Catches a silent classifier blow-up in one number."""
    lo, hi = (rng or [0, 10 ** 9])[0], (rng or [0, 10 ** 9])[1]
    n = len(notes)
    return lo <= n <= hi, {"notes": n, "range": [lo, hi]}


def determinism_gate(first: list, second: list, pkg_a: dict | None = None,
                     pkg_b: dict | None = None):
    """G-DET. Two consecutive builds must produce identical notes and media.

    The .apkg itself is not byte-comparable -- it is a zip around a SQLite file
    that carries ids and mtimes -- so the comparison is over the note payloads
    (guid + all 8 fields) and the media manifest, read back out of the two
    packages when they were written.
    """
    a = canonical_json([[n["guid"]] + n["fields"] for n in first])
    b = canonical_json([[n["guid"]] + n["fields"] for n in second])
    detail = {"notes_equal": a == b, "n_notes": [len(first), len(second)]}
    ok = a == b
    if pkg_a is not None and pkg_b is not None:
        detail["package_notes_equal"] = pkg_a["notes"] == pkg_b["notes"]
        detail["package_media_equal"] = pkg_a["media"] == pkg_b["media"]
        ok = ok and detail["package_notes_equal"] and detail["package_media_equal"]
    if not detail["notes_equal"]:
        for i, (x, y) in enumerate(zip(first, second)):
            if x != y:
                detail["first_difference"] = {"index": i,
                                             "family_id": x.get("family_id")}
                break
    return ok, detail


def sound_names_of(notes: list) -> list:
    """Every [sound:NAME] the notes actually contain."""
    out = []
    for n in notes:
        for field in n["fields"]:
            start = 0
            while True:
                i = field.find("[sound:", start)
                if i < 0:
                    break
                j = field.find("]", i)
                if j < 0:
                    break
                out.append(field[i + 7:j])
                start = j + 1
    return out


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------

def write_package(cfg: Config, lang: str, notes: list, media_paths: list,
                  path: Path) -> dict:
    import genanki

    meta = deck_meta(lang, cfg.copyright_year)
    model = build_model(lang, cfg.copyright_year)
    deck = genanki.Deck(deck_id(lang), meta["deck_name"],
                        description=meta["deck_description"])
    for n in notes:
        deck.add_note(genanki.Note(model=model, fields=n["fields"], guid=n["guid"]))
    pkg = genanki.Package(deck)
    pkg.media_files = list(media_paths)  # already a sorted list
    path.parent.mkdir(parents=True, exist_ok=True)
    pkg.write_to_file(str(path))
    return {"path": str(path), "notes": len(notes), "media": len(media_paths)}


def read_package(path: Path) -> dict:
    """Read (guid, fields, tags) and the media map back out of an .apkg with
    nothing but the standard library. Used by the determinism check and by
    tools/guid_diff.py."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            db_name = ("collection.anki21" if "collection.anki21" in names
                       else "collection.anki2")
            z.extract(db_name, tmp)
            media = {}
            if "media" in names:
                media = json.loads(z.read("media").decode("utf-8") or "{}")
        con = sqlite3.connect(str(Path(tmp) / db_name))
        try:
            rows = con.execute("SELECT guid, flds, tags FROM notes").fetchall()
        finally:
            con.close()
    return {"notes": sorted([r[0], r[1], r[2]] for r in rows),
            "media": sorted(media.values())}


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------

def apkg_path(cfg: Config, lang: str) -> Path:
    return Path(cfg.dist_dir) / ("DDO_Danish_Frequency_Deck_%s.apkg" % lang)


def run(cfg: Config, registry, lang: str, check_determinism: bool = False) -> dict:
    if not lang:
        raise FatalError("export needs --lang (one of %s)" % ", ".join(cfg.langs))
    if lang not in cfg.langs:
        # `export --lang Gemran` used to mint a fresh deck_id and model_id from
        # adler32("Gemran") and only fail later, at G-COV.
        raise FatalError(
            "unknown --lang %r. Configured languages are: %s. The deck id and "
            "the notetype id are derived from this exact string, so a typo mints "
            "a brand-new deck instead of upgrading the existing one."
            % (lang, ", ".join(cfg.langs)))
    gates_cfg = registry.gates
    built = build_all(cfg, registry, lang)
    notes, media, misses = built["notes"], built["media"], built["misses"]
    if not notes:
        raise FatalError("no renderable families -- run the build stages first")

    second = second_media = None
    pkg_a = pkg_b = None
    if check_determinism:
        second_built = build_all(cfg, registry, lang)
        second, second_media = second_built["notes"], second_built["media"]

    v2_querywords = read_json(cfg.json_dir / "legacy" / "v2_querywords.json",
                              default={})
    # Written by stage 30, enforced here: the guide puts the retry in stage 12
    # phase C and the gate at export, because a shortfall at merge time has no
    # remedy the merge can apply.
    shortfall = read_json(cfg.json_dir / "sitemap_shortfall.json",
                          default={"rows": [], "n_families": 0})
    counts = {"notes": len(notes),
              "senses_rendered": built["stats"].get("senses_rendered", 0),
              "expressions_rendered": built["stats"].get("expressions_rendered", 0),
              "pos_keys_rendered": built["stats"].get("pos_keys_rendered", 0),
              "translation_rows": built["translations"]}
    media_paths = media.sorted_paths()
    sound_names = sound_names_of(notes)
    family_ids = [n["family_id"] for n in notes]

    write_json(cfg.report_dir / ("coverage_misses_%s.json" % lang),
               {"language": lang, "n": len(misses), "misses": misses})

    # A stage-70 gate carries the language in `extra` IF AND ONLY IF its verdict
    # is per-language. Merging a per-language gate on the bare id is what let a
    # passing German export overwrite a failing Chinese row, so `ankidkdeck
    # gates` reported the release all-green with the Chinese coverage failure
    # erased.
    #
    # G-SEED and G-RANK are the exceptions and stay UNSCOPED: the seeds come from
    # the append-only registry and the ranks from words.json, so both verdicts are
    # identical for every language by construction. Scoping them wrote four
    # byte-identical rows per release, which reads like four separate checks.
    scope = {"lang": lang}
    # read_json() treats default=None as "required", so an absent report has to
    # be an empty dict here.
    guid_diff = read_json(cfg.report_dir / "guid_diff.json", default={})
    gate_list = [
        Gate(G_EMPTY_C, "no note has an empty Content field",
             lambda: empty_content_gate(notes), stage="70", extra=scope),
        Gate(G_RATE, "no field's empty RATE exceeds the v2.1 baseline plus "
                     "tolerance",
             lambda: empty_rate_gate(notes,
                                     gates_cfg.get("empty_rate_baseline_pct", {}),
                                     float(gates_cfg.get("empty_rate_tolerance_pp", 1.0))),
             stage="70", extra=scope),
        Gate(G_COV, "every rendered sense, expression and pos_key has a "
                    "translation in this language",
             lambda: coverage_gate(misses, counts), stage="70", extra=scope),
        Gate(G_GUID, "every note GUID is unique",
             lambda: guid_gate(notes), stage="70", extra=scope),
        # Language-independent (see the note above `scope`): one row, no extra.
        Gate(G_SEED, "every carried guid_seed is NFC and byte-equal to a v2.1 "
                     "QueryWord",
             lambda: registry_seed_bytes(registry.card_keys, v2_querywords,
                                         family_ids), stage="70"),
        Gate(G_RANK, "FrequencyRank is dense 1..N and unique over the notes",
             lambda: dense_unique_ranks([int(n["fields"][7]) for n in notes],
                                        len(notes)), stage="70"),
        Gate(G_MEDIA, "every [sound:] tag resolves, media_files is a sorted "
                      "list, the count clears the floor, and the "
                      "upstream-dead audio registry is inside its baseline",
             lambda: media_gate(
                 media, int(gates_cfg.get("media_floor", 0)), sound_names,
                 int(gates_cfg.get("known_missing_audio_max", 0))),
             stage="70", extra=scope),
        Gate(G_NOTE, "the note count is inside the declared range",
             lambda: note_count_gate(notes, gates_cfg.get("note_count_range")),
             stage="70", extra=scope),
        Gate(G_SITEMAP, "the per-family homograph shortfall against the sitemap "
                        "inventory is under the declared rate",
             lambda: sitemap_shortfall(
                 shortfall.get("rows") or [],
                 shortfall.get("n_families") or len(notes),
                 float(gates_cfg.get("sitemap_shortfall_max_rate", 0.01))),
             stage="70", extra=scope),
        # G-SEP on the EXPORT path, not only under pytest. A one-character drift
        # in extract.SEP voids 22,734 cells x 4 languages and its symptom is a
        # collapsed bind rate, not a blocked build. Fixtures absent FAILS: a
        # release host that cannot run this check has not run it.
        Gate(G_SEP, "the separator table still reproduces the 2025 Danish "
                    "strings the translation asset is keyed by, and a wrong "
                    "table provably does not",
             lambda: separator_golden(registry, cfg.work_dir / "fixtures"),
             stage="70", extra=scope),
    ]
    if guid_diff:
        # Recorded only when the tool has been run: it needs a released .apkg,
        # which a fresh checkout does not have.
        gate_list.append(
            Gate(G_REL, "reports/guid_diff.json describes this language and the "
                        "same card count the exporter is about to write",
                 lambda: guid_diff_reconciles(guid_diff, len(notes), lang),
                 stage="70", extra=scope))
    if check_determinism:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.apkg"
            b = Path(tmp) / "b.apkg"
            # Build B writes its OWN media list. Passing build A's made
            # package_media_equal a list compared against itself -- a gate half
            # that could not fail, and the media ORDERING bug this gate exists
            # for (v2.1's set()) is exactly what it would have missed.
            write_package(cfg, lang, notes, media_paths, a)
            write_package(cfg, lang, second, second_media.sorted_paths(), b)
            pkg_a, pkg_b = read_package(a), read_package(b)
            gate_list.append(
                Gate(G_DET, "two consecutive builds produce identical notes and "
                            "media",
                     lambda: determinism_gate(notes, second, pkg_a, pkg_b),
                     stage="70", extra=scope))
            run_gates(gate_list, cfg, stage="70")
    else:
        run_gates(gate_list, cfg, stage="70")

    out = apkg_path(cfg, lang)
    pkg_info = write_package(cfg, lang, notes, media_paths, out)

    report = {
        "language": lang,
        "apkg": pkg_info["path"],
        "notes": len(notes),
        "media_files": len(media_paths),
        "deck_id": deck_id(lang),
        "model_id": model_id(lang),
        "copyright_year": cfg.copyright_year,
        "coverage_misses": len(misses),
        "determinism_checked": bool(check_determinism),
        # Visible emptiness, the same measure G-RATE uses.
        "empty_rates_pct": {name: round(100.0 * sum(
            1 for n in notes if not visible(n["fields"][i]).strip()) / len(notes), 2)
            for i, name in enumerate(FIELD_NAMES)},
        **built["stats"], **built["report"],
    }
    write_json(cfg.report_dir / ("export_report_%s.json" % lang), report)
    return report
