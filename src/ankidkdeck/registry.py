"""Checked-in, human-reviewed registry files.

Defaults ship as package data (src/ankidkdeck/registry/); a run may layer local
additions from <work>/registry/ (same filenames). card_keys.json is append-only:
an existing family's guid_seed can never change -- it is the users' study
progress. These files contain word identities and rules only, never DDO text.
"""

import json
from importlib import resources
from pathlib import Path

from .util import FatalError, read_json, write_json

FILES = [
    "card_keys.json",
    "form_to_lemma.json",
    "known_no_entry.json",
    "alias_pairs.json",
    "demoted_pos_keys.json",
    "paradigm_slots.json",
    "pos_translations.json",
    "gates.json",
]


def _merge_dicts(base: dict, extra: dict) -> dict:
    """Overlay `extra` onto `base`, recursing into nested DICTS only.

    A shallow `{**base, **extra}` made every advertised partial overlay a trap: a
    work/registry/gates.json that touches `empty_rate_baseline_pct` at all
    replaced the whole 5-field baseline dict with whatever it named, and a
    pos_translations overlay adding one Chinese term would drop the other 21.

    A nested LIST is replaced, not appended: `note_count_range: [1, 10]` means
    that range, not "2800, 3100, 1 and 10". Appending is the right rule only for
    a whole registry file that IS a list (alias_pairs, demoted_pos_keys), which
    _merge() handles one level up.
    """
    out = dict(base)
    for k, v in extra.items():
        if isinstance(base.get(k), dict) and isinstance(v, dict):
            out[k] = _merge_dicts(base[k], v)
        else:
            out[k] = v
    return out


def _merge(base, extra):
    if isinstance(base, dict) and isinstance(extra, dict):
        return _merge_dicts(base, extra)
    if isinstance(base, list) and isinstance(extra, list):
        return base + [x for x in extra if x not in base]
    return extra


def _package_default(name: str):
    # Address the data directory through the top-level package: this module is
    # itself named `registry`, so files("ankidkdeck.registry") resolves to
    # registry.py and hands back src/ankidkdeck/ instead of the data dir.
    ref = resources.files("ankidkdeck").joinpath("registry", name)
    with ref.open("r", encoding="utf-8") as f:
        return json.load(f)


class Registry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = {}
        for name in FILES:
            base = _package_default(name)
            local = Path(cfg.registry_local) / name
            if local.exists():
                base = _merge(base, read_json(local))
            self.data[name.removesuffix(".json")] = base

    @property
    def card_keys(self) -> dict:
        return self.data["card_keys"]

    @property
    def form_to_lemma(self) -> dict:
        return self.data["form_to_lemma"]

    @property
    def known_no_entry(self) -> dict:
        return self.data["known_no_entry"]

    @property
    def alias_pairs(self) -> set:
        return {tuple(p) for p in self.data["alias_pairs"]}

    @property
    def demoted_pos_keys(self) -> set:
        return set(self.data["demoted_pos_keys"])

    def paradigm_labels(self, pos_key: str | None, shape: tuple) -> list | None:
        key = f"{pos_key}|{','.join(str(n) for n in shape)}"
        return self.data["paradigm_slots"].get(key)

    @property
    def pos_translations(self) -> dict:
        """{lang: {pos_key: translated}} -- hand-written, human-reviewed, and
        the reason an .apkg can be built with no LLM call at all.

        The ~14 live `data-pos-key` values x 4 languages are a table a
        lexicographer can write in an afternoon; without it G-COV blocked every
        export until someone paid for a Gemini POS call, which made the offline
        deck hostage to an API key. The 2025 `pos_translations_*_gemini.json`
        (41 mangled display strings) is deliberately NOT reused -- guide 1.11f.

        Keys starting with "_" are file-level documentation, not languages.
        """
        return {k: v for k, v in (self.data.get("pos_translations") or {}).items()
                if not str(k).startswith("_")}

    def pos_for(self, lang: str) -> dict:
        return dict(self.pos_translations.get(lang) or {})

    @property
    def gates(self) -> dict:
        return self.data["gates"]

    def freeze_card_keys(self, new_entries: dict, source_path: Path) -> dict:
        """Append-only merge of new families into card_keys.json.

        Existing family_ids are NEVER modified (re-anchoring silently destroys
        review history). Returns {added, carried, unchanged} counts and writes
        the merged file to source_path for human review + commit.
        """
        current = dict(self.card_keys)
        added = 0
        for fid, row in new_entries.items():
            if fid in current:
                if current[fid]["guid_seed"] != row["guid_seed"]:
                    raise FatalError(
                        f"registry violation: family {fid} would change guid_seed "
                        f"{current[fid]['guid_seed']!r} -> {row['guid_seed']!r}"
                    )
                continue
            current[fid] = row
            added += 1
        write_json(source_path, current)
        self.data["card_keys"] = current
        return {"added": added, "total": len(current)}
