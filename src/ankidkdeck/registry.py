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
    "gates.json",
]


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
                extra = read_json(local)
                if isinstance(base, dict) and isinstance(extra, dict):
                    base = {**base, **extra}
                elif isinstance(base, list) and isinstance(extra, list):
                    base = base + [x for x in extra if x not in base]
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
