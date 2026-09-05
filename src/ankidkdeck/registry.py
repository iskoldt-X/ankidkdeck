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
    "known_missing_audio.json",
    "wordlist_invalid.json",
    "alias_pairs.json",
    "alias_merge_pending.json",
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
        """Words DDO genuinely has no entry for. Stage 21 stops reporting them.

        Filled 2026-08-26 (owner decision, round-2 open item 10.45) from the 545
        verified nohit rows: every one had fetch status `nohit`, its archived page
        carries DDO's own "matcher ingen opslag i ordbogen" string, and none of
        them appears among the 96,956 lemma keys of DDO's published sitemap. The
        67 OCR-corrupted wordlist rows are deliberately NOT here -- 53 of them are
        in wordlist_invalid.json and 14 stay visible in reports/unresolved.json,
        because "DDO has no such word" is the wrong thing to record about a row
        that is a misspelling of a word DDO does have.

        The population is baselined (gates.json:known_no_entry_max, G-SUPPRESS):
        this file removes words from the ONLY report that shows them, so it must
        not be able to grow silently.
        """
        return self.data["known_no_entry"]

    @property
    def known_missing_audio(self) -> dict:
        """Audio slots DDO DECLARES and DDO's audio host cannot serve.
        {audio_url: {url_slot, entry_id, lemma, entry_keeps_slots, reason,
        evidence, verified_by}}.

        Only the KEY -- the exact audio_url -- is read by code. Both halves of
        G-MEDIA (s60_audio._media_gate over the cache, s70_export.media_gate over
        the notes) baseline themselves against it: a slot in here is reported as
        `known_missing_upstream` instead of counted as a failure, and a missing
        slot that is NOT in here still fails both.

        Filled 2026-08-27 from the four slots the stage-60 delta run could not
        fetch: HTTP 200, content-length 0, content-type text/html and the same
        etag "5b06c6d8-0" -- nginx's etag for a zero-byte file -- on all four,
        deterministic over 3 attempts each, while sibling slots of the same
        entries answered with real mp3 bodies. That is an upstream data defect
        per SLOT; no retry ladder reaches it.

        The population is baselined (gates.json:known_missing_audio_max) and the
        file's `_note` carries the review contract, including the two rules that
        keep it from becoming a suppression list: a file missing for any reason of
        OURS is not an entry here, and a slot that starts working is reported as
        `recovered` and must then be deleted from the file. Keys starting with
        "_" are file-level documentation, not URLs.
        """
        return {k: v for k, v in (self.data.get("known_missing_audio") or {}).items()
                if not str(k).startswith("_")}

    @property
    def wordlist_invalid(self) -> dict:
        """Wordlist rows that are not words: OCR damage in the frozen 5,000-row
        subtitle wordlist. {row: {correct, reason, rule}}.

        Only the KEY is read by code. Stage 21 skips these rows before any layer
        runs, so they never bind and never reach unresolved.json.

        This is the registry route for owner decision 10.5, chosen over editing
        the wordlist because wordlist_sha256 is the foundation of every GUID:
        the file stays byte-identical, so no row's rank moves and no family's
        guid_seed can change. Each row's `correct` field names the word the OCR
        damaged, and 52 of the 53 corrections already ship a card at a BETTER
        rank -- which is the proof that invalidating the row loses no Danish
        content.

        The one exception is `fbl` -> `fbi`, where the correction is itself a DDO
        nohit and ships nothing: neither string is a word in any language, so
        that row is invalid on its own terms rather than by the ships-a-card
        rule, and it carries a `note` saying so. `vei` also carries a `note`, but
        about WHICH correction (`vel`, rank 188; `vej` would need an i->j rule
        this registry does not have) -- `vel` ships a card normally, so `vei` is
        not an exception to the rule above. `ali` -> `all` fails the same test as
        `fbl` and is deliberately NOT here: `Ali` is a name, so that row stays
        visible in reports/unresolved.json for a human.
        """
        return self.data["wordlist_invalid"]

    @property
    def alias_pairs(self) -> set:
        return {tuple(p) for p in self.data["alias_pairs"]}

    @property
    def alias_merge_pending(self) -> list:
        """Alias pairs the CLASSIFIER honours but the MERGE must keep as two
        heads -- for now.

        Both sides of these three pairs have their own DDO article and their own
        already-frozen card_keys.json row, so merging them retires one of the two
        guid_seeds (`check`, a real v2.1 QueryWord, disappears into `tjek`).
        card_keys.json is append-only, so the pipeline can neither rewrite nor
        un-retire that row: which GUID retires is a release decision with a
        deadline, and it belongs in the same single re-freeze as the 22 families
        whose seed was frozen before the unresolved list was curated.

        Emptying this file is the switch that lands the merge -- one line, and
        the re-freeze procedure names it as its first step. It is deliberately
        NOT the alias registry: the classifier must keep admitting these pairs,
        or `naeh` and `o.k.` lose the only edge that reaches them at all.
        """
        return [list(p) for p in (self.data.get("alias_merge_pending") or [])]

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

    def freeze_card_keys(self, new_entries: dict, source_path: Path,
                         proposed_seeds: dict | None = None) -> dict:
        """Append-only merge of new families into card_keys.json.

        Existing family_ids are NEVER modified (re-anchoring silently destroys
        review history). Returns {added, total, unchanged, stale_seeds,
        not_in_the_committed_registry} and writes the merged file to source_path
        for human review + commit.

        `added` counts what THIS call appended, which is 0 on any rerun: the rows
        were appended by an earlier run and append-only means they stay. That is
        correct and it is also unreadable as evidence -- round 2 appended 4 rows
        and then two idempotent reruns overwrote the report with `added: 0`, so
        the file an owner opens says this workspace froze nothing. The count that
        survives a rerun is `not_in_the_committed_registry`: rows the overlay
        holds that src/ankidkdeck/registry/card_keys.json does not. It is the
        size of the diff a human still has to review and commit, so it answers
        "what has this workspace frozen" no matter how many times the chain ran.

        `proposed_seeds` is {family_id: seed} for EVERY family in this build,
        already-frozen ones included -- i.e. what today's data would choose. It
        exists because this method could not previously see them: the caller
        filtered every existing family_id out before calling, so the guard below
        was dead code AND the append-only rule had no voice. That combination is
        how 22 families froze the LESS frequent of two spellings -- the freeze
        ran before the unresolved list was curated, so the more frequent member
        word did not exist yet -- and locked it in with no error, no warning and
        no report line. The rows still do not change (append-only is the point;
        the users' study progress is in those bytes), but the divergence is now
        RETURNED, so stage 30 can write it out and a re-freeze decision can be
        made on evidence instead of on a re-derivation nobody had run.
        """
        current = dict(self.card_keys)
        added = 0
        for fid, row in new_entries.items():
            if fid in current:
                # Reachable: a caller may hand over a family that is already
                # frozen (tests do, and any future single-pass freeze would).
                # Refusing loudly is right here -- unlike proposed_seeds below,
                # this dict is a WRITE request.
                if current[fid]["guid_seed"] != row["guid_seed"]:
                    raise FatalError(
                        f"registry violation: family {fid} would change guid_seed "
                        f"{current[fid]['guid_seed']!r} -> {row['guid_seed']!r}"
                    )
                continue
            current[fid] = row
            added += 1
        stale = []
        for fid, seed in sorted((proposed_seeds or {}).items()):
            row = current.get(fid)
            if row is not None and row.get("guid_seed") != seed:
                stale.append({"family_id": fid, "frozen_seed": row.get("guid_seed"),
                              "seed_today": seed,
                              "frozen_since": row.get("since"),
                              "lemma_at_freeze": row.get("lemma_at_freeze")})
        write_json(source_path, current)
        self.data["card_keys"] = current
        # Re-read the SHIPPED default rather than remembering it: self.card_keys
        # is the merged view, and the question here is exactly what the overlay
        # adds on top of what is committed.
        committed = _package_default("card_keys.json")
        return {"added": added, "total": len(current),
                "unchanged": len(current) - added,
                "not_in_the_committed_registry":
                    sum(1 for fid in current if fid not in committed),
                "stale_seeds": stale}
