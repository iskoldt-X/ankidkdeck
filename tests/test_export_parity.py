"""Parity with v2.1, at the two levels where it actually means something.

LEVEL 1, always runnable: the ported CSS / QFMT / AFMT blocks are BYTE-EQUAL to
`git show v2.1-pipeline:09_export_apkg.py`. The sha256 of each block is pinned
below, computed from that branch at test-authoring time. Round 2 found the claim
"ported verbatim" was not quite true -- an editor had stripped a trailing space
from two CSS lines and three AFMT <script> lines. Inert on the rendered card, but
the notetype is rewritten on import from these exact strings, and a docstring
that says "verbatim" has to be checkable. A stripping editor now fails here
instead of quietly changing everyone's notetype.

LEVEL 2, fixture/workspace-gated: the identity constants against the REAL released
.apkg files, when the recovered 2025 workspace is on the host
(ANKIDKDECK_LEGACY_WORKSPACE, or --legacy-workspace's usual path). deck id, model
id, model name, the 8 field names and sortf must be byte-identical to what the
users already have in their collections.

WHAT IS DELIBERATELY NOT TESTED: note-for-note byte parity with v2.1. v3 changed
the card UNIT from one query word to one word family, so the note count, the GUID
set and every field differ by design (guide D10). tools/compare_apkg.py is the
harness for full byte parity and it is meaningful only against the v2.1-pipeline
branch's own output -- see its docstring.
"""

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from ankidkdeck.stages import s70_export as S70

REPO_ROOT = Path(__file__).resolve().parents[1]

# sha256 of each block as it stands on the v2.1-pipeline branch, in
# 09_export_apkg.py. Recompute with:
#   git show v2.1-pipeline:09_export_apkg.py
# and hash the CSS / QFMT string literals and the two string operands of the
# AFMT concatenation (the third operand is COPYRIGHT_HTML, which is per-language
# data and carries the year).
V21_SHA256 = {
    "CSS": "15a6d8ea0699bbcaa0d442614815cb49d1596756801f98dda5be4c2baa9d2aee",
    "QFMT": "07a2ecece0ca264115cc524707588120f35ec7144ebdd23895979b762b232a1f",
    "AFMT_HEAD": "ddfe95e03698fe1d504dbb0c63c0d92c07e9bcd1c03a14ea6d9b7e171900c9bf",
    "AFMT_SCRIPT": "75d54f2d1fc7e8f0eba18ecf3a3950490ee044b4ae3e8cef897a354f64c59a5c",
}

# Where the v3 CSS stops being a port and starts being v3.
V3_CSS_MARKER = "\n/* --- V3 ADDITIONS"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_ported_css_prefix_is_byte_equal_to_v2_1():
    ported = S70.CSS.split(V3_CSS_MARKER)[0]
    assert _sha(ported) == V21_SHA256["CSS"], (
        "the v2.1 CSS block has drifted. Trailing whitespace counts: two "
        "`details > summary` lines carry a trailing space in "
        "v2.1-pipeline:09_export_apkg.py.")
    # ...and the v3 additions are a clearly separated block after it
    assert V3_CSS_MARKER in S70.CSS
    assert ".searchable-forms { display: none; }" in S70.CSS


def test_qfmt_is_byte_equal_to_v2_1():
    assert _sha(S70.QFMT) == V21_SHA256["QFMT"]


def test_the_afmt_blocks_are_byte_equal_to_v2_1():
    assert _sha(S70.AFMT_HEAD) == V21_SHA256["AFMT_HEAD"]
    assert _sha(S70.AFMT_SCRIPT) == V21_SHA256["AFMT_SCRIPT"], (
        "the AFMT <script> has drifted; three lines carry trailing whitespace "
        "in v2.1-pipeline:09_export_apkg.py")


def test_afmt_puts_the_copyright_between_the_two_ported_blocks():
    html = S70.deck_meta("German", 2026)["copyright_html"]
    assert S70.afmt(html) == S70.AFMT_HEAD + html + S70.AFMT_SCRIPT


# ---------------------------------------------------- against the branch

def _v21_blocks():
    """The blocks read straight out of the v2.1-pipeline branch, if git and the
    branch are both here. Parsed with ast: importing that script would pull in
    genanki and its module-level config."""
    try:
        src = subprocess.run(
            ["git", "show", "v2.1-pipeline:09_export_apkg.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if src.returncode != 0 or not src.stdout:
        return None
    out = {}
    for node in ast.parse(src.stdout).body:
        if not (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in ("CSS", "QFMT"):
            out[name] = ast.literal_eval(node.value)
        elif name == "AFMT":
            parts = []

            def walk(n):
                if isinstance(n, ast.BinOp):
                    walk(n.left)
                    walk(n.right)
                elif isinstance(n, ast.Constant):
                    parts.append(n.value)
            walk(node.value)
            if len(parts) == 2:
                out["AFMT_HEAD"], out["AFMT_SCRIPT"] = parts
    return out or None


def test_the_pinned_hashes_still_describe_the_branch():
    """The pins are only trustworthy if they match the source they claim to come
    from. Skips where the branch is not available (a shallow clone)."""
    blocks = _v21_blocks()
    if not blocks:
        pytest.skip("v2.1-pipeline:09_export_apkg.py is not readable here")
    for name, sha in V21_SHA256.items():
        assert name in blocks, name
        assert _sha(blocks[name]) == sha, (
            "the pinned sha for %s does not match the branch any more" % name)


def test_the_v3_exporter_matches_the_branch_block_for_block():
    blocks = _v21_blocks()
    if not blocks:
        pytest.skip("v2.1-pipeline:09_export_apkg.py is not readable here")
    assert S70.CSS.split(V3_CSS_MARKER)[0] == blocks["CSS"]
    assert S70.QFMT == blocks["QFMT"]
    assert S70.AFMT_HEAD == blocks["AFMT_HEAD"]
    assert S70.AFMT_SCRIPT == blocks["AFMT_SCRIPT"]


# --------------------------------------- against the released .apkg files

LEGACY_ENV = "ANKIDKDECK_LEGACY_WORKSPACE"
RELEASED = "DDO_Danish_Frequency_Deck_%s.apkg"


def _legacy_workspace():
    p = os.environ.get(LEGACY_ENV)
    if p and Path(p).is_dir():
        return Path(p)
    default = (Path.home() / "GitHub" / "danish_pipelines" / "data"
               / "recovered-v2.1-workspace")
    return default if default.is_dir() else None


def _released(lang):
    ws = _legacy_workspace()
    if ws is None:
        return None
    p = ws / (RELEASED % lang)
    return p if p.exists() else None


def _model_and_deck(path):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            db = ("collection.anki21" if "collection.anki21" in names
                  else "collection.anki2")
            z.extract(db, tmp)
        con = sqlite3.connect(str(Path(tmp) / db))
        try:
            models, decks = con.execute(
                "SELECT models, decks FROM col").fetchone()
            notes = con.execute("SELECT guid, flds FROM notes").fetchall()
        finally:
            con.close()
    models, decks = json.loads(models), json.loads(decks)
    deck = [d for k, d in decks.items() if str(k) != "1"][0]
    return list(models.values())[0], deck, notes


@pytest.mark.parametrize("lang", ["Chinese", "English", "German", "Spanish"])
def test_the_identity_constants_match_the_released_deck(lang):
    """The one comparison that decides "upgrade" vs "4,442 duplicates"."""
    path = _released(lang)
    if path is None:
        pytest.skip("released %s deck is not on this host (set %s)"
                    % (lang, LEGACY_ENV))
    model, deck, notes = _model_and_deck(path)
    assert int(deck["id"]) == S70.deck_id(lang)
    assert int(model["id"]) == S70.model_id(lang)
    assert model["name"] == S70.model_name(lang)
    assert [f["name"] for f in model["flds"]] == S70.FIELD_NAMES
    assert model["sortf"] == S70.SORT_FIELD_INDEX
    assert deck["name"] == S70.deck_meta(lang, 2025)["deck_name"]


@pytest.mark.parametrize("lang", ["German"])
def test_every_released_guid_is_guid_for_its_own_first_field(lang):
    """The GUID formula, checked against real shipped data: guid_for(field0,
    lang). v3 carries the seed forward from the registry precisely so this stays
    true for a card the user already studies."""
    genanki = pytest.importorskip("genanki")
    path = _released(lang)
    if path is None:
        pytest.skip("released %s deck is not on this host (set %s)"
                    % (lang, LEGACY_ENV))
    _, _, notes = _model_and_deck(path)
    assert notes
    bad = [(g, f.split("\x1f")[0]) for g, f in notes
           if genanki.guid_for(f.split("\x1f")[0], lang) != g]
    assert bad == [], bad[:5]
