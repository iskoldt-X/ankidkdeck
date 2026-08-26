"""Stage 41: telling the three ways of losing a translation row apart, and
keeping the one retranslation trigger the design has.

`n_bound + n_dropped == n_legacy` balanced before these fixes too -- which is
exactly the failure mode the gate was written against: the loss was explained
WRONGLY, not left unexplained. Every legacy row whose article the classifier had
rejected was labelled `article_gone_from_ddo`, and `rejected_article` was a code
in the closed set that could never fire.
"""

from conftest import make_entry, make_expression, make_sense

from ankidkdeck.stages.s41_bind import run as bind_run
from ankidkdeck.util import NFC, read_json, sha256_str, write_json


def _legacy_row(lemma, gloss, text):
    return {"lemma": lemma, "gloss": gloss,
            "src_sha": sha256_str(NFC(text)),
            "provenance": "migrated:2025:x.html"}


def _workspace(cfg, entries, classification, families, defs=None, exprs=None):
    write_json(cfg.json_dir / "entries.json", entries)
    write_json(cfg.json_dir / "classification.json", classification)
    write_json(cfg.json_dir / "words.json", families)
    write_json(cfg.json_dir / "legacy" / "sense_paths.json", {})
    for lang in cfg.langs:
        write_json(cfg.json_dir / "legacy" / ("legacy_%s_definitions.json" % lang),
                   defs or {})
        write_json(cfg.json_dir / "legacy" / ("legacy_%s_expressions.json" % lang),
                   exprs or {})


def _one_lang(cfg):
    cfg.langs = ["English"]
    return cfg


def test_a_rejected_article_is_reported_as_rejected_not_as_deleted(cfg):
    """Guide 6.3 population 3 -- ~445 cards whose only DDO article the
    classifier rejected -- has to be measurable from the pipeline's own output.

    `11000101` is present in entries.json but every word that saw it rejected
    it; `99999999` really is gone from DDO. Those are different losses.
    """
    _one_lang(cfg)
    kept = make_entry("11000100", "hus", pos_key="sb.",
                      senses=[make_sense("21000001", "bygning")])
    rejected = make_entry("11000101", "-hus", pos_key="sidsteled",
                          senses=[make_sense("21000002", "sidsteled")])
    entries = {e["entry_id"]: e for e in (kept, rejected)}
    classification = {
        "hus": {"members": [{"entry_id": "11000100", "bucket": "exact_cs",
                             "demoted": False}],
                "xrefs": [], "rejected": [{"entry_id": "11000101",
                                           "headword": "-hus",
                                           "pos_key": "sidsteled",
                                           "reason": "affix"}],
                "resolved_by": "forward"},
    }
    families = {"11000100": {"family_id": "11000100",
                             "anchor_entry_id": "11000100",
                             "entry_ids": ["11000100"]}}
    legacy_defs = {
        "11000100": {"bygning": _legacy_row("house", "a building", "bygning")},
        "11000101": {"sidsteled": _legacy_row("suffix", "a suffix", "sidsteled")},
        "99999999": {"vaek": _legacy_row("gone", "not here", "vaek")},
    }
    _workspace(cfg, entries, classification, families, defs=legacy_defs)
    report = bind_run(cfg)

    reasons = report["per_language"]["English"]["reasons"]
    assert reasons == {"rejected_article": 1, "article_gone_from_ddo": 1}
    dropped = read_json(cfg.json_dir / "translations" / "English" / "dropped.json")
    by_eid = {d["entry_id"]: d["reason"] for d in dropped}
    assert by_eid["11000101"] == "rejected_article"
    assert by_eid["99999999"] == "article_gone_from_ddo"
    # accounting still balances, which is the point: 1 bound + 2 dropped == 3
    per = report["per_language"]["English"]
    assert per["n_bound"] + per["n_dropped"] == per["n_legacy"] == 3
    assert per["n_unexplained"] == 0


def test_an_entry_recovered_by_stage_21_is_not_rejected(cfg):
    """The rejected set is read from classification.json AFTER stage 21, so an
    article recovered through the reverse index or an override still binds."""
    _one_lang(cfg)
    e = make_entry("11000110", "være", pos_key="vb.", forms=["er", "var"],
                   senses=[make_sense("21000010", "eksistere")])
    entries = {"11000110": e}
    classification = {
        # `er` saw nothing forward and rejected everything on its own page
        "er": {"members": [{"entry_id": "11000110", "bucket": "form",
                            "demoted": False, "evidence": "reverse_index"}],
               "xrefs": [], "rejected": [], "resolved_by": "reverse_index"},
        # ...and some other word did reject it
        "vare": {"members": [], "xrefs": [],
                 "rejected": [{"entry_id": "11000110", "headword": "være",
                               "pos_key": "vb.", "reason": "unrelated"}],
                 "resolved_by": None},
    }
    families = {"11000110": {"family_id": "11000110",
                             "anchor_entry_id": "11000110",
                             "entry_ids": ["11000110"]}}
    legacy = {"11000110": {"eksistere": _legacy_row("to be", "to exist",
                                                    "eksistere")}}
    _workspace(cfg, entries, classification, families, defs=legacy)
    report = bind_run(cfg)
    per = report["per_language"]["English"]
    assert per["n_bound"] == 1 and per["reasons"] == {}


def test_expression_rows_keep_the_expression_texts_sha(cfg):
    """src_sha is the sha of the EXPRESSION TEXT -- what 2025 actually sent to
    the LLM (payload field `expr`; the definition was only an optional `hint`).

    Stage 41 overwrote it with the sha of the idiom's DEFINITION for 736 of 736
    rows, which disables the single retranslation trigger (D7/1.7) for all
    12,716 expression cells x 4 languages: editing an idiom never retranslates,
    editing its definition retranslates for nothing.
    """
    _one_lang(cfg)
    expr = make_expression("21000020", "af hus", "fra en bestemt familie")
    e = make_entry("11021722", "hus", pos_key="sb.", expressions=[expr],
                   senses=[make_sense("21000021", "bygning")])
    families = {"11021722": {"family_id": "11021722",
                             "anchor_entry_id": "11021722",
                             "entry_ids": ["11021722"]}}
    legacy_exprs = {"11021722": {"af hus": _legacy_row("of the house",
                                                       "from a family",
                                                       "af hus")}}
    _workspace(cfg, {"11021722": e}, {}, families, exprs=legacy_exprs)
    bind_run(cfg)
    rows = read_json(cfg.json_dir / "translations" / "English" / "expressions.json")
    stored = rows["21000020"]["src_sha"]
    assert stored == sha256_str(NFC("af hus"))
    assert stored != sha256_str(NFC("fra en bestemt familie"))

    # stage 42 must compute the SAME formula, or every migrated cell reads as
    # changed and is paid for twice
    from ankidkdeck.stages.s42_translate import compute_todo, expression_src_sha
    assert expression_src_sha(expr) == stored
    todo = compute_todo(cfg, {"11021722": e},
                        {"definitions": {}, "expressions": rows}, "English")
    assert [r["kind"] for r in todo] == ["definition"]   # the expression is done


def test_an_equal_rebind_does_not_reassign_the_incumbents_sha(cfg):
    """Two legacy rows can share a dannetid and come from DIFFERENT Danish
    strings (16-17 real collapses per language). Overwriting the incumbent's
    src_sha makes the row's sha stop describing the string its gloss came
    from."""
    _one_lang(cfg)
    x = make_expression("21000030", "kold krig", "spaendt tilstand")
    a = make_entry("11000200", "krig", pos_key="sb.", expressions=[x],
                   senses=[make_sense("21000031", "vaebnet konflikt")])
    b = make_entry("11000201", "kold", pos_key="adj.", expressions=[x],
                   senses=[make_sense("21000032", "lav temperatur")])
    families = {"11000200": {"family_id": "11000200",
                             "anchor_entry_id": "11000200",
                             "entry_ids": ["11000200", "11000201"]}}
    same = {"lemma": "cold war", "gloss": "a tense standoff"}
    legacy_exprs = {
        "11000200": {"kold krig": {**same, "src_sha": sha256_str(NFC("kold krig")),
                                   "provenance": "migrated:2025:krig.html"}},
        # the SAME gloss arriving from a different Danish string
        "11000201": {"kold krig": {**same, "src_sha": "deadbeef",
                                   "provenance": "migrated:2025:kold.html"}},
    }
    _workspace(cfg, {"11000200": a, "11000201": b}, {}, families,
               exprs=legacy_exprs)
    bind_run(cfg)
    rows = read_json(cfg.json_dir / "translations" / "English" / "expressions.json")
    assert rows["21000030"]["src_sha"] == sha256_str(NFC("kold krig"))
    assert rows["21000030"]["provenance"] == "migrated:2025:krig.html"


def test_rows_on_entries_no_family_renders_are_reported(cfg):
    """Bind-then-GC is the guide's order (stage 42 archives the orphans and
    G-ORPH enforces it), so binding them is deliberate -- but the volume has to
    be reported rather than discovered by the GC."""
    _one_lang(cfg)
    used = make_entry("11000300", "hus", pos_key="sb.",
                      senses=[make_sense("21000040", "bygning")])
    unused = make_entry("11000301", "allé", pos_key="sb.",
                        senses=[make_sense("21000041", "vej med traeer")])
    families = {"11000300": {"family_id": "11000300",
                             "anchor_entry_id": "11000300",
                             "entry_ids": ["11000300"]}}
    legacy = {"11000300": {"bygning": _legacy_row("house", "a building",
                                                  "bygning")},
              "11000301": {"vej med traeer": _legacy_row("avenue", "a road",
                                                         "vej med traeer")}}
    _workspace(cfg, {"11000300": used, "11000301": unused}, {}, families,
               defs=legacy)
    report = bind_run(cfg)
    per = report["per_language"]["English"]
    assert per["n_bound"] == 2
    assert per["n_bound_on_unused_entries"] == 1
    assert "entry-scoped by design" in report["bind_scope"]


def test_a_paid_translate_cannot_silently_pollute_the_migration_audit(cfg):
    """Spec 5.9. n_legacy / n_bound / n_dropped are the audit of the recovered
    2025 asset, but they are computed against the LIVE translation tables -- so
    once a paid v3 translate has written rows there, a re-bind (or a `build`,
    which runs bind) answers a different question under the same name: the key a
    legacy row would have taken is occupied, `_place` keeps the incumbent, and
    the row is filed as a shared_dannetid_conflict. The bind rate then falls for
    a reason that has nothing to do with 2025.

    The counts are SPLIT and labelled rather than refused: bind is part of
    `build`, so a hard stop here would brick a rebuild after the first paid
    wave.
    """
    _one_lang(cfg)
    e = make_entry("11000400", "hus", pos_key="sb.",
                   senses=[make_sense("21000050", "bygning")])
    families = {"11000400": {"family_id": "11000400",
                             "anchor_entry_id": "11000400",
                             "entry_ids": ["11000400"]}}
    legacy = {"11000400": {"bygning": _legacy_row("house", "a building",
                                                  "bygning")}}
    _workspace(cfg, {"11000400": e}, {}, families, defs=legacy)

    clean = bind_run(cfg)
    assert clean["audit_is_pre_llm"] is True
    per = clean["per_language"]["English"]
    assert per["keys_from_paid_translate"] == {"definitions": 0,
                                              "expressions": 0}
    assert per["audit_integrity"].startswith("clean")

    # ...now a paid v3 run has written its own row over that key
    write_json(cfg.json_dir / "translations" / "English" / "definitions.json",
               {"11000400:21000050": {
                   "lemma": "building", "gloss": "a structure",
                   "src_sha": sha256_str(NFC("bygning")),
                   "provenance": "gemini:gemini-3.7-flash+v4-frozen+LOW@2026-08-26"}})
    polluted = bind_run(cfg)
    per = polluted["per_language"]["English"]
    assert polluted["audit_is_pre_llm"] is False
    assert per["keys_from_paid_translate"]["definitions"] == 1
    assert per["audit_integrity"].startswith("SPLIT")
    assert any("paid v3 translate" in w for w in polluted["warnings"])
    # the legacy accounting itself still balances: it is read from json/legacy/
    assert per["n_bound"] + per["n_dropped"] == per["n_legacy"]
    assert per["reasons"] == {"shared_dannetid_conflict": 1}
    # ...and the migrated row was NOT overwritten by the legacy one
    row = read_json(cfg.json_dir / "translations" / "English"
                    / "definitions.json")["11000400:21000050"]
    assert row["lemma"] == "building"
