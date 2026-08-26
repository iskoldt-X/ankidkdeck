"""The money stack: prices, the ledger, the arithmetic and the money gates.

Every test here is offline and none of them may import the SDK. The named tests
the patch plan asks for (section 4.1) are:

    test_money_arithmetic          the one formula, both conventions, and a
                                   2027 doubling case
    test_g_cache_predicate         the new criterion passes, and the audit's
                                   cached/prompt >= 0.90 is pinned as WRONG
    test_g_think_predicate         a measured zero passes, a MEDIUM row fails
    test_gate_provenance           the consumption rules refuse a spend

plus the ones the money stack needs to be trustworthy on its own: that no
measured constant is hard-coded in it, that a crashed wave still reaches the
month-to-date total, and that the two dollar implementations in this codebase
agree to the cent.
"""

import ast
import datetime
import json
import sys

import pytest

from ankidkdeck import billing, gates
from ankidkdeck.prices import (CACHED_INPUT_OPEN_QUESTION, DOUBLING_DATE,
                               PRICING_PAGE_READ_AT, cache_storage_usd,
                               priced_models, rate_card)
from ankidkdeck.util import FatalError, write_json

MODEL = "gemini-3.7-flash"
IN_2026 = datetime.date(2026, 12, 31)
IN_2027 = datetime.date(2027, 1, 1)


def usage_row(**kw) -> dict:
    """One normalize_usage()-shaped row. cached is a SUBSET of prompt here."""
    row = {"label": "test", "kind": "definition", "model": MODEL,
           "mode": "batch", "prompt_id": "v4-frozen", "cache_name": None,
           "finish_reason": "STOP", "n_expected": 8,
           "prompt_tokens": 1350, "cached_tokens": 0,
           "candidates_tokens": 300, "tool_use_tokens": 0,
           "thinking_tokens": 0}
    row.update(kw)
    row["total_tokens"] = (row["prompt_tokens"] + row["candidates_tokens"]
                           + row["thinking_tokens"] + row["tool_use_tokens"])
    return row


# --------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------

def test_a_rate_card_is_a_fact_about_a_page_on_a_day():
    """The allow-list, the read date and the two windows."""
    from ankidkdeck.config import VERIFIED_MODELS
    assert priced_models() == (MODEL,)
    # A model may not be allow-listed for constants with its card read on one
    # date and priced from a card read on another.
    assert VERIFIED_MODELS[MODEL]["rate_card_read_at"] == PRICING_PAGE_READ_AT
    with pytest.raises(FatalError) as exc:
        rate_card("gemini-2.0-flash", "batch")
    assert "no rate card for model" in str(exc.value)
    with pytest.raises(FatalError):
        rate_card(MODEL, "interactive")

    std = rate_card(MODEL, "standard", on_date=IN_2026)
    batch = rate_card(MODEL, "batch", on_date=IN_2026)
    flex = rate_card(MODEL, "flex", on_date=IN_2026)
    assert batch["input_usd_per_mtok"] == std["input_usd_per_mtok"] / 2
    assert batch["output_usd_per_mtok"] == std["output_usd_per_mtok"] / 2
    # flex is a service tier on the standard surface, priced like batch --
    # penny for penny, which is what the pricing page shows.
    assert {k: flex[k] for k in ("input_usd_per_mtok", "output_usd_per_mtok",
                                 "cached_input_usd_per_mtok")} == \
        {k: batch[k] for k in ("input_usd_per_mtok", "output_usd_per_mtok",
                               "cached_input_usd_per_mtok")}
    # The conservative reading of the one unsettled rate, and it says so.
    assert batch["cached_input_usd_per_mtok"] == 0.075
    assert "0.0375" in batch["cached_input_unsettled"]
    assert CACHED_INPUT_OPEN_QUESTION == batch["cached_input_unsettled"]


def test_the_price_doubles_on_the_first_of_january_2027():
    """The page states two prices with a boundary and never calls the lower one
    promotional, so the card is a function of the date."""
    assert DOUBLING_DATE == datetime.date(2027, 1, 1)
    for mode in ("standard", "batch", "flex"):
        before = rate_card(MODEL, mode, on_date=IN_2026)
        after = rate_card(MODEL, mode, on_date=IN_2027)
        for key in ("input_usd_per_mtok", "output_usd_per_mtok",
                    "cached_input_usd_per_mtok"):
            assert after[key] == pytest.approx(before[key] * 2), (mode, key)
        assert "2026" in before["window"] and "2027" in after["window"]


def test_cache_storage_is_negligible_and_computed_rather_than_asserted():
    """A 1,135-token definition prompt held for a whole 24-hour batch drain."""
    usd = cache_storage_usd(1135, 24, MODEL, "batch", on_date=IN_2026)
    assert usd == pytest.approx(0.01362, rel=1e-6)
    # 1.4 cents per language for a whole day of drain -- small next to a $2-4
    # wave, and NOT the 0.001 cents a "storage is negligible" hand-wave would
    # have put in the report. This is why it is computed.
    assert cache_storage_usd(4564, 24, MODEL, "batch") > usd


# --------------------------------------------------------------------------
# the arithmetic (spec 2.5 / W0-5)
# --------------------------------------------------------------------------

def test_money_arithmetic():
    """input$ = (prompt - cached) x r_uncached + cached x r_cached;
    output$ = (candidates + thoughts) x r_out; total == the four token counts.

    Includes a CACHED row (where the subset convention decides the answer) and
    a row priced on the 2027 card.
    """
    rates = rate_card(MODEL, "batch", on_date=IN_2026)
    row = usage_row(prompt_tokens=1350, cached_tokens=1135,
                    candidates_tokens=300, thinking_tokens=0)
    money = billing.row_dollars(row, rates)
    # cached is a SUBSET of prompt: the uncached half is the difference, never
    # the sum. Getting this backwards is the single easiest way to misbill.
    assert money["uncached_input_tokens"] == 1350 - 1135
    assert money["cached_input_tokens"] == 1135
    expected_in = (215 * rates["input_usd_per_mtok"]
                   + 1135 * rates["cached_input_usd_per_mtok"]) / 1e6
    assert money["input_usd"] == pytest.approx(expected_in, rel=1e-9)
    assert money["output_usd"] == pytest.approx(
        300 * rates["output_usd_per_mtok"] / 1e6, rel=1e-9)
    assert money["usd"] == pytest.approx(money["input_usd"]
                                         + money["output_usd"], rel=1e-9)
    assert money["identity_ok"] is True

    # thinking is billed as OUTPUT, never as input.
    thinking = billing.row_dollars(usage_row(thinking_tokens=500), rates)
    plain = billing.row_dollars(usage_row(thinking_tokens=0), rates)
    assert thinking["usd"] - plain["usd"] == pytest.approx(
        500 * rates["output_usd_per_mtok"] / 1e6, rel=1e-9)

    # the same row on the 2027 card costs exactly twice as much
    later = billing.row_dollars(row, rate_card(MODEL, "batch", on_date=IN_2027))
    assert later["usd"] == pytest.approx(money["usd"] * 2, rel=1e-9)

    # a row whose counts do not add up is not silently priced
    broken = usage_row()
    broken["total_tokens"] += 77
    ok, detail = billing.token_identity(broken)
    assert ok is False and detail["difference"] == 77
    with pytest.raises(FatalError):
        billing.row_dollars(broken, rates, strict=True)
    assert billing.row_dollars(broken, rates, strict=False)["identity_ok"] \
        is False

    # cached can never exceed prompt: that would mean the two were added
    with pytest.raises(FatalError):
        billing.row_dollars(usage_row(prompt_tokens=100, cached_tokens=200),
                            rates)


def test_the_two_dollar_implementations_agree():
    """Spec 2.5 says the billing arithmetic is written ONCE.

    Stage 42's dollar_figures() (the forecast, in DISJOINT token blocks) and
    billing.usd_for_tokens() (the ledger, after the subset subtraction) are two
    call sites of one formula. They are currently two implementations, so this
    test is the thing that keeps them one formula: if either changes, it fails.
    """
    from ankidkdeck.stages import s42_translate as S42
    rates = rate_card(MODEL, "batch", on_date=IN_2026)
    tokens = {"available": True}
    for i, scenario in enumerate(billing.SCENARIOS):
        tokens[scenario] = {"requests": 10 + i,
                            "cached_input_tokens": 1135 * (10 + i),
                            "uncached_input_tokens": 2000 + i,
                            "output_tokens": 3000 + i,
                            "thinking_tokens": 0}
    theirs = S42.dollar_figures(tokens, rates, ceiling_usd=10.0)
    for scenario in billing.SCENARIOS:
        t = tokens[scenario]
        mine = billing.usd_for_tokens(
            uncached_input=t["uncached_input_tokens"],
            cached_input=t["cached_input_tokens"],
            output=t["output_tokens"] + t["thinking_tokens"],
            rates=rates, ndigits=4)
        assert theirs[scenario] == mine, scenario
    # and the scenario vocabulary itself is one vocabulary
    assert billing.SCENARIOS == S42.BILL_SCENARIOS
    assert billing.FORBIDDEN_SCENARIO == S42.FORBIDDEN_SCENARIO


def test_a_bill_with_no_rate_card_states_the_absence():
    """The other half of the same contract: without a card, every figure is None
    and `why` says so. A made-up price gets read; a missing one gets asked
    about."""
    from ankidkdeck.stages import s42_translate as S42
    tokens = {"available": True,
              "cache_works": {"requests": 1, "cached_input_tokens": 1,
                              "uncached_input_tokens": 1, "output_tokens": 1,
                              "thinking_tokens": 0}}
    tokens["lean_uncached"] = tokens["rich_uncached"] = tokens["cache_works"]
    out = S42.dollar_figures(tokens, None, ceiling_usd=10.0)
    assert out["cache_works"] is None and "no rate card" in out["why"]


# --------------------------------------------------------------------------
# the ledger (spec 2.4)
# --------------------------------------------------------------------------

def _write_usage(cfg, name, rows):
    path = cfg.report_dir / name
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_the_ledger_counts_a_crashed_wave(cfg):
    """A wave that died still spent money.

    Stage 42 fsyncs every call to reports/translate_usage.jsonl BEFORE anything
    else, so those rows are on disk even when the run never reached a sink and
    never wrote a report. The ledger ingests that file, which is why
    month-to-date is right after a crash and not only after a clean run.
    """
    _write_usage(cfg, "translate_usage.jsonl",
                 [usage_row(label="a", cached_tokens=1135),
                  usage_row(label="b"), usage_row(label="c")])
    ledger = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 26))
    first = ledger.ingest()
    assert first["absorbed"] == {"translate_usage.jsonl": 3}
    assert len(ledger.rows()) == 3
    total = ledger.period_to_date()
    assert total["period_key"] == "2026-08"
    assert total["requests"] == 3
    assert total["usd"] > 0
    # the roll-up spec 2.4 names is DERIVED, and deleting it is safe
    assert (cfg.report_dir / billing.LEDGER_JSON).exists()

    # ingesting again absorbs nothing: idempotent by (source, line, sha)
    again = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 26))
    assert again.ingest()["absorbed"] == {}
    assert again.period_to_date()["usd"] == total["usd"]

    # a new call appends one line; only that line is absorbed
    _write_usage(cfg, "translate_usage.jsonl", [usage_row(label="d")])
    third = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 26))
    assert third.ingest()["absorbed"] == {"translate_usage.jsonl": 1}
    assert third.period_to_date()["requests"] == 4

    # losing the derived roll-up cannot cause a double count: the cursors live
    # in the append-only JSONL itself
    (cfg.report_dir / billing.LEDGER_JSON).unlink()
    fourth = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 26))
    assert fourth.ingest()["absorbed"] == {}
    assert fourth.period_to_date()["requests"] == 4


def test_the_ledger_notices_a_usage_file_that_is_not_the_one_it_absorbed(cfg):
    """Deleted and recreated, or truncated: a new GENERATION, read from the top.

    Over-counting refuses a wave that would have fit; under-counting spends
    money nobody accepted. So the ambiguous case resolves towards counting.
    """
    _write_usage(cfg, "priority_usage.jsonl", [usage_row(kind="rank")] * 2)
    led = billing.SpendLedger(cfg)
    led.ingest()
    assert led.period_to_date()["requests"] == 2
    (cfg.report_dir / "priority_usage.jsonl").unlink()
    _write_usage(cfg, "priority_usage.jsonl", [usage_row(kind="rank")])
    fresh = billing.SpendLedger(cfg)
    out = fresh.ingest()
    assert out["new_generations"] == ["priority_usage.jsonl#g2"]
    assert fresh.period_to_date()["requests"] == 3


def test_the_sink_is_the_per_call_half(cfg):
    """billing.ledger_sink(cfg) is what stage 42 takes as `usage_sink`."""
    sink = billing.ledger_sink(cfg, on_date=datetime.date(2026, 8, 26))
    sink(usage_row(label="one"))
    sink(usage_row(label="two", cached_tokens=1135, cache_name="caches/x"))
    rows = billing.SpendLedger(cfg).rows()
    assert [r["source"] for r in rows] == ["sink", "sink"]
    assert all(r["usd"] > 0 for r in rows)
    assert rows[1]["cached_input_tokens"] == 1135


def _paid_call(label="call", kind="definition", **usage):
    """One usage row the way the STAGE makes it: through normalize_usage, so it
    carries the (ts, seq) stamp the ledger's identity is built on."""
    from ankidkdeck.stages.s42_translate import normalize_usage
    counts = {"promptTokenCount": 1350, "candidatesTokenCount": 300,
              "totalTokenCount": 1650}
    counts.update(usage)
    return normalize_usage(counts, model=MODEL, label=label, kind=kind,
                           mode="batch", prompt_id="v4-frozen")


def test_the_sink_and_the_ingest_are_one_channel(cfg):
    """Reviewer A's BLOCKER and reviewer B's MAJOR-4, both measured at EXACTLY
    2.000x: 3 calls -> 6 ledger rows -> $0.002016 doubled to $0.004032.

    UsageLog.record() writes reports/translate_usage.jsonl and THEN calls the
    sink -- both always happen, and the disk half cannot be dropped because it is
    the crash-safe one. The sink's rows carried source="sink" with no line
    number, the ingest cursor skipped exactly those, so wiring the sink (which is
    instruction 1 of the integration note) made every call count twice. In a
    $10 cap with $0.22 of headroom that refuses the third language while
    reporting the fourth as paid for: the gate becomes a number that lies.

    The fix is one authoritative identity per call, stamped by normalize_usage.
    This test wires BOTH doors exactly as the integration note says to and
    asserts the ratio is 1.000.
    """
    from ankidkdeck.stages.s42_translate import UsageLog
    usage = UsageLog(sink=billing.ledger_sink(cfg),
                     path=cfg.report_dir / "translate_usage.jsonl")
    for i in range(3):
        usage.record(_paid_call(label="call-%d" % i))

    sink_half = billing.SpendLedger(cfg).summary()
    assert sink_half["rows"] == 3 and sink_half["usd_total"] > 0

    after = billing.SpendLedger(cfg)
    out = after.ingest()
    assert out["absorbed"] == {}
    assert out["already_on_the_ledger"] == {"translate_usage.jsonl": 3}
    both = after.summary()
    assert both["rows"] == 3
    assert both["usd_total"] / sink_half["usd_total"] == 1.0

    # ...and in the other order too: ingest first, then the sink fires for a row
    # already absorbed. append() is uid-idempotent, so the order cannot matter.
    row = _paid_call(label="late")
    _write_usage(cfg, "translate_usage.jsonl", [row])
    third = billing.SpendLedger(cfg)
    assert third.ingest()["absorbed"] == {"translate_usage.jsonl": 1}
    billing.ledger_sink(cfg)(row)
    assert billing.SpendLedger(cfg).summary()["rows"] == 4


def test_every_stage_usage_file_reaches_the_ledger(cfg):
    """Reviewer B's MAJOR-3, measured: 7 rows on disk, 5 absorbed.

    USAGE_GLOB was "*_usage.jsonl", which does not match
    review_usage_German.jsonl -- so every dollar the review subcommand spent was
    invisible to G-BUDGET, in the UNDER-counting direction (the one that spends
    money nobody approved). The old tests used only the two filenames that
    happened to match. This one enumerates all three writers.
    """
    from ankidkdeck.stages import s42_translate, s50_priority
    written = {"translate_usage.jsonl": 4, "review_usage_German.jsonl": 2,
               "priority_usage.jsonl": 1}
    for name, n in written.items():
        _write_usage(cfg, name, [_paid_call(label="%s-%d" % (name, i))
                                 for i in range(n)])
    led = billing.SpendLedger(cfg)
    out = led.ingest()
    assert out["absorbed"] == written
    assert led.summary()["rows"] == sum(written.values()) == 7

    # and the pattern list is the WRITERS, not a guess: every usage-jsonl
    # filename constructed anywhere in the package has to be matched by it
    import re
    from pathlib import Path
    names = set()
    for mod in (s42_translate, s50_priority):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        for raw in re.findall(r'"([a-z_]*usage[a-z_%s]*\.jsonl)"', text):
            names.add(raw.replace("%s", "German"))
    assert names, "the writers moved; this test has to be re-pointed"
    for name in names:
        assert any(Path(name).match(p) for p in billing.USAGE_GLOBS), name


def test_a_rotated_usage_file_is_never_silently_skipped(cfg):
    """Reviewer B's MAJOR-5, measured: 8 rows on disk, 5 in the ledger, zero
    warnings.

    The cursor decided "still the same file" from ONE line's sha. normalize_usage
    rows carried no timestamp and no sequence, so two identical retranslate
    requests -- and every failed call, whose counts are all zero -- produced
    byte-identical lines: a deleted-and-recreated file whose Nth line matched was
    resumed at the old offset and everything before it was lost. Under-counting
    is the direction that spends money nobody approved.

    Now the rows carry (ts, seq), a rotated file is re-read FROM THE TOP with the
    uids deciding what is new, and the rotation is an explicit event that G-BUDGET
    refuses to authorise a spend through.
    """
    _write_usage(cfg, "translate_usage.jsonl",
                 [_paid_call(label="w1-%d" % i) for i in range(3)])
    led = billing.SpendLedger(cfg)
    assert led.ingest()["absorbed"] == {"translate_usage.jsonl": 3}

    # The file is rotated away and rebuilt with FIVE further paid calls -- the
    # reviewer's repro: 8 calls really happened. It used to absorb 2 of the 5,
    # because line 3 of the new file was byte-identical to line 3 of the old one
    # and the cursor resumed there.
    (cfg.report_dir / "translate_usage.jsonl").unlink()
    _write_usage(cfg, "translate_usage.jsonl",
                 [_paid_call(label="w2-%d" % i) for i in range(5)])
    after = billing.SpendLedger(cfg)
    out = after.ingest()
    assert out["new_generations"] == ["translate_usage.jsonl#g2"]
    assert out["absorbed"] == {"translate_usage.jsonl#g2": 5}
    assert after.summary()["rows"] == 8, "8 calls on disk, 8 in the ledger"

    # ...and it is an EXPLICIT event, on disk and in the gate that authorises
    # the next spend
    assert len(out["anomalies"]) == 1
    assert out["anomalies"][0]["reconciled_by"] == "per-call uid"
    roll = json.loads((cfg.report_dir / billing.LEDGER_JSON)
                      .read_text(encoding="utf-8"))
    assert roll["ingest_anomalies"]
    ok, detail = gates.budget_has_room(1.0, 1.0, 10.0,
                                       ledger_anomalies=out["anomalies"])
    assert ok is False and "changed shape underneath it" in detail["why"]

    # A file whose rows carry NO (ts, seq) stamp cannot be reconciled at all: a
    # copied line and a new paid call are indistinguishable, so it is re-read in
    # full (over-counting refuses a wave that would have fit; under-counting
    # spends money nobody accepted) and the anomaly says so instead of implying
    # the total is trustworthy.
    _write_usage(cfg, "priority_usage.jsonl", [usage_row(kind="rank")] * 2)
    old = billing.SpendLedger(cfg)
    old.ingest()
    (cfg.report_dir / "priority_usage.jsonl").unlink()
    _write_usage(cfg, "priority_usage.jsonl", [usage_row(kind="rank")])
    out = billing.SpendLedger(cfg).ingest()
    stale = [a for a in out["anomalies"]
             if a["source"] == "priority_usage.jsonl"]
    assert stale and stale[0]["lines_without_a_call_uid"] == 1
    assert "NOTHING" in stale[0]["reconciled_by"]


def test_a_call_is_filed_in_the_period_it_was_paid_for(cfg):
    """The ledger used to file every INGESTED row on the day of the ingest, so a
    wave that ran over midnight -- or crashed and was absorbed next morning --
    read as zero in the month whose cap it actually consumed."""
    row = dict(_paid_call(label="july"), ts="2026-07-31T23:58:00.000000+00:00")
    _write_usage(cfg, "translate_usage.jsonl", [row])
    led = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 2))
    led.ingest()
    assert led.period_to_date(today=datetime.date(2026, 7, 31))["requests"] == 1
    assert led.period_to_date(today=datetime.date(2026, 8, 2))["requests"] == 0


def test_a_row_on_an_unpriced_model_is_visible_not_free(cfg):
    out = billing.rows_usd_priced([usage_row(model="gemini-2.0-flash"),
                                   usage_row()])
    assert out["rows"] == 2 and out["rows_priced"] == 1
    assert out["unpriced"][0]["model"] == "gemini-2.0-flash"


def test_an_unknown_cap_period_is_refused(cfg):
    cfg.spend_cap_period = "fortnight"
    with pytest.raises(FatalError):
        billing.SpendLedger(cfg).period_to_date()


# --------------------------------------------------------------------------
# the gates (spec 2.6 / 2.7)
# --------------------------------------------------------------------------

# The measured table from probe 8.3: on a wave whose cache hit 1.00 every time,
# `cached` is CONSTANT and `prompt` grows with the payload.
MEASURED_CACHE_ROWS = [(1, 1214), (5, 1255), (8, 1350), (12, 1580), (20, 1795)]
DECLARED = 1135


def test_g_cache_predicate():
    """The new criterion passes the measured wave; the audit's is pinned WRONG."""
    rows = [usage_row(label="n=%d" % n, n_expected=n, prompt_tokens=prompt,
                      cached_tokens=DECLARED, cache_name="caches/x")
            for n, prompt in MEASURED_CACHE_ROWS]
    ok, detail = gates.cache_hit_is_complete(rows, DECLARED)
    assert ok is True
    assert detail["rows_exactly_declared"] == 5
    assert detail["sum_cached_over_declared_x_requests"] == 1.0

    # The metric the audit proposed, computed on the SAME healthy rows. It
    # condemns three of the five. This assertion is the permanent record that
    # cached/prompt >= 0.90 is not a criterion.
    ratios = [r["cached_tokens"] / r["prompt_tokens"] for r in rows]
    assert [round(x, 3) for x in ratios] == [0.935, 0.904, 0.841, 0.718, 0.632]
    assert sum(1 for x in ratios if x < 0.90) == 3
    assert detail["cached_over_prompt_range"] == [0.6323, 0.9349]

    # a real miss (an expired cache: cached collapses to 0) fails
    broken = rows[:-1] + [usage_row(prompt_tokens=1795, cached_tokens=0,
                                    cache_name="caches/x")]
    bad_ok, bad_detail = gates.cache_hit_is_complete(broken, DECLARED)
    assert bad_ok is False and bad_detail["rows_exactly_declared"] == 4

    # no cache declared: n/a, and it SAYS n/a rather than passing quietly
    na_ok, na_detail = gates.cache_hit_is_complete([usage_row()], None)
    assert na_ok is True and na_detail["checked"] is False
    assert na_detail["verdict"] == "n/a"
    # ...unless a cache was configured, in which case an uncached wave is the
    # failure the gate exists for
    conf_ok, conf_detail = gates.cache_hit_is_complete([usage_row()], None,
                                                       cache_expected=True)
    assert conf_ok is False and "full uncached rate" in conf_detail["why"]


def test_g_cache_counts_the_whole_wave_not_just_the_hits():
    """Reviewer B's MAJOR-6, measured: share = 1.00 and PASS on a wave where 99
    of 100 requests paid full price.

    The denominator was len(rows carrying a cache name), which makes the share a
    ratio of the hits to themselves. The patch plan's formula is
    sum(cached) / (declared x REQUESTS), and the failure it exists to catch is
    exactly the partial one: an expired cache cannot be updated, only recreated,
    and a recreate changes the resource name -- so some jobs carry the new name
    and some fall back to an inlined prompt. G-CACHE is the load-bearing wall of
    the budget argument (an uncached program is $17.36 against a $10 cap), so
    this hole was the size of the whole argument.
    """
    hit = usage_row(label="hit", prompt_tokens=1350, cached_tokens=DECLARED,
                    cache_name="caches/x")
    missed = [usage_row(label="miss-%d" % i, prompt_tokens=1350,
                        cached_tokens=0) for i in range(99)]

    ok, detail = gates.cache_hit_is_complete([hit] + missed, DECLARED,
                                             cache_expected=True)
    assert ok is False, "1 cached + 99 uncached is not a healthy wave"
    assert detail["rows_checked"] == 100
    assert detail["requests_with_a_cache"] == 1
    assert detail["requests_without_a_cache"] == 99
    assert detail["sum_cached_over_declared_x_requests"] == 0.01

    # the denominator that produced the false PASS, for the record
    assert round(DECLARED / float(DECLARED * 1), 4) == 1.0

    # and a healthy MIXED wave still passes: the expression prompt is 336 tokens
    # against a measured 1,024-token floor, so expression rows are uncached BY
    # DESIGN and cannot be in the denominator.
    mixed = [usage_row(label="d%d" % n, n_expected=n, prompt_tokens=p,
                       cached_tokens=DECLARED, cache_name="caches/x")
             for n, p in MEASURED_CACHE_ROWS] \
        + [usage_row(label="e%d" % i, kind="expression", prompt_tokens=700,
                     cached_tokens=0) for i in range(20)]
    ok, detail = gates.cache_hit_is_complete(mixed, DECLARED,
                                             cache_expected=True)
    assert ok is True
    assert detail["rows_checked"] == 5 and detail["requests"] == 25
    assert detail["cached_kinds"] == ["definition"]


# ---- G-CACHE under the batch transport's recovery paths -------------------
#
# CROSS-OWNER TESTS (made for the batch transport; the gate lives in this file). The denominator change these pin
# shipped without one test of its own: `grep rows_that_never_executed tests/`
# found nothing, and the two tests above only proved the change had not BROKEN
# them. Both directions are asserted here, plus the two holes the acceptance
# reviewers measured.

def test_g_cache_does_not_count_a_row_that_never_executed():
    """The denominator change, pinned in the direction it was made for.

    On the batch surface a per-row failure is a bare gRPC status at prompt=0,
    billed $0 (probe W3-4: a cache deleted after a successful submit failed all
    21 rows that way), and the wave RETRIES it. Counting the failed attempt in
    the denominator scored one error plus one successful retry at
    1135/(1135 x 2) = 0.5 against a 0.95 threshold and condemned a healthy wave.
    """
    dead = usage_row(label="attempt 1", prompt_tokens=0, cached_tokens=0,
                     cache_name="caches/x", error="batch_row_error")
    retried = usage_row(label="attempt 2", prompt_tokens=1350,
                        cached_tokens=DECLARED, cache_name="caches/x")
    ok, detail = gates.cache_hit_is_complete([dead, retried], DECLARED,
                                             cache_expected=True)
    assert ok is True
    assert detail["rows_that_never_executed"] == 1
    assert detail["rows_checked"] == 1
    assert detail["sum_cached_over_declared_x_requests"] == 1.0

    # the arithmetic the OLD denominator produced, for the record
    assert round(DECLARED / float(DECLARED * 2), 4) == 0.5

    # and a whole wave of them is STILL a refusal: those rows are of a cacheable
    # kind, so the wave declared something to cache and got nothing back
    allof = [usage_row(label="r%d" % i, prompt_tokens=0, cached_tokens=0,
                       cache_name="caches/x", error="batch_row_error")
             for i in range(21)]
    ok, detail = gates.cache_hit_is_complete(allof, DECLARED,
                                             cache_expected=True)
    assert ok is False
    assert detail["rows_that_never_executed"] == 21
    assert detail["rows_of_a_cacheable_kind"] == 21


def test_g_cache_counts_a_billed_row_that_also_reported_an_error():
    """The exclusion is `prompt_tokens > 0`, and nothing else.

    It used to be `prompt_tokens > 0 and not error`, which is wider than the
    reason needs and unsafe in the one direction that matters: a row that was
    BILLED, used no cache and also carried an error -- a truncation, an
    unparseable body -- did miss the cache, and dropping it from the denominator
    hides exactly the full-price row this gate looks for. Measured by an
    acceptance reviewer: twenty such rows left rows_checked at 1 and the gate
    PASSED.
    """
    hit = usage_row(label="hit", prompt_tokens=1350, cached_tokens=DECLARED,
                    cache_name="caches/x")
    billed_and_broken = [usage_row(label="miss-%d" % i, prompt_tokens=2485,
                                   cached_tokens=0, error="batch_row_error")
                         for i in range(20)]
    ok, detail = gates.cache_hit_is_complete([hit] + billed_and_broken,
                                             DECLARED, cache_expected=True)
    assert ok is False, "20 billed uncached rows are not a healthy wave"
    assert detail["rows_checked"] == 21
    assert detail["requests_without_a_cache"] == 20


def test_g_cache_passes_a_wave_with_nothing_cacheable_in_it():
    """cache_expected is scoped to waves that DECLARED a cacheable request.

    The expression prompt is ~336 tokens against a measured 1,024-token floor,
    so it is uncached BY DESIGN -- which means an incremental wave with only
    expression cells left has nothing to cache. Failing it made the
    documented recovery path unreachable: "the unrecovered cells stay missing and
    the next run picks them up" was followed by a run with no definition rows,
    which died on this gate before it could write anything.
    """
    exprs = [usage_row(label="e%d" % i, kind="expression", prompt_tokens=700,
                       cached_tokens=0) for i in range(12)]
    ok, detail = gates.cache_hit_is_complete(exprs, None, cache_expected=True)
    assert ok is True
    assert detail["rows_of_a_cacheable_kind"] == 0
    assert detail["verdict"] == "n/a: no request of a cacheable kind in this wave"

    # ONE definition row is enough to bring the criterion back
    with_one = exprs + [usage_row(label="d", prompt_tokens=1350,
                                  cached_tokens=0)]
    ok, detail = gates.cache_hit_is_complete(with_one, None,
                                             cache_expected=True)
    assert ok is False
    assert detail["rows_of_a_cacheable_kind"] == 1


def test_g_cache_still_fails_the_original_failure_at_every_real_denominator():
    """The denominator is now this LANGUAGE's own declaration.

    So the values that can reach the gate are the measured prompt sizes -- and at
    every one of them the failure the gate was rewritten for still fails. The
    values that made it PASS (a stale declaration of 5, pulled in by a min()
    across every job the workspace ever ran) are no longer reachable, and the
    direction is the whole point: a smaller divisor RAISES the share, so the
    unsafe error was the one that used to be the normal path.
    """
    hit = usage_row(label="hit", prompt_tokens=1350, cached_tokens=DECLARED,
                    cache_name="caches/x")
    missed = [usage_row(label="miss-%d" % i, prompt_tokens=1350,
                        cached_tokens=0) for i in range(99)]
    for declared in (1135, 1092, 500, 100):
        ok, detail = gates.cache_hit_is_complete([hit] + missed, declared,
                                                 cache_expected=True)
        assert ok is False, declared
        assert detail["sum_cached_over_declared_x_requests"] < 0.15, declared
    # the stale-denominator PASS, kept as the record of what was wrong
    ok, _ = gates.cache_hit_is_complete([hit] + missed, 5, cache_expected=True)
    assert ok is True, ("a denominator of 5 passes the broken wave -- which is "
                        "why the transport may no longer take min() across "
                        "languages and across history")


def test_g_think_predicate():
    """A measured zero passes; a MEDIUM-scale row fails.

    And the finding that made this gate three-way: at the SAME thinkingLevel=LOW
    the ranking prompt produced 236-275 thought tokens (field present,
    finishReason STOP) while the definition prompt produced 0 in 62 observations.
    A gate that failed on any non-zero row would fail every healthy ranking
    wave -- the same mistake as cached/prompt.
    """
    zeros = [usage_row(thinking_tokens=0) for _ in range(5)]
    ok, detail = gates.thinking_is_at_the_measured_level(zeros, "LOW", 0.0)
    assert ok is True and detail["distinct_thinking_values"] == [0]

    # accidentally MEDIUM on the measured kind
    medium = zeros + [usage_row(label="oops", thinking_tokens=578)]
    ok, detail = gates.thinking_is_at_the_measured_level(medium, "LOW", 0.0)
    assert ok is False
    assert detail["violations"][0]["thinking_tokens"] == 578

    # the ranking prompt at LOW: recorded, warned about, NOT failed
    rank = zeros + [usage_row(label="rank", kind="rank", thinking_tokens=275)]
    ok, detail = gates.thinking_is_at_the_measured_level(
        rank, "LOW", 0.0, strict_kinds=("definition",), alarm_at=578.7)
    assert ok is True
    assert detail["warnings"][0]["thinking_tokens"] == 275
    assert detail["warning_note"]

    # but the same unmeasured kind reaching the MEDIUM band still fails
    ok, _ = gates.thinking_is_at_the_measured_level(
        zeros + [usage_row(kind="rank", thinking_tokens=900)], "LOW", 0.0,
        strict_kinds=("definition",), alarm_at=578.7)
    assert ok is False


def test_g_bill_predicate():
    ok, detail = gates.bill_within_ceiling(2.0, 2.15, 1.10)
    assert ok is True and detail["allowed_usd"] == 2.2
    ok, detail = gates.bill_within_ceiling(2.0, 2.4, 1.10)
    assert ok is False and detail["overrun_usd"] == pytest.approx(0.2)
    # never quoted -> never approved
    ok, detail = gates.bill_within_ceiling(None, 0.01, 1.10)
    assert ok is False and "never quoted" in detail["why"]


def test_g_prompt_predicate():
    shas = {"definition": "d" * 64, "expression": "e" * 64}
    good = [usage_row(prompt_sha256="d" * 64),
            usage_row(kind="expression", prompt_sha256="e" * 64)]
    ok, _ = gates.one_prompt_per_wave(good, shas, "v4-frozen")
    assert ok is True
    # a second prompt text on the wire
    bad = good + [usage_row(label="x", prompt_sha256="f" * 64)]
    ok, detail = gates.one_prompt_per_wave(bad, shas, "v4-frozen")
    assert ok is False and detail["rows_with_a_different_sha"]
    # a pack version the bill did not quote
    ok, detail = gates.one_prompt_per_wave(
        [usage_row(prompt_id="rich-v1", prompt_sha256="d" * 64)], shas,
        "v4-frozen")
    assert ok is False and detail["prompt_id_consistent"] is False
    # A row with no sha of its own, on a kind the bill DID quote a sha for, is
    # not a pass: this is the cached path, and it used to report ok=True with
    # nothing checked and no field saying so. rows_checked is always reported.
    ok, detail = gates.one_prompt_per_wave([usage_row(prompt_sha256=None)],
                                           shas, "v4-frozen")
    assert ok is False and detail["rows_checked"] == 0
    assert "checked nothing" in detail["why"]
    # a call that legitimately has no system prompt AND whose kind the bill
    # quotes no sha for (the manual review call) is still not a violation
    ok, detail = gates.one_prompt_per_wave(
        [usage_row(kind="review", prompt_sha256=None)], shas, "v4-frozen")
    assert ok is True and detail["rows_checked"] == 0
    assert detail["rows_with_no_system_prompt"] == 1
    assert detail["rows_whose_cached_prompt_is_unverified_count"] == 0


def test_g_prompt_checks_the_cache_when_the_prompt_lives_in_one(cfg,
                                                                probe_stats):
    """A-M-2. On the CACHED path -- the only path with a discount, i.e. the one
    this program is going to run -- systemInstruction and cachedContent are
    mutually exclusive (hard 400), so every row's prompt_sha256 is None. G-PROMPT
    used to `continue` past all of them and report

        G-PROMPT ok=True {"requests": 2, "rows_with_a_different_sha": [], ...}

    with no field anywhere saying it had compared nothing. Two fixes: the cache's
    own recorded prompt sha is the thing compared, and rows_checked is always on
    the record.
    """
    shas = {"definition": "d" * 64}
    cached = [usage_row(label="a", prompt_sha256=None, cache_name="caches/x"),
              usage_row(label="b", prompt_sha256=None, cache_name="caches/x")]

    # 1. nothing recorded about the cache: the wave is UNVERIFIED, not healthy
    ok, detail = gates.one_prompt_per_wave(cached, shas, "v4-frozen")
    assert ok is False
    assert detail["rows_checked"] == 0
    assert detail["rows_whose_cached_prompt_is_unverified_count"] == 2
    assert "cache_prompt_sha256" in detail["why"]

    # 2. the transport stamps the cache's prompt sha on each row (preferred: it
    #    is per row, so a wave that used two cache objects stays checkable)
    stamped = [dict(r, cache_prompt_sha256="d" * 64) for r in cached]
    ok, detail = gates.one_prompt_per_wave(stamped, shas, "v4-frozen")
    assert ok is True
    assert detail["rows_checked"] == 2
    assert detail["rows_checked_via"]["cache_row_sha"] == 2

    # 3. or hands the gate a {cache_name: sha} map
    ok, detail = gates.one_prompt_per_wave(cached, shas, "v4-frozen",
                                           {"caches/x": "d" * 64})
    assert ok is True and detail["rows_checked_via"]["cache_name_map"] == 2

    # 4. and a cache holding the WRONG prompt is caught, which is the whole
    #    point: this is the only reading that can catch it at all
    ok, detail = gates.one_prompt_per_wave(cached, shas, "v4-frozen",
                                           {"caches/x": "f" * 64})
    assert ok is False and detail["rows_with_a_different_sha"][0][
        "checked_via"] == "cache_name_map"

    # 5. end to end through the builder, the way a stage calls it
    write_json(cfg.report_dir / "translate_bill_German.json",
               {"language": "German", "prompt_id": "v4-frozen",
                "prompt_sha256": {"definition": "d" * 64}})
    thin = {"German": {"dollars": {"lean_uncached": 99.0}}}
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, thin, cached, lang="German",
                                   stats=probe_stats)}
    assert built["G-PROMPT"][0] is False
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, thin, cached, lang="German",
                                   cache_prompt_shas={"caches/x": "d" * 64},
                                   stats=probe_stats)}
    assert built["G-PROMPT"][0] is True


def test_g_budget_refuses_a_run_that_would_break_the_cap():
    ok, detail = gates.budget_has_room(3.0, 4.0, 10.0)
    assert ok is True and detail["headroom_usd"] == 3.0
    ok, detail = gates.budget_has_room(8.0, 4.0, 10.0)
    assert ok is False and "12.0000 > the 10.00 cap" in detail["why"]
    # an unpriceable forecast is a refusal, not a cheap run
    ok, detail = gates.budget_has_room(0.0, None, 10.0)
    assert ok is False and "cannot be checked" in detail["why"]
    ok, _ = gates.budget_has_room(0.0, 1.0, None)
    assert ok is False


def test_g_scope_frozen_refuses_without_a_stamp(cfg, not_refrozen):
    """Spec 2.7. No refreeze signature, no spending -- and the signature is a
    signature: nothing in this package writes it."""
    stamp, where = gates.read_refreeze_stamp(cfg)
    assert stamp is None, "the refreeze has not happened; nothing may forge it"
    ok, detail = gates.scope_is_frozen(stamp, 2909, {"1": {}}, where)
    assert ok is False and "no refreeze stamp" in detail["why"]

    card_keys = {str(i): {"guid_seed": "w%d" % i} for i in range(2909)}
    good = {"refrozen_at": "2026-09-01", "families": 2909,
            "card_keys_rows": 2909, "by": "Binghan"}
    ok, _ = gates.scope_is_frozen(good, 2909, card_keys, "test")
    assert ok is True
    # the scope moved after the signature
    ok, detail = gates.scope_is_frozen(good, 2910, card_keys, "test")
    assert ok is False and "signed for 2909 families" in detail["violations"][0]
    # the registry moved after the signature
    ok, detail = gates.scope_is_frozen(good, 2909, dict(list(card_keys.items())
                                                        [:2900]), "test")
    assert ok is False and "moved after the signature" in detail["violations"][0]
    # a stamp with no date is not a signature
    ok, detail = gates.scope_is_frozen({"families": 2909}, 2909, card_keys, "x")
    assert ok is False

    # and the local registry copy is what a run host would carry
    write_json(cfg.registry_local / gates.REFREEZE_STAMP, good)
    stamp, where = gates.read_refreeze_stamp(cfg)
    assert stamp == good and gates.REFREEZE_STAMP in where


def test_the_money_gates_are_declared_and_reported(cfg):
    """A gate id that is not in ALL_GATE_IDS is invisible in the report's
    "never run" accounting, which is where a missing gate hides."""
    assert set(gates.MONEY_GATE_IDS) <= set(gates.ALL_GATE_IDS)
    assert len(gates.ALL_GATE_IDS) == len(set(gates.ALL_GATE_IDS))
    for gid in ("G-BILL", "G-THINK", "G-PROMPT", "G-CACHE", "G-BUDGET",
                "G-SCOPE-FROZEN"):
        assert gid in gates.ALL_GATE_IDS

    from ankidkdeck.registry import Registry
    policy = Registry(cfg).gates
    assert policy["bill_tolerance_factor"] == 1.1
    assert policy["cache_hit_min_share"] == 0.95
    assert "cached/prompt" in policy["_note_money_gates"]


def test_the_registry_policy_numbers_are_not_dead_data(cfg, probe_stats):
    """Reviewer A's M-3 / reviewer B's MINOR-11: registry/gates.json declares
    itself a human sign-off point for the two money policy numbers, and NOTHING
    read them. No production caller passed a policy dict, and the values that
    decided G-BILL and G-CACHE were the module constants. A review gate that
    cannot change behaviour is worse than none: it manufactures the belief that
    the numbers were reviewed.

    The check is a DEAD-DATA check, not a presence check: change the number in
    the registry and the verdict has to change.
    """
    rows = [usage_row(label="a", prompt_sha256="d" * 64)]
    actual = billing.rows_usd_priced(rows)["usd"]
    # quoted just low enough that the DEFAULT 1.10 tolerance refuses it
    quote = actual / 1.2
    bill = {"German": {"prompt_id": "v4-frozen",
                       "prompt_sha256": {"definition": "d" * 64},
                       "dollars": {"lean_uncached": quote}}}

    def verdict():
        built = {g.id: g.fn() for g in
                 gates.post_wave_gates(cfg, bill, rows, lang="German",
                                       stats=probe_stats)}
        return built["G-BILL"]

    ok, detail = verdict()
    assert ok is False and detail["tolerance_factor"] == 1.1

    # a human widens the tolerance in the file they signed -- and it takes effect
    write_json(cfg.registry_local / "gates.json",
               {"bill_tolerance_factor": 1.5})
    ok, detail = verdict()
    assert ok is True and detail["tolerance_factor"] == 1.5

    # ...and the same for the cache share
    write_json(cfg.registry_local / "gates.json",
               {"cache_hit_min_share": 0.5})
    rows2 = [usage_row(label="c%d" % i, prompt_tokens=1350,
                       cached_tokens=1135 if i < 3 else 0,
                       cache_name="caches/x" if i < 3 else None)
             for i in range(5)]
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, bill, rows2, lang="German",
                                   declared_cache_tokens=1135,
                                   stats=probe_stats)}
    assert built["G-CACHE"][1]["min_share"] == 0.5
    assert built["G-CACHE"][0] is True          # 3/5 = 0.60 clears 0.50
    write_json(cfg.registry_local / "gates.json",
               {"cache_hit_min_share": 0.95})
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, bill, rows2, lang="German",
                                   declared_cache_tokens=1135,
                                   stats=probe_stats)}
    assert built["G-CACHE"][0] is False         # 0.60 does not clear 0.95


def test_pre_spend_gates_refuse_a_run_nobody_signed_for(cfg, probe_stats):
    """The two gates that stand between --confirm-spend and the first call, on a
    workspace in exactly the state this patch leaves it in: no refreeze stamp,
    an empty ledger, and a bill that has been priced."""
    bill = {"German": {"dollars": {"cache_works": 2.0, "lean_uncached": 4.0,
                                   "rich_uncached": 9.0}}}
    built = gates.pre_spend_gates(cfg, bill, families=2909)
    assert [g.id for g in built] == ["G-SCOPE-FROZEN", "G-BUDGET"]
    verdicts = {g.id: g.fn() for g in built}
    assert verdicts["G-SCOPE-FROZEN"][0] is False      # no stamp yet
    assert verdicts["G-BUDGET"][0] is True             # 0 spent + 4.00 < 10
    assert verdicts["G-BUDGET"][1]["forecast_usd"] == 4.0

    # ...and run_gates turns that into a refusal, with the report on disk
    with pytest.raises(FatalError) as exc:
        gates.run_gates(built, cfg, stage="42")
    assert "G-SCOPE-FROZEN" in str(exc.value)
    report = json.loads((cfg.report_dir / "gates_report.json")
                        .read_text(encoding="utf-8"))
    assert "G-SCOPE-FROZEN" in report["failed"]

    # a month already at the cap refuses even with a stamp in place
    _write_usage(cfg, "translate_usage.jsonl",
                 [usage_row(prompt_tokens=10 ** 7, candidates_tokens=10 ** 7)])
    again = gates.pre_spend_gates(cfg, bill, families=2909)
    budget = [g for g in again if g.id == "G-BUDGET"][0].fn()
    assert budget[0] is False
    assert budget[1]["spent_usd"] > 0


def test_post_wave_gates_adjudicate_a_wave_that_was_paid_for(cfg, probe_stats):
    rows = [usage_row(label="a", prompt_sha256="d" * 64),
            usage_row(label="b", prompt_sha256="d" * 64)]
    priced = billing.rows_usd_priced(rows)
    bill = {"German": {"prompt_id": "v4-frozen",
                       "prompt_sha256": {"definition": "d" * 64},
                       "dollars": {"cache_works": None,
                                   "lean_uncached": priced["usd"],
                                   "rich_uncached": 1.0}}}
    built = gates.post_wave_gates(cfg, bill, rows, lang="German",
                                  stats=probe_stats)
    assert [g.id for g in built] == ["G-BILL", "G-THINK", "G-PROMPT", "G-CACHE"]
    results = {g.id: g.fn() for g in built}
    assert all(ok for ok, _ in results.values()), results
    assert results["G-CACHE"][1]["verdict"] == "n/a"
    # the MEDIUM alarm band was read off the artifact, not hard-coded
    assert results["G-THINK"][1]["medium_band_alarm_at"] == 578.7
    assert results["G-THINK"][1]["kinds_held_to_the_measured_zero"] \
        == ["definition"]


# --------------------------------------------------------------------------
# N-09: the consumption rules
# --------------------------------------------------------------------------

def rules_by_id(rows) -> dict:
    return {r["rule"]: r for r in rows}


def measured_artifact(probe_stats: dict) -> dict:
    """The conftest fixture plus the three fields the N-09 backfill wrote onto
    the real artifact: which prompt PACK it measured, how many characters that
    prompt was, and the measured expiry behaviour that licenses a 1.5x TTL.
    Values from work/probes/stats.json after tools/backfill_probe_stats.py."""
    stats = dict(probe_stats)
    stats["prompt_id"] = "v4-frozen"
    stats["prompt_lineage"] = {"prompt_id": "v4-frozen",
                               "measured_prompt_chars": [5123, 5124]}
    stats["wave2"] = dict(stats["wave2"],
                          CACHE_EXPIRY_BEHAVIOUR="LOUD_ERROR_403")
    # The prompt body is English whatever the target language is, so English is
    # the ratio that sizes an English system prompt.
    stats["CHARS_PER_TOKEN"] = {"Chinese": 4.314, "English": 4.322,
                                "German": 4.295, "Spanish": 4.314}
    return stats


def test_gate_provenance(cfg, probe_stats):
    """N-09 rule 1 and rule 6: the artifact has to be the one that measured the
    thing being spent on, or the spend is refused (not warned about)."""
    from ankidkdeck.stages.s42_translate import definition_prompt
    stats = measured_artifact(probe_stats)
    prompts = {"definition": definition_prompt("German")}

    rows = rules_by_id(billing.consumption_rules(cfg, stats, prompts=prompts))
    assert all(r["ok"] for r in rows.values()), rows
    billing.assert_ready_to_spend(cfg, stats, prompts=prompts)

    # 1. a missing constant
    thin = {k: v for k, v in stats.items() if k != "EXPECTED_OUTPUT"}
    assert rules_by_id(billing.consumption_rules(
        cfg, thin, prompts=prompts))["R1-constants"]["ok"] is False

    # 2. measured on another model
    other = dict(stats, model="gemini-2.0-flash")
    assert rules_by_id(billing.consumption_rules(
        cfg, other, prompts=prompts))["R1-model"]["ok"] is False

    # 3. measured before the model existed
    old = dict(stats, measured_at="2026-08-12T23:59+02:00")
    row = rules_by_id(billing.consumption_rules(cfg, old,
                                                prompts=prompts))["R2-measured-at"]
    assert row["ok"] is False
    assert row["detail"]["not_before"] == billing.CONSTANTS_NOT_BEFORE
    with pytest.raises(FatalError) as exc:
        billing.assert_ready_to_spend(cfg, old, prompts=prompts)
    assert "R2-measured-at" in str(exc.value)

    # 4. an artifact that does not say which prompt pack it measured
    nameless = {k: v for k, v in stats.items() if k != "prompt_id"}
    assert rules_by_id(billing.consumption_rules(
        cfg, nameless, prompts=prompts))["R6-prompt-id"]["ok"] is False

    # 5. a pack version the constants were not measured on (this is what will
    #    happen the day the RICH pack lands, and it is meant to)
    cfg.prompt_id = "rich-core-v1"
    assert rules_by_id(billing.consumption_rules(
        cfg, stats, prompts=prompts))["R6-prompt-id"]["ok"] is False
    cfg.prompt_id = "v4-frozen"


def test_the_consumption_guard_actually_stops_a_spend(cfg, probe_stats):
    """Reviewer A's m-6. tools/backfill_probe_stats.py sets CONSUMPTION_GUARD
    when the artifact is not fit to authorise a spend, and its docstring said
    "doctor and the spend gate both refuse while it is present". doctor did. The
    spend path did not read the key at all -- not probe_stats(), not
    consumption_rules() -- so the one switch built to stop a spend could not.
    """
    from ankidkdeck.stages.s42_translate import definition_prompt
    stats = measured_artifact(probe_stats)
    prompts = {"definition": definition_prompt("German")}
    assert rules_by_id(billing.consumption_rules(
        cfg, stats, prompts=prompts))["R1-guard"]["ok"] is True

    guarded = dict(stats, CONSUMPTION_GUARD="the basis was relabelled by hand")
    row = rules_by_id(billing.consumption_rules(
        cfg, guarded, prompts=prompts))["R1-guard"]
    assert row["ok"] is False and row["blocking"] is True
    with pytest.raises(FatalError) as exc:
        billing.assert_ready_to_spend(cfg, guarded, prompts=prompts)
    assert "R1-guard" in str(exc.value)


def test_the_size_band_basis_is_scoped_to_a_prompt_family(cfg, probe_stats):
    """Reviewer B's MAJOR-9. The basis was max() over EVERY prompt size in the
    ledger. The patch plan's 4.4 A/B measures LEAN and RICH on the same model, so
    after that run the flat maximum is the RICH size -- and LEAN, which 4.4 calls
    a FREE rollback because LEAN is a pure prefix of RICH, drifts 57% from its own
    basis and gets refused. A gate that makes the documented rollback impossible
    is a gate that gets switched off.
    """
    from ankidkdeck.stages.s42_translate import definition_prompt
    lean = definition_prompt("German")
    stats = measured_artifact(probe_stats)
    # the ledger AFTER the 4.4 A/B: it holds both prompts
    rich_chars = int(len(lean) * 2.33)
    stats["prompt_lineage"] = {
        "prompt_id": "v4-frozen",
        "measured_prompt_chars": [1503, 5123, 5124, rich_chars],
        "size_band_basis": {
            "prompt_id": "v4-frozen",
            "by_family": {"definition": [5123, 5124, rich_chars],
                          "other": [1503]}},
    }
    row = rules_by_id(billing.consumption_rules(
        cfg, stats, prompts={"definition": lean}))["R6-prompt-size"]
    item = row["detail"]["prompts"][0]
    assert row["ok"] is True, "the free LEAN rollback has to stay possible"
    assert item["measured_chars"] == 5124          # NEAREST, not max()
    assert item["basis_scope"] == "prompt family 'definition'"

    # the flat max() basis, on the same artifact, is what used to happen
    flat = dict(stats)
    flat["prompt_lineage"] = {k: v for k, v
                              in stats["prompt_lineage"].items()
                              if k != "size_band_basis"}
    row = rules_by_id(billing.consumption_rules(
        cfg, flat, prompts={"definition": lean}))["R6-prompt-size"]
    assert row["ok"] is False
    assert row["detail"]["prompts"][0]["measured_chars"] == rich_chars

    # and the ranking prompt's 1,503 characters are not a definition size: a
    # family-scoped basis never compares the two
    assert 1503 not in stats["prompt_lineage"]["size_band_basis"][
        "by_family"]["definition"]


def test_the_backfill_tool_will_not_relabel_a_measurement_by_itself():
    """Reviewer B's MAJOR-8. --declare-prompt-id rewrote prompt_id to any string
    with exit 0 and no warning, while measured_prompt_chars and the thinking
    constants stayed the LEAN ones -- so the stronger half of consumption rule 6
    ("measured on LEAN, spent on RICH") could be switched off with one argument.

    Offline: the tool is driven over a synthetic ledger, no probe artifact needed.
    """
    import importlib.util
    import json as _json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools" / "backfill_probe_stats.py"
    spec = importlib.util.spec_from_file_location("backfill_probe_stats", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    raw = {"prompt_sha256_distinct": ["a" * 64], "prompt_chars_seen": [5123],
           "calls_with_usage": 60, "prompt_chars_by_family":
               {"definition": [5123]}}
    raw["ledger_fingerprint"] = "fingerprint-1"
    stats = {"prompt_id": "v4-frozen",
             "prompt_lineage": {"prompt_id": "v4-frozen",
                                "size_band_basis": {
                                    "prompt_id": "v4-frozen",
                                    "ledger_fingerprint": "fingerprint-1"}}}

    # re-declaring the SAME pack is not a relabel
    assert tool.relabel_refusal(stats, raw, "v4-frozen", False) is None
    # a different pack, with a ledger that has not moved, is refused
    refusal = tool.relabel_refusal(stats, raw, "v5-rich", False)
    assert refusal is not None
    assert refusal["ledger_state"].startswith("UNCHANGED")
    assert "consumption rule 6 exists to prevent" in refusal["why"]
    # ...and the escape hatch is explicit, named and recorded
    assert tool.relabel_refusal(stats, raw, "v5-rich", True) is None
    lineage = tool.prompt_lineage(stats, raw, "v5-rich",
                                  rebased_from="v4-frozen")
    assert lineage["rebased_from"] == "v4-frozen"
    assert lineage["size_band_basis"]["prompt_id"] == "v5-rich"
    assert lineage["size_band_basis"]["by_family"] == {"definition": [5123]}
    _json.dumps(lineage)          # it has to be serialisable into the artifact

    # the causality note is stated once, and it is the right way round
    assert "16 minutes LATER" in tool.FLOOR_NOTE
    assert "overwritten by a later" not in tool.FLOOR_NOTE


def test_a_prompt_that_grew_four_times_may_not_use_the_old_constants(cfg,
                                                                    probe_stats):
    """Rule 6 without a sha check, because a sha check can never pass.

    The probes sent SEVEN different system prompts (one per batch size, 1-2
    characters apart); item 1.7 replaced that with one constant prompt, so the
    sha that will be sent is not among them and never can be. Size is the
    offline proxy that still works: it accepts the constantisation edit and
    refuses an enrichment.
    """
    from ankidkdeck.stages.s42_translate import definition_prompt
    stats = measured_artifact(probe_stats)
    lean = definition_prompt("German")
    row = rules_by_id(billing.consumption_rules(
        cfg, stats, prompts={"definition": lean}))["R6-prompt-size"]
    assert row["ok"] is True
    item = row["detail"]["prompts"][0]
    # the character basis: 5,134 characters now against 5,124 measured
    assert item["basis"] == "characters"
    assert item["drift"] < 0.01

    # with no record of what the probes sent, the coarser token basis via the
    # measured chars/token ratio -- and without THAT, no verdict at all
    tokens_only = {k: v for k, v in stats.items() if k != "prompt_lineage"}
    tokens_only["CHARS_PER_TOKEN"] = {"English": 4.322}
    item = rules_by_id(billing.consumption_rules(
        cfg, tokens_only,
        prompts={"definition": lean}))["R6-prompt-size"]["detail"]["prompts"][0]
    assert item["basis"] == "tokens" and item["drift"] < 0.10
    blind = {k: v for k, v in tokens_only.items() if k != "CHARS_PER_TOKEN"}
    row = rules_by_id(billing.consumption_rules(
        cfg, blind, prompts={"definition": lean}))["R6-prompt-size"]
    assert row["ok"] is False
    assert "CHARS_PER_TOKEN" in row["detail"]["prompts"][0]["why"]

    rich = lean + ("\nENRICHMENT BLOCK. " * 400)
    row = rules_by_id(billing.consumption_rules(
        cfg, stats, prompts={"definition": rich}))["R6-prompt-size"]
    assert row["ok"] is False
    # and with no prompt at all, nothing was compared -- which is a refusal
    row = rules_by_id(billing.consumption_rules(
        cfg, stats))["R6-prompt-size"]
    assert row["ok"] is False


def test_a_configured_cache_under_the_measured_floor_is_refused(cfg,
                                                                probe_stats):
    stats = measured_artifact(probe_stats)
    cfg.cache_enabled = True
    row = rules_by_id(billing.consumption_rules(
        cfg, stats, prompts={"definition": "tiny"}))["R2-cache-floor"]
    assert row["ok"] is False
    assert row["detail"]["explicit_cache_floor"] == 1024
    # the real prompt is over the floor, which is why no enrichment is needed
    from ankidkdeck.stages.s42_translate import definition_prompt
    row = rules_by_id(billing.consumption_rules(
        cfg, stats,
        prompts={"definition": definition_prompt("German")}))["R2-cache-floor"]
    assert row["ok"] is True


def test_the_ttl_margin_is_licensed_by_the_measured_expiry_behaviour(cfg,
                                                                     probe_stats):
    """Rule 4: 1.5x is allowed BECAUSE expiry was measured loud and free. Take
    that measurement away and the margin has to go back to 3x."""
    stats = measured_artifact(probe_stats)
    cfg.cache_enabled = True
    cfg.cache_ttl_factor = 1.5
    assert rules_by_id(billing.consumption_rules(
        cfg, stats))["R4-ttl-margin"]["ok"] is True
    silent = dict(stats)
    silent["wave2"] = {k: v for k, v in stats["wave2"].items()
                       if k != "CACHE_EXPIRY_BEHAVIOUR"}
    row = rules_by_id(billing.consumption_rules(cfg, silent))["R4-ttl-margin"]
    assert row["ok"] is False and row["detail"]["required"] == 3.0


def test_the_bill_uses_a_p95_and_the_artifact_has_to_carry_one(cfg,
                                                               probe_stats):
    """Rule 3. A ceiling half the requests are above is not a ceiling."""
    stats = measured_artifact(probe_stats)
    assert rules_by_id(billing.consumption_rules(
        cfg, stats))["R3-p95"]["ok"] is True
    mean_only = dict(stats)
    mean_only["thinking"] = {"THINKING_PER_REQUEST_LOW": {"mean": 0}}
    assert rules_by_id(billing.consumption_rules(
        cfg, mean_only))["R3-p95"]["ok"] is False


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

MEASURED_LITERALS = {35.964, 23.07, 23.917, 1164.2, 1135, 1092, 4564, 4779,
                     578.7, 1042.0, 1156, 4.314, 4.322, 4.295, 0.985, 0.632}


def test_the_money_stack_hard_codes_no_measured_constant():
    """The rule that makes a re-measurement reach the code: measured values live
    on disk. Prices are NOT measured values -- they are read off a page on a
    date, which is why prices.py may hold numbers and billing.py may not.

    Numeric LITERALS only, so the prose that explains a measurement (and must
    quote it) is untouched.
    """
    import ankidkdeck.billing as B
    import ankidkdeck.gates as G
    for module in (B, G):
        tree = ast.parse(open(module.__file__, encoding="utf-8").read())
        found = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant)
                 and isinstance(node.value, (int, float))
                 and not isinstance(node.value, bool)}
        assert not (found & MEASURED_LITERALS), (module.__name__,
                                                 found & MEASURED_LITERALS)


def test_the_money_path_imports_no_llm_module():
    """The bill-only path is offline by contract, and the money stack is on it.

    Checked in a SUBPROCESS, not in this one: the suite installs a fake
    google.genai into sys.modules, so an in-process check would pass or skip
    depending on which tests ran before it -- i.e. it would not be a check.
    """
    import subprocess
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = ("import sys; sys.path.insert(0, %r);"
            "import ankidkdeck.prices, ankidkdeck.billing, ankidkdeck.gates;"
            "bad=[n for n in sys.modules if n.split('.')[0]=='google'];"
            "print(bad); sys.exit(1 if bad else 0)" % src)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_required_constants_are_one_list():
    """REQUIRED_STATS_KEYS is defined elsewhere; N-09 reuses it rather than writing a
    second list that can drift out of step."""
    import importlib.util
    from pathlib import Path

    from ankidkdeck.stages.s42_translate import REQUIRED_STATS_KEYS
    path = Path(__file__).resolve().parents[1] / "tools" / "backfill_probe_stats.py"
    spec = importlib.util.spec_from_file_location("backfill_probe_stats", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    CONSUMED = tool.CONSUMED

    required = {k for k, _ in REQUIRED_STATS_KEYS}
    # every required key is either named in the tool's provenance table or is a
    # component of a fit that is (EXPECTED_OUTPUT.a/.b)
    for key in required:
        assert key in CONSUMED or key.split(".")[0] in {
            c.split(".")[0] for c in CONSUMED}, key


def test_g_prompt_reads_the_quote_from_the_bill_file(cfg, probe_stats):
    """report["bill"][lang] does not carry prompt_id or prompt_sha256; the bill
    FILE does, and that file is what the human read. Reading the quote from
    there is what stops G-PROMPT from comparing the configuration against
    itself."""
    write_json(cfg.report_dir / "translate_bill_German.json",
               {"language": "German", "prompt_id": "v4-frozen",
                "prompt_sha256": {"definition": "d" * 64,
                                  "expression": "e" * 64}})
    rows = [usage_row(prompt_sha256="d" * 64)]
    thin_bill = {"German": {"dollars": {"lean_uncached": 1.0}}}
    row = gates.billed_row(cfg, thin_bill, "German")
    assert row["prompt_id"] == "v4-frozen"
    assert row["prompt_sha256"]["definition"] == "d" * 64
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, thin_bill, rows, lang="German",
                                   stats=probe_stats)}
    assert built["G-PROMPT"][0] is True
    # a wave written with a prompt the bill never quoted fails
    other = [usage_row(prompt_sha256="0" * 64)]
    built = {g.id: g.fn() for g in
             gates.post_wave_gates(cfg, thin_bill, other, lang="German",
                                   stats=probe_stats)}
    assert built["G-PROMPT"][0] is False


def test_an_unpriced_row_is_recorded_not_dropped(cfg):
    """The ledger spans model changes. A row on a model with no rate card must
    not crash the ingest (which runs inside a spend gate) and must not be
    counted as free."""
    _write_usage(cfg, "translate_usage.jsonl",
                 [usage_row(model="gemini-2.0-flash"), usage_row()])
    led = billing.SpendLedger(cfg, on_date=datetime.date(2026, 8, 26))
    led.ingest()
    total = led.period_to_date()
    assert total["requests"] == 2 and total["unpriced_requests"] == 1
    assert total["usd"] > 0
    assert any(r.get("unpriced_why") for r in led.rows())
