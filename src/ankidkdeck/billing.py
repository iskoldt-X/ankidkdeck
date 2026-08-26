"""The money stack: one arithmetic, one ledger, one set of consumption rules.

Three jobs, deliberately in one file because they are one subject:

  1. THE ARITHMETIC (spec 2.5), written once. Every dollar figure in this
     program -- the forecast on the bill, the actual on the ledger, the number
     G-BILL compares them with -- goes through usd_for_tokens(). The rule that
     is easiest to get wrong, and that three reviewers flagged separately:
     cachedContentTokenCount is a SUBSET of promptTokenCount on a usage row, not
     an addend. Charging (prompt + cached) at full rate over-states the bill by
     the cache; charging prompt at full rate and adding cached at the discount
     rate DOUBLE-charges the cached half. row_dollars() does the subtraction in
     one place so no caller has to remember which convention it is holding.

  2. THE LEDGER (spec 2.4). reports/spend_ledger.jsonl, append-only, one line
     per API interaction, fsync'd. It is what makes the $10/month cap
     enforceable at all: Google's project-level cap does NOT stop an
     already-submitted batch wave (measured -- the same job was accepted twice,
     back to back, and both were billed), so the ceiling has to be arithmetic we
     do before we submit, against a total we kept ourselves.

  3. THE CONSUMPTION RULES (N-09). The six rules that decide whether the
     measured constants on disk may be used to authorise a spend at all.

None of this imports the SDK, and none of it may: the bill-only path is
offline by contract, and there is a test that asserts no google* module is
imported on it.
"""

import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from .prices import rate_card
from .util import FatalError, read_json, write_json

# The authoritative record: append-only, one JSON object per line, fsync'd on
# every append. Everything else about the ledger is derived from it, including
# the ingest cursors -- so deleting the roll-up is safe and re-deriving it
# cannot double-count.
LEDGER_JSONL = "spend_ledger.jsonl"
# The derived roll-up spec 2.4 names. Rewritten from the JSONL after every
# ingest; never read back as a source of truth.
LEDGER_JSON = "spend_ledger.json"

# The stage ledgers this module absorbs. Written by stage 42 and stage 50, one
# fsync'd line per call, and they survive a crash -- which is the whole point:
# a wave that died halfway still spent money, and month-to-date has to know.
#
# ENUMERATED FROM THE WRITERS, not guessed with a wildcard. The previous glob
# was "*_usage.jsonl", which does not match review_usage_German.jsonl -- so
# every dollar the review subcommand spent was invisible to G-BUDGET, and the
# tests happened to use only the two filenames the glob did match. One line per
# writer, with its call site, so adding a writer without adding it here is a
# visible omission rather than a silent one:
#
#   s42_translate.py  reports/translate_usage.jsonl
#   s42_translate.py  reports/review_usage_<lang>.jsonl   (review(), per language)
#   s50_priority.py   reports/priority_usage.jsonl
#
# The trailing catch-all is deliberate belt-and-braces: under-counting spends
# money nobody approved, so a NEW writer must be absorbed by default rather than
# wait for someone to notice it is missing.
USAGE_GLOBS = ("translate_usage.jsonl", "review_usage_*.jsonl",
               "priority_usage.jsonl", "*usage*.jsonl")


def usage_paths(report_dir) -> list:
    """Every stage usage ledger in a report directory, de-duplicated, sorted."""
    report_dir = Path(report_dir)
    out: dict = {}
    for pattern in USAGE_GLOBS:
        for path in report_dir.glob(pattern):
            if path.is_file():
                out[path.name] = path
    return [out[name] for name in sorted(out)]


def usage_row_uid(row: dict):
    """The row's OWN identity, independent of the file and line it arrived on.

    normalize_usage() stamps every usage row with (ts, seq): a microsecond UTC
    timestamp and a monotonic per-process sequence number. That pair is what lets
    the ledger have TWO entry points without double counting -- the per-call sink
    and the ingest of the stage's own fsync'd file are two views of the same
    call, and before this they were two ledger rows (measured: 3 calls, 6 rows,
    exactly 2.000x, and the $10 cap became a number that lied).

    Returns None for a row without the stamp (a row written before it existed).
    Those still dedupe on (source, line, sha), which is why the cursor stays.
    """
    ts, seq = row.get("ts"), row.get("seq")
    if isinstance(ts, str) and ts and isinstance(seq, int):
        return "%s#%d" % (ts, seq)
    return None

# Earliest measured_at that may authorise a spend: the model's publication date.
# A constant measured before the model existed was measured on another model.
CONSTANTS_NOT_BEFORE = "2026-08-13"

# How far the prompt that will actually be SENT may drift from the prompt the
# constants were MEASURED on before the constants stop applying (consumption
# rule 6, "measured on LEAN, spent on RICH"). 10% of the system prompt: wide
# enough for the 1.7 constantisation edit (the German definition prompt is 5,134
# characters against a measured 5,124, i.e. +10 characters / 0.2%) and far too
# narrow for an enrichment (5,124 -> ~11,970 characters, +134%).
PROMPT_DRIFT_TOLERANCE = 0.10

# Which measured prompt family a request kind is sized against. The basis has to
# be per family or the rule inverts itself: `measured_prompt_chars` is a flat
# list across every prompt the probe ledger saw, and the patch plan's 4.4 A/B
# measures LEAN *and* RICH on the same model. After that A/B a flat max() basis
# jumps to the RICH size, and the LEAN prompt -- which the plan calls a FREE
# rollback, LEAN being a pure prefix of RICH -- drifts 57% from its own basis and
# gets refused. A gate that makes the documented rollback impossible is a gate
# that will be switched off.
PROMPT_FAMILY_OF_KIND = {"definition": "definition"}

# The scenario names the bill quotes (stage 42's BILL_SCENARIOS). Repeated here
# as the money stack's own vocabulary rather than imported, because importing
# the stage from this module would close the cycle gates -> billing -> stage ->
# gates. The test asserts the two tuples are equal.
SCENARIOS = ("cache_works", "lean_uncached", "rich_uncached")
FORBIDDEN_SCENARIO = "rich_uncached"


# --------------------------------------------------------------------------
# 1. The arithmetic. One implementation, used by every caller.
# --------------------------------------------------------------------------

def usd_for_tokens(*, uncached_input: int, cached_input: int, output: int,
                   rates: dict, ndigits: int = 9) -> float:
    """input$ + output$ for three DISJOINT token counts.

        input$  = uncached_input x input_rate + cached_input x cached_rate
        output$ = output x output_rate

    `output` must ALREADY include thinking tokens: thinking is billed at the
    OUTPUT rate, which is why the ledger adds thoughts to candidates and never
    to prompt. The three counts are disjoint here by contract -- callers holding
    the usage-row convention (cached is a subset of prompt) must go through
    row_dollars(), which does the subtraction.
    """
    missing = [k for k in ("input_usd_per_mtok", "cached_input_usd_per_mtok",
                           "output_usd_per_mtok")
               if not isinstance(rates, dict) or rates.get(k) is None]
    if missing:
        raise FatalError(
            "cannot price %d uncached + %d cached input and %d output tokens: "
            "the rate card is missing %s. A made-up price on a bill is worse "
            "than a missing one, because the missing one gets read."
            % (uncached_input, cached_input, output, ", ".join(missing)))
    total = (uncached_input * rates["input_usd_per_mtok"]
             + cached_input * rates["cached_input_usd_per_mtok"]
             + output * rates["output_usd_per_mtok"]) / 1e6
    # Nine decimals by default -- nano-dollars. One definition request costs
    # about $0.0002, so rounding a ROW to six decimals loses 0.15% of it, and
    # 5,565 of those roundings is a real drift in the one total that authorises
    # the next wave. Presentation rounds; the ledger does not.
    return round(total, ndigits)


def token_identity(row: dict) -> tuple:
    """(ok, detail) for total == prompt + candidates + thinking + tool_use.

    On a row whose thinking came from derived_thinking() the identity holds by
    construction, so a VIOLATION is information: it means this row's thinking
    did not come from the derivation. That is the defect worth catching --
    protobuf omits zero-valued fields, so a directly-read thoughtsTokenCount is
    0 exactly when the field is absent, which is exactly when the question
    ("did this request think?") is being asked.

    A row with no usage at all (a failed call: every count 0) is not a violation;
    it is a row with nothing to check, and it says so.
    """
    prompt = int(row.get("prompt_tokens") or 0)
    cand = int(row.get("candidates_tokens") or 0)
    think = int(row.get("thinking_tokens") or 0)
    tools = int(row.get("tool_use_tokens") or 0)
    total = int(row.get("total_tokens") or 0)
    parts = prompt + cand + think + tools
    if total == 0 and parts == 0:
        return True, {"checked": False,
                      "why": "no usage on this row (a failed call)"}
    if total != parts:
        return False, {"checked": True, "total_tokens": total,
                       "prompt+candidates+thinking+tool_use": parts,
                       "difference": total - parts,
                       "why": ("the row's thinking did not come from "
                               "total - prompt - candidates - toolUse; never "
                               "read thoughtsTokenCount directly")}
    return True, {"checked": True, "total_tokens": total}


def row_dollars(row: dict, rates: dict, *, strict: bool = True) -> dict:
    """What one usage row cost, in the usage-row convention.

    A usage row carries cached_tokens as a SUBSET of prompt_tokens. This is the
    one place that turns that into the three disjoint counts the arithmetic
    wants, and the one place that checks the token identity.

    `strict` refuses to price a row whose counts do not add up. Off, the row is
    still priced (a paid call is paid whether or not its bookkeeping is
    self-consistent) and carries `identity` so the gate can see it.
    """
    ok, detail = token_identity(row)
    if strict and not ok:
        raise FatalError(
            "usage row %r does not satisfy total == prompt + candidates + "
            "thinking + tool_use: %s" % (row.get("label"), detail))
    prompt = int(row.get("prompt_tokens") or 0)
    cached = int(row.get("cached_tokens") or 0)
    if cached > prompt:
        raise FatalError(
            "usage row %r has cached_tokens (%d) > prompt_tokens (%d). "
            "cachedContentTokenCount is a SUBSET of promptTokenCount; a row "
            "where it is not means the two were added somewhere upstream."
            % (row.get("label"), cached, prompt))
    uncached = prompt - cached
    output = (int(row.get("candidates_tokens") or 0)
              + int(row.get("thinking_tokens") or 0))
    return {
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "billable_output_tokens": output,
        "input_usd": usd_for_tokens(uncached_input=uncached,
                                    cached_input=cached, output=0,
                                    rates=rates),
        "output_usd": usd_for_tokens(uncached_input=0, cached_input=0,
                                     output=output, rates=rates),
        "usd": usd_for_tokens(uncached_input=uncached, cached_input=cached,
                              output=output, rates=rates),
        "identity_ok": ok,
        "identity": detail,
    }


def rows_usd(rows, rates: dict, *, strict: bool = False) -> float:
    """The total for a set of usage rows, priced one at a time."""
    return round(sum(row_dollars(r, rates, strict=strict)["usd"]
                     for r in rows), 6)


def rows_usd_priced(rows, on_date=None, default_model: str = "",
                    default_mode: str = "standard") -> dict:
    """Price a set of usage rows, EACH ON ITS OWN model and mode.

    A wave can be mixed (definitions on one model, expressions and the POS
    table on another) and a ledger spans configuration changes, so pricing the
    whole set on today's config would file half of it wrong. A row whose model
    has no rate card is NOT priced at zero: it is counted separately, because an
    unpriceable row has to be visible rather than free.
    """
    usd = 0.0
    unpriced = []
    for row in rows:
        model = row.get("model") or default_model
        mode = row.get("mode") or default_mode
        try:
            rates = rate_card(model, mode, on_date=on_date)
        except FatalError as exc:
            unpriced.append({"label": row.get("label"), "model": model,
                             "mode": mode, "why": str(exc).split(".")[0]})
            continue
        usd += row_dollars(row, rates, strict=False)["usd"]
    return {"usd": round(usd, 6), "rows": len(rows),
            "rows_priced": len(rows) - len(unpriced),
            "unpriced": unpriced[:20]}


# --------------------------------------------------------------------------
# 2. The ledger.
# --------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loads(raw: str):
    """One JSONL line, or None. A half-written last line is not a paid call."""
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _append_jsonl(path: Path, obj: dict) -> None:
    """One line, flushed and fsync'd before returning.

    Duplicated from stage 42's append_jsonl on purpose: this module must not
    import the stage (it would close the gates -> billing -> stage -> gates
    cycle), and a spend record that is only in a buffer is not a spend record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _row_day(row: dict, default: datetime.date) -> datetime.date:
    """A row's date, or the caller's default. A row with an unreadable date is
    counted in the CURRENT period rather than dropped: dropping it would make
    the cap wider than the human set it."""
    try:
        return datetime.date.fromisoformat(str(row.get("recorded_at")))
    except (TypeError, ValueError):
        return default


def call_day(row: dict):
    """The day the CALL happened, from the row's own `ts`, or None.

    The ledger used to file every ingested row on the day it was INGESTED, so a
    wave that ran over midnight -- or crashed and was absorbed the next morning
    -- landed in the wrong capped period, and the month the money was actually
    spent in read as zero. normalize_usage() stamps the call; this reads it.
    """
    raw = row.get("ts")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw).date()
    except ValueError:
        try:
            return datetime.date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _period_key(period: str, day: datetime.date) -> str:
    if period == "month":
        return day.strftime("%Y-%m")
    if period == "day":
        return day.isoformat()
    if period == "all":
        return "all"
    raise FatalError(
        "spend_cap_period = %r is not one of month, day, all. The cap is the "
        "only thing standing between a typo and a paid wave; an unrecognised "
        "period would silently widen it." % (period,))


class SpendLedger:
    """reports/spend_ledger.jsonl -- append-only, per call, crash-safe.

    Two ways in, and they are the SAME channel seen twice:

      append()  the per-call hook (stage 42's `usage_sink`). Exact: the row is
                written the moment the call returns.
      ingest()  absorbs the stage's own reports/<stage>_usage.jsonl. This is the
                half that makes a CRASHED wave count: the stage fsyncs every
                call to that file before it does anything else, so the rows are
                on disk even when nothing got as far as a sink. Without it, the
                ledger of record would be missing exactly the waves whose money
                is hardest to account for.

    ONE AUTHORITATIVE IDENTITY per call, so wiring both is safe. Every usage row
    carries (ts, seq) from normalize_usage(); the ledger records that as
    `row_uid` and ingest() skips any line whose uid is already on file. Before
    this, the sink wrote `source="sink"` with no line number, the cursor skipped
    exactly those rows, and a wave with the sink wired up counted twice --
    measured at exactly 2.000x, which turned the $10 cap into a number that
    refused the third language while claiming the fourth had been paid for.

    For a row with no uid (written before the stamp existed) the fallback is the
    old (source file, line number, line sha) cursor, DERIVED from the ledger
    itself so losing the roll-up cannot cause a double count.

    Rotation is no longer decided by one line's sha alone: when a source file no
    longer matches the prefix we absorbed, the file becomes a NEW GENERATION
    (`name#g2`) and is re-read FROM THE TOP, with the uids deciding what is
    actually new. That closes the measured hole where a deleted-and-recreated
    file whose Nth line happened to be byte-identical was resumed from the old
    offset -- 8 rows on disk, 5 in the ledger, no warning. Every such event is
    reported in `anomalies`, and G-BUDGET refuses the run that sees one: an
    unexplained ledger is not a ledger.
    """

    def __init__(self, cfg, rates=None, on_date=None):
        self.cfg = cfg
        self.path = Path(cfg.report_dir) / LEDGER_JSONL
        self.rollup_path = Path(cfg.report_dir) / LEDGER_JSON
        self.on_date = on_date
        self._rates = rates
        self._rows = None
        self._seen = None

    # ---- rates ----

    def rates(self) -> dict:
        """The rate card for the CONFIGURED model and mode, on the ledger's date.

        Refuses rather than quoting an unpriced model: an unpriceable row must
        stay unpriced and visible, not be filed at zero.
        """
        if self._rates is None:
            self._rates = rate_card(self.cfg.gemini_model, self.cfg.mode,
                                    on_date=self.on_date)
        return self._rates

    def _rates_for(self, row: dict) -> dict:
        """A row is priced on ITS OWN model and mode, not on today's config.

        The ledger spans configuration changes -- that is what a month-to-date
        total is -- so a batch row stays a batch row after someone switches the
        default to standard.
        """
        model = row.get("model") or self.cfg.gemini_model
        mode = row.get("mode") or self.cfg.mode
        return rate_card(model, mode, on_date=self.on_date)

    # ---- reading ----

    def rows(self, reload: bool = False) -> list:
        if self._rows is None or reload:
            self._rows = []
            self._seen = None
            if self.path.exists():
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._rows.append(json.loads(line))
        return self._rows

    def _cursors(self) -> dict:
        """{source: (max_line, sha_of_that_line)} derived from the ledger."""
        out: dict = {}
        for row in self.rows():
            src, line, sha = (row.get("source"), row.get("source_line"),
                              row.get("source_sha256"))
            if not src or not isinstance(line, int):
                continue
            if src not in out or line > out[src][0]:
                out[src] = (line, sha)
        return out

    def seen_uids(self) -> set:
        """Every per-call uid already on the ledger, whichever door it came in.

        This is the set that makes the sink and the ingest one channel instead
        of two. Cached and kept in step by append().
        """
        if self._seen is None:
            self._seen = {uid for uid in (r.get("row_uid")
                                          for r in self.rows()) if uid}
        return self._seen

    # ---- writing ----

    def append(self, row: dict, *, source: str = "sink",
               source_line: int | None = None,
               source_sha256: str | None = None,
               recorded_at: str | None = None) -> dict:
        """Price one usage row and append it. Never raises on a bad row's math:
        a paid call is paid whether or not its counts add up, and the identity
        result travels with the row so G-THINK and the audit can see it.

        The row is filed on the day the CALL happened when it says so (its own
        `ts`), and only then on the caller's date. That is what makes a wave that
        ran over midnight, or crashed and was ingested the next morning, land in
        the period whose cap it actually consumed.
        """
        day = (datetime.date.fromisoformat(recorded_at) if recorded_at
               else (call_day(row) or self.on_date or datetime.date.today()))
        uid = usage_row_uid(row)
        if uid is not None and uid in self.seen_uids():
            # This call is already in the ledger of record, whichever door it
            # came in by. Refusing it HERE and not only in ingest() is what makes
            # "wire the sink and the ingest and the total does not move" true in
            # either order, rather than true only while nobody reorders them.
            return next(r for r in self.rows() if r.get("row_uid") == uid)
        try:
            rates = self._rates_for(row)
        except FatalError as exc:
            # A row on a model with no rate card is RECORDED and marked, never
            # dropped and never priced at zero. The ledger spans model changes
            # (there are 2.0-flash rows in this project's history) and a spend
            # gate that crashed on one of them would take the whole run with it.
            out = dict(row)
            out.update({"usd": None, "input_usd": None, "output_usd": None,
                        "unpriced_why": str(exc).split(".")[0],
                        "recorded_at": day.isoformat(),
                        "period_month": day.strftime("%Y-%m"),
                        "source": source, "source_line": source_line,
                        "source_sha256": source_sha256, "row_uid": uid})
            self._write(out, uid)
            return out
        money = row_dollars(row, rates, strict=False)
        out = dict(row)
        out.update({
            "usd": money["usd"],
            "input_usd": money["input_usd"],
            "output_usd": money["output_usd"],
            "uncached_input_tokens": money["uncached_input_tokens"],
            "cached_input_tokens": money["cached_input_tokens"],
            "billable_output_tokens": money["billable_output_tokens"],
            "identity_ok": money["identity_ok"],
            "rate_window": rates["window"],
            "rate_input_usd_per_mtok": rates["input_usd_per_mtok"],
            "rate_cached_input_usd_per_mtok":
                rates["cached_input_usd_per_mtok"],
            "rate_output_usd_per_mtok": rates["output_usd_per_mtok"],
            "recorded_at": day.isoformat(),
            "period_month": day.strftime("%Y-%m"),
            "source": source,
            "source_line": source_line,
            "source_sha256": source_sha256,
            # The per-call identity, so the two doors into this ledger are one
            # channel. See usage_row_uid().
            "row_uid": uid,
        })
        self._write(out, uid)
        return out

    def _write(self, out: dict, uid) -> None:
        _append_jsonl(self.path, out)
        if self._rows is not None:
            self._rows.append(out)
        if uid and self._seen is not None:
            self._seen.add(uid)

    def ingest(self, write_rollup: bool = True) -> dict:
        """Absorb every stage usage line this ledger has not seen yet.

        Idempotent on the row's own uid first and on (source, line, sha) second,
        so calling this after a run that also had the per-call sink wired up
        absorbs NOTHING and the month-to-date total does not move.
        """
        cursors = self._cursors()
        seen = self.seen_uids()
        absorbed: dict = {}
        already: dict = {}
        generations: list = []
        anomalies: list = []
        for path in usage_paths(self.cfg.report_dir):
            name = path.name
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            # Which generation of this file are we looking at? The newest one we
            # have rows for, if its prefix still matches.
            gen = 1
            while ("%s#g%d" % (name, gen + 1)) in cursors:
                gen += 1
            key = name if gen == 1 else "%s#g%d" % (name, gen)
            have_line, have_sha = cursors.get(key, (0, None))
            rotated_why = ""
            if have_line:
                if len(lines) < have_line:
                    rotated_why = ("the file holds %d lines and %d were already "
                                   "absorbed from it: it was truncated or "
                                   "replaced" % (len(lines), have_line))
                elif have_sha and _sha(lines[have_line - 1]) != have_sha:
                    rotated_why = ("line %d is not the line absorbed at that "
                                   "offset: the file was replaced" % have_line)
            if rotated_why:
                # Not the file we absorbed. A NEW GENERATION, re-read FROM THE
                # TOP -- never resumed at the old offset, which is how a
                # recreated file with a byte-identical Nth line silently lost
                # every paid row before it. The uids decide what is new.
                gen += 1
                key = "%s#g%d" % (name, gen)
                have_line, have_sha = 0, None
                generations.append(key)
                uidless = sum(1 for ln in lines
                              if not usage_row_uid(_loads(ln) or {}))
                anomalies.append({
                    "source": name, "generation": key, "why": rotated_why,
                    "lines_now": len(lines), "lines_without_a_call_uid": uidless,
                    "reconciled_by": ("per-call uid" if not uidless
                                      else "NOTHING: %d line(s) carry no "
                                           "(ts, seq) stamp, so a copied line "
                                           "and a new paid call are "
                                           "indistinguishable and this file is "
                                           "re-read in full" % uidless),
                })
            start = 0 if rotated_why else have_line
            n = dup = 0
            for i in range(start, len(lines)):
                raw = lines[i]
                row = _loads(raw)
                if row is None:
                    continue
                uid = usage_row_uid(row)
                if uid is not None and uid in seen:
                    dup += 1
                    continue
                self.append(row, source=key, source_line=i + 1,
                            source_sha256=_sha(raw))
                n += 1
            if n:
                absorbed[key] = n
            if dup:
                already[key] = dup
        out = {"absorbed": absorbed, "already_on_the_ledger": already,
               "new_generations": generations, "anomalies": anomalies,
               "rows_total": len(self.rows()),
               "sources": [p.name for p in usage_paths(self.cfg.report_dir)]}
        if write_rollup:
            out["rollup"] = self.write_rollup(anomalies=anomalies)
        return out

    def write_rollup(self, anomalies=None) -> dict:
        """reports/spend_ledger.json -- DERIVED, never a source of truth."""
        roll = self.summary()
        roll["note"] = ("derived from %s, which is the append-only record. This "
                        "file can be deleted and regenerated; the ingest "
                        "cursors live in the JSONL rows themselves."
                        % LEDGER_JSONL)
        # On disk as well as in the return value: a rotated or truncated usage
        # file is the one event that can make this total wrong, so it may not
        # live only in a function's return value.
        roll["ingest_anomalies"] = list(anomalies or [])
        write_json(self.rollup_path, roll)
        return roll

    # ---- totals ----

    def summary(self) -> dict:
        rows = self.rows()
        by_month: dict = {}
        by_model: dict = {}
        by_mode: dict = {}
        for row in rows:
            usd = float(row.get("usd") or 0.0)
            by_month.setdefault(row.get("period_month") or "?",
                                {"usd": 0.0, "requests": 0})
            by_month[row.get("period_month") or "?"]["usd"] += usd
            by_month[row.get("period_month") or "?"]["requests"] += 1
            by_model[row.get("model") or "?"] = round(
                by_model.get(row.get("model") or "?", 0.0) + usd, 6)
            by_mode[row.get("mode") or "?"] = round(
                by_mode.get(row.get("mode") or "?", 0.0) + usd, 6)
        for month in by_month:
            by_month[month]["usd"] = round(by_month[month]["usd"], 6)
        return {
            "rows": len(rows),
            "usd_total": round(sum(float(r.get("usd") or 0.0) for r in rows), 6),
            "by_month": by_month,
            "by_model": by_model,
            "by_mode": by_mode,
            "identity_violations": sum(1 for r in rows
                                       if r.get("identity_ok") is False),
            "tokens": {
                key: sum(int(r.get(key) or 0) for r in rows)
                for key in ("prompt_tokens", "cached_tokens",
                            "candidates_tokens", "thinking_tokens",
                            "total_tokens")},
        }

    def period_to_date(self, period: str | None = None, today=None) -> dict:
        """What has been spent in the capped period so far.

        The period of a row is the day the CALL happened when the row says so
        (normalize_usage stamps `ts`), and the day it was recorded otherwise. A
        wave that ran over midnight, or crashed and was ingested the next
        morning, therefore lands in the period whose cap it actually consumed --
        it used to land in the period it was ingested in, which read the spending
        month as zero.
        """
        period = period or getattr(self.cfg, "spend_cap_period", "month")
        day = today or self.on_date or datetime.date.today()
        want = _period_key(period, day)
        rows = [r for r in self.rows()
                if _period_key(period, _row_day(r, day)) == want]
        return {
            "period": period,
            "period_key": want,
            "usd": round(sum(float(r.get("usd") or 0.0) for r in rows), 6),
            "requests": len(rows),
            # Rows the ledger could not price are counted separately rather
            # than folded in as zero: "we do not know what this cost" and "this
            # cost nothing" are different answers to a budget question.
            "unpriced_requests": sum(1 for r in rows if r.get("usd") is None),
            "models": sorted({r.get("model") or "?" for r in rows}),
            "modes": sorted({r.get("mode") or "?" for r in rows}),
        }


def ledger_sink(cfg, rates=None, on_date=None):
    """The callable stage 42 takes as `usage_sink`: one ledger line per call.

    Wiring this is one line at the two call sites in cli.py, and it is SAFE to
    wire alongside ingest(): both write the same per-call uid, so whichever runs
    second absorbs nothing. That is the whole point of usage_row_uid() -- the
    same wiring used to produce exactly 2.000x.

    What it buys is immediacy, not correctness: the row reaches the ledger of
    record as the call returns rather than at the next pre-spend gate, so a
    process killed with -9 between the stage's fsync and the next ingest still
    has its money in the ledger's own file.
    """
    ledger = SpendLedger(cfg, rates=rates, on_date=on_date)
    return ledger.append


# --------------------------------------------------------------------------
# 3. The forecast side: which scenario a run is actually about to take.
# --------------------------------------------------------------------------

def expected_scenario(cfg) -> str:
    """The bill column this configuration will actually be billed on.

    cache_works when an explicit cache is configured, lean_uncached otherwise.
    Never rich_uncached: that column exists so the number is on the page, and no
    branch of this program may run there (spec 2.2).
    """
    return "cache_works" if getattr(cfg, "cache_enabled", False) \
        else "lean_uncached"


def forecast(bill: dict, cfg) -> dict:
    """What the run just quoted, on the scenario it will actually take.

    `bill` is stage 42's report["bill"] -- {lang: bill_row + tokens + dollars}.
    Returns the summed forecast and, when it cannot be computed, WHY. An
    unpriceable forecast is not zero: it is a refusal, because the ceiling
    cannot be checked against a number that does not exist.
    """
    scenario = expected_scenario(cfg)
    usd = 0.0
    langs = []
    problems = []
    for lang in sorted(bill):
        money = (bill[lang] or {}).get("dollars") or {}
        value = money.get(scenario)
        if value is None:
            problems.append("%s: %s" % (lang, money.get("why")
                                        or "no %s figure on the bill"
                                        % scenario))
            continue
        usd += float(value)
        langs.append(lang)
    return {"scenario": scenario, "usd": round(usd, 6) if not problems else None,
            "languages": langs, "problems": problems,
            "forbidden_scenario": FORBIDDEN_SCENARIO}


# --------------------------------------------------------------------------
# 4. N-09: the six consumption rules.
# --------------------------------------------------------------------------

def _measured_at_date(stats: dict):
    raw = str(stats.get("measured_at") or "")[:10]
    try:
        return datetime.date.fromisoformat(raw)
    except ValueError:
        return None


def _chars_per_token(stats: dict):
    """The measured chars/token of the PROMPT, or None.

    The prompt body is English whatever the target language is, so English is
    the ratio to size an English system prompt with. No fallback: a default
    ratio here would change a gate's verdict on the strength of a number nobody
    measured -- and it does, by 13%, which is more than the tolerance it would
    be compared against.
    """
    table = stats.get("CHARS_PER_TOKEN") or {}
    value = table.get("English")
    return float(value) if isinstance(value, (int, float)) else None


def estimated_prompt_tokens(text: str, stats: dict):
    """Offline size of a prompt, in the measured chars/token of the probe set.

    Offline on purpose: the whole point of rule 6 is to catch "measured on LEAN,
    spent on RICH" BEFORE a call is placed, and countTokens is a call.
    """
    ratio = _chars_per_token(stats)
    if not ratio:
        return None
    return int(math.ceil(len(text) / ratio))


def consumption_rules(cfg, stats: dict, prompts: dict | None = None,
                      today=None) -> list:
    """The N-09 consumption rules, one row per CHECK: (id, spec_rule, ok,
    blocking, detail).

    Nine rows for six spec rules, because three of the six decompose into
    independently checkable halves (rule 1 -> the constants exist / the model
    matches / the measurement is not older than the model / the artifact carries
    no consumption guard; rule 6 -> pack version and size band). `spec_rule` is
    the authority for which plan rule a row belongs to. Spec rule 5 is not here:
    it is a POST-wave assertion and lives in gates.cache_hit_is_complete.

    `prompts` is {kind: the system prompt text that will actually be sent} --
    from stage 42's system_prompt(kind, lang), which is the single source the
    wire, the bill and the ledger all read. Without it the prompt-lineage rules
    report that they could not run, which is a refusal and not a pass.
    """
    rows = []

    def add(rule, spec, ok, detail, blocking=True):
        rows.append({"rule": rule, "spec_rule": spec, "ok": bool(ok),
                     "blocking": bool(blocking), "detail": detail})

    # Rule 1a: the artifact exists, names the configured model, and carries
    # every constant the money math reads. ONE list of required keys, the
    # stage's, so this cannot disagree with probe_stats() or with doctor.
    from .stages.s42_translate import (REQUIRED_STATS_KEYS,  # noqa: PLC0415
                                       missing_stats_keys,
                                       thinking_per_request)
    why = dict(REQUIRED_STATS_KEYS)
    missing = missing_stats_keys(stats or {})
    add("R1-constants", "1", not missing,
        {"missing": [{"key": k, "consumed_by": why[k]} for k in missing],
         "required": [k for k, _ in REQUIRED_STATS_KEYS]})
    add("R1-model", "1", (stats or {}).get("model") == cfg.gemini_model,
        {"measured_on": (stats or {}).get("model"),
         "configured": cfg.gemini_model})

    # Rule 1b: measured_at is not older than the model.
    day = _measured_at_date(stats or {})
    floor = datetime.date.fromisoformat(CONSTANTS_NOT_BEFORE)
    add("R2-measured-at", "1", day is not None and day >= floor,
        {"measured_at": (stats or {}).get("measured_at"),
         "not_before": CONSTANTS_NOT_BEFORE,
         "why": ("a constant measured before the model was published was "
                 "measured on a different model")})

    # Rule 6: the prompt the constants were measured on is the prompt we are
    # about to send. Two halves: the PACK VERSION must match (that is what
    # prompt_id is), and the SIZE must not have drifted (that is what catches an
    # enrichment whose pack version somebody forgot to bump).
    declared = (stats or {}).get("prompt_id")
    add("R6-prompt-id", "6", bool(declared) and declared == cfg.prompt_id,
        {"artifact_prompt_id": declared, "configured_prompt_id": cfg.prompt_id,
         "why": ("the thinking constant, the prompt-token fit and the system "
                 "prompt size are all properties of one prompt pack. A pack "
                 "version change invalidates them until somebody re-measures. "
                 "An artifact that does not declare which pack it was measured "
                 "on cannot authorise a spend on any pack.")})

    # The guard the backfill tool sets when the artifact is not fit to authorise
    # a spend. It had no reader at all: the tool's docstring said "doctor and the
    # spend gate both refuse while it is present", doctor did and the spend path
    # did not, so the one switch designed to stop a spend stopped nothing. Now
    # the switch is wired to the thing it names.
    guard = (stats or {}).get("CONSUMPTION_GUARD")
    add("R1-guard", "1", not guard,
        {"consumption_guard": guard,
         "why": ("tools/backfill_probe_stats.py sets CONSUMPTION_GUARD when a "
                 "consumption-critical constant is missing, disagrees with the "
                 "raw ledger, or was relabelled onto a prompt pack nobody "
                 "re-measured. It is cleared by fixing that, never by editing "
                 "it out.")})

    system_only = (stats or {}).get("PROMPT_TOKENS_system_only") or {}
    # The best available basis for "is this the prompt that was measured", in
    # order of directness:
    #   1. CHARACTERS, against the prompt the probes actually sent. Recorded by
    #      the backfill tool from the raw ledger's request fingerprints, and free
    #      of any tokenizer approximation -- the current German definition prompt
    #      is 5,134 characters against a measured 5,124, i.e. 0.2%.
    #   2. TOKENS, via the measured chars/token ratio. Coarser (it carries a
    #      systematic offset of a few percent), used when the artifact does not
    #      record what it sent.
    lineage = (stats or {}).get("prompt_lineage") or {}
    band = lineage.get("size_band_basis") or {}
    by_family = band.get("by_family") if isinstance(band.get("by_family"),
                                                    dict) else {}
    measured_chars = [c for c in lineage.get("measured_prompt_chars") or []
                      if isinstance(c, (int, float))]
    measured_tokens = [v for v in system_only.values()
                       if isinstance(v, (int, float))]
    if prompts:
        drift = []
        for kind in sorted(prompts):
            text = prompts[kind] or ""
            item = {"kind": kind, "chars_now": len(text),
                    "sha256_now": _sha(text), "checked": False,
                    "tolerance": PROMPT_DRIFT_TOLERANCE}
            family = PROMPT_FAMILY_OF_KIND.get(kind)
            family_chars = [c for c in (by_family.get(family) or [])
                            if isinstance(c, (int, float))] if family else []
            # Only the DEFINITION prompt was ever probed. The expression prompt
            # has no measured size at all, which the bill also says out loud.
            if kind != "definition":
                item["why"] = "never probed; no measured size to compare with"
            elif family_chars:
                # The NEAREST measured size in this prompt's OWN family, not the
                # largest size in the ledger: after the 4.4 A/B the ledger holds
                # both LEAN and RICH, and max() would make the free LEAN
                # rollback fail its own gate.
                base = min(family_chars, key=lambda c: abs(len(text) - c))
                item.update({"basis": "characters",
                             "basis_scope": "prompt family %r" % family,
                             "measured_chars": base,
                             "family_measured_chars": sorted(family_chars),
                             "drift": round(abs(len(text) - base) / base, 4),
                             "checked": True})
            elif measured_chars:
                base = max(measured_chars)
                item.update({"basis": "characters",
                             "basis_scope": ("every prompt in the ledger (the "
                                             "artifact records no per-family "
                                             "size band)"),
                             "measured_chars": base,
                             "drift": round(abs(len(text) - base) / base, 4),
                             "checked": True})
            elif measured_tokens:
                est = estimated_prompt_tokens(text, stats or {})
                base = max(measured_tokens)
                if est is None:
                    item["why"] = ("no measured CHARS_PER_TOKEN, so the prompt "
                                   "cannot be sized offline")
                else:
                    item.update({"basis": "tokens", "estimated_tokens": est,
                                 "measured_tokens": base,
                                 "drift": round(abs(est - base) / base, 4),
                                 "checked": True})
            else:
                item["why"] = "the artifact records no prompt size at all"
            drift.append(item)
        checked = [d for d in drift if d.get("checked")]
        add("R6-prompt-size", "6",
            bool(checked) and all(d["drift"] <= PROMPT_DRIFT_TOLERANCE
                                  for d in checked),
            {"prompts": drift,
             "measured_shas": (stats or {}).get("prompt_sha256_per_n"),
             "size_band_basis_prompt_id": band.get("prompt_id"),
             "size_band_basis_families": sorted(by_family),
             "why": ("size is the offline proxy for identity. The probe set "
                     "measured %s different prompt shas (one per batch size) on "
                     "a prompt this program has since made constant, so a sha "
                     "equality check can never pass; a %d%% size band accepts "
                     "that edit and rejects an enrichment."
                     % (len((stats or {}).get("prompt_sha256_per_n") or []),
                        int(PROMPT_DRIFT_TOLERANCE * 100)))})
    else:
        add("R6-prompt-size", "6", False,
            {"why": "no prompt text was passed, so nothing was compared"})

    # Rule 3: the bill quotes a CEILING, so the thinking constant it uses has
    # to be a p95 and the artifact has to carry one.
    level = getattr(cfg, "thinking_level", "LOW")
    p95 = thinking_per_request(stats or {}, level, "p95")
    add("R3-p95", "3", p95 is not None,
        {"thinking_level": level, "p95": p95,
         "mean": thinking_per_request(stats or {}, level, "mean"),
         "why": ("half the requests being above the number a human accepted is "
                 "not a ceiling")})

    # Rule 2: the cache constructor reads the measured floor, and a configured
    # cache has to be over it.
    floor_tok = ((stats or {}).get("wave2") or {}).get("EXPLICIT_CACHE_FLOOR")
    if getattr(cfg, "cache_enabled", False):
        est = (estimated_prompt_tokens(prompts.get("definition") or "",
                                       stats or {})
               if prompts and prompts.get("definition") else None)
        add("R2-cache-floor", "2",
            isinstance(floor_tok, (int, float)) and est is not None
            and est >= floor_tok,
            {"explicit_cache_floor": floor_tok, "estimated_prompt_tokens": est,
             "why": ("a cache object under the floor is refused by the server "
                     "(400 INVALID_ARGUMENT, the server states the minimum), so "
                     "the wave would run uncached at full price"
                     if est is not None else
                     "the prompt could not be sized offline (no measured "
                     "CHARS_PER_TOKEN and no prompt text), so nothing can say "
                     "whether the cache object would clear the floor")})
    else:
        add("R2-cache-floor", "2", isinstance(floor_tok, (int, float)),
            {"explicit_cache_floor": floor_tok, "cache_enabled": False,
             "why": "no cache configured; the floor still has to be on file"})

    # Rule 4: the measured expiry behaviour licenses a 1.5x TTL margin instead
    # of the 3x the audit asked for -- but only because the failure is LOUD and
    # free. If a future measurement ever says otherwise, the margin goes back up.
    wave2 = (stats or {}).get("wave2") or {}
    wave3 = (stats or {}).get("wave3") or {}
    behaviours = [str(wave2.get("CACHE_EXPIRY_BEHAVIOUR") or ""),
                  str(((wave3.get("W3_4_EXPIRY") or {})
                       if isinstance(wave3.get("W3_4_EXPIRY"), dict) else {})
                      .get("CACHE_EXPIRY_BEHAVIOUR") or "")]
    loud = any("403" in b or "per_line_error" in b or "ERROR" in b.upper()
               for b in behaviours if b)
    factor = float(getattr(cfg, "cache_ttl_factor", 1.5) or 1.5)
    need = 1.5 if loud else 3.0
    # Blocking only when a cache is actually configured: with no cache there is
    # no TTL for the margin to be a margin of, and a gate that refuses a run
    # over an irrelevant setting teaches people to ignore gates.
    add("R4-ttl-margin", "4", factor >= need,
        {"cache_enabled": bool(getattr(cfg, "cache_enabled", False)),
         "cache_ttl_factor": factor, "required": need,
         "measured_expiry_behaviour": [b for b in behaviours if b],
         "why": ("expiry is a loud, free failure (403 interactive, gRPC code 7 "
                 "per row in batch, prompt=0, $0 billed) and SILENT_UNCACHED "
                 "was never observed, so 1.5x covers the drain window. Without "
                 "that measurement the margin would have to be 3x."
                 if loud else
                 "nothing on file says an expired cache fails loudly, so the "
                 "margin has to cover a silent full-price wave")},
        blocking=bool(getattr(cfg, "cache_enabled", False)))
    return rows


def assert_ready_to_spend(cfg, stats: dict, prompts: dict | None = None,
                          today=None) -> list:
    """Refuse the spend unless every blocking consumption rule passes.

    Call this next to probe_stats(cfg) on the paid path. probe_stats already
    covers the file, the model and the required keys; this adds the rules it
    does not -- the measurement date and the prompt lineage -- and returns the
    full row list for the report.
    """
    rows = consumption_rules(cfg, stats, prompts=prompts, today=today)
    bad = [r for r in rows if r["blocking"] and not r["ok"]]
    if bad:
        raise FatalError(
            "%d consumption rule(s) refuse this spend:\n%s"
            % (len(bad), "\n".join(
                "  %s (patch plan N-09 rule %s): %s"
                % (r["rule"], r["spec_rule"], json.dumps(r["detail"],
                                                         ensure_ascii=False,
                                                         sort_keys=True))
                for r in bad)))
    return rows


def probe_stats_state(cfg) -> dict:
    """A one-glance summary of the measured-constants artifact, for doctor and
    for the working notes. Never raises: its job is to describe a file that may
    be missing or wrong."""
    path = Path(cfg.probe_stats_path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stats = read_json(path, default={})
    from .stages.s42_translate import missing_stats_keys  # noqa: PLC0415
    return {"path": str(path), "exists": True,
            "schema": stats.get("schema"),
            "measured_at": stats.get("measured_at"),
            "model": stats.get("model"),
            "prompt_id": stats.get("prompt_id"),
            "consumption_guard": stats.get("CONSUMPTION_GUARD"),
            "missing_required_keys": missing_stats_keys(stats)}
