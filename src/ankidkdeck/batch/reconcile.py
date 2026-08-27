"""Line-by-line reconciliation of a batch result file (patch plan 5.3).

BY KEY. The join is the echoed `key`. The position is a cross-check that is
reported and never gated on, because the documented order guarantee is false.

Measured on the first real wave -- job `Chinese-def-w0-00`, 3,644 rows,
2026-08-27:

    result[   0:1000] -> plan    0 .. 999    ascending, no gaps
    result[1000:1644] -> plan 3000 .. 3643   ascending, no gaps
    result[1644:2644] -> plan 2000 .. 2999   ascending, no gaps
    result[2644:3644] -> plan 1000 .. 1999   ascending, no gaps

The service completes the input in ~1000-row shards and concatenates the shards
OUT OF ORDER; each shard is internally in input order. Positions 0-999 agreed
and the divergence starts at exactly row 1000, which is why no probe could see
it: every probe wave was <= 32 rows, i.e. a single shard, where the order does
hold. The docs' "responses will be written in the same order as the input
requests" was therefore unfalsified rather than confirmed, and at >1000 rows it
is simply untrue. Nothing here relies on it.

The bijection is the guard, and on the real wave it held exactly: 3,644 lines
for 3,644 planned rows, a `key` on every line (3,644/3,644), the two key sets an
exact bijection, zero duplicates. So a duplicate key in the results is FATAL, a
key that is not in the plan is FATAL, and a planned key with no result is FATAL.
There is no partial-credit reading of a broken bijection: under a shard
permutation a positional write shifts glosses onto the wrong senses from row
1000 on, which is catastrophic and invisible.

What a failed row actually looks like, measured:

    {"code": 7, "message": "The caller does not have permission"}
    {"code": 3, "message": "Request contains an invalid argument."}

A bare gRPC status with no detail, and on the probe evidence possibly with no
`key` either -- the key echo on an error row is documented nowhere. An error row
WITH a key joins by key like any other. An error row WITHOUT a key can only be
attributed to what is left over, and only when the leftover is forced: one
keyless row against one unclaimed plan row, or several keyless rows that are
byte-identical to one another. That second case is the dead cache (code 7 on
every row of a job, prompt=0, billed $0), where the assignment cannot change any
outcome because every candidate line is the same line -- the failure this
reconciliation has to survive without losing the rows around it. Anything else
is FATAL naming the line.

Interpolating a keyless row from its keyed neighbours was considered and
rejected. Within a shard the order holds, so bracketing looks sound -- but a
bracket that spans a shard boundary can enclose exactly one unclaimed plan row
that is the WRONG one, and a unique-looking wrong answer is the precise failure
mode this module exists to prevent. For the same reason a keyless row carrying a
response rather than an error is FATAL: an unkeyed success row would write cells
on an assumption.

Partial failure is judged from `batchStats.failedRequestCount` plus these
per-row error objects, NEVER from the job state: PARTIALLY_SUCCEEDED is a Vertex
enum that the Developer API does not have, so a job with failed rows reports
SUCCEEDED.
"""

import dataclasses
import json

from ..util import FatalError

# The array the count lock guards, per request kind. Named here because the
# count-lock check on the batch path is LOCAL (patch plan 5.6: the retry waves
# check the count lock on our side, the schema having already asked for it).
RESULT_ARRAY = {"definition": "definitions", "expression": "fixed_expressions"}


@dataclasses.dataclass
class RowOutcome:
    """One input row and the result line that carried its key.

    `pos` is the row's position in the INPUT plan and is the identity used
    everywhere downstream. `result_pos` is the line number the answer actually
    arrived on, kept only so the order cross-check can be reported.
    """
    pos: int
    key: str
    kind: str
    n_expected: int
    ok: bool = False
    error: dict | None = None
    finish_reason: str = ""
    usage: dict = dataclasses.field(default_factory=dict)
    items: list | None = None
    key_echoed: str | None = None
    parse_error: str | None = None
    result_pos: int | None = None

    @property
    def count_ok(self) -> bool:
        return isinstance(self.items, list) and len(self.items) == self.n_expected

    @property
    def got(self):
        return len(self.items) if isinstance(self.items, list) else self.items

    def why(self) -> str:
        """One line, for the report and for the retry decision."""
        if self.error:
            return "error %s: %s" % (self.error.get("code"),
                                     self.error.get("message"))
        if self.parse_error:
            return "unparseable response body: %s" % self.parse_error
        if self.finish_reason and self.finish_reason != "STOP":
            return "finishReason=%s" % self.finish_reason
        if not self.count_ok:
            return ("count lock: %s objects, expected %d (finishReason=%s)"
                    % (self.got, self.n_expected, self.finish_reason or "?"))
        return ""


def _error_of(line: dict):
    # A line that is not a JSON object at all is read as a failed row rather
    # than allowed to raise an AttributeError three frames down. It has no key,
    # so it can only be attributed if the leftover slot is forced -- and if it
    # is not, the caller refuses with the line named.
    if not isinstance(line, dict):
        return {"code": None,
                "message": "result line is not a JSON object (%s)"
                           % type(line).__name__}
    err = line.get("error")
    if err is None:
        err = line.get("status")
    if err is None:
        return None
    if isinstance(err, dict):
        return err
    return {"code": None, "message": str(err)}


def _finish_reason(response: dict) -> str:
    cands = response.get("candidates") or []
    if not cands:
        return "NO_CANDIDATE"
    first = cands[0] or {}
    return str(first.get("finishReason") or first.get("finish_reason")
               or "UNKNOWN")


def _text_of(response: dict) -> str:
    cands = response.get("candidates") or []
    if not cands:
        return ""
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
    return "".join(p.get("text") or "" for p in parts if isinstance(p, dict))


def _canonical(line) -> str:
    """A line reduced to a comparable string, for "are these two identical".

    Only used to decide whether several keyless error rows are interchangeable.
    A line that will not serialise falls back to repr, which is still a
    conservative comparison: unequal strings mean "assume they differ".
    """
    try:
        return json.dumps(line, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        return repr(line)


def _fill(outcome: RowOutcome, line) -> RowOutcome:
    """Read one result line into an already-attributed outcome."""
    error = _error_of(line)
    if error is not None:
        outcome.error = error
        return outcome
    response = line.get("response") if isinstance(line, dict) else None
    if not isinstance(response, dict):
        outcome.error = {"code": None,
                         "message": "no response and no error on this line"}
        return outcome
    outcome.usage = response.get("usageMetadata") or {}
    outcome.finish_reason = _finish_reason(response)
    # finishReason BEFORE json.loads, same order as the interactive path: a
    # MAX_TOKENS truncation is a cap error, and reading it as a parse error
    # sends the identical request again.
    if outcome.finish_reason != "STOP":
        return outcome
    text = _text_of(response)
    try:
        body = json.loads(text)
    except ValueError as exc:
        outcome.parse_error = str(exc)[:200]
        return outcome
    array = RESULT_ARRAY.get(outcome.kind)
    items = body.get(array) if isinstance(body, dict) and array else None
    outcome.items = items
    outcome.ok = outcome.count_ok
    return outcome


def _plan_index(plan: list) -> dict:
    """{key: index into plan}, refusing a plan whose keys are not unique."""
    by_key = {}
    for index, row in enumerate(plan):
        key = row["key"]
        if key in by_key:
            raise FatalError(
                "the stored plan carries key %r at both row %s and row %s. The "
                "result file is joined on that key, so a duplicate makes the "
                "join ambiguous. keys.validate_keys refuses this before a byte "
                "is uploaded, so a plan holding one did not come from this "
                "code." % (key, plan[by_key[key]].get("pos"), row.get("pos")))
        by_key[key] = index
    return by_key


def _attribute_keyless(keyless: list, unclaimed: list, plan: list) -> list:
    """[(plan index, result line number, line)] for rows that came back keyless.

    Forced or fatal, never plausible. See the module docstring for why the
    keyed neighbours of a keyless row are not allowed to place it.
    """
    if len(keyless) != len(unclaimed):
        raise FatalError(
            "%d result line(s) carry no key but %d planned row(s) have no "
            "result. The keyless rows cannot account for the gap, so the plan "
            "and the result file are not a bijection on the key. Nothing was "
            "written." % (len(keyless), len(unclaimed)))
    if len(keyless) > 1 and len({_canonical(ln) for _, ln in keyless}) > 1:
        raise FatalError(
            "%d result line(s) carry no key and they are not identical to one "
            "another: line(s) %s, against %d unclaimed planned row(s) (%s). "
            "The output order is not the input order -- measured on the first "
            "real wave, the service concatenates ~1000-row shards out of order "
            "-- so there is no position that attributes them, and guessing is "
            "the mis-attribution this guard exists to prevent. Nothing was "
            "written."
            % (len(keyless), ", ".join(str(r) for r, _ in keyless[:5]),
               len(unclaimed),
               ", ".join(plan[i]["key"] for i in unclaimed[:5])))
    return [(index, rpos, line)
            for index, (rpos, line) in zip(unclaimed, keyless)]


def reconcile(plan: list, lines: list, *, report: dict | None = None) -> list:
    """[RowOutcome] for one job, one per PLANNED row, in PLAN order.

    The join is `key` -> `key`; the return order is the plan's and not the
    file's, because `_absorb` zips the stored plan against this list. That
    contract is unchanged by the fix -- what changed is that the file is no
    longer assumed to arrive in that order.

    `plan` is what the registry stored at submit time: [{pos, key, kind, n,
    cells: [...]}]. Recomputing it from entries.json at ingest time would work
    right up until the source text changed between the submit and the ingest,
    at which point every row would be attributed to a request that was never
    sent.

    `report`, if given, is filled with the order cross-check: how many rows
    arrived where the docs say they would, where that first stopped being true,
    and how many ascending runs (shards) the file decomposes into. It is
    evidence for the run log, not a gate -- a shuffled file with an intact
    bijection is a correct file.
    """
    if len(lines) != len(plan):
        raise FatalError(
            "the result file has %d line(s) for a job of %d row(s). One result "
            "line per planned row is the hard check that nothing was lost or "
            "duplicated in transit -- the attribution itself is by key -- and "
            "a count that does not match means some planned row has no answer "
            "at all. Nothing was written." % (len(lines), len(plan)))
    by_key = _plan_index(plan)
    claimed: dict = {}
    keyless: list = []
    extra: list = []
    for rpos, line in enumerate(lines):
        echoed = line.get("key") if isinstance(line, dict) else None
        if echoed is None:
            keyless.append((rpos, line))
            continue
        index = by_key.get(str(echoed))
        if index is None:
            extra.append((rpos, echoed))
            continue
        if index in claimed:
            raise FatalError(
                "result line %d echoes key %r, which line %d already claimed. "
                "The result file is joined on the key, so a duplicate key is "
                "two answers for one request with no way to tell which one is "
                "the answer. Nothing was written."
                % (rpos, echoed, claimed[index][0]))
        claimed[index] = (rpos, line)
    unclaimed = [i for i in range(len(plan)) if i not in claimed]
    if extra:
        missing = [plan[i]["key"] for i in unclaimed]
        raise FatalError(
            "%d result line(s) echo a key this job never sent: %s. %d "
            "planned key(s) have no result: %s. The plan and the result file "
            "have to be an exact bijection on the key -- the first real wave "
            "measured 3,644 for 3,644, zero either way -- and these are not. "
            "Nothing was written."
            % (len(extra),
               ", ".join("line %d %r" % pair for pair in extra[:5]),
               len(missing),
               ", ".join(repr(k) for k in missing[:5]) or "none"))
    unkeyed_payload = [rpos for rpos, line in keyless
                       if _error_of(line) is None]
    if unkeyed_payload:
        raise FatalError(
            "%d result line(s) carry no key and no error: line(s) %s. A row "
            "with a response body is a row whose cells would be WRITTEN, and "
            "the only thing that could place it is the input order -- which "
            "the first real wave measured to be false past row 1000. The key "
            "echo is missing, so the row cannot be attributed. Nothing was "
            "written." % (len(unkeyed_payload),
                          ", ".join(str(r) for r in unkeyed_payload[:5])))
    for index, rpos, line in _attribute_keyless(keyless, unclaimed, plan):
        claimed[index] = (rpos, line)
    out = []
    for index, row in enumerate(plan):
        rpos, line = claimed[index]
        outcome = RowOutcome(
            pos=row["pos"], key=row["key"], kind=row["kind"],
            n_expected=int(row["n"]), result_pos=rpos,
            key_echoed=(line.get("key") if isinstance(line, dict) else None))
        out.append(_fill(outcome, line))
    if report is not None:
        report.update(order_cross_check(out, keyless=len(keyless)))
    return out


def order_cross_check(outcomes: list, *, keyless: int = 0) -> dict:
    """What the position WOULD have said, for the record only.

    `agreeing_prefix` is the interesting number: it was 1000 on the first real
    wave and 32 on every probe, which is the whole reason the false guarantee
    survived until a paid job. `ascending_runs` counts the shards the file
    decomposes into (1 == the file really was in input order).
    """
    n = len(outcomes)
    agree = sum(1 for i, o in enumerate(outcomes) if o.result_pos == i)
    first = next((i for i, o in enumerate(outcomes) if o.result_pos != i), None)
    prefix = n if first is None else first
    order = [0] * n
    for i, outcome in enumerate(outcomes):
        if outcome.result_pos is not None and 0 <= outcome.result_pos < n:
            order[outcome.result_pos] = i
    descents = sum(1 for i in range(n - 1) if order[i] > order[i + 1])
    runs = 1 + descents if n else 0
    return {"rows": n, "joined_by_key": n - keyless,
            "joined_without_key": keyless, "in_input_order": first is None,
            "positions_agreeing": agree, "agreeing_prefix": prefix,
            "first_divergence": first, "ascending_runs": runs}


def order_note(report: dict) -> str:
    """One line for the run log. Never a refusal: the bijection is the gate."""
    if not report:
        return ""
    if report.get("in_input_order"):
        return ("order cross-check: all %d row(s) arrived in input order"
                % report.get("rows", 0))
    return ("order cross-check: the result file is NOT in input order -- %d/%d "
            "positions agree, divergence starts at row %s, %d ascending run(s) "
            "(shards). Joined on the key instead, bijection intact."
            % (report.get("positions_agreeing", 0), report.get("rows", 0),
               report.get("first_divergence"), report.get("ascending_runs", 0)))


def failed_request_count(job) -> int | None:
    """batchStats.failedRequestCount, both spellings, or None if absent.

    Recorded next to the per-row errors, never instead of them: the field was
    never exercised by a probe (the probe waves judged partial failure from the
    rows), so it is evidence rather than the criterion.
    """
    stats = getattr(job, "batch_stats", None)
    if stats is None:
        stats = getattr(job, "batchStats", None)
    if stats is None:
        return None
    for name in ("failed_request_count", "failedRequestCount"):
        value = (stats.get(name) if isinstance(stats, dict)
                 else getattr(stats, name, None))
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def batch_stats_dict(job) -> dict | None:
    """batchStats as plain JSON for the registry, or None."""
    stats = getattr(job, "batch_stats", None) or getattr(job, "batchStats", None)
    if stats is None:
        return None
    if isinstance(stats, dict):
        return {k: stats.get(k) for k in sorted(stats)}
    out = {}
    for name in ("request_count", "requestCount", "successful_request_count",
                 "successfulRequestCount", "failed_request_count",
                 "failedRequestCount", "pending_request_count",
                 "pendingRequestCount"):
        value = getattr(stats, name, None)
        if value is not None:
            out[name] = value
    return out or None
