"""Line-by-line reconciliation of a batch result file (patch plan 5.3).

BY POSITION. The output order is the input order and that is documented and was
measured (32 rows in, 32 rows out, order preserved), whereas the `key` echo on
an ERROR row is not documented at all -- it was present in the probe wave and
matched the positional expectation, but a behaviour that happens to hold is not
a contract. So the position is the attribution and the key is the cross-check,
in that order and never the other way round.

What a failed row actually looks like, measured:

    {"code": 7, "message": "The caller does not have permission"}
    {"code": 3, "message": "Request contains an invalid argument."}

A bare gRPC status with no detail. Code 7 is what a dead cache reference
produces on every row of a job (prompt=0, billed $0), which is the failure this
reconciliation has to survive without losing the rows around it.

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
    """One input row and whatever came back in its position."""
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


def reconcile(plan: list, lines: list) -> list:
    """[RowOutcome] for one job, one per PLANNED row, in input order.

    `plan` is what the registry stored at submit time: [{pos, key, kind, n,
    cells: [...]}]. Recomputing it from entries.json at ingest time would work
    right up until the source text changed between the submit and the ingest,
    at which point every row would be attributed to a request that was never
    sent.
    """
    if len(lines) != len(plan):
        raise FatalError(
            "the result file has %d line(s) for a job of %d row(s). The "
            "reconciliation is POSITIONAL -- output order is input order -- so "
            "a length mismatch is not something to work around: every "
            "attribution after the first missing line would be wrong. Nothing "
            "was written." % (len(lines), len(plan)))
    out = []
    for row, line in zip(plan, lines):
        expected = row["key"]
        echoed = line.get("key")
        if echoed is not None and echoed != expected:
            raise FatalError(
                "batch row %d echoed key %r where the input had %r. The key is "
                "the cross-check on the positional attribution, and a "
                "disagreement means one of the two is wrong -- there is no "
                "reading of this that lets the wave be written."
                % (row["pos"], echoed, expected))
        outcome = RowOutcome(pos=row["pos"], key=expected, kind=row["kind"],
                             n_expected=int(row["n"]), key_echoed=echoed)
        error = _error_of(line)
        if error is not None:
            outcome.error = error
            out.append(outcome)
            continue
        response = line.get("response")
        if not isinstance(response, dict):
            outcome.error = {"code": None,
                             "message": "no response and no error on this line"}
            out.append(outcome)
            continue
        outcome.usage = response.get("usageMetadata") or {}
        outcome.finish_reason = _finish_reason(response)
        # finishReason BEFORE json.loads, same order as the interactive path: a
        # MAX_TOKENS truncation is a cap error, and reading it as a parse error
        # sends the identical request again.
        if outcome.finish_reason != "STOP":
            out.append(outcome)
            continue
        text = _text_of(response)
        try:
            body = json.loads(text)
        except ValueError as exc:
            outcome.parse_error = str(exc)[:200]
            out.append(outcome)
            continue
        array = RESULT_ARRAY.get(outcome.kind)
        items = body.get(array) if isinstance(body, dict) and array else None
        outcome.items = items
        outcome.ok = outcome.count_ok
        out.append(outcome)
    return out


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
