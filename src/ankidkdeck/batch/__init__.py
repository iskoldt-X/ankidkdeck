"""The batch transport: JSONL writer, job registry, wave splitter, cache
lifecycle, positional reconciliation and the bounded retry loop.

Nothing here imports google.genai at module level. The bill-only path and the
money stack must stay import-free of the SDK (there is a test that asserts it),
and this package is only reached from stage 42 past the line where money is
spent -- so every SDK import is inside the function that needs it.

Module map:

    keys.py         the row key: {kind}__{lang}__{entry_id}__{chunk:02d} (5.2)
    jsonl.py        one row from one LlmRequest, field placement asserted (N-07)
    registry.py     the job state machine, the wave fingerprint, dedupe (5.1)
    reconcile.py    output-order attribution, error rows, count lock (5.3)
    caches.py       create / extend / recreate / delete an explicit cache (5.4)
    waves.py        the splitter, the enqueued arithmetic, the retry bound (5.5,
                    5.6)
    transport.py    the driver: submit -> poll -> download -> ingest -> retry
"""

from . import caches, jsonl, keys, reconcile, registry, transport, waves

__all__ = ["caches", "jsonl", "keys", "reconcile", "registry", "transport",
           "waves"]
