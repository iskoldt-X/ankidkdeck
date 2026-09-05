"""The rate card: dollars per million tokens, copied off the pricing page.

Nothing here is measured and nothing here is guessed: every number was read off
Google's Gemini API pricing page on PRICING_PAGE_READ_AT and is stored with that
date, because a price is a fact about a page on a day. The three properties that
make this module worth having as its own file:

  1. PER-MODEL ALLOW-LIST. A model whose card nobody has read is REFUSED, not
     quoted. Quoting an unread model is how a run pays 2.0-flash prices for a
     3.7-flash bill and files the difference as rounding.
  2. DATED WINDOWS. The pricing page states two prices for this model with a
     boundary at 2027-01-01 ("$0.375 through December 31, 2026. $0.75 starting
     January 1, 2027."), and it never calls the lower one promotional. A bill
     quoted for a wave that runs in January must use the January card, so the
     card is a function of the date and not a constant.
  3. ONE CONSERVATIVE UNKNOWN, NAMED, AND STILL OPEN. The BATCH cached-input
     rate is the one number the sources disagree on: the pricing page's batch
     row says $0.0375/M and the batch guide says "you pay the standard context
     caching rates", i.e. $0.075/M. Only the INVOICE can settle it. Until it
     does, this module returns the HIGHER figure and says so in
     `cached_input_unsettled`, because the direction of a pricing error matters:
     over-quoting refuses a wave that would have fit, under-quoting spends money
     nobody accepted. A month-total comparison against a dashboard cannot close
     it and was tried -- see CACHED_INPUT_OPEN_QUESTION for the arithmetic and
     for the half of the question that IS settled.

The dollar arithmetic is NOT here. This module answers "what does a token
cost"; billing.py answers "what does this wave cost", once, in one place.
"""

import datetime

from .config import MODES, VERIFIED_MODELS
from .util import FatalError

# The date the numbers below were read off
# https://ai.google.dev/gemini-api/docs/pricing . Same date as
# config.VERIFIED_MODELS[...]["rate_card_read_at"], and the two are asserted
# equal by the tests: a model may not be on the allow-list with a card read on
# one date and priced from a card read on another.
PRICING_PAGE_READ_AT = "2026-08-13"

# The page's own boundary, quoted verbatim on the batch input line:
#   "$0.375 through December 31, 2026. $0.75 starting January 1, 2027."
# The page never says "promotional", "introductory" or "limited time", so the
# second figure is the price and the first one is the exception. A wave that
# drains across midnight on new year is billed on both cards; the ceiling has to
# be quoted on the later one.
DOUBLING_DATE = datetime.date(2027, 1, 1)

# Cache storage is billed per token-HOUR, not per token, and it is the only line
# here that the doubling sentence does not restate. Carried forward unchanged
# into the 2027 window and flagged there rather than silently doubled.
_CACHE_STORAGE_USD_PER_MTOK_HOUR = 0.50

# Per model, per window, per mode. Written out as literals in both windows
# instead of computing the second one as `* 2`: a derived price cannot be
# checked against a page, and this table's whole job is to be checkable.
#
# WHAT WAS QUOTED VERBATIM vs WHAT IS THE SPEC'S READING OF THE PAGE:
#   verbatim   the 2026 numbers (all four lines, both surfaces) and the batch
#              INPUT line's two dates.
#   reading    that the doubling applies to the whole row set rather than to the
#              batch input line alone. The patch plan (section 1.3) and the
#              audit (2.1, judgement #4) both take that reading. It is the
#              conservative one: if only the batch input line doubles, every
#              2027 figure here is an over-quote, which refuses waves rather
#              than under-charging them.
_CARDS = {
    "gemini-3.7-flash": {
        "read_at": "2026-08-13",
        "windows": (
            {
                "from": None,
                "until": datetime.date(2026, 12, 31),
                "label": "2026 (through December 31, 2026)",
                "source": "pricing page, read 2026-08-13, verbatim",
                "modes": {
                    # Interactive. Cached input is the standard context-caching
                    # rate: one tenth of uncached input.
                    "standard": {"input": 0.75, "output": 3.75,
                                 "cached_input": 0.075},
                    # Half price on both sides of the wire. The cached-input
                    # figure is the CONSERVATIVE one of the two the sources give
                    # (see the module docstring): the page's batch row says
                    # 0.0375, the batch guide says standard caching rates. The
                    # page's figure is the better documentary claim and may well
                    # be right -- but only the invoice's per-project
                    # cached-input line item can settle it, and until it does
                    # this line over-quotes rather than under-quotes. See
                    # CACHED_INPUT_OPEN_QUESTION.
                    "batch": {"input": 0.375, "output": 1.875,
                              "cached_input": 0.075},
                    # Input and output are penny-for-penny identical to batch on
                    # the page. CACHED INPUT IS THE SAME CONSERVATIVE FIGURE FOR
                    # A DIFFERENT REASON, and the two must not be assumed to
                    # move together: flex is a service tier on the STANDARD
                    # surface, not a separate price list, and the batch guide's
                    # "standard context caching rates" is uncontested for that
                    # surface. Whatever the invoice says about BATCH cached
                    # input says nothing about flex. It is a forecast-only
                    # number in any case: s42.transport_guard refuses
                    # cache_enabled on every mode except batch, so no flex
                    # request ever bills a cached input token.
                    "flex": {"input": 0.375, "output": 1.875,
                             "cached_input": 0.075},
                },
            },
            {
                "from": DOUBLING_DATE,
                "until": None,
                "label": "2027 (starting January 1, 2027)",
                "source": ("pricing page, read 2026-08-13: the batch input "
                           "line's doubling is verbatim; applying it to the "
                           "whole row set is the patch plan's reading"),
                "modes": {
                    "standard": {"input": 1.50, "output": 7.50,
                                 "cached_input": 0.15},
                    "batch": {"input": 0.75, "output": 3.75,
                              "cached_input": 0.15},
                    "flex": {"input": 0.75, "output": 3.75,
                             "cached_input": 0.15},
                },
            },
        ),
    },
}

# The whole reason the cached-input figure is not simply "the price": both
# readings are on the record, the difference is under $0.02 per language at the
# measured volume, and only the invoice can close it. Recorded here so the bill
# carries the ambiguity instead of a reader having to remember it.
#
# WHAT SETTLES IT, AND WHAT DOES NOT. The only instrument that can decide this
# is the invoice's PER-PROJECT CACHED-INPUT LINE ITEM, compared against
# sum(cachedContentTokenCount). That is the procedure this project
# pre-registered in work/probes/stats.json W4_RECONCILIATION, and it is still
# the procedure. A month-TOTAL comparison against the AI Studio dashboard was
# tried on 2026-08-30 and CANNOT discriminate, for two independent reasons:
#
#   the direction came out backwards. Re-priced with this package's own billing
#   code, the 2026-08 ledger totals $2.845 at $0.075/M and $2.689 at $0.0375/M
#   against a dashboard month-to-date of $3.04 -- gaps of -$0.195 and -$0.351.
#   The HIGHER rate is the CLOSER one, by $0.156, which is to within rounding
#   the rate delta itself (4,155,072 cached batch tokens x $0.0375/M = $0.156):
#   the delta had been mistaken for a residual.
#   the comparison is underdetermined anyway. reports/*usage*.jsonl is not the
#   complete record of the project's August spend -- the wave-1/wave-2 probe
#   calls live only in work/probes/calls.jsonl and were never ingested. That
#   term is uncertain between ~$0.33 (contemporaneous report) and ~$0.80
#   (re-priced), which is larger than the $0.156 signal under test.
#
# THE OTHER HALF OF THE QUESTION IS SETTLED, and the same dashboard reading is
# what settles it. Batch responses come back reporting serviceTier STANDARD,
# which looked like evidence that the batch discount was not being applied at
# all. Priced undiscounted this ledger would total about $5.68 against a
# dashboard of $3.04, so the discount plainly IS applied and the reported tier
# is a label on the response rather than a statement about the bill. Do not
# "fix" a batch bill to standard rates on the strength of that field.
CACHED_INPUT_OPEN_QUESTION = (
    "$0.0375/M (pricing page, batch row -- the better documentary claim) vs "
    "$0.075/M (batch guide: \"the standard context caching rates\"). Quoting "
    "the higher one until the INVOICE's per-project cached-input line settles "
    "it; see stats.json W4_RECONCILIATION for the pre-registered comparison. A "
    "month-total check against the AI Studio dashboard was tried on 2026-08-30 "
    "and does not discriminate (the ledger is $2.845 at 0.075 and $2.689 at "
    "0.0375 against $3.04, so the higher rate is the closer one, and an "
    "un-ingested probe wave leaves a gap larger than the delta). SETTLED by "
    "that same reading: the batch discount IS applied even though responses "
    "report serviceTier STANDARD -- undiscounted the month would be ~$5.68.")


def priced_models() -> tuple:
    """Every model this module will quote. The allow-list, as data."""
    return tuple(sorted(_CARDS))


def is_priced(model: str) -> bool:
    return model in _CARDS


def _window(model: str, on_date: datetime.date) -> dict:
    for win in _CARDS[model]["windows"]:
        if win["from"] is not None and on_date < win["from"]:
            continue
        if win["until"] is not None and on_date > win["until"]:
            continue
        return win
    raise FatalError(
        "no price window covers %s for model %r. The card has windows %s; a "
        "date outside all of them means the page was read again and this table "
        "was not updated."
        % (on_date.isoformat(), model,
           ", ".join(w["label"] for w in _CARDS[model]["windows"])))


def rate_card(model: str, mode: str, on_date=None) -> dict:
    """Dollars per million tokens for (model, mode) on a date.

    This is the signature stage 42 already calls (`rate_card(model, mode)`), so
    the bill's three dollar figures light up as soon as this module exists. The
    date defaults to TODAY on purpose: the price of a wave is the price on the
    day the wave runs, and the 2027 window is a real change rather than a
    formatting detail. Pass `on_date` to quote another day (a wave that will
    drain in January, or a test).

    Returns the three keys the bill needs plus the provenance that makes them
    auditable in reports/translate_bill_<lang>.json:

        input_usd_per_mtok           uncached input
        cached_input_usd_per_mtok    the cache discount line
        output_usd_per_mtok          output, INCLUDING thinking tokens: they are
                                     billed as output, which is why the ledger
                                     adds them to candidates and not to prompt
        cache_storage_usd_per_mtok_hour
        model / mode / effective_on / window / page_read_at
        cached_input_unsettled       the one number the sources disagree on

    Refuses (FatalError) rather than quoting a model whose card nobody read, or
    a transport that is not one of the three.
    """
    if model not in _CARDS:
        raise FatalError(
            "no rate card for model %r. Priced models are: %s. A price is a "
            "fact about a page on a day -- read the pricing page, add the "
            "model to prices.py with the date, and only then quote it. "
            "(Allow-listed for constants: %s.)"
            % (model, ", ".join(priced_models()),
               ", ".join(sorted(VERIFIED_MODELS))))
    if mode not in MODES:
        raise FatalError("no rate card for mode %r; the transports are %s"
                         % (mode, ", ".join(MODES)))
    on = on_date or datetime.date.today()
    if isinstance(on, datetime.datetime):
        on = on.date()
    win = _window(model, on)
    rates = win["modes"][mode]
    return {
        "input_usd_per_mtok": rates["input"],
        "cached_input_usd_per_mtok": rates["cached_input"],
        "output_usd_per_mtok": rates["output"],
        "cache_storage_usd_per_mtok_hour": _CACHE_STORAGE_USD_PER_MTOK_HOUR,
        "model": model,
        "mode": mode,
        "effective_on": on.isoformat(),
        "window": win["label"],
        "source": win["source"],
        "page_read_at": _CARDS[model]["read_at"],
        "cached_input_unsettled": CACHED_INPUT_OPEN_QUESTION,
    }


def cache_storage_usd(tokens: int, hours: float, model: str, mode: str,
                      on_date=None) -> float:
    """What holding an explicit cache costs while a wave drains.

    Small but not zero, and it is the one cost that grows with the WALL CLOCK
    rather than with the work: a 1,135-token definition prompt held for a
    24-hour batch drain is $0.0136, and four of them is $0.055. Against a
    $2.45 wave that is small; against the $0.22 of headroom four languages
    leave under a $10 cap it is not nothing. Which is the whole argument for
    computing it instead of calling it negligible.
    """
    card = rate_card(model, mode, on_date=on_date)
    return round(tokens * hours
                 * card["cache_storage_usd_per_mtok_hour"] / 1e6, 8)
