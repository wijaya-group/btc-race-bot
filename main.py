"""Trend-rider bot for the BTC race (agent-fun/bars-replay/v1, Python 3.13, stdlib only).

Idea: one asset, one position, long only. The bot only rides a trend that has
already proved itself, and it gets off fast when the trend breaks. Three
independent filters must agree before it buys, so sideways chop and one-bar
spikes never trigger a trade:

  1. alignment - a fast exponential average sits above a slow one;
  2. regime - the latest close sits above a long slow average;
  3. push - the return over the last hour clears a minimum size, and the
     return over the last two hours is positive too, so a one-hour bounce
     inside a falling market never counts as a trend.
  4. breakout - the close is the highest of the last hour: the bot buys a
     move that is actually making new ground, never a bounce back toward a
     high the market already made and rejected.
  5. four-hour proof - the return over the last four hours clears one
     percent: a rally that only exists inside the last hour is a spike,
     and spikes are where this market hands money back.

While it holds, the bot does nothing at all unless the trend genuinely
breaks: the fast average falls back under the slow one while the hourly
push has turned clearly negative, or the two-hour push itself has turned
clearly negative (a slow bleed the hourly push is too slow to catch), or
the position has lost four percent from what it cost. There is no
trailing stop: one-minute prices jump around far too much for a tight
stop, and every stop-out in the chop is a round trip of fees the trend
then has to earn back. After an exit the bot waits out a cooldown, so a
whipsaw cannot make it buy straight back into the same falling knife. It never trades on the first bars (the averages
need history), never reads the clock, the environment or any randomness, and
every decision is a pure function of the closes it has been sent, so two runs
on the same bars answer byte for byte alike.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- tuning knobs (all in bars / basis points) -------------------------------
FAST_SPAN = 5          # fast exponential average, in bars
SLOW_SPAN = 13         # slow exponential average, in bars
TREND_SPAN = 60        # long average that defines the regime, in bars
MOM_BARS = 60          # bars the entry push is measured over
MOM_MIN_BPS = 80       # smallest 60-bar push worth buying, in bps
LONG_BARS = 120        # bars the longer push is measured over
LONG_MIN_BPS = 50      # smallest 120-bar push worth buying, in bps
PROOF_BARS = 240       # bars the four-hour proof is measured over
PROOF_MIN_BPS = 150    # smallest 240-bar push worth buying, in bps
BRAKE_BPS = -150       # 120-bar push below which a held trend is called broken, in bps
NEW_HIGH_BARS = 60     # bars the breakout high is measured over
NO_ENTRY_AFTER = 1200  # last batch the bot may open a position on
HARD_STOP_BPS = 400    # worst loss on one position before bailing, in bps
EXIT_MOM_BPS = -150    # 60-bar push below which a held trend is called dead, in bps
COOLDOWN_BARS = 60     # bars to sit out after an exit before buying again
MAX_SLIP_BPS = 200     # widest fill move from the latest close the bot accepts

_BPS = 10_000

# --- running state -----------------------------------------------------------
_closes = []           # closes seen so far, as floats
_fast = None           # fast average value (None until seeded)
_slow = None           # slow average value (None until seeded)
_fast_seed = []        # closes collected while the fast average seeds
_slow_seed = []        # closes collected while the slow average seeds
_cooldown = 0          # batches left before the bot may buy again


def reset():
    """Clear every scrap of running state (fresh run, and unit tests)."""
    global _closes, _fast, _slow, _fast_seed, _slow_seed, _cooldown
    _closes = []
    _fast = None
    _slow = None
    _fast_seed = []
    _slow_seed = []
    _cooldown = 0


def _step_average(value, seed, span, close):
    """Advance one exponential average: seed it with the mean of its first
    span of closes, then move 2 / (span + 1) of the way to each new close."""
    if value is not None:
        return value + (2.0 / (span + 1)) * (close - value), seed
    seed.append(close)
    if len(seed) == span:
        return sum(seed) / span, seed
    return None, seed


def _long_average(closes, span):
    """Plain mean of the last `span` closes, or None before that many exist."""
    if len(closes) < span:
        return None
    return sum(closes[-span:]) / span


def _push_bps(closes, bars):
    """Return over the last `bars` bars, in basis points, or None."""
    if len(closes) < bars + 1:
        return None
    return (closes[-1] / closes[-1 - bars] - 1.0) * _BPS


def _is_new_high(closes, bars):
    """True when the latest close beats every close of the `bars` before it."""
    if len(closes) < bars + 1:
        return False
    return closes[-1] > max(closes[-bars - 1:-1])


def on_batch(state, batch):
    """Decide this batch's orders. `state` is the SDK's BarsState, current
    before this batch; `batch` is the harness batch. Returns a list of order
    dicts (empty when the bot sits out)."""
    global _fast, _slow, _fast_seed, _slow_seed, _cooldown
    if state.warming_up:
        return []
    bar = None
    for b in batch["bars"]:
        if b["symbol"] == SYMBOL:
            bar = b
            break
    if bar is None:
        return []
    close = float(bar["c"])
    _closes.append(close)
    _fast, _fast_seed = _step_average(_fast, _fast_seed, FAST_SPAN, close)
    _slow, _slow_seed = _step_average(_slow, _slow_seed, SLOW_SPAN, close)
    if _cooldown > 0:
        _cooldown -= 1

    held = state.positions.get(SYMBOL)
    if held is not None:
        # Bail only on a real break: the averages have crossed back down
        # while the hourly push is clearly negative, the two-hour push
        # itself has turned clearly negative (a slow bleed), or the
        # position has lost four percent from what it cost. Dips and
        # chop are ridden out.
        worth = state.value(SYMBOL, held.qty)
        aligned = _fast is not None and _slow is not None and _fast >= _slow
        push = _push_bps(_closes, MOM_BARS)
        long_push = _push_bps(_closes, LONG_BARS)
        broken = (not aligned) and push is not None and push < EXIT_MOM_BPS
        bleeding = long_push is not None and long_push < BRAKE_BPS
        if broken or bleeding or worth * _BPS <= held.cost * (_BPS - HARD_STOP_BPS):
            _cooldown = COOLDOWN_BARS
            return [{"side": "sell", "symbol": SYMBOL, "qty": held.qty,
                     "max_slippage_bps": MAX_SLIP_BPS}]
        return []

    trend = _long_average(_closes, TREND_SPAN)
    push = _push_bps(_closes, MOM_BARS)
    long_push = _push_bps(_closes, LONG_BARS)
    proof = _push_bps(_closes, PROOF_BARS)
    if (
        _fast is not None
        and _slow is not None
        and _fast > _slow
        and trend is not None
        and close > trend
        and push is not None
        and push > MOM_MIN_BPS
        and long_push is not None
        and long_push > LONG_MIN_BPS
        and proof is not None
        and proof > PROOF_MIN_BPS
        and _is_new_high(_closes, NEW_HIGH_BARS)
        and _cooldown == 0
        and state.seq <= NO_ENTRY_AFTER
    ):
        room = state.weight_room(SYMBOL)
        amount = state.cash if state.cash < room else room
        if amount >= int(state.hello["fees"]["min_notional_quote_units"]):
            return [{"side": "buy", "symbol": SYMBOL,
                     "notional_quote_units": amount,
                     "max_slippage_bps": MAX_SLIP_BPS}]
    return []


if __name__ == "__main__":
    makefun_bars.run(on_batch)
