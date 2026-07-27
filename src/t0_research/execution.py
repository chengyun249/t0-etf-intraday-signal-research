"""Conservative OHLC-bar execution rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitResult:
    bar_position: int
    price: float
    reason: str


def simulate_long_exit(
    bars: pd.DataFrame,
    *,
    entry_pos: int,
    max_exit_pos: int,
    entry_price: float,
    take_profit_price: float,
    hard_stop_price: float,
    trailing_stop: float,
    vwap_stop_band: float,
    opening_range_stop_price: float,
    stop_slippage_bp: float = 1.0,
    known_vwap_col: str = "known_vwap",
    force_flat_col: str = "force_flat_bar",
) -> ExitResult:
    """Simulate a long exit without using current-bar close-derived VWAP.

    ``known_vwap_col`` must be information known at the bar open (normally the
    previous bar's cumulative VWAP). A gap through a stop fills at the adverse
    bar open. An intrabar stop fills below the trigger by ``stop_slippage_bp``.
    If both stop and target are touched in one OHLC bar, the stop is assumed to
    occur first. The trailing high-water mark is updated only after the bar
    survives, avoiding an unknowable high/low path assumption.
    """

    if not 0 <= entry_pos <= max_exit_pos < len(bars):
        raise ValueError("invalid entry/exit positions")
    if entry_price <= 0 or stop_slippage_bp < 0:
        raise ValueError("entry price must be positive and slippage non-negative")

    high_water = float(entry_price)
    fallback = float(bars.iloc[max_exit_pos]["close"])
    result = ExitResult(max_exit_pos, fallback, "max_hold")
    adverse_multiplier = 1.0 - stop_slippage_bp / 10000.0

    for pos in range(entry_pos, max_exit_pos + 1):
        row = bars.iloc[pos]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if not all(np.isfinite(x) and x > 0 for x in [open_price, high, low, close]):
            continue

        if int(row.get(force_flat_col, 0)) == 1:
            return ExitResult(pos, close, "force_flat")

        known_vwap = row.get(known_vwap_col, np.nan)
        known_vwap = float(known_vwap) if pd.notna(known_vwap) else np.nan
        trail_price = high_water * (1.0 - trailing_stop)
        vwap_price = (
            known_vwap * (1.0 - vwap_stop_band) if np.isfinite(known_vwap) else -np.inf
        )
        stop_price = max(hard_stop_price, trail_price, vwap_price, opening_range_stop_price)

        if open_price <= stop_price:
            return ExitResult(pos, open_price, "stop_gap")
        if low <= stop_price:
            return ExitResult(pos, stop_price * adverse_multiplier, "stop_intrabar")
        if open_price >= take_profit_price:
            return ExitResult(pos, open_price, "take_profit_gap")
        if high >= take_profit_price:
            return ExitResult(pos, take_profit_price, "take_profit")

        high_water = max(high_water, high)

    return result
