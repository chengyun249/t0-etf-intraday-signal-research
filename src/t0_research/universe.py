"""Point-in-time dynamic universe construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_dynamic_universe(
    panel: pd.DataFrame,
    *,
    code_col: str = "ts_code",
    date_col: str = "trade_date",
    amount_col: str = "amount",
    lookback_days: int = 20,
    min_history_days: int = 20,
    min_median_daily_amount: float = 0.0,
) -> pd.DataFrame:
    """Return daily eligibility using only information before each date."""

    if lookback_days <= 0 or min_history_days < 0 or min_median_daily_amount < 0:
        raise ValueError("invalid dynamic-universe parameters")
    needed = {code_col, date_col, amount_col}
    if missing := needed.difference(panel.columns):
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    daily = (
        panel.assign(**{date_col: panel[date_col].astype(str)})
        .groupby([code_col, date_col], as_index=False)[amount_col]
        .sum(min_count=1)
        .rename(columns={amount_col: "daily_amount"})
        .sort_values([code_col, date_col])
    )
    daily["prior_history_days"] = daily.groupby(code_col).cumcount()
    daily["trailing_median_daily_amount"] = daily.groupby(code_col)["daily_amount"].transform(
        lambda x: x.shift(1).rolling(lookback_days, min_periods=min_history_days).median()
    )
    daily["universe_eligible"] = (
        (daily["prior_history_days"] >= min_history_days)
        & (daily["trailing_median_daily_amount"] >= min_median_daily_amount)
    )
    daily["universe_reason"] = np.select(
        [
            daily["prior_history_days"] < min_history_days,
            daily["trailing_median_daily_amount"] < min_median_daily_amount,
        ],
        ["insufficient_history", "insufficient_liquidity"],
        default="eligible",
    )
    return daily
