import numpy as np
import pandas as pd

from t0_research.execution import simulate_long_exit
from t0_research.universe import build_dynamic_universe
from t0_research.validation import PurgedDateSplit, clustered_mean_lower_bound


def test_stop_uses_known_vwap_and_gap_open():
    bars = pd.DataFrame(
        {
            "open": [100.0, 97.0],
            "high": [101.0, 98.0],
            "low": [99.0, 96.0],
            "close": [100.5, 97.5],
            "known_vwap": [100.0, 100.0],
            "force_flat_bar": [0, 0],
        }
    )
    out = simulate_long_exit(
        bars,
        entry_pos=0,
        max_exit_pos=1,
        entry_price=100.0,
        take_profit_price=110.0,
        hard_stop_price=98.0,
        trailing_stop=0.05,
        vwap_stop_band=0.02,
        opening_range_stop_price=0.0,
    )
    assert out.reason == "stop_gap"
    assert out.price == 97.0


def test_dynamic_universe_never_uses_same_day_liquidity():
    panel = pd.DataFrame(
        {
            "ts_code": ["A"] * 3,
            "trade_date": ["1", "2", "3"],
            "amount": [10.0, 10.0, 1000.0],
        }
    )
    out = build_dynamic_universe(panel, lookback_days=2, min_history_days=2, min_median_daily_amount=100)
    assert not out.loc[out["trade_date"] == "3", "universe_eligible"].iloc[0]


def test_purged_split_separates_date_blocks():
    dates = pd.Series(np.repeat([f"d{i:03}" for i in range(100)], 2))
    splitter = PurgedDateSplit(min_train_days=40, test_days=20, purge_days=2, embargo_days=1)
    train_idx, test_idx = next(splitter.split(dates))
    train_dates = set(dates.iloc[train_idx])
    test_dates = set(dates.iloc[test_idx])
    assert train_dates.isdisjoint(test_dates)
    assert max(train_dates) < min(test_dates)


def test_clustered_lower_bound_counts_days_not_trades():
    values = pd.Series([10.0, 10.0, -2.0, -2.0])
    days = pd.Series(["d1", "d1", "d2", "d2"])
    mean, lower, n_days = clustered_mean_lower_bound(values, days)
    assert mean == 4.0
    assert n_days == 2
    assert lower < mean
