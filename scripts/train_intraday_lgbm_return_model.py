#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_intraday_lgbm_return_model.py

收益—风险感知的分钟级 ETF 日内模型。

相对上一版 train_intraday_lgbm_direction_model.py 的变化：
1. 不再只预测 future_ret > 4bp 的二分类；
2. 新增收益回归模型：pred_ret_bp = E[future_ret_bp | features]；
3. 新增下跌风险模型：downside_prob = P(future_ret_bp < -downside_threshold_bp)；
4. 用 trade_score = pred_ret_bp - lambda * downside_prob 做交易排序；
5. 新增候选事件过滤 candidate_event，避免每一分钟都交易；
6. 新增商品池内部相对强弱特征、同一 bar_index 成交额冲击、动量加速度等短周期特征；
7. 回测同时比较：
   - no_event：不加事件过滤；
   - event：只在 candidate_event == True 的样本里交易。

默认运行：
    python ".\\train_intraday_lgbm_return_model.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --scope commodity_focus

输出目录默认：
    data_t0_2022_2024\\ml_intraday_return_commodity_focus_h15

重要说明：
    该脚本仍然是研究回测，不构成投资建议。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class ReturnModelConfig:
    horizon_bars: int = 15
    downside_threshold_bp: float = 4.0

    train_start: str = "20220101"
    train_end: str = "20231231"
    test_start: str = "20240101"
    test_end: str = "20241231"

    entry_start: str = "09:40"
    entry_end: str = "14:15"
    force_flat_time: str = "14:55"

    max_train_rows: int = 900000
    random_state: int = 42

    # portfolio
    position_weight: float = 0.20
    max_positions: int = 3
    max_total_exposure: float = 1.00
    same_category_max_open: int = 2
    etf_daily_trade_limit: int = 2
    cooldown_minutes: int = 10

    # optional path exits for trading simulation
    use_path_exit: bool = True
    take_profit_bp: float = 8.0
    stop_loss_bp: float = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train return-aware intraday LightGBM model.")

    p.add_argument("--root-dir", type=str, default="./data_t0_2022_2024")
    p.add_argument("--panel-file", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")

    p.add_argument("--scope", type=str, default="commodity_focus",
                   choices=["commodity_focus", "cross_commodity", "no_bond_money", "all"])

    p.add_argument("--horizon-bars", type=int, default=15)
    p.add_argument("--downside-threshold-bp", type=float, default=4.0)

    p.add_argument("--train-start", type=str, default="20220101")
    p.add_argument("--train-end", type=str, default="20231231")
    p.add_argument("--test-start", type=str, default="20240101")
    p.add_argument("--test-end", type=str, default="20241231")

    p.add_argument("--entry-start", type=str, default="09:40")
    p.add_argument("--entry-end", type=str, default="14:15")
    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")

    # trade thresholds
    p.add_argument("--pred-ret-thresholds", type=str, default="2,4,6,8,10,12")
    p.add_argument("--downside-prob-thresholds", type=str, default="0.25,0.30,0.35,0.40,0.50")
    p.add_argument("--score-quantiles", type=str, default="0.90,0.95,0.97,0.99")
    p.add_argument("--lambdas", type=str, default="5,8,10,15,20")

    p.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "histgb"],
                   help="默认 LightGBM；若没安装 lightgbm，自动退回 sklearn。")
    p.add_argument("--max-train-rows", type=int, default=900000)

    p.add_argument("--position-weight", type=float, default=0.20)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--same-category-max-open", type=int, default=2)
    p.add_argument("--etf-daily-trade-limit", type=int, default=2)
    p.add_argument("--cooldown-minutes", type=int, default=10)

    p.add_argument("--no-path-exit", action="store_true",
                   help="默认启用路径止盈止损；加此参数则固定持有 horizon。")
    p.add_argument("--take-profit-bp", type=float, default=8.0)
    p.add_argument("--stop-loss-bp", type=float, default=5.0)

    return p.parse_args()


def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.root_dir)
    panel_file = Path(args.panel_file) if args.panel_file else root / "processed" / "t0_intraday_bar_panel.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else root / f"ml_intraday_return_{args.scope}_h{args.horizon_bars}"
    return panel_file, out_dir


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def scope_mask(d: pd.DataFrame, scope: str) -> pd.Series:
    cat = d["t0_category"].astype(str)
    cat_cn = d["t0_category_cn"].astype(str)
    if scope == "all":
        return pd.Series(True, index=d.index)
    if scope == "no_bond_money":
        return ~(cat.isin(["bond", "money_market"]) | cat_cn.str.contains("债券|货币", regex=True, na=False))
    if scope == "commodity_focus":
        return cat.isin(["gold_commodity"]) | cat_cn.str.contains("黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    if scope == "cross_commodity":
        return cat.isin(["cross_border", "gold_commodity"]) | cat_cn.str.contains("跨境|黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    raise ValueError(scope)


def read_panel(path: Path, scope: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time", "bar_index",
        "open", "high", "low", "close", "vol", "amount",
        "intraday_vwap", "valid_entry_bar", "force_flat_bar",
        "day_open", "daily_ret",
    ]

    import pyarrow.parquet as pq
    cols = pq.read_schema(path).names
    read_cols = [c for c in needed if c in cols]
    d = pd.read_parquet(path, columns=read_cols)

    for c in needed:
        if c not in d.columns:
            if c in ["name", "t0_category", "t0_category_cn"]:
                d[c] = ""
            else:
                d[c] = np.nan

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["name"] = d["name"].fillna("").astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)

    d = d[scope_mask(d, scope)].copy()

    for c in ["open", "high", "low", "close", "vol", "amount", "intraday_vwap", "day_open", "daily_ret"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype(int)

    if d["day_open"].isna().all():
        d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")

    if d["intraday_vwap"].isna().all():
        tmp_vol = d["vol"].clip(lower=0)
        cum_vol = tmp_vol.groupby([d["ts_code"], d["trade_date"]]).cumsum()
        cum_cv = (d["close"] * tmp_vol).groupby([d["ts_code"], d["trade_date"]]).cumsum()
        d["intraday_vwap"] = cum_cv / cum_vol.replace(0, np.nan)
        d["intraday_vwap"] = d["intraday_vwap"].fillna(
            d.groupby(["ts_code", "trade_date"])["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
        )

    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def rolling_z(s: pd.Series, win: int, minp: int | None = None) -> pd.Series:
    if minp is None:
        minp = max(5, win // 2)
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd.replace(0, np.nan)


def add_daily_long_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["ret_1m_tmp"] = x.groupby(["ts_code", "trade_date"])["close"].pct_change()

    daily = x.groupby(["ts_code", "trade_date"], as_index=False).agg(
        day_amount=("amount", "sum"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_open=("open", "first"),
        day_close=("close", "last"),
        rv_1m_day=("ret_1m_tmp", "std"),
    )
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["day_ret"] = daily.groupby("ts_code")["day_close"].pct_change()
    daily["day_range"] = daily["day_high"] / daily["day_low"] - 1.0
    daily["abs_day_ret"] = daily["day_ret"].abs()
    daily["trendiness"] = daily["abs_day_ret"] / daily["day_range"].replace(0, np.nan)

    g = daily.groupby("ts_code", group_keys=False)
    for win in [5, 10, 20, 30]:
        daily[f"hist_amount_med_{win}d"] = g["day_amount"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_rv_med_{win}d"] = g["rv_1m_day"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_range_med_{win}d"] = g["day_range"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_trend_med_{win}d"] = g["trendiness"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())

    keep = ["ts_code", "trade_date"] + [c for c in daily.columns if c.startswith("hist_")]
    return x.merge(daily[keep], on=["ts_code", "trade_date"], how="left").drop(columns=["ret_1m_tmp"])


def add_features_and_targets(d: pd.DataFrame, cfg: ReturnModelConfig) -> Tuple[pd.DataFrame, List[str]]:
    x = d.copy().sort_values(["ts_code", "trade_date", "trade_time"]).reset_index(drop=True)
    gb = x.groupby(["ts_code", "trade_date"], group_keys=False)

    x["log_close"] = np.log(x["close"].replace(0, np.nan))
    x["ret_1m"] = gb["close"].pct_change(1)

    # lagged 1m returns: raw short sequence
    for lag in range(1, 21):
        x[f"ret_lag_{lag}"] = gb["ret_1m"].shift(lag - 1)

    # momentum windows
    for n in [2, 3, 5, 10, 15, 20, 30]:
        x[f"ret_{n}m"] = gb["close"].pct_change(n)
        x[f"ret_{n}m_bp"] = x[f"ret_{n}m"] * 10000.0
        x[f"logret_{n}m"] = gb["log_close"].diff(n)

    # acceleration / curvature
    x["mom_accel_3_10"] = x["ret_3m"] - x["ret_10m"]
    x["mom_accel_5_20"] = x["ret_5m"] - x["ret_20m"]
    x["mom_turn_3"] = x["ret_3m"] - gb["ret_3m"].shift(3)
    x["mom_turn_5"] = x["ret_5m"] - gb["ret_5m"].shift(5)
    x["slope_5"] = x["logret_5m"] / 5.0
    x["slope_10"] = x["logret_10m"] / 10.0
    x["slope_20"] = x["logret_20m"] / 20.0
    x["curvature_5_20"] = x["slope_5"] - x["slope_20"]

    # volatility / range
    for n in [5, 10, 20, 30]:
        x[f"rv_{n}m"] = gb["ret_1m"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).std())
        x[f"range_{n}m"] = (
            gb["high"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).max())
            / gb["low"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).min())
            - 1.0
        )
    x["rv_ratio_5_20"] = x["rv_5m"] / x["rv_20m"].replace(0, np.nan)
    x["rv_ratio_10_30"] = x["rv_10m"] / x["rv_30m"].replace(0, np.nan)
    x["range_ratio_5_20"] = x["range_5m"] / x["range_20m"].replace(0, np.nan)

    # amount features: rolling and same-time bar_index normalisation
    for n in [5, 10, 20, 30]:
        x[f"amount_sum_{n}m"] = gb["amount"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).sum())
        x[f"amount_z_{n}m"] = gb["amount"].transform(lambda s, n=n: rolling_z(s.fillna(0), n))

    # same bar_index amount shock based only on previous days
    x = x.sort_values(["ts_code", "bar_index", "trade_date"]).reset_index(drop=True)
    for win in [10, 20, 30]:
        med = x.groupby(["ts_code", "bar_index"], group_keys=False)["amount"].transform(
            lambda s, win=win: s.shift(1).rolling(win, min_periods=max(5, win // 2)).median()
        )
        x[f"amount_same_time_ratio_{win}d"] = x["amount"] / med.replace(0, np.nan)
        x[f"amount_same_time_logratio_{win}d"] = np.log(x[f"amount_same_time_ratio_{win}d"].replace(0, np.nan))
    x = x.sort_values(["ts_code", "trade_date", "trade_time"]).reset_index(drop=True)
    gb = x.groupby(["ts_code", "trade_date"], group_keys=False)

    x["amount_acc_5_20"] = x["amount_z_5m"] - x["amount_z_20m"]
    x["amount_acc_10_30"] = x["amount_z_10m"] - x["amount_z_30m"]
    x["ret_amount_confirm_5m"] = np.sign(x["ret_5m"].fillna(0)) * x["amount_z_5m"]
    x["ret_amount_confirm_10m"] = np.sign(x["ret_10m"].fillna(0)) * x["amount_z_10m"]
    x["price_volume_confirm_5m"] = x["ret_5m"] * x["amount_z_10m"]
    x["price_volume_confirm_10m"] = x["ret_10m"] * x["amount_z_20m"]

    # VWAP
    x["vwap_gap"] = x["close"] / x["intraday_vwap"] - 1.0
    x["vwap_gap_bp"] = x["vwap_gap"] * 10000.0
    for n in [1, 3, 5, 10, 20]:
        x[f"vwap_gap_chg_{n}m"] = gb["vwap_gap"].diff(n)
        x[f"vwap_slope_{n}m"] = gb["intraday_vwap"].pct_change(n)

    # price position / breakout / reversal
    for n in [5, 10, 20, 30, 60]:
        roll_high = gb["high"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).max())
        roll_low = gb["low"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).min())
        prev_high = gb["high"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=max(3, n // 2)).max())
        prev_low = gb["low"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=max(3, n // 2)).min())
        x[f"pos_{n}m"] = ((x["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)).clip(0, 1)
        x[f"dist_high_{n}m"] = x["close"] / roll_high.replace(0, np.nan) - 1.0
        x[f"dist_low_{n}m"] = x["close"] / roll_low.replace(0, np.nan) - 1.0
        x[f"break_high_{n}m"] = (x["close"] > prev_high).astype(int)
        x[f"break_low_{n}m"] = (x["close"] < prev_low).astype(int)

    x["intraday_high_so_far"] = gb["high"].cummax()
    x["intraday_low_so_far"] = gb["low"].cummin()
    x["dist_intraday_high"] = x["close"] / x["intraday_high_so_far"].replace(0, np.nan) - 1.0
    x["rebound_from_intraday_low"] = x["close"] / x["intraday_low_so_far"].replace(0, np.nan) - 1.0
    x["intraday_ret_from_open"] = x["close"] / x["day_open"] - 1.0

    # candle structure
    bar_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["range_1m_bp"] = (x["high"] / x["low"].replace(0, np.nan) - 1.0) * 10000.0
    x["body_ratio"] = (x["close"] - x["open"]) / bar_range
    x["abs_body_ratio"] = (x["close"] - x["open"]).abs() / bar_range
    x["close_location"] = (x["close"] - x["low"]) / bar_range
    x["upper_shadow_ratio"] = (x["high"] - x[["open", "close"]].max(axis=1)) / bar_range
    x["lower_shadow_ratio"] = (x[["open", "close"]].min(axis=1) - x["low"]) / bar_range
    x["is_green"] = (x["close"] > x["open"]).astype(int)
    x["green_count_3"] = gb["is_green"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    x["green_count_5"] = gb["is_green"].transform(lambda s: s.rolling(5, min_periods=1).sum())

    for n in [5, 10, 20]:
        x[f"new_high_count_{n}m"] = gb[f"break_high_{n}m"].transform(lambda s, n=n: s.rolling(n, min_periods=1).sum())

    # time
    x["minutes"] = pd.to_datetime(x["trade_time"]).dt.hour * 60 + pd.to_datetime(x["trade_time"]).dt.minute
    x["minutes_since_open"] = np.where(
        x["clock_time"] <= "11:30",
        x["minutes"] - (9 * 60 + 30),
        x["minutes"] - (13 * 60) + 121,
    )
    x["time_sin"] = np.sin(2 * np.pi * x["minutes_since_open"] / 241.0)
    x["time_cos"] = np.cos(2 * np.pi * x["minutes_since_open"] / 241.0)

    # cross-sectional relative strength within current scope/category at same timestamp
    # Use current timestamp information only; it is observable at bar close.
    for n in [3, 5, 10, 20]:
        ret_col = f"ret_{n}m"
        group = x.groupby(["trade_time", "t0_category"], group_keys=False)
        mean_ret = group[ret_col].transform("mean")
        cnt = group[ret_col].transform("count")
        x[f"cat_mean_ret_{n}m"] = mean_ret
        x[f"rel_ret_{n}m"] = x[ret_col] - mean_ret
        x[f"rank_ret_{n}m"] = group[ret_col].rank(pct=True)
        # leave-one-out approximate mean
        x[f"loo_cat_mean_ret_{n}m"] = (mean_ret * cnt - x[ret_col]) / (cnt - 1).replace(0, np.nan)

    group = x.groupby(["trade_time", "t0_category"], group_keys=False)
    x["rank_amount_z_10m"] = group["amount_z_10m"].rank(pct=True)
    x["rank_vwap_gap"] = group["vwap_gap"].rank(pct=True)
    x["leader_ret_5m"] = group["ret_5m"].transform("max")
    x["leader_ret_10m"] = group["ret_10m"].transform("max")
    x["laggard_ret_5m"] = group["ret_5m"].transform("min")
    x["laggard_ret_10m"] = group["ret_10m"].transform("min")

    # categorical
    x["ts_code_code"] = pd.Categorical(x["ts_code"]).codes
    x["category_code"] = pd.Categorical(x["t0_category"]).codes

    # future target: signal at bar t, next bar open entry, horizon path
    h = cfg.horizon_bars
    x["entry_open_next"] = gb["open"].shift(-1)
    x["future_close_h"] = gb["close"].shift(-(h + 1))
    x["future_time_h"] = gb["trade_time"].shift(-(h + 1))
    x["future_clock_h"] = gb["clock_time"].shift(-(h + 1))
    x["future_ret"] = x["future_close_h"] / x["entry_open_next"] - 1.0
    x["future_ret_bp"] = x["future_ret"] * 10000.0
    x["label_downside"] = (x["future_ret_bp"] < -cfg.downside_threshold_bp).astype(int)

    # path high/low after entry, used for optional path exit and diagnostics
    # A loop over groups is safer than many shifts for high/low path.
    path_high = np.full(len(x), np.nan, dtype=float)
    path_low = np.full(len(x), np.nan, dtype=float)
    path_tp_hit = np.zeros(len(x), dtype=int)
    path_sl_hit = np.zeros(len(x), dtype=int)
    path_exit_ret_bp = np.full(len(x), np.nan, dtype=float)
    path_exit_reason = np.array([""] * len(x), dtype=object)

    tp = cfg.take_profit_bp / 10000.0
    sl = cfg.stop_loss_bp / 10000.0

    for _, idx in x.groupby(["ts_code", "trade_date"], sort=False).groups.items():
        idx_arr = np.asarray(list(idx))
        opens = x.loc[idx_arr, "open"].to_numpy(dtype=float)
        highs = x.loc[idx_arr, "high"].to_numpy(dtype=float)
        lows = x.loc[idx_arr, "low"].to_numpy(dtype=float)
        closes = x.loc[idx_arr, "close"].to_numpy(dtype=float)
        clocks = x.loc[idx_arr, "clock_time"].astype(str).to_numpy()

        m = len(idx_arr)
        for local_i in range(m):
            entry_i = local_i + 1
            end_i = local_i + h + 1
            if entry_i >= m or end_i >= m:
                continue
            entry_price = opens[entry_i]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            # skip if horizon exits after force flat
            if clocks[end_i] >= cfg.force_flat_time:
                continue

            window_hi = highs[entry_i:end_i + 1]
            window_lo = lows[entry_i:end_i + 1]
            path_high[idx_arr[local_i]] = np.nanmax(window_hi) / entry_price - 1.0
            path_low[idx_arr[local_i]] = np.nanmin(window_lo) / entry_price - 1.0

            exit_ret = closes[end_i] / entry_price - 1.0
            reason = "time"

            if cfg.use_path_exit:
                # Conservative: if stop and take profit same bar, stop first.
                for j in range(entry_i, end_i + 1):
                    if lows[j] <= entry_price * (1.0 - sl):
                        exit_ret = -sl
                        reason = "stop"
                        path_sl_hit[idx_arr[local_i]] = 1
                        break
                    if highs[j] >= entry_price * (1.0 + tp):
                        exit_ret = tp
                        reason = "take_profit"
                        path_tp_hit[idx_arr[local_i]] = 1
                        break

            path_exit_ret_bp[idx_arr[local_i]] = exit_ret * 10000.0
            path_exit_reason[idx_arr[local_i]] = reason

    x["path_high_bp"] = path_high * 10000.0
    x["path_low_bp"] = path_low * 10000.0
    x["path_tp_hit"] = path_tp_hit
    x["path_sl_hit"] = path_sl_hit
    x["path_exit_ret_bp"] = path_exit_ret_bp
    x["path_exit_reason"] = path_exit_reason

    # candidate events
    x["candidate_trend"] = (
        (x["amount_same_time_ratio_20d"].fillna(0) > 1.0)
        & (x["ret_3m_bp"].abs().fillna(0) > 2.0)
        & (
            (x["break_high_20m"] == 1)
            | (x["vwap_slope_10m"].fillna(0) > 0)
            | (x["rank_ret_5m"].fillna(0) > 0.70)
        )
    ).astype(int)

    x["candidate_reclaim"] = (
        (x["amount_same_time_ratio_20d"].fillna(0) > 1.0)
        & (x["vwap_gap_chg_5m"].fillna(0) > 0)
        & (x["close"] > x["intraday_vwap"])
        & (x["lower_shadow_ratio"].fillna(0) > 0.20)
    ).astype(int)

    x["candidate_event"] = ((x["candidate_trend"] == 1) | (x["candidate_reclaim"] == 1)).astype(int)

    valid = (
        (x["valid_entry_bar"].fillna(0).astype(int) == 1)
        & (x["clock_time"] >= cfg.entry_start)
        & (x["clock_time"] <= cfg.entry_end)
        & (x["amount"].fillna(0) > 0)
        & (x["entry_open_next"].fillna(0) > 0)
        & (x["future_close_h"].fillna(0) > 0)
        & (x["future_clock_h"].astype(str) < cfg.force_flat_time)
        & x["future_ret_bp"].notna()
    )
    x["ml_valid_row"] = valid.astype(int)

    feature_cols = [
        "ts_code_code", "category_code", "bar_index", "minutes_since_open", "time_sin", "time_cos",

        # raw lag sequence
        *[f"ret_lag_{i}" for i in range(1, 21)],

        # returns / acceleration
        "ret_1m", "ret_2m", "ret_3m", "ret_5m", "ret_10m", "ret_15m", "ret_20m", "ret_30m",
        "ret_1m_bp", "ret_2m_bp", "ret_3m_bp", "ret_5m_bp", "ret_10m_bp", "ret_15m_bp", "ret_20m_bp", "ret_30m_bp",
        "mom_accel_3_10", "mom_accel_5_20", "mom_turn_3", "mom_turn_5",
        "slope_5", "slope_10", "slope_20", "curvature_5_20",

        # volatility
        "rv_5m", "rv_10m", "rv_20m", "rv_30m", "rv_ratio_5_20", "rv_ratio_10_30",
        "range_1m_bp", "range_5m", "range_10m", "range_20m", "range_30m", "range_ratio_5_20",

        # amount
        "amount_z_5m", "amount_z_10m", "amount_z_20m", "amount_z_30m",
        "amount_same_time_ratio_10d", "amount_same_time_ratio_20d", "amount_same_time_ratio_30d",
        "amount_same_time_logratio_10d", "amount_same_time_logratio_20d", "amount_same_time_logratio_30d",
        "amount_acc_5_20", "amount_acc_10_30",
        "ret_amount_confirm_5m", "ret_amount_confirm_10m",
        "price_volume_confirm_5m", "price_volume_confirm_10m",

        # VWAP
        "vwap_gap", "vwap_gap_bp",
        "vwap_gap_chg_1m", "vwap_gap_chg_3m", "vwap_gap_chg_5m", "vwap_gap_chg_10m", "vwap_gap_chg_20m",
        "vwap_slope_1m", "vwap_slope_3m", "vwap_slope_5m", "vwap_slope_10m", "vwap_slope_20m",

        # price position
        "pos_5m", "pos_10m", "pos_20m", "pos_30m", "pos_60m",
        "dist_high_5m", "dist_high_10m", "dist_high_20m", "dist_high_30m", "dist_high_60m",
        "dist_low_5m", "dist_low_10m", "dist_low_20m", "dist_low_30m", "dist_low_60m",
        "break_high_5m", "break_high_10m", "break_high_20m", "break_high_30m", "break_high_60m",
        "break_low_5m", "break_low_10m", "break_low_20m", "break_low_30m", "break_low_60m",
        "dist_intraday_high", "rebound_from_intraday_low", "intraday_ret_from_open",

        # candle
        "body_ratio", "abs_body_ratio", "close_location", "upper_shadow_ratio", "lower_shadow_ratio",
        "green_count_3", "green_count_5",
        "new_high_count_5m", "new_high_count_10m", "new_high_count_20m",

        # relative strength
        "cat_mean_ret_3m", "cat_mean_ret_5m", "cat_mean_ret_10m", "cat_mean_ret_20m",
        "rel_ret_3m", "rel_ret_5m", "rel_ret_10m", "rel_ret_20m",
        "rank_ret_3m", "rank_ret_5m", "rank_ret_10m", "rank_ret_20m",
        "loo_cat_mean_ret_3m", "loo_cat_mean_ret_5m", "loo_cat_mean_ret_10m", "loo_cat_mean_ret_20m",
        "rank_amount_z_10m", "rank_vwap_gap", "leader_ret_5m", "leader_ret_10m", "laggard_ret_5m", "laggard_ret_10m",

        # candidate flags as features
        "candidate_trend", "candidate_reclaim", "candidate_event",

        # long state
        "hist_amount_med_5d", "hist_amount_med_10d", "hist_amount_med_20d", "hist_amount_med_30d",
        "hist_rv_med_5d", "hist_rv_med_10d", "hist_rv_med_20d", "hist_rv_med_30d",
        "hist_range_med_5d", "hist_range_med_10d", "hist_range_med_20d", "hist_range_med_30d",
        "hist_trend_med_5d", "hist_trend_med_10d", "hist_trend_med_20d", "hist_trend_med_30d",
    ]

    feature_cols = [c for c in feature_cols if c in x.columns]
    return x, feature_cols


def clean_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def sample_train_data(train: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train
    return train.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def train_return_model(train: pd.DataFrame, feature_cols: List[str], cfg: ReturnModelConfig, model_name: str):
    train_s = sample_train_data(train, cfg.max_train_rows, cfg.random_state)
    X = clean_matrix(train_s, feature_cols)
    y = train_s["future_ret_bp"].astype("float32")

    if model_name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                objective="huber",
                alpha=0.85,
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=8.0,
                min_child_samples=100,
                random_state=cfg.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X, y)
            return model, "lightgbm_regressor_huber", train_s
        except Exception as e:
            print(f"[WARN] LGBMRegressor 失败，回退 sklearn。原因：{e}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=cfg.random_state,
    )
    model.fit(X, y)
    return model, "hist_gradient_boosting_regressor", train_s


def train_downside_model(train: pd.DataFrame, feature_cols: List[str], cfg: ReturnModelConfig, model_name: str):
    train_s = sample_train_data(train, cfg.max_train_rows, cfg.random_state)
    X = clean_matrix(train_s, feature_cols)
    y = train_s["label_downside"].astype(int)

    pos = int(y.sum())
    neg = int((1 - y).sum())
    scale_pos_weight = neg / max(pos, 1)

    if model_name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective="binary",
                n_estimators=550,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=8.0,
                min_child_samples=100,
                scale_pos_weight=scale_pos_weight,
                random_state=cfg.random_state + 7,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X, y)
            return model, "lightgbm_downside_classifier", train_s
        except Exception as e:
            print(f"[WARN] LGBMClassifier 失败，回退 sklearn。原因：{e}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    sample_weight = np.where(y == 1, scale_pos_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=cfg.random_state + 7,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, "hist_gradient_boosting_downside_classifier", train_s


def predict_downside_prob(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = clean_matrix(df, feature_cols)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def model_metrics(df: pd.DataFrame) -> Dict:
    out = {
        "rows": int(len(df)),
        "mean_future_ret_bp": float(df["future_ret_bp"].mean()),
        "median_future_ret_bp": float(df["future_ret_bp"].median()),
        "downside_rate": float(df["label_downside"].mean()),
        "pred_ret_corr": np.nan,
        "downside_auc": np.nan,
        "downside_average_precision": np.nan,
    }
    if len(df) > 5:
        out["pred_ret_corr"] = float(pd.Series(df["pred_ret_bp"]).corr(pd.Series(df["future_ret_bp"])))
    if df["label_downside"].nunique() > 1:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            out["downside_auc"] = float(roc_auc_score(df["label_downside"], df["downside_prob"]))
            out["downside_average_precision"] = float(average_precision_score(df["label_downside"], df["downside_prob"]))
        except Exception:
            pass
    return out


def decile_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    x = df.copy()
    try:
        x["score_bin"] = pd.qcut(x[score_col], q=10, duplicates="drop")
    except Exception:
        x["score_bin"] = "all"

    return (
        x.groupby("score_bin", dropna=False)
        .agg(
            rows=("future_ret_bp", "size"),
            mean_score=(score_col, "mean"),
            mean_pred_ret_bp=("pred_ret_bp", "mean"),
            mean_downside_prob=("downside_prob", "mean"),
            mean_future_ret_bp=("future_ret_bp", "mean"),
            median_future_ret_bp=("future_ret_bp", "median"),
            win_rate=("future_ret_bp", lambda s: (s > 0).mean()),
            downside_rate=("label_downside", "mean"),
            p25_bp=("future_ret_bp", lambda s: s.quantile(0.25)),
            p75_bp=("future_ret_bp", lambda s: s.quantile(0.75)),
            candidate_rate=("candidate_event", "mean"),
        )
        .reset_index()
    )


def add_trade_scores(df: pd.DataFrame, lambdas: List[float]) -> pd.DataFrame:
    x = df.copy()
    for lam in lambdas:
        name = f"trade_score_lam{int(lam) if float(lam).is_integer() else str(lam).replace('.', '_')}"
        x[name] = x["pred_ret_bp"] - lam * x["downside_prob"]
    return x


def make_trades(test: pd.DataFrame, score_col: str, score_threshold: float,
                pred_ret_min: float, downside_max: float, use_event_filter: bool,
                cfg: ReturnModelConfig) -> pd.DataFrame:
    m = (
        (test[score_col] >= score_threshold)
        & (test["pred_ret_bp"] >= pred_ret_min)
        & (test["downside_prob"] <= downside_max)
    )
    if use_event_filter:
        m &= (test["candidate_event"] == 1)

    sig = test[m].copy()
    if sig.empty:
        return pd.DataFrame()

    sig = sig.sort_values(["trade_time", score_col], ascending=[True, False]).reset_index(drop=True)

    # Use fixed future return or path exit return depending on config.
    if cfg.use_path_exit and "path_exit_ret_bp" in sig.columns:
        gross_ret = sig["path_exit_ret_bp"] / 10000.0
        exit_reason = sig["path_exit_reason"]
    else:
        gross_ret = sig["future_ret_bp"] / 10000.0
        exit_reason = "time"

    trades = pd.DataFrame({
        "ts_code": sig["ts_code"].values,
        "name": sig["name"].values,
        "t0_category": sig["t0_category"].values,
        "t0_category_cn": sig["t0_category_cn"].values,
        "trade_date": sig["trade_date"].values,
        "signal_time": sig["trade_time"].values,
        "signal_clock_time": sig["clock_time"].values,
        "entry_time": sig["trade_time"] + pd.Timedelta(minutes=1),
        "exit_time": sig["future_time_h"].values,
        "entry_price": sig["entry_open_next"].values,
        "exit_price": sig["future_close_h"].values,
        "gross_ret": gross_ret.values,
        "future_ret_bp": sig["future_ret_bp"].values,
        "pred_ret_bp": sig["pred_ret_bp"].values,
        "downside_prob": sig["downside_prob"].values,
        "trade_score": sig[score_col].values,
        "score_col": score_col,
        "score_threshold": score_threshold,
        "pred_ret_min": pred_ret_min,
        "downside_max": downside_max,
        "use_event_filter": use_event_filter,
        "candidate_event": sig["candidate_event"].values,
        "candidate_trend": sig["candidate_trend"].values,
        "candidate_reclaim": sig["candidate_reclaim"].values,
        "exit_reason": exit_reason.values if hasattr(exit_reason, "values") else exit_reason,
        "weight": cfg.position_weight,
        "holding_bars": cfg.horizon_bars,
    })

    trades = trades.dropna(subset=["entry_price", "exit_price", "gross_ret", "exit_time"])
    trades = trades[(trades["entry_price"] > 0) & (trades["exit_price"] > 0)].copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    return apply_portfolio_constraints(trades, cfg)


def apply_portfolio_constraints(trades: pd.DataFrame, cfg: ReturnModelConfig) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.sort_values(["entry_time", "trade_score"], ascending=[True, False]).reset_index(drop=True)
    open_pos = []
    cooldown_until: Dict[str, pd.Timestamp] = {}
    etf_day_count: Dict[Tuple[str, str], int] = {}
    accepted = []

    for _, row in c.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_t = pd.Timestamp(row["exit_time"])
        code = row["ts_code"]
        date = row["trade_date"]
        cat = row["t0_category"]
        weight = float(row["weight"])

        open_pos = [p for p in open_pos if p["exit_time"] > entry]
        current_exposure = sum(p["weight"] for p in open_pos)

        key = (code, date)
        if etf_day_count.get(key, 0) >= cfg.etf_daily_trade_limit:
            continue
        if code in cooldown_until and entry < cooldown_until[code]:
            continue
        if any(p["ts_code"] == code for p in open_pos):
            continue
        if len(open_pos) >= cfg.max_positions:
            continue
        if sum(1 for p in open_pos if p["t0_category"] == cat) >= cfg.same_category_max_open:
            continue
        if current_exposure + weight > cfg.max_total_exposure:
            continue

        accepted.append(row.to_dict())
        open_pos.append({"ts_code": code, "t0_category": cat, "exit_time": exit_t, "weight": weight})
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=cfg.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1

    return pd.DataFrame(accepted)


def profit_factor(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str]) -> Tuple[Dict, pd.DataFrame]:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = t["weight"] * t["net_ret"]
        daily = t.groupby("trade_date", as_index=False)["pnl"].sum().rename(columns={"pnl": "daily_ret"})
        daily = pd.DataFrame({"trade_date": dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = (1.0 + daily["daily_ret"]).cumprod()

    n = len(daily)
    final_nav = float(daily["nav"].iloc[-1]) if n else 1.0
    ann_return = final_nav ** (252.0 / n) - 1.0 if n and final_nav > 0 else np.nan
    ann_vol = float(daily["daily_ret"].std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
    dd = daily["nav"] / daily["nav"].cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    if trades.empty:
        s = {
            "cost_bp": cost_bp,
            "trade_count": 0,
            "final_nav": final_nav,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": np.nan,
            "avg_gross_ret": np.nan,
            "avg_net_ret": np.nan,
            "profit_factor": np.nan,
            "avg_pred_ret_bp": np.nan,
            "avg_downside_prob": np.nan,
            "avg_trade_score": np.nan,
        }
    else:
        net = trades["gross_ret"] - 2.0 * cost_bp / 10000.0
        s = {
            "cost_bp": cost_bp,
            "trade_count": int(len(trades)),
            "final_nav": final_nav,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": float((net > 0).mean()),
            "avg_gross_ret": float(trades["gross_ret"].mean()),
            "avg_net_ret": float(net.mean()),
            "profit_factor": profit_factor(net),
            "avg_pred_ret_bp": float(trades["pred_ret_bp"].mean()),
            "avg_downside_prob": float(trades["downside_prob"].mean()),
            "avg_trade_score": float(trades["trade_score"].mean()),
        }
    return s, daily


def group_summary(trades: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(group_cols, dropna=False)
        .agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            win_rate=("gross_ret", lambda s: (s > 0).mean()),
            profit_factor=("gross_ret", profit_factor),
            avg_pred_ret_bp=("pred_ret_bp", "mean"),
            avg_downside_prob=("downside_prob", "mean"),
            avg_trade_score=("trade_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_gross_ret", ascending=False)
    )


def feature_importance(model, feature_cols: List[str], label: str) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": feature_cols, f"importance_{label}": model.feature_importances_})
    return pd.DataFrame({"feature": feature_cols, f"importance_{label}": np.nan})


def write_report(out_dir: Path, config: Dict, metric_df: pd.DataFrame,
                 decile_tables: Dict[str, pd.DataFrame], bt_summary: pd.DataFrame,
                 best_etf: pd.DataFrame, best_exit: pd.DataFrame, fi: pd.DataFrame) -> None:
    lines = []
    lines.append("# Intraday Return-Aware LightGBM Model Report\n")
    lines.append("## 1. 目的\n")
    lines.append("本脚本从二分类方向预测改为收益—风险感知模型：回归 future_ret_bp，同时预测 downside_prob，并用 trade_score = pred_ret_bp - lambda * downside_prob 进行交易筛选。")
    lines.append("")
    lines.append("## 2. 配置\n")
    lines.append(pd.DataFrame([config]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 模型预测能力\n")
    lines.append(metric_df.to_markdown(index=False))
    lines.append("")
    for name, tab in decile_tables.items():
        lines.append(f"## 4.{name} 分层表现")
        lines.append(tab.to_markdown(index=False))
        lines.append("")
    lines.append("## 5. 回测汇总")
    show_cols = [
        "cost_bp", "final_nav", "trade_count", "avg_gross_ret", "avg_net_ret", "win_rate",
        "profit_factor", "score_col", "score_quantile", "pred_ret_min", "downside_max", "use_event_filter"
    ]
    show_cols = [c for c in show_cols if c in bt_summary.columns]
    lines.append(bt_summary[show_cols].head(100).to_markdown(index=False))
    lines.append("")
    if not best_etf.empty:
        lines.append("## 6. 最优参数组合下 ETF 归因")
        lines.append(best_etf.to_markdown(index=False))
        lines.append("")
    if not best_exit.empty:
        lines.append("## 7. 最优参数组合下退出归因")
        lines.append(best_exit.to_markdown(index=False))
        lines.append("")
    if not fi.empty:
        lines.append("## 8. 特征重要性 Top 60")
        lines.append(fi.head(60).to_markdown(index=False))
        lines.append("")
    lines.append("## 9. 读数标准")
    lines.append("- 重点看 2bp 成本后的 final_nav、avg_net_ret、trade_count。")
    lines.append("- 如果 event 过滤后的 top score 桶平均收益明显提升，说明候选事件机制有效。")
    lines.append("- 如果 pred_ret_bp 分层和实际 future_ret_bp 没有单调关系，说明回归模型仍然没有抓住可交易收益。")
    (out_dir / "ml_return_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    panel_file, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReturnModelConfig(
        horizon_bars=args.horizon_bars,
        downside_threshold_bp=args.downside_threshold_bp,
        train_start=norm_date(args.train_start),
        train_end=norm_date(args.train_end),
        test_start=norm_date(args.test_start),
        test_end=norm_date(args.test_end),
        entry_start=args.entry_start,
        entry_end=args.entry_end,
        max_train_rows=args.max_train_rows,
        position_weight=args.position_weight,
        max_positions=args.max_positions,
        same_category_max_open=args.same_category_max_open,
        etf_daily_trade_limit=args.etf_daily_trade_limit,
        cooldown_minutes=args.cooldown_minutes,
        use_path_exit=not args.no_path_exit,
        take_profit_bp=args.take_profit_bp,
        stop_loss_bp=args.stop_loss_bp,
    )

    cost_bps = parse_float_list(args.cost_bps)
    pred_ret_thresholds = parse_float_list(args.pred_ret_thresholds)
    downside_prob_thresholds = parse_float_list(args.downside_prob_thresholds)
    score_quantiles = parse_float_list(args.score_quantiles)
    lambdas = parse_float_list(args.lambdas)

    print("=" * 100)
    print("Train Intraday Return-Aware Model")
    print("=" * 100)
    print(f"panel_file : {panel_file.resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"scope      : {args.scope}")
    print(f"horizon    : {cfg.horizon_bars}")
    print("=" * 100)

    print("Reading panel...")
    panel = read_panel(panel_file, args.scope)
    print(f"panel rows after scope: {len(panel):,}, ETFs={panel['ts_code'].nunique()}, days={panel['trade_date'].nunique()}")

    print("Adding daily long-state features...")
    panel = add_daily_long_features(panel)

    print("Adding short-term, relative-strength features, targets...")
    data, feature_cols = add_features_and_targets(panel, cfg)
    data = data[data["ml_valid_row"] == 1].copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    train = data[(data["trade_date"] >= cfg.train_start) & (data["trade_date"] <= cfg.train_end)].copy()
    test = data[(data["trade_date"] >= cfg.test_start) & (data["trade_date"] <= cfg.test_end)].copy()

    if train.empty or test.empty:
        raise RuntimeError(f"训练集或测试集为空：train={len(train)}, test={len(test)}")

    print(f"valid rows: train={len(train):,}, test={len(test):,}")
    print(f"train mean ret={train['future_ret_bp'].mean():.3f}bp, downside={train['label_downside'].mean():.3f}")
    print(f"test mean ret={test['future_ret_bp'].mean():.3f}bp, downside={test['label_downside'].mean():.3f}")
    print(f"feature count={len(feature_cols)}")

    print("Training return model...")
    ret_model, ret_model_name, train_ret_sample = train_return_model(train, feature_cols, cfg, args.model)
    print(f"return model: {ret_model_name}, sample rows={len(train_ret_sample):,}")

    print("Training downside model...")
    down_model, down_model_name, train_down_sample = train_downside_model(train, feature_cols, cfg, args.model)
    print(f"downside model: {down_model_name}, sample rows={len(train_down_sample):,}")

    print("Predicting...")
    train_eval = sample_train_data(train, min(len(train), 300000), cfg.random_state).copy()
    test = test.copy()

    train_eval["pred_ret_bp"] = ret_model.predict(clean_matrix(train_eval, feature_cols))
    train_eval["downside_prob"] = predict_downside_prob(down_model, train_eval, feature_cols)

    test["pred_ret_bp"] = ret_model.predict(clean_matrix(test, feature_cols))
    test["downside_prob"] = predict_downside_prob(down_model, test, feature_cols)

    # clip extreme predicted returns to reduce silly thresholds in reports
    train_eval["pred_ret_bp"] = train_eval["pred_ret_bp"].clip(-50, 50)
    test["pred_ret_bp"] = test["pred_ret_bp"].clip(-50, 50)

    # add base scores
    test = add_trade_scores(test, lambdas)
    train_eval = add_trade_scores(train_eval, lambdas)

    metric_rows = []
    for name, df in [("train_eval_sample", train_eval), ("test", test)]:
        m = model_metrics(df)
        m["period"] = name
        metric_rows.append(m)
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "return_model_prediction_metrics.csv", index=False, encoding="utf-8-sig")

    # deciles
    decile_tables = {
        "_pred_ret": decile_table(test, "pred_ret_bp"),
        "_downside_prob_ascending": decile_table(test.assign(neg_downside=-test["downside_prob"]), "neg_downside"),
    }
    # main score decile for lambda=10 if exists
    main_lam = 10.0 if 10.0 in lambdas else lambdas[0]
    main_score_col = f"trade_score_lam{int(main_lam) if float(main_lam).is_integer() else str(main_lam).replace('.', '_')}"
    decile_tables[f"_{main_score_col}"] = decile_table(test, main_score_col)

    for name, tab in decile_tables.items():
        tab.to_csv(out_dir / f"decile_table{name}.csv", index=False, encoding="utf-8-sig")

    # Compact predictions
    compact_cols = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_date", "trade_time", "clock_time",
        "candidate_event", "candidate_trend", "candidate_reclaim",
        "entry_open_next", "future_close_h", "future_time_h",
        "future_ret_bp", "path_exit_ret_bp", "path_exit_reason",
        "label_downside", "pred_ret_bp", "downside_prob",
    ] + [c for c in test.columns if c.startswith("trade_score_lam")]
    test[compact_cols].to_csv(out_dir / "return_model_test_predictions_compact.csv", index=False, encoding="utf-8-sig")

    # feature importance
    fi_ret = feature_importance(ret_model, feature_cols, "ret")
    fi_down = feature_importance(down_model, feature_cols, "downside")
    fi = fi_ret.merge(fi_down, on="feature", how="outer")
    if "importance_ret" in fi.columns:
        fi["importance_total"] = fi[["importance_ret", "importance_downside"]].fillna(0).sum(axis=1)
        fi = fi.sort_values("importance_total", ascending=False)
    fi.to_csv(out_dir / "return_model_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("Backtesting thresholds...")
    test_dates = sorted(test["trade_date"].unique().tolist())
    bt_rows = []
    all_trades_parts = []

    for lam in lambdas:
        score_col = f"trade_score_lam{int(lam) if float(lam).is_integer() else str(lam).replace('.', '_')}"
        for q in score_quantiles:
            score_threshold = float(test[score_col].quantile(q))
            for pred_min in pred_ret_thresholds:
                for down_max in downside_prob_thresholds:
                    for use_event in [False, True]:
                        trades = make_trades(
                            test=test,
                            score_col=score_col,
                            score_threshold=score_threshold,
                            pred_ret_min=pred_min,
                            downside_max=down_max,
                            use_event_filter=use_event,
                            cfg=cfg,
                        )
                        if not trades.empty:
                            trades["score_quantile"] = q
                            trades["lambda"] = lam
                            all_trades_parts.append(trades)

                        for cost in cost_bps:
                            s, nav = performance(trades, cost, test_dates)
                            s.update({
                                "score_col": score_col,
                                "lambda": lam,
                                "score_quantile": q,
                                "score_threshold": score_threshold,
                                "pred_ret_min": pred_min,
                                "downside_max": down_max,
                                "use_event_filter": use_event,
                            })
                            bt_rows.append(s)

    bt_summary = pd.DataFrame(bt_rows)
    bt_summary = bt_summary.sort_values(["cost_bp", "final_nav"], ascending=[True, False]).reset_index(drop=True)
    bt_summary.to_csv(out_dir / "return_model_backtest_summary.csv", index=False, encoding="utf-8-sig")

    all_trades = pd.concat(all_trades_parts, ignore_index=True) if all_trades_parts else pd.DataFrame()
    # Keep all trades file can be large. Save only if not huge.
    if len(all_trades) <= 500000:
        all_trades.to_csv(out_dir / "return_model_all_threshold_trades.csv", index=False, encoding="utf-8-sig")

    # select exploratory best at 2bp
    main_cost = 2.0 if 2.0 in cost_bps else cost_bps[0]
    best = bt_summary[bt_summary["cost_bp"] == main_cost].sort_values("final_nav", ascending=False).iloc[0].to_dict()

    best_trades = make_trades(
        test=test,
        score_col=best["score_col"],
        score_threshold=float(best["score_threshold"]),
        pred_ret_min=float(best["pred_ret_min"]),
        downside_max=float(best["downside_max"]),
        use_event_filter=bool(best["use_event_filter"]),
        cfg=cfg,
    )
    best_trades.to_csv(out_dir / "return_model_best_trades.csv", index=False, encoding="utf-8-sig")

    etf_sum = group_summary(best_trades, ["ts_code", "name", "t0_category_cn"])
    exit_sum = group_summary(best_trades, ["exit_reason"])
    etf_sum.to_csv(out_dir / "return_model_best_etf_summary.csv", index=False, encoding="utf-8-sig")
    exit_sum.to_csv(out_dir / "return_model_best_exit_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel_file": str(panel_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "scope": args.scope,
        "return_model": ret_model_name,
        "downside_model": down_model_name,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_mean_future_ret_bp": float(train["future_ret_bp"].mean()),
        "test_mean_future_ret_bp": float(test["future_ret_bp"].mean()),
        "train_downside_rate": float(train["label_downside"].mean()),
        "test_downside_rate": float(test["label_downside"].mean()),
        "cost_bps": cost_bps,
        "pred_ret_thresholds": pred_ret_thresholds,
        "downside_prob_thresholds": downside_prob_thresholds,
        "score_quantiles": score_quantiles,
        "lambdas": lambdas,
        "best_at_2bp_exploratory": best,
        "config": asdict(cfg),
    }
    (out_dir / "return_model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(out_dir, config, metric_df, decile_tables, bt_summary, etf_sum, exit_sum, fi)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"report  : {(out_dir / 'ml_return_model_report.md').resolve()}")
    print(f"summary : {(out_dir / 'return_model_backtest_summary.csv').resolve()}")
    print(f"best    : {(out_dir / 'return_model_best_trades.csv').resolve()}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        print("\n[FATAL] 程序异常退出：", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
