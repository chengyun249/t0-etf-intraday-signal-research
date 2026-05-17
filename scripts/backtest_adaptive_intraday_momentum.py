#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_adaptive_intraday_momentum.py

自适应日内动量策略：
1. 长期状态层：每个交易日前，只用过去 rolling N 日数据判断 ETF 是否值得交易，并动态决定阈值/仓位。
2. 短期触发层：每根 bar 收盘后，用当天已发生的 bar 计算短期动量、成交放大、VWAP 斜率、价格位置。
3. 动态退出层：入场后使用动态 stop、VWAP stop、trailing、score decay、max_hold、尾盘强平。

默认读取：
    ./data_t0_2022_2024/processed/t0_intraday_bar_panel.parquet

运行示例：
    python ".\\backtest_adaptive_intraday_momentum.py" --root-dir ".\\data_t0_2022_2024"
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class AdaptiveParams:
    long_lookback_days: int = 30
    min_long_obs: int = 20
    min_long_score: float = 0.50
    min_hist_amount_quantile: float = 0.20

    entry_start: str = "09:45"
    entry_end: str = "11:30"
    force_flat_time: str = "14:55"

    ret_fast_n: int = 5
    ret_slow_n: int = 10
    vol_fast_n: int = 10
    vol_slow_n: int = 20
    amount_z_n: int = 20
    price_pos_n: int = 20
    vwap_slope_n: int = 10

    base_short_threshold: float = 1.10
    min_short_threshold: float = 0.75
    max_short_threshold: float = 1.60
    k_vwap_band: float = 0.80
    k_recent_high_band: float = 0.20
    min_expected_edge_bp: float = 4.0
    max_vwap_gap_bp: float = 120.0

    w_ret_fast: float = 0.25
    w_ret_slow: float = 0.25
    w_amount: float = 0.20
    w_vwap_slope: float = 0.15
    w_price_pos: float = 0.10
    w_bar_strength: float = 0.05

    min_hold_bars: int = 3
    max_hold_bars: int = 60
    exit_score_threshold: float = -0.20
    hard_stop_mult: float = 2.50
    trailing_mult: float = 2.00
    vwap_stop_mult: float = 1.00
    min_stop_bp: float = 12.0
    max_stop_bp: float = 45.0
    min_trailing_bp: float = 10.0
    max_trailing_bp: float = 60.0

    base_weight: float = 0.20
    max_weight: float = 0.50
    min_weight: float = 0.05
    max_positions: int = 3
    max_total_exposure: float = 1.00
    same_category_max_open: int = 2
    etf_daily_trade_limit: int = 2
    cooldown_minutes: int = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive intraday momentum backtest.")
    p.add_argument("--root-dir", type=str, default="./data_t0_2022_2024")
    p.add_argument("--panel-file", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--start-date", type=str, default="")
    p.add_argument("--end-date", type=str, default="")
    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")
    p.add_argument("--scope", type=str, default="all",
                   choices=["all", "commodity_focus", "cross_commodity", "no_bond_money"])

    p.add_argument("--long-lookback-days", type=int, default=30)
    p.add_argument("--min-long-score", type=float, default=0.50)
    p.add_argument("--entry-start", type=str, default="09:45")
    p.add_argument("--entry-end", type=str, default="11:30")
    p.add_argument("--base-short-threshold", type=float, default=1.10)
    p.add_argument("--max-hold-bars", type=int, default=60)
    p.add_argument("--base-weight", type=float, default=0.20)
    p.add_argument("--max-weight", type=float, default=0.50)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--same-category-max-open", type=int, default=2)
    return p.parse_args()


def norm_date(x) -> str:
    if x is None or str(x).strip() == "":
        return ""
    return str(x).replace("-", "")[:8]


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.root_dir)
    panel_file = Path(args.panel_file) if args.panel_file else root / "processed" / "t0_intraday_bar_panel.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else root / "backtest_adaptive_intraday_momentum"
    return panel_file, out_dir


def read_panel(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "vol", "amount",
        "intraday_vwap", "bar_index",
        "valid_entry_bar", "force_flat_bar",
        "day_open", "daily_close", "daily_ret",
    ]

    import pyarrow.parquet as pq
    cols = pq.read_schema(path).names
    read_cols = [c for c in needed if c in cols]
    d = pd.read_parquet(path, columns=read_cols)

    for c in needed:
        if c not in d.columns:
            d[c] = "" if c in ["name", "t0_category", "t0_category_cn"] else np.nan

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["name"] = d["name"].fillna("").astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)

    if start_date:
        d = d[d["trade_date"] >= start_date].copy()
    if end_date:
        d = d[d["trade_date"] <= end_date].copy()

    for c in ["open", "high", "low", "close", "vol", "amount", "intraday_vwap", "day_open", "daily_close", "daily_ret"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype(int)

    if d["day_open"].isna().all():
        d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")

    if d["intraday_vwap"].isna().all():
        vol = d["vol"].clip(lower=0)
        cum_vol = vol.groupby([d["ts_code"], d["trade_date"]]).cumsum()
        cum_cv = (d["close"] * vol).groupby([d["ts_code"], d["trade_date"]]).cumsum()
        d["intraday_vwap"] = cum_cv / cum_vol.replace(0, np.nan)
        d["intraday_vwap"] = d["intraday_vwap"].fillna(
            d.groupby(["ts_code", "trade_date"])["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
        )

    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


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


def build_daily_state(panel: pd.DataFrame, p: AdaptiveParams) -> pd.DataFrame:
    x = panel.copy()
    x["ret_1m_tmp"] = x.groupby(["ts_code", "trade_date"])["close"].pct_change()

    daily = x.groupby(["ts_code", "name", "t0_category", "t0_category_cn", "trade_date"], as_index=False).agg(
        day_amount=("amount", "sum"),
        day_vol=("vol", "sum"),
        day_open=("open", "first"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
        realised_vol_1m=("ret_1m_tmp", "std"),
        rows=("trade_time", "count"),
    )

    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["daily_ret"] = daily.groupby("ts_code")["day_close"].pct_change()
    daily["day_range"] = daily["day_high"] / daily["day_low"] - 1.0
    daily["abs_daily_ret"] = daily["daily_ret"].abs()
    daily["trendiness"] = daily["abs_daily_ret"] / daily["day_range"].replace(0, np.nan)

    g = daily.groupby("ts_code", group_keys=False)
    daily["hist_amount_med"] = g["day_amount"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=p.min_long_obs).median())
    daily["hist_vol_med"] = g["realised_vol_1m"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=p.min_long_obs).median())
    daily["hist_range_med"] = g["day_range"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=p.min_long_obs).median())
    daily["hist_absret_med"] = g["abs_daily_ret"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=p.min_long_obs).median())
    daily["hist_trendiness_med"] = g["trendiness"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=p.min_long_obs).median())
    daily["hist_obs"] = g["day_amount"].transform(lambda s: s.shift(1).rolling(p.long_lookback_days, min_periods=1).count())

    for col, out in [
        ("hist_amount_med", "rank_liquidity"),
        ("hist_vol_med", "rank_vol"),
        ("hist_range_med", "rank_range"),
        ("hist_absret_med", "rank_absret"),
        ("hist_trendiness_med", "rank_trendiness"),
    ]:
        daily[out] = daily.groupby("trade_date")[col].rank(pct=True)

    daily["long_score"] = (
        0.25 * daily["rank_liquidity"].fillna(0)
        + 0.25 * daily["rank_vol"].fillna(0)
        + 0.20 * daily["rank_range"].fillna(0)
        + 0.15 * daily["rank_absret"].fillna(0)
        + 0.15 * daily["rank_trendiness"].fillna(0)
    )

    daily["hist_amount_cs_rank"] = daily.groupby("trade_date")["hist_amount_med"].rank(pct=True)
    daily["long_tradeable"] = (
        (daily["hist_obs"] >= p.min_long_obs)
        & (daily["long_score"] >= p.min_long_score)
        & (daily["hist_amount_cs_rank"].fillna(0) >= p.min_hist_amount_quantile)
        & (daily["hist_vol_med"].fillna(0) > 0)
        & (daily["rows"] >= 200)
    ).astype(int)

    keep = [
        "ts_code", "trade_date",
        "hist_amount_med", "hist_vol_med", "hist_range_med", "hist_absret_med", "hist_trendiness_med",
        "hist_obs", "rank_liquidity", "rank_vol", "rank_range", "rank_absret", "rank_trendiness",
        "long_score", "long_tradeable",
    ]
    return daily[keep]


def add_intraday_adaptive_features(panel: pd.DataFrame, daily_state: pd.DataFrame, p: AdaptiveParams) -> pd.DataFrame:
    x = panel.merge(daily_state, on=["ts_code", "trade_date"], how="left")
    x = x.sort_values(["ts_code", "trade_date", "trade_time"]).reset_index(drop=True)
    gb = x.groupby(["ts_code", "trade_date"], group_keys=False)

    x["ret_1m"] = gb["close"].pct_change()
    x[f"ret_{p.ret_fast_n}m"] = gb["close"].pct_change(p.ret_fast_n)
    x[f"ret_{p.ret_slow_n}m"] = gb["close"].pct_change(p.ret_slow_n)

    x[f"short_vol_{p.vol_fast_n}m"] = gb["ret_1m"].transform(lambda s: s.rolling(p.vol_fast_n, min_periods=max(3, p.vol_fast_n // 2)).std())
    x[f"short_vol_{p.vol_slow_n}m"] = gb["ret_1m"].transform(lambda s: s.rolling(p.vol_slow_n, min_periods=max(5, p.vol_slow_n // 2)).std())

    def rolling_z(s: pd.Series, win: int) -> pd.Series:
        m = s.rolling(win, min_periods=max(5, win // 2)).mean()
        sd = s.rolling(win, min_periods=max(5, win // 2)).std()
        return (s - m) / sd.replace(0, np.nan)

    x["amount_z"] = gb["amount"].transform(lambda s: rolling_z(s.fillna(0), p.amount_z_n)).replace([np.inf, -np.inf], np.nan).fillna(0)

    x["vwap_gap"] = x["close"] / x["intraday_vwap"] - 1.0
    x["vwap_slope"] = gb["intraday_vwap"].pct_change(p.vwap_slope_n)

    x["prev_high_n"] = gb["high"].transform(lambda s: s.shift(1).rolling(p.price_pos_n, min_periods=max(5, p.price_pos_n // 2)).max())
    x["prev_low_n"] = gb["low"].transform(lambda s: s.shift(1).rolling(p.price_pos_n, min_periods=max(5, p.price_pos_n // 2)).min())
    denom = (x["prev_high_n"] - x["prev_low_n"]).replace(0, np.nan)
    x["price_pos_n"] = ((x["close"] - x["prev_low_n"]) / denom).clip(0, 1)

    bar_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["close_location"] = ((x["close"] - x["low"]) / bar_range).clip(0, 1)
    x["candle_body"] = ((x["close"] - x["open"]) / bar_range).clip(-1, 1)
    x["bar_strength"] = 0.5 * (2 * x["close_location"] - 1) + 0.5 * x["candle_body"]

    x["vol_scale_fast"] = x[f"short_vol_{p.vol_fast_n}m"].fillna(x["hist_vol_med"])
    x["vol_scale_slow"] = x[f"short_vol_{p.vol_slow_n}m"].fillna(x["hist_vol_med"])

    eps = 1e-8
    fast_scale = (x["hist_vol_med"].fillna(x["vol_scale_fast"]) * np.sqrt(p.ret_fast_n)).replace(0, np.nan)
    slow_scale = (x["hist_vol_med"].fillna(x["vol_scale_slow"]) * np.sqrt(p.ret_slow_n)).replace(0, np.nan)
    vwap_scale = (x["hist_vol_med"].fillna(x["vol_scale_slow"]) * np.sqrt(p.vwap_slope_n)).replace(0, np.nan)

    x["ret_fast_score"] = (x[f"ret_{p.ret_fast_n}m"] / (fast_scale + eps)).clip(-4, 4)
    x["ret_slow_score"] = (x[f"ret_{p.ret_slow_n}m"] / (slow_scale + eps)).clip(-4, 4)
    x["amount_score"] = x["amount_z"].clip(-3, 3)
    x["vwap_slope_score"] = (x["vwap_slope"] / (vwap_scale + eps)).clip(-4, 4)
    x["price_pos_score"] = (2.0 * x["price_pos_n"] - 1.0).fillna(0)
    x["bar_strength_score"] = x["bar_strength"].fillna(0)

    x["short_score"] = (
        p.w_ret_fast * x["ret_fast_score"].fillna(0)
        + p.w_ret_slow * x["ret_slow_score"].fillna(0)
        + p.w_amount * x["amount_score"].fillna(0)
        + p.w_vwap_slope * x["vwap_slope_score"].fillna(0)
        + p.w_price_pos * x["price_pos_score"].fillna(0)
        + p.w_bar_strength * x["bar_strength_score"].fillna(0)
    )

    x["adaptive_threshold"] = (
        p.base_short_threshold - 0.55 * (x["long_score"].fillna(0.5) - 0.5)
    ).clip(p.min_short_threshold, p.max_short_threshold)

    vwap_band = x["intraday_vwap"] * (1.0 + p.k_vwap_band * x["vol_scale_slow"].fillna(0).clip(lower=0))
    recent_high_band = x["prev_high_n"] * (1.0 + p.k_recent_high_band * x["vol_scale_fast"].fillna(0).clip(lower=0))
    x["dynamic_upper_band"] = pd.concat([vwap_band, recent_high_band], axis=1).max(axis=1)

    x["breakout_extra_bp"] = ((x["close"] / x["dynamic_upper_band"] - 1.0).clip(lower=0) * 10000.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    x["vol_edge_bp"] = (
        np.maximum(
            x["hist_vol_med"].fillna(0) * np.sqrt(max(p.max_hold_bars, 1)),
            x["vol_scale_slow"].fillna(0) * np.sqrt(max(p.max_hold_bars, 1)),
        ) * 10000.0
    )
    x["expected_edge_bp"] = 0.65 * x["vol_edge_bp"] + 0.35 * x["breakout_extra_bp"]

    score_intensity = (x["short_score"] / x["adaptive_threshold"].replace(0, np.nan)).clip(0.5, 1.5)
    long_intensity = (0.6 + x["long_score"].fillna(0.5)).clip(0.5, 1.5)
    med_hist_vol = x["hist_vol_med"].median(skipna=True)
    vol_target_adjust = (med_hist_vol / x["hist_vol_med"].replace(0, np.nan)).clip(0.5, 1.5)
    x["raw_weight"] = p.base_weight * score_intensity * long_intensity * vol_target_adjust
    x["trade_weight"] = x["raw_weight"].clip(p.min_weight, p.max_weight).fillna(0)

    return x


def build_entry_signal(x: pd.DataFrame, p: AdaptiveParams, scope: str) -> pd.Series:
    m = (
        scope_mask(x, scope)
        & (x["long_tradeable"].fillna(0).astype(int) == 1)
        & (x["valid_entry_bar"].fillna(0).astype(int) == 1)
        & (x["clock_time"] >= p.entry_start)
        & (x["clock_time"] <= p.entry_end)
        & (x["amount"].fillna(0) > 0)
        & (x["close"].fillna(0) > 0)
        & x["hist_vol_med"].notna()
        & x["dynamic_upper_band"].notna()
        & (x["close"] > x["intraday_vwap"])
        & (x["close"] > x["dynamic_upper_band"])
        & (x["vwap_slope"].fillna(0) > 0)
        & (x["short_score"] >= x["adaptive_threshold"])
        & (x["expected_edge_bp"] >= p.min_expected_edge_bp)
        & ((x["vwap_gap"].fillna(0) * 10000.0) <= p.max_vwap_gap_bp)
    )
    prev = m.groupby([x["ts_code"], x["trade_date"]]).shift(1).fillna(False)
    return m & (~prev)


def dynamic_bp_from_vol(vol: float, mult: float, min_bp: float, max_bp: float) -> float:
    if not np.isfinite(vol) or vol <= 0:
        return min_bp
    return float(np.clip(vol * mult * 10000.0, min_bp, max_bp))


def simulate_trade(day: pd.DataFrame, signal_pos: int, p: AdaptiveParams) -> Dict:
    entry_pos = signal_pos + 1
    if entry_pos >= len(day):
        return {}

    sig = day.iloc[signal_pos]
    ent = day.iloc[entry_pos]
    if ent["clock_time"] >= p.force_flat_time:
        return {}

    entry_price = float(ent["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    entry_vol = float(sig.get("vol_scale_slow", np.nan))
    stop_bp = dynamic_bp_from_vol(entry_vol, p.hard_stop_mult, p.min_stop_bp, p.max_stop_bp)
    trail_bp = dynamic_bp_from_vol(entry_vol, p.trailing_mult, p.min_trailing_bp, p.max_trailing_bp)
    stop_price = entry_price * (1.0 - stop_bp / 10000.0)
    high_water = entry_price

    max_exit_pos = min(len(day) - 1, entry_pos + p.max_hold_bars)
    exit_pos = max_exit_pos
    exit_price = float(day.iloc[exit_pos]["close"])
    exit_reason = "max_hold"

    for j in range(entry_pos, max_exit_pos + 1):
        row = day.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        vwap = float(row["intraday_vwap"]) if np.isfinite(row["intraday_vwap"]) else np.nan
        cur_vol = float(row.get("vol_scale_slow", entry_vol))
        cur_score = float(row.get("short_score", np.nan))

        if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
            continue

        if row["clock_time"] >= p.force_flat_time or int(row.get("force_flat_bar", 0)) == 1:
            exit_pos = j
            exit_price = close
            exit_reason = "force_flat"
            break

        high_water = max(high_water, high)
        cur_trail_bp = dynamic_bp_from_vol(cur_vol, p.trailing_mult, p.min_trailing_bp, p.max_trailing_bp)
        trail_price = high_water * (1.0 - cur_trail_bp / 10000.0)

        vwap_stop_bp = dynamic_bp_from_vol(cur_vol, p.vwap_stop_mult, 2.0, 30.0)
        vwap_stop = vwap * (1.0 - vwap_stop_bp / 10000.0) if np.isfinite(vwap) else -np.inf
        effective_stop = max(stop_price, trail_price, vwap_stop)

        if low <= effective_stop:
            exit_pos = j
            exit_price = effective_stop
            if effective_stop == vwap_stop:
                exit_reason = "vwap_stop"
            elif effective_stop == trail_price:
                exit_reason = "trailing_stop"
            else:
                exit_reason = "hard_stop"
            break

        if (j - entry_pos + 1) >= p.min_hold_bars and np.isfinite(cur_score) and cur_score < p.exit_score_threshold:
            exit_pos = j
            exit_price = close
            exit_reason = "score_decay"
            break

    if not np.isfinite(exit_price) or exit_price <= 0:
        return {}

    ex = day.iloc[exit_pos]
    gross_ret = exit_price / entry_price - 1.0

    return {
        "ts_code": sig["ts_code"],
        "name": sig.get("name", ""),
        "t0_category": sig.get("t0_category", ""),
        "t0_category_cn": sig.get("t0_category_cn", ""),
        "trade_date": sig["trade_date"],
        "signal_time": sig["trade_time"],
        "signal_clock_time": sig["clock_time"],
        "entry_time": ent["trade_time"],
        "entry_clock_time": ent["clock_time"],
        "exit_time": ex["trade_time"],
        "exit_clock_time": ex["clock_time"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "weight": float(sig["trade_weight"]) if np.isfinite(sig["trade_weight"]) else p.base_weight,
        "long_score": float(sig["long_score"]) if np.isfinite(sig["long_score"]) else np.nan,
        "short_score": float(sig["short_score"]) if np.isfinite(sig["short_score"]) else np.nan,
        "adaptive_threshold": float(sig["adaptive_threshold"]) if np.isfinite(sig["adaptive_threshold"]) else np.nan,
        "expected_edge_bp": float(sig["expected_edge_bp"]) if np.isfinite(sig["expected_edge_bp"]) else np.nan,
        "vwap_gap_bp": float(sig["vwap_gap"] * 10000.0) if np.isfinite(sig["vwap_gap"]) else np.nan,
        "hist_amount_med": float(sig["hist_amount_med"]) if np.isfinite(sig["hist_amount_med"]) else np.nan,
        "hist_vol_med": float(sig["hist_vol_med"]) if np.isfinite(sig["hist_vol_med"]) else np.nan,
        "vol_scale_slow": float(sig["vol_scale_slow"]) if np.isfinite(sig["vol_scale_slow"]) else np.nan,
        "stop_bp_init": stop_bp,
        "trail_bp_init": trail_bp,
    }


def generate_candidate_trades(x: pd.DataFrame, p: AdaptiveParams, scope: str) -> pd.DataFrame:
    y = x.copy()
    y["_signal"] = build_entry_signal(y, p, scope).astype(int)

    trades = []
    groups = y.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups
    for i, (_, day) in enumerate(groups, 1):
        if i % 3000 == 0:
            print(f"  ETF-days simulated {i}/{total}")
        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        for sp in sig_pos:
            tr = simulate_trade(day, int(sp), p)
            if tr:
                trades.append(tr)
    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, p: AdaptiveParams) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.copy()
    c = c.sort_values(["entry_time", "short_score", "long_score", "expected_edge_bp"], ascending=[True, False, False, False]).reset_index(drop=True)

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

        open_pos = [pos for pos in open_pos if pos["exit_time"] > entry]
        current_exposure = sum(pos["weight"] for pos in open_pos)

        key = (code, date)
        if etf_day_count.get(key, 0) >= p.etf_daily_trade_limit:
            continue
        if code in cooldown_until and entry < cooldown_until[code]:
            continue
        if any(pos["ts_code"] == code for pos in open_pos):
            continue
        if len(open_pos) >= p.max_positions:
            continue
        if sum(1 for pos in open_pos if pos["t0_category"] == cat) >= p.same_category_max_open:
            continue
        if current_exposure + weight > p.max_total_exposure:
            remaining = p.max_total_exposure - current_exposure
            if remaining < p.min_weight:
                continue
            row = row.copy()
            row["weight"] = remaining
            weight = remaining

        accepted.append(row.to_dict())
        open_pos.append({"ts_code": code, "t0_category": cat, "exit_time": exit_t, "weight": weight})
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=p.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1

    return pd.DataFrame(accepted)


def profit_factor(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return np.nan
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
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
        summary = {
            "cost_bp": cost_bp, "trade_count": 0, "final_nav": final_nav,
            "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": max_dd, "win_rate": np.nan, "avg_gross_ret": np.nan,
            "avg_net_ret": np.nan, "profit_factor": np.nan, "avg_weight": np.nan,
            "avg_holding_bars": np.nan,
        }
    else:
        net = trades["gross_ret"] - 2.0 * cost_bp / 10000.0
        summary = {
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
            "avg_weight": float(trades["weight"].mean()),
            "avg_holding_bars": float(trades["holding_bars"].mean()),
        }
    return summary, daily


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
            avg_weight=("weight", "mean"),
            avg_holding_bars=("holding_bars", "mean"),
            avg_long_score=("long_score", "mean"),
            avg_short_score=("short_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_gross_ret", ascending=False)
    )


def write_report(out_dir: Path, p: AdaptiveParams, config: Dict, summary: pd.DataFrame,
                 etf_sum: pd.DataFrame, exit_sum: pd.DataFrame, time_sum: pd.DataFrame) -> None:
    lines = []
    lines.append("# Adaptive Intraday Momentum 回测报告\n")
    lines.append("## 1. 策略说明\n")
    lines.append("本策略使用长期状态层和短期盘中状态层共同决定开仓、仓位和退出。长期状态由当前交易日前的 rolling 历史流动性、波动、趋势性计算；短期状态由当天已发生的分钟 bar 计算。信号在 bar close 后确认，并在下一根 1min open 成交。")
    lines.append("")
    lines.append("## 2. 配置\n")
    lines.append(pd.DataFrame([config]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 参数\n")
    lines.append(pd.DataFrame([asdict(p)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not etf_sum.empty:
        lines.append("## 5. ETF 归因\n")
        lines.append(etf_sum.to_markdown(index=False))
        lines.append("")
    if not exit_sum.empty:
        lines.append("## 6. 退出原因归因\n")
        lines.append(exit_sum.to_markdown(index=False))
        lines.append("")
    if not time_sum.empty:
        lines.append("## 7. 入场时间段归因\n")
        lines.append(time_sum.to_markdown(index=False))
        lines.append("")
    lines.append("## 8. 读数标准\n")
    lines.append("- 重点看 1bp、2bp、3bp 单边成本后的 final_nav、avg_net_ret 和 profit_factor。")
    lines.append("- 如果 2bp 后仍为正，说明当前动态信号有继续细化价值。")
    lines.append("- 如果 0bp 下都很弱，说明当前公式还没有抓住有效日内边际。")
    (out_dir / "adaptive_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start_date = norm_date(args.start_date)
    end_date = norm_date(args.end_date)
    panel_file, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = AdaptiveParams(
        long_lookback_days=args.long_lookback_days,
        min_long_score=args.min_long_score,
        entry_start=args.entry_start,
        entry_end=args.entry_end,
        base_short_threshold=args.base_short_threshold,
        max_hold_bars=args.max_hold_bars,
        base_weight=args.base_weight,
        max_weight=args.max_weight,
        max_positions=args.max_positions,
        same_category_max_open=args.same_category_max_open,
    )

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]

    print("=" * 100)
    print("Adaptive Intraday Momentum Backtest")
    print("=" * 100)
    print(f"panel_file : {panel_file.resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"scope      : {args.scope}")
    print(f"date range : {start_date or 'ALL'} -> {end_date or 'ALL'}")
    print("=" * 100)

    panel = read_panel(panel_file, start_date, end_date)
    dates = sorted(panel["trade_date"].unique().tolist())

    print("Building daily long-state features...")
    daily_state = build_daily_state(panel, p)

    print("Building intraday adaptive features...")
    feat = add_intraday_adaptive_features(panel, daily_state, p)

    print("Generating candidate trades...")
    raw_trades = generate_candidate_trades(feat, p, args.scope)
    raw_trades.to_csv(out_dir / "adaptive_raw_trades_before_portfolio.csv", index=False, encoding="utf-8-sig")

    print(f"Raw trades before portfolio constraints: {len(raw_trades):,}")
    print("Applying portfolio constraints...")
    trades = apply_portfolio_constraints(raw_trades, p)
    print(f"Accepted trades: {len(trades):,}")

    trades.to_csv(out_dir / "adaptive_trades.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    nav_parts = []
    for c in cost_bps:
        s, nav = performance(trades, c, dates)
        summary_rows.append(s)
        nav["cost_bp"] = c
        nav_parts.append(nav)

    summary = pd.DataFrame(summary_rows)
    daily_nav = pd.concat(nav_parts, ignore_index=True)

    summary.to_csv(out_dir / "adaptive_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    daily_nav.to_csv(out_dir / "adaptive_daily_nav_by_cost.csv", index=False, encoding="utf-8-sig")

    etf_sum = group_summary(trades, ["ts_code", "name", "t0_category_cn"])
    exit_sum = group_summary(trades, ["exit_reason"])

    if not trades.empty:
        tr_tmp = trades.copy()
        mins = pd.to_datetime(tr_tmp["entry_time"]).dt.hour * 60 + pd.to_datetime(tr_tmp["entry_time"]).dt.minute
        tr_tmp["entry_time_bucket"] = pd.cut(
            mins,
            bins=[0, 600, 630, 690, 780, 870, 1440],
            labels=["before10", "10:00-10:30", "10:30-11:30", "13:00-14:30", "14:30-14:55", "other"],
            include_lowest=True,
        )
        time_sum = group_summary(tr_tmp, ["entry_time_bucket"])
    else:
        time_sum = pd.DataFrame()

    etf_sum.to_csv(out_dir / "adaptive_etf_summary.csv", index=False, encoding="utf-8-sig")
    exit_sum.to_csv(out_dir / "adaptive_exit_reason_summary.csv", index=False, encoding="utf-8-sig")
    time_sum.to_csv(out_dir / "adaptive_time_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel_file": str(panel_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "scope": args.scope,
        "start_date": start_date,
        "end_date": end_date,
        "trade_dates": len(dates),
        "raw_trades": int(len(raw_trades)),
        "accepted_trades": int(len(trades)),
        "cost_bps": cost_bps,
    }
    (out_dir / "adaptive_config.json").write_text(json.dumps({"config": config, "params": asdict(p)}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, p, config, summary, etf_sum, exit_sum, time_sum)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"trades  : {(out_dir / 'adaptive_trades.csv').resolve()}")
    print(f"summary : {(out_dir / 'adaptive_summary_by_cost.csv').resolve()}")
    print(f"report  : {(out_dir / 'adaptive_report.md').resolve()}")
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
