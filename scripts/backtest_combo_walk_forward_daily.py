#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
backtest_combo_walk_forward_daily.py

用途：
    对 T+0 ETF 日内组合策略做“滚动 walk-forward”回测。

核心改动：
    不再简单使用：
        2022-2023 找参数，2024 固定测试
    而是改成：
        每个交易日 d，只使用 d 之前的历史窗口来选择参数和可交易 ETF，
        然后在 d 当天盘中交易。

策略结构：
    ORB 开盘区间突破
  + Noise Boundary 噪声边界确认
  + VWAP 确认
  + 类别同步确认
  + Relative Value 过滤
  + Expected Edge 成本覆盖过滤
  + Dynamic Exit 动态退出

重要说明：
    1. 当前脚本是规则策略，不是机器学习模型。
    2. 信号本身逐分钟更新。
    3. 参数和 ETF 适用范围默认每日滚动更新；也可以改为 weekly/monthly。
    4. 所有参数选择、ETF筛选只使用当前交易日前的数据，避免未来函数。
    5. 为了速度，脚本先预生成候选参数的交易，再做 walk-forward 选择。

默认输入：
    .\data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet

默认输出：
    .\data_t0_2022_2024\backtest_combo_walk_forward_daily\

运行：
    python ".\backtest_combo_walk_forward_daily.py" `
      --panel-file ".\data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet" `
      --out-dir ".\data_t0_2022_2024\backtest_combo_walk_forward_daily"

快速测试前 N 天：
    python ".\backtest_combo_walk_forward_daily.py" `
      --panel-file ".\data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet" `
      --out-dir ".\data_t0_2022_2024\backtest_combo_walk_forward_debug" `
      --limit-trade-days 120
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. 参数结构
# ============================================================

@dataclass(frozen=True)
class ComboParams:
    opening_start: str = "09:30"
    opening_end: str = "09:45"
    entry_start: str = "09:46"
    entry_end: str = "14:15"

    relvol_lookback_days: int = 10
    min_relvol_obs: int = 5
    rel_open_amount_min: float = 1.30
    top_relvol_n: int = 10
    breakout_buffer: float = 0.0002

    noise_lookback_days: int = 14
    min_noise_obs: int = 7
    noise_k: float = 1.25
    require_noise_break: int = 1

    amount_z_min: float = 0.0
    category_strength_min: float = 0.55
    require_ret5_positive: int = 1
    require_above_vwap: int = 1
    signal_rising_edge: int = 1

    use_relative_value_filter: int = 1
    rv_z_max: float = 1.50
    rv_rank_pct_max: float = 0.80

    expected_edge_mult: float = 1.50
    assumed_roundtrip_cost_bp: float = 4.0

    take_profit: float = 0.0060
    hard_stop_loss: float = 0.0020
    trailing_stop: float = 0.0030
    vwap_stop_band: float = 0.0005
    or_high_stop_band: float = 0.0005
    max_hold_bars: int = 30

    max_positions: int = 3
    same_category_max_open: int = 1
    position_weight: float = 0.20
    cooldown_minutes: int = 15
    etf_daily_trade_limit: int = 2


# ============================================================
# 2. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward daily backtest for combo intraday ETF strategy.")

    p.add_argument("--panel-file", type=str, default="./data_t0_2022_2024/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0_2022_2024/backtest_combo_walk_forward_daily")

    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")
    p.add_argument("--main-cost-bp", type=float, default=2.0)

    p.add_argument("--scopes", type=str, default="commodity_focus",
                   help="逗号分隔：commodity_focus,cross_commodity,no_bond_money,all。默认只跑 commodity_focus，速度更快。")
    p.add_argument("--grid-size", type=str, default="mini", choices=["single", "mini", "small", "medium"])

    # walk-forward
    p.add_argument("--train-lookback-days", type=int, default=120, help="每次选择参数/ETF时向前看的交易日数。")
    p.add_argument("--min-train-days", type=int, default=60)
    p.add_argument("--min-candidate-trades", type=int, default=20)
    p.add_argument("--min-etf-trades", type=int, default=3)
    p.add_argument("--min-etf-avg-net-bp", type=float, default=-1.0)
    p.add_argument("--min-etf-profit-factor", type=float, default=0.8)
    p.add_argument("--fallback-all-if-no-etf-pass", action="store_true",
                   help="若历史窗口没有ETF通过筛选，则使用该candidate当天全部交易；默认不交易。")
    p.add_argument("--update-frequency", type=str, default="daily", choices=["daily", "weekly", "monthly"],
                   help="参数/ETF筛选更新频率。daily最贴近每日更新，但最慢。")
    p.add_argument("--limit-trade-days", type=int, default=None, help="调试用，只跑前 N 个可交易日。")

    return p.parse_args()


# ============================================================
# 3. 基础工具
# ============================================================

def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def next_minute(hhmm: str) -> str:
    hh, mm = hhmm.split(":")
    total = int(hh) * 60 + int(mm) + 1
    return f"{total // 60:02d}:{total % 60:02d}"


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def make_period_key(date_str: str, freq: str) -> str:
    dt = pd.Timestamp(date_str)
    if freq == "daily":
        return date_str
    if freq == "weekly":
        iso = dt.isocalendar()
        return f"{iso.year}-W{int(iso.week):02d}"
    if freq == "monthly":
        return dt.strftime("%Y-%m")
    raise ValueError(freq)


# ============================================================
# 4. 参数网格
# ============================================================

def make_param_grid(grid_size: str) -> List[ComboParams]:
    if grid_size == "single":
        return [ComboParams()]

    if grid_size == "mini":
        grid = {
            "opening_end": ["09:45", "10:00"],
            "rel_open_amount_min": [1.3],
            "noise_k": [1.0, 1.25, 1.5],
            "rv_z_max": [1.0, 1.5],
            "take_profit": [0.005, 0.008],
            "hard_stop_loss": [0.002, 0.003],
            "max_hold_bars": [30],
        }
    elif grid_size == "small":
        grid = {
            "opening_end": ["09:40", "09:45", "10:00"],
            "rel_open_amount_min": [1.0, 1.3, 1.6],
            "noise_k": [1.0, 1.25, 1.5],
            "rv_z_max": [1.0, 1.5, 2.0],
            "take_profit": [0.004, 0.006, 0.008],
            "hard_stop_loss": [0.0015, 0.002, 0.003],
            "max_hold_bars": [15, 30, 60],
        }
    else:  # medium
        grid = {
            "opening_end": ["09:40", "09:45", "10:00"],
            "rel_open_amount_min": [1.0, 1.3, 1.6],
            "top_relvol_n": [5, 10],
            "noise_k": [1.0, 1.25, 1.5],
            "rv_z_max": [1.0, 1.5, 2.0],
            "rv_rank_pct_max": [0.7, 0.8, 0.9],
            "take_profit": [0.004, 0.006, 0.008],
            "hard_stop_loss": [0.0015, 0.002, 0.003],
            "trailing_stop": [0.002, 0.003, 0.004],
            "max_hold_bars": [15, 30, 60],
        }

    keys = list(grid.keys())
    out: List[ComboParams] = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        opening_end = kw.get("opening_end", "09:45")
        rel_lb = kw.get("relvol_lookback_days", 10)
        noise_lb = kw.get("noise_lookback_days", 14)
        out.append(ComboParams(
            **kw,
            entry_start=next_minute(opening_end),
            min_relvol_obs=max(5, int(rel_lb) // 2),
            min_noise_obs=max(5, int(noise_lb) // 2),
        ))
    return out


# ============================================================
# 5. 读取面板
# ============================================================

def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount", "vol",
        "intraday_vwap", "bar_index", "day_open", "intraday_move", "abs_intraday_move",
        "ret_5m", "amount_z_20m",
        "valid_entry_bar", "force_flat_bar",
        "daily_adj_factor_change", "daily_is_extreme_return_day",
        "cat_positive_ratio", "cat_ret5_positive_ratio", "rv_z", "rv_rank_pct",
    ]

    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(path).names
    cols = [c for c in needed if c in schema_cols]
    d = pd.read_parquet(path, columns=cols)

    for c in needed:
        if c not in d.columns:
            if c in ["name", "t0_category", "t0_category_cn"]:
                d[c] = ""
            else:
                d[c] = np.nan

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)

    for c in ["open", "high", "low", "close", "amount", "vol", "intraday_vwap",
              "day_open", "intraday_move", "abs_intraday_move", "ret_5m",
              "amount_z_20m", "rv_z", "rv_rank_pct", "cat_positive_ratio", "cat_ret5_positive_ratio"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype("Int64")

    # 如果旧面板缺少类别同步/rv字段，兜底重算。
    if d["cat_positive_ratio"].isna().all():
        d["positive_intraday"] = (d["intraday_move"] > 0).astype(int)
        d["positive_5m"] = (d["ret_5m"] > 0).astype(int)
        d["cat_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
        d["cat_ret5_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")

    if d["rv_z"].isna().all():
        g = d.groupby(["trade_time", "t0_category"])
        mean = g["intraday_move"].transform("mean")
        std = g["intraday_move"].transform("std")
        d["rv_z"] = (d["intraday_move"] - mean) / std.replace(0, np.nan)
        d["rv_rank_pct"] = g["intraday_move"].rank(pct=True)

    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return d


# ============================================================
# 6. 特征计算：ORB / Noise / Edge
# ============================================================

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
    raise ValueError(f"未知 scope: {scope}")


def add_orb_features(panel: pd.DataFrame, p: ComboParams) -> pd.DataFrame:
    d = panel.copy()
    or_mask = (d["clock_time"] >= p.opening_start) & (d["clock_time"] <= p.opening_end)
    opening = d[or_mask].copy()

    daily = opening.groupby(["ts_code", "trade_date"], as_index=False).agg(
        or_high=("high", "max"),
        or_low=("low", "min"),
        or_open=("open", "first"),
        or_close=("close", "last"),
        or_amount=("amount", "sum"),
        or_bar_count=("clock_time", "count"),
    )

    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["or_range"] = daily["or_high"] / daily["or_low"] - 1.0
    daily["or_return"] = daily["or_close"] / daily["or_open"] - 1.0

    daily["or_amount_ma"] = (
        daily.groupby("ts_code")["or_amount"]
        .transform(lambda s: s.shift(1).rolling(p.relvol_lookback_days, min_periods=p.min_relvol_obs).mean())
    )
    daily["rel_open_amount"] = daily["or_amount"] / daily["or_amount_ma"].replace(0, np.nan)
    daily["rel_open_amount_rank"] = daily.groupby("trade_date")["rel_open_amount"].rank(ascending=False, method="first")
    daily["is_top_relvol"] = (daily["rel_open_amount_rank"] <= p.top_relvol_n).astype(int)

    d = d.merge(daily, on=["ts_code", "trade_date"], how="left")
    d["orb_upper"] = d["or_high"] * (1.0 + p.breakout_buffer)
    d["price_to_or_high"] = d["close"] / d["orb_upper"] - 1.0
    return d


def add_noise_features(panel: pd.DataFrame, p: ComboParams) -> pd.DataFrame:
    d = panel.copy()
    d = d.sort_values(["ts_code", "bar_index", "trade_date"]).reset_index(drop=True)

    noise = (
        d.groupby(["ts_code", "bar_index"], group_keys=False)["abs_intraday_move"]
        .apply(lambda s: s.shift(1).rolling(p.noise_lookback_days, min_periods=p.min_noise_obs).mean())
    )

    d["noise"] = noise.to_numpy()
    d["noise_upper"] = d["day_open"] * (1.0 + p.noise_k * d["noise"])
    d["price_to_noise_upper"] = d["close"] / d["noise_upper"] - 1.0

    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def add_expected_edge(d: pd.DataFrame, p: ComboParams) -> pd.DataFrame:
    x = d.copy()
    x["or_range_bp"] = x["or_range"].fillna(0) * 10000.0
    x["noise_bp"] = x["noise"].fillna(0) * p.noise_k * 10000.0
    x["breakout_extra_bp"] = x[["price_to_or_high", "price_to_noise_upper"]].max(axis=1).fillna(0).clip(lower=0) * 10000.0
    x["expected_edge_bp"] = np.maximum(x["or_range_bp"], x["noise_bp"]) + 0.5 * x["breakout_extra_bp"]
    x["edge_ok"] = x["expected_edge_bp"] >= p.expected_edge_mult * p.assumed_roundtrip_cost_bp
    return x


# ============================================================
# 7. 信号、成交、退出
# ============================================================

def build_signal_mask(d: pd.DataFrame, p: ComboParams, scope: str) -> pd.Series:
    m = (
        scope_mask(d, scope)
        & (d["valid_entry_bar"] == 1)
        & (d["force_flat_bar"] == 0)
        & (d["clock_time"] >= p.entry_start)
        & (d["clock_time"] <= p.entry_end)
        & d["or_high"].notna()
        & d["rel_open_amount"].notna()
        & d["noise"].notna()
        & (d["amount"].fillna(0) > 0)
        & (d["or_amount"].fillna(0) > 0)
        & (d["close"] > d["orb_upper"])
        & ((d["rel_open_amount"] >= p.rel_open_amount_min) | (d["is_top_relvol"] == 1))
        & (d["amount_z_20m"].fillna(0) >= p.amount_z_min)
        & (d["cat_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["cat_ret5_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["daily_adj_factor_change"].fillna(0).astype(int) == 0)
        & (d["daily_is_extreme_return_day"].fillna(0).astype(int) == 0)
        & (d["edge_ok"] == 1)
    )

    if p.require_noise_break:
        m &= (d["close"] > d["noise_upper"])

    if p.require_above_vwap:
        m &= d["close"] > d["intraday_vwap"]

    if p.require_ret5_positive:
        m &= d["ret_5m"].fillna(0) > 0

    if p.use_relative_value_filter:
        rv_z_ok = d["rv_z"].isna() | (d["rv_z"] <= p.rv_z_max)
        rv_rank_ok = d["rv_rank_pct"].isna() | (d["rv_rank_pct"] <= p.rv_rank_pct_max)
        m &= rv_z_ok & rv_rank_ok

    if p.signal_rising_edge:
        prev = m.groupby([d["ts_code"], d["trade_date"]]).shift(1).fillna(False)
        m = m & (~prev)

    return m


def simulate_trade(day: pd.DataFrame, signal_pos: int, p: ComboParams) -> Dict:
    entry_pos = signal_pos + 1
    if entry_pos >= len(day):
        return {}

    sig = day.iloc[signal_pos]
    ent = day.iloc[entry_pos]

    if int(ent.get("force_flat_bar", 0)) == 1:
        return {}

    entry_price = float(ent["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    max_exit_pos = min(len(day) - 1, entry_pos + p.max_hold_bars)
    tp_price = entry_price * (1.0 + p.take_profit)
    hard_stop_price = entry_price * (1.0 - p.hard_stop_loss)
    high_water = entry_price
    or_high = float(sig["or_high"]) if np.isfinite(sig["or_high"]) else np.nan

    exit_pos = max_exit_pos
    exit_price = float(day.iloc[exit_pos]["close"])
    exit_reason = "max_hold"

    for j in range(entry_pos, max_exit_pos + 1):
        row = day.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        vwap = float(row["intraday_vwap"]) if np.isfinite(row["intraday_vwap"]) else np.nan

        if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
            continue

        if int(row.get("force_flat_bar", 0)) == 1:
            exit_pos = j
            exit_price = close
            exit_reason = "force_flat"
            break

        trail_stop_price = high_water * (1.0 - p.trailing_stop)
        vwap_stop_price = vwap * (1.0 - p.vwap_stop_band) if np.isfinite(vwap) else -np.inf
        or_stop_price = or_high * (1.0 - p.or_high_stop_band) if np.isfinite(or_high) else -np.inf
        stop_price = max(hard_stop_price, trail_stop_price, vwap_stop_price, or_stop_price)

        # 保守：同一bar内先止损后止盈
        if low <= stop_price:
            exit_pos = j
            exit_price = stop_price
            exit_reason = "stop"
            break

        if high >= tp_price:
            exit_pos = j
            exit_price = tp_price
            exit_reason = "take_profit"
            break

        high_water = max(high_water, high)

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
        "entry_time": ent["trade_time"],
        "exit_time": ex["trade_time"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "signal_rel_open_amount": float(sig["rel_open_amount"]) if np.isfinite(sig["rel_open_amount"]) else np.nan,
        "signal_price_to_or_high": float(sig["price_to_or_high"]) if np.isfinite(sig["price_to_or_high"]) else np.nan,
        "signal_price_to_noise_upper": float(sig["price_to_noise_upper"]) if np.isfinite(sig["price_to_noise_upper"]) else np.nan,
        "signal_rv_z": float(sig["rv_z"]) if np.isfinite(sig["rv_z"]) else np.nan,
        "signal_rv_rank_pct": float(sig["rv_rank_pct"]) if np.isfinite(sig["rv_rank_pct"]) else np.nan,
        "signal_expected_edge_bp": float(sig["expected_edge_bp"]) if np.isfinite(sig["expected_edge_bp"]) else np.nan,
    }


def generate_trades_for_candidate(panel: pd.DataFrame, p: ComboParams, scope: str, candidate_id: int) -> pd.DataFrame:
    d = add_orb_features(panel, p)
    d = add_noise_features(d, p)
    d = add_expected_edge(d, p)
    d["_signal"] = build_signal_mask(d, p, scope).astype(int)

    trades = []
    groups = d.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups

    for i, (_, day) in enumerate(groups, 1):
        if i % 3000 == 0:
            print(f"    candidate {candidate_id}: ETF-days {i}/{total}")
        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        for sp in sig_pos:
            tr = simulate_trade(day, int(sp), p)
            if tr:
                tr["candidate_id"] = candidate_id
                tr["scope"] = scope
                trades.append(tr)

    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, p: ComboParams) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.copy()
    c["rv_rank_sort"] = c["signal_rv_rank_pct"].fillna(0.5)
    c = c.sort_values(
        ["entry_time", "signal_expected_edge_bp", "signal_rel_open_amount", "rv_rank_sort"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    open_pos = []
    cooldown_until = {}
    etf_day_count = {}
    accepted = []

    for _, row in c.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_t = pd.Timestamp(row["exit_time"])
        code = row["ts_code"]
        date = row["trade_date"]
        cat = row["t0_category"]

        open_pos = [pos for pos in open_pos if pos["exit_time"] > entry]

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

        accepted.append(row.to_dict())
        open_pos.append({"ts_code": code, "t0_category": cat, "exit_time": exit_t})
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=p.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1

    return pd.DataFrame(accepted)


# ============================================================
# 8. 历史窗口评分与ETF筛选
# ============================================================

def score_trades(trades: pd.DataFrame, cost_bp: float) -> Dict:
    if trades.empty:
        return {
            "trade_count": 0,
            "avg_gross_ret": np.nan,
            "avg_net_ret": np.nan,
            "avg_net_bp": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "score": -1e9,
        }

    net = trades["gross_ret"] - 2.0 * cost_bp / 10000.0
    pf = profit_factor(net)
    avg_net = float(net.mean())
    avg_net_bp = avg_net * 10000.0
    n = int(len(trades))
    wr = float((net > 0).mean())

    if not np.isfinite(pf):
        pf_for_score = 3.0
    else:
        pf_for_score = min(pf, 3.0)

    # 评分偏向平均净收益，其次 profit factor 和样本数。
    score = avg_net_bp + 0.35 * pf_for_score + 0.02 * min(n, 100)

    return {
        "trade_count": n,
        "avg_gross_ret": float(trades["gross_ret"].mean()),
        "avg_net_ret": avg_net,
        "avg_net_bp": avg_net_bp,
        "win_rate": wr,
        "profit_factor": pf,
        "score": score,
    }


def select_candidate_and_etfs(
    all_trades: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    hist_dates: List[str],
    cost_bp: float,
    min_candidate_trades: int,
    min_etf_trades: int,
    min_etf_avg_net_bp: float,
    min_etf_profit_factor: float,
    fallback_all_if_no_etf_pass: bool,
) -> Dict:
    hist = all_trades[all_trades["trade_date"].isin(hist_dates)].copy()
    if hist.empty:
        return {
            "selected": False,
            "candidate_id": None,
            "selected_etfs": [],
            "reason": "no_hist_trades",
        }

    rows = []
    for cid, g in hist.groupby("candidate_id"):
        s = score_trades(g, cost_bp)
        s["candidate_id"] = int(cid)
        rows.append(s)
    score_df = pd.DataFrame(rows)

    score_df = score_df[score_df["trade_count"] >= min_candidate_trades].copy()
    if score_df.empty:
        return {
            "selected": False,
            "candidate_id": None,
            "selected_etfs": [],
            "reason": "no_candidate_enough_trades",
        }

    best_row = score_df.sort_values(["score", "avg_net_bp", "profit_factor"], ascending=False).iloc[0]
    cid = int(best_row["candidate_id"])
    chosen_hist = hist[hist["candidate_id"] == cid].copy()

    etf_rows = []
    for code, g in chosen_hist.groupby("ts_code"):
        s = score_trades(g, cost_bp)
        s["ts_code"] = code
        etf_rows.append(s)
    etf_df = pd.DataFrame(etf_rows)

    pass_df = etf_df[
        (etf_df["trade_count"] >= min_etf_trades)
        & (etf_df["avg_net_bp"] >= min_etf_avg_net_bp)
        & ((etf_df["profit_factor"] >= min_etf_profit_factor) | (~np.isfinite(etf_df["profit_factor"])))
    ].copy()

    if pass_df.empty:
        if fallback_all_if_no_etf_pass:
            selected_etfs = sorted(chosen_hist["ts_code"].dropna().unique().tolist())
            reason = "fallback_all_etfs"
        else:
            return {
                "selected": False,
                "candidate_id": cid,
                "selected_etfs": [],
                "candidate_score": best_row.to_dict(),
                "reason": "no_etf_pass",
            }
    else:
        selected_etfs = sorted(pass_df["ts_code"].tolist())
        reason = "ok"

    meta = candidate_meta[candidate_meta["candidate_id"] == cid].iloc[0].to_dict()
    return {
        "selected": True,
        "candidate_id": cid,
        "selected_etfs": selected_etfs,
        "candidate_score": best_row.to_dict(),
        "candidate_meta": meta,
        "reason": reason,
        "etf_score_table": etf_df,
    }


# ============================================================
# 9. 绩效
# ============================================================

def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str], position_weight: float) -> Tuple[Dict, pd.DataFrame]:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = position_weight * t["net_ret"]
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
            "avg_holding_bars": np.nan,
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
            "avg_holding_bars": float(trades["holding_bars"].mean()),
        }
    return s, daily


def summary_by_group(trades: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(group_cols, dropna=False)
        .agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            win_rate=("gross_ret", lambda s: (s > 0).mean()),
            profit_factor=("gross_ret", profit_factor),
            avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
        .sort_values("avg_gross_ret", ascending=False)
    )


# ============================================================
# 10. 主流程
# ============================================================

def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]
    scopes = [x.strip() for x in str(args.scopes).split(",") if x.strip()]
    params = make_param_grid(args.grid_size)

    print("=" * 100)
    print("Walk-forward daily combo strategy")
    print("=" * 100)
    print(f"panel_file          : {Path(args.panel_file).resolve()}")
    print(f"out_dir             : {out_dir.resolve()}")
    print(f"scopes              : {scopes}")
    print(f"grid_size           : {args.grid_size}")
    print(f"params count        : {len(params)}")
    print(f"train_lookback_days : {args.train_lookback_days}")
    print(f"update_frequency    : {args.update_frequency}")
    print("=" * 100)

    panel = read_panel(Path(args.panel_file))
    all_dates = sorted(panel["trade_date"].unique().tolist())

    # 可交易日期从有足够历史之后开始
    start_idx = args.min_train_days
    trade_dates = all_dates[start_idx:]
    if args.limit_trade_days is not None:
        trade_dates = trade_dates[:args.limit_trade_days]

    # 生成候选交易
    candidate_meta_rows = []
    all_candidate_trade_parts = []
    cid = 0

    for scope in scopes:
        for p in params:
            cid += 1
            print(f"\nGenerating candidate {cid}: scope={scope}, params={asdict(p)}")
            tr = generate_trades_for_candidate(panel, p, scope, cid)
            if not tr.empty:
                tr = apply_portfolio_constraints(tr, p)
                all_candidate_trade_parts.append(tr)
            meta = asdict(p)
            meta.update({
                "candidate_id": cid,
                "scope": scope,
                "candidate_trade_count": int(len(tr)),
            })
            candidate_meta_rows.append(meta)
            print(f"  candidate {cid} accepted trades={len(tr):,}")

    candidate_meta = pd.DataFrame(candidate_meta_rows)
    candidate_meta.to_csv(out_dir / "wf_candidate_metadata.csv", index=False, encoding="utf-8-sig")

    if all_candidate_trade_parts:
        all_trades = pd.concat(all_candidate_trade_parts, ignore_index=True)
        all_trades["trade_date"] = all_trades["trade_date"].astype(str).map(norm_date)
        all_trades = all_trades.sort_values(["trade_date", "entry_time"]).reset_index(drop=True)
    else:
        all_trades = pd.DataFrame()

    all_trades.to_csv(out_dir / "wf_all_candidate_trades.csv", index=False, encoding="utf-8-sig")

    if all_trades.empty:
        raise RuntimeError("所有候选参数都没有产生交易。请放宽参数或检查面板字段。")

    # Walk-forward
    selected_parts = []
    select_logs = []
    period_cache: Dict[str, Dict] = {}

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for k, d in enumerate(trade_dates, 1):
        if k % 50 == 0:
            print(f"Walk-forward day {k}/{len(trade_dates)}: {d}")

        idx = date_to_idx[d]
        hist_start_idx = max(0, idx - args.train_lookback_days)
        hist_dates = all_dates[hist_start_idx:idx]

        period_key = make_period_key(d, args.update_frequency)

        if period_key in period_cache:
            sel = period_cache[period_key]
        else:
            sel = select_candidate_and_etfs(
                all_trades=all_trades,
                candidate_meta=candidate_meta,
                hist_dates=hist_dates,
                cost_bp=args.main_cost_bp,
                min_candidate_trades=args.min_candidate_trades,
                min_etf_trades=args.min_etf_trades,
                min_etf_avg_net_bp=args.min_etf_avg_net_bp,
                min_etf_profit_factor=args.min_etf_profit_factor,
                fallback_all_if_no_etf_pass=args.fallback_all_if_no_etf_pass,
            )
            period_cache[period_key] = sel

        log = {
            "trade_date": d,
            "period_key": period_key,
            "hist_start": hist_dates[0] if hist_dates else "",
            "hist_end": hist_dates[-1] if hist_dates else "",
            "selected": bool(sel.get("selected", False)),
            "reason": sel.get("reason", ""),
            "candidate_id": sel.get("candidate_id", ""),
            "selected_etf_count": len(sel.get("selected_etfs", [])),
            "selected_etfs": ",".join(sel.get("selected_etfs", [])),
        }

        if sel.get("selected", False):
            cs = sel.get("candidate_score", {})
            for kk, vv in cs.items():
                log[f"candidate_{kk}"] = vv
            meta = sel.get("candidate_meta", {})
            for kk in ["scope", "opening_end", "rel_open_amount_min", "noise_k", "rv_z_max", "take_profit", "hard_stop_loss", "max_hold_bars"]:
                if kk in meta:
                    log[f"param_{kk}"] = meta[kk]

            day_tr = all_trades[
                (all_trades["trade_date"] == d)
                & (all_trades["candidate_id"] == int(sel["candidate_id"]))
                & (all_trades["ts_code"].isin(sel["selected_etfs"]))
            ].copy()

            if not day_tr.empty:
                day_tr["wf_period_key"] = period_key
                day_tr["wf_hist_start"] = log["hist_start"]
                day_tr["wf_hist_end"] = log["hist_end"]
                selected_parts.append(day_tr)

        select_logs.append(log)

    wf_trades = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selection_log = pd.DataFrame(select_logs)

    wf_trades.to_csv(out_dir / "wf_trades.csv", index=False, encoding="utf-8-sig")
    selection_log.to_csv(out_dir / "wf_selection_log.csv", index=False, encoding="utf-8-sig")

    # 绩效
    eval_dates = trade_dates
    summary_rows = []
    nav_parts = []

    # position_weight 取候选参数默认值；若每日不同，这里仍用每笔固定 20%，与生成交易一致。
    position_weight = 0.20

    for c in cost_bps:
        s, daily = performance(wf_trades, c, eval_dates, position_weight)
        summary_rows.append(s)
        daily["cost_bp"] = c
        nav_parts.append(daily)

    summary = pd.DataFrame(summary_rows)
    daily_nav = pd.concat(nav_parts, ignore_index=True)

    summary.to_csv(out_dir / "wf_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    daily_nav.to_csv(out_dir / "wf_daily_nav_by_cost.csv", index=False, encoding="utf-8-sig")

    cat_sum = summary_by_group(wf_trades, ["t0_category_cn"])
    etf_sum = summary_by_group(wf_trades, ["ts_code", "name", "t0_category_cn"])
    exit_sum = summary_by_group(wf_trades, ["exit_reason"])

    if not cat_sum.empty:
        cat_sum.to_csv(out_dir / "wf_category_summary.csv", index=False, encoding="utf-8-sig")
    if not etf_sum.empty:
        etf_sum.to_csv(out_dir / "wf_etf_summary.csv", index=False, encoding="utf-8-sig")
    if not exit_sum.empty:
        exit_sum.to_csv(out_dir / "wf_exit_reason_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel_file": str(Path(args.panel_file).resolve()),
        "out_dir": str(out_dir.resolve()),
        "scopes": scopes,
        "grid_size": args.grid_size,
        "params_count": len(params),
        "candidate_count": int(len(candidate_meta)),
        "walk_forward_trade_dates": int(len(trade_dates)),
        "train_lookback_days": args.train_lookback_days,
        "min_train_days": args.min_train_days,
        "update_frequency": args.update_frequency,
        "main_cost_bp": args.main_cost_bp,
        "min_candidate_trades": args.min_candidate_trades,
        "min_etf_trades": args.min_etf_trades,
        "min_etf_avg_net_bp": args.min_etf_avg_net_bp,
        "min_etf_profit_factor": args.min_etf_profit_factor,
        "fallback_all_if_no_etf_pass": bool(args.fallback_all_if_no_etf_pass),
    }
    (out_dir / "wf_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报告
    lines = []
    lines.append("# Walk-forward Daily 组合策略回测报告\n")
    lines.append("## 1. 核心逻辑\n")
    lines.append("本脚本不再简单把样本切成一个开发期和一个测试期，而是每天只用当前交易日前的滚动历史窗口选择参数和可交易 ETF，然后在当天盘中交易。信号仍然是 ORB + Noise Boundary + VWAP + 类别同步 + 相对价值过滤 + 成本覆盖过滤 + 动态退出。")
    lines.append("")
    lines.append("## 2. 配置\n")
    lines.append(pd.DataFrame([config]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not cat_sum.empty:
        lines.append("## 4. 类别归因\n")
        lines.append(cat_sum.to_markdown(index=False))
        lines.append("")
    if not etf_sum.empty:
        lines.append("## 5. ETF 归因\n")
        lines.append(etf_sum.head(50).to_markdown(index=False))
        lines.append("")
    if not exit_sum.empty:
        lines.append("## 6. 退出原因归因\n")
        lines.append(exit_sum.to_markdown(index=False))
        lines.append("")
    lines.append("## 7. 解释口径\n")
    lines.append("- 这是滚动样本外结果，比固定 2022-2023 找参数、2024 测试更接近真实交易。")
    lines.append("- 如果 2bp 单边成本后仍为正，说明滚动更新有实际改善。")
    lines.append("- 如果仍只在 0bp/1bp 有效，说明当前 long-only 日内突破框架的收益边际仍偏薄。")
    lines.append("- 若交易数很少，不能直接认为策略稳定。")
    (out_dir / "wf_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Finished")
    print(f"wf trades   : {(out_dir / 'wf_trades.csv').resolve()}")
    print(f"summary     : {(out_dir / 'wf_summary_by_cost.csv').resolve()}")
    print(f"selection   : {(out_dir / 'wf_selection_log.csv').resolve()}")
    print(f"report      : {(out_dir / 'wf_report.md').resolve()}")
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
