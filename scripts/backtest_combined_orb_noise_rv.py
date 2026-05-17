#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_combined_orb_noise_rv.py

用途：
    用现有 T+0 ETF 1min 面板，测试组合策略：

        ORB 开盘区间突破
      + Noise Boundary 噪声边界确认
      + Relative Value / Cluster Z 相对价值过滤
      + Category Breadth 类别同步确认
      + Expected Edge 成本覆盖过滤
      + Dynamic Exit 动态退出

为什么写这个脚本：
    前面单独测试过：
        1. VWAP 均值回归：失败；
        2. long-only 配对：太弱；
        3. long-short 配对：理论有效但做空不现实；
        4. 噪声边界动量：全池边际薄；
        5. ORB：自动收缩到 commodity_focus，但 2bp 后失效。

    所以现在用面板做一个”组合策略原型测试”，看组合过滤后
    能否提高平均单笔毛收益和成本承受能力。

输入：
    默认读取：
        .\\data_t0\\processed\\t0_intraday_bar_panel.parquet

输出：
    默认写入：
        .\\data_t0\\backtest_combined_orb_noise_rv\\

运行默认参数：
    python ".\\backtest_combined_orb_noise_rv.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_combined_orb_noise_rv"

运行小网格：
    python ".\\backtest_combined_orb_noise_rv.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_combined_orb_noise_rv_grid" `
      --run-grid

建议：
    先跑默认版；如果 test 期 1bp/2bp 仍有改善，再跑 --run-grid。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ComboParams:
    # ---------- 样本内特征 ----------
    opening_start: str = "09:30"
    opening_end: str = "09:45"
    entry_start: str = "09:46"
    entry_end: str = "14:15"

    # ORB / relative volume
    relvol_lookback_days: int = 10
    min_relvol_obs: int = 5
    rel_open_amount_min: float = 1.30
    top_relvol_n: int = 10
    breakout_buffer: float = 0.0002

    # noise boundary
    noise_lookback_days: int = 14
    min_noise_obs: int = 7
    noise_k: float = 1.25
    require_noise_break: int = 1

    # category breadth / momentum
    amount_z_min: float = 0.0
    category_strength_min: float = 0.55
    require_ret5_positive: int = 1
    require_above_vwap: int = 1
    signal_rising_edge: int = 1

    # relative value filter
    use_relative_value_filter: int = 1
    rv_z_max: float = 1.50
    rv_rank_pct_max: float = 0.80

    # cost / edge filter
    expected_edge_mult: float = 1.50
    assumed_roundtrip_cost_bp: float = 4.0

    # exit
    take_profit: float = 0.0080
    hard_stop_loss: float = 0.0030
    trailing_stop: float = 0.0035
    vwap_stop_band: float = 0.0005
    or_high_stop_band: float = 0.0005
    max_hold_bars: int = 60

    # portfolio
    max_positions: int = 3
    same_category_max_open: int = 1
    position_weight: float = 0.20
    cooldown_minutes: int = 15
    etf_daily_trade_limit: int = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest combined ORB + noise + relative-value filter strategy.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_combined_orb_noise_rv")

    p.add_argument("--dev-start", type=str, default="20220101")
    p.add_argument("--dev-end", type=str, default="20231231")
    p.add_argument("--test-start", type=str, default="20240101")
    p.add_argument("--test-end", type=str, default="20241231")

    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")
    p.add_argument("--main-cost-bp", type=float, default=2.0)
    p.add_argument("--run-grid", action="store_true")

    p.add_argument(
        "--scopes",
        type=str,
        default="commodity_focus,cross_commodity,no_bond_money,all",
        help="逗号分隔：all,no_bond_money,commodity_focus,cross_commodity"
    )

    # 默认参数可手动覆盖
    p.add_argument("--opening-end", type=str, default="09:45")
    p.add_argument("--relvol-lookback-days", type=int, default=10)
    p.add_argument("--rel-open-amount-min", type=float, default=1.30)
    p.add_argument("--top-relvol-n", type=int, default=10)
    p.add_argument("--noise-lookback-days", type=int, default=14)
    p.add_argument("--noise-k", type=float, default=1.25)
    p.add_argument("--rv-z-max", type=float, default=1.50)
    p.add_argument("--expected-edge-mult", type=float, default=1.50)
    p.add_argument("--assumed-roundtrip-cost-bp", type=float, default=4.0)
    p.add_argument("--take-profit", type=float, default=0.0080)
    p.add_argument("--hard-stop-loss", type=float, default=0.0030)
    p.add_argument("--trailing-stop", type=float, default=0.0035)
    p.add_argument("--max-hold-bars", type=int, default=60)
    return p.parse_args()


def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def next_minute(hhmm: str) -> str:
    hh, mm = hhmm.split(":")
    total = int(hh) * 60 + int(mm) + 1
    return f"{total // 60:02d}:{total % 60:02d}"


def make_params(args: argparse.Namespace) -> List[ComboParams]:
    if not args.run_grid:
        return [ComboParams(
            opening_end=args.opening_end,
            entry_start=next_minute(args.opening_end),
            relvol_lookback_days=args.relvol_lookback_days,
            min_relvol_obs=max(5, args.relvol_lookback_days // 2),
            rel_open_amount_min=args.rel_open_amount_min,
            top_relvol_n=args.top_relvol_n,
            noise_lookback_days=args.noise_lookback_days,
            min_noise_obs=max(5, args.noise_lookback_days // 2),
            noise_k=args.noise_k,
            rv_z_max=args.rv_z_max,
            expected_edge_mult=args.expected_edge_mult,
            assumed_roundtrip_cost_bp=args.assumed_roundtrip_cost_bp,
            take_profit=args.take_profit,
            hard_stop_loss=args.hard_stop_loss,
            trailing_stop=args.trailing_stop,
            max_hold_bars=args.max_hold_bars,
        )]

    # 小网格：不要太大，先用于判断组合过滤有没有价值。
    grid = {
        "opening_end": ["09:45", "10:00"],
        "relvol_lookback_days": [5, 10, 14],
        "rel_open_amount_min": [1.0, 1.3, 1.6],
        "noise_lookback_days": [10, 14],
        "noise_k": [1.0, 1.25, 1.5],
        "rv_z_max": [1.0, 1.5, 2.0],
        "expected_edge_mult": [1.0, 1.5, 2.0],
        "take_profit": [0.006, 0.008],
        "max_hold_bars": [30, 60],
    }

    keys = list(grid.keys())
    params = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        params.append(ComboParams(
            **kw,
            entry_start=next_minute(kw["opening_end"]),
            min_relvol_obs=max(3, kw["relvol_lookback_days"] // 2),
            min_noise_obs=max(5, kw["noise_lookback_days"] // 2),
            hard_stop_loss=args.hard_stop_loss,
            trailing_stop=args.trailing_stop,
            assumed_roundtrip_cost_bp=args.assumed_roundtrip_cost_bp,
        ))
    return params


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "intraday_vwap", "bar_index",
        "ret_5m", "ret_10m", "ret_20m",
        "amount_z_20m", "breakout_high_20m",
        "valid_entry_bar", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day",
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

    for c in ["open", "high", "low", "close", "amount", "intraday_vwap", "ret_5m", "ret_10m", "ret_20m", "amount_z_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day", "breakout_high_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype("Int64")

    d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")
    d["intraday_move"] = d["close"] / d["day_open"] - 1.0
    d["abs_intraday_move"] = d["intraday_move"].abs()

    # 类别同步：同一时刻同类别，日内为正/5m 为正的比例
    d["positive_intraday"] = (d["intraday_move"] > 0).astype(int)
    d["positive_5m"] = (d["ret_5m"] > 0).astype(int)
    d["cat_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
    d["cat_ret5_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")

    # cluster relative value：同一类别同一时刻相对强弱。
    # 突破时如果该 ETF 相对同类已经太强，容易追高，过滤掉。
    g = d.groupby(["trade_time", "t0_category"])
    d["cluster_move_mean"] = g["intraday_move"].transform("mean")
    d["cluster_move_std"] = g["intraday_move"].transform("std")
    d["rv_z"] = (d["intraday_move"] - d["cluster_move_mean"]) / d["cluster_move_std"].replace(0, np.nan)

    # 类别内 percentile rank，越高代表越强/越贵。
    d["rv_rank_pct"] = g["intraday_move"].rank(pct=True)

    return d


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

    # 粗略估计本次信号可用空间：
    # 取 OR range、noise move、当日已突破幅度的组合。只做过滤，不作为收益预测。
    x["or_range_bp"] = x["or_range"].fillna(0) * 10000.0
    x["noise_bp"] = x["noise"].fillna(0) * p.noise_k * 10000.0
    x["breakout_extra_bp"] = x[["price_to_or_high", "price_to_noise_upper"]].max(axis=1).fillna(0).clip(lower=0) * 10000.0

    x["expected_edge_bp"] = np.maximum(x["or_range_bp"], x["noise_bp"]) + 0.5 * x["breakout_extra_bp"]
    x["edge_ok"] = x["expected_edge_bp"] >= p.expected_edge_mult * p.assumed_roundtrip_cost_bp

    return x


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
        # 对于只有 1 只的类别，rv_z 可能是 NaN；不因此剔除。
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

    exit_pos = max_exit_pos
    exit_price = float(day.iloc[exit_pos]["close"])
    exit_reason = "max_hold"

    or_high = float(sig["or_high"]) if np.isfinite(sig["or_high"]) else np.nan

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

        # 保守：同一 bar 同时触发止损/止盈，先止损。
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
        "signal_close": float(sig["close"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "signal_or_high": float(sig["or_high"]) if np.isfinite(sig["or_high"]) else np.nan,
        "signal_or_range": float(sig["or_range"]) if np.isfinite(sig["or_range"]) else np.nan,
        "signal_noise": float(sig["noise"]) if np.isfinite(sig["noise"]) else np.nan,
        "signal_rel_open_amount": float(sig["rel_open_amount"]) if np.isfinite(sig["rel_open_amount"]) else np.nan,
        "signal_relvol_rank": float(sig["rel_open_amount_rank"]) if np.isfinite(sig["rel_open_amount_rank"]) else np.nan,
        "signal_price_to_or_high": float(sig["price_to_or_high"]) if np.isfinite(sig["price_to_or_high"]) else np.nan,
        "signal_price_to_noise_upper": float(sig["price_to_noise_upper"]) if np.isfinite(sig["price_to_noise_upper"]) else np.nan,
        "signal_rv_z": float(sig["rv_z"]) if np.isfinite(sig["rv_z"]) else np.nan,
        "signal_rv_rank_pct": float(sig["rv_rank_pct"]) if np.isfinite(sig["rv_rank_pct"]) else np.nan,
        "signal_expected_edge_bp": float(sig["expected_edge_bp"]) if np.isfinite(sig["expected_edge_bp"]) else np.nan,
        "signal_cat_positive_ratio": float(sig["cat_positive_ratio"]) if np.isfinite(sig["cat_positive_ratio"]) else np.nan,
        "signal_cat_ret5_positive_ratio": float(sig["cat_ret5_positive_ratio"]) if np.isfinite(sig["cat_ret5_positive_ratio"]) else np.nan,
        "signal_amount_z_20m": float(sig["amount_z_20m"]) if np.isfinite(sig["amount_z_20m"]) else np.nan,
    }


def generate_trades(d: pd.DataFrame, p: ComboParams, scope: str) -> pd.DataFrame:
    x = d.copy()
    x["_signal"] = build_signal_mask(x, p, scope).astype(int)

    trades = []
    groups = x.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups

    for i, (_, day) in enumerate(groups, 1):
        if i % 1000 == 0:
            print(f"    simulated ETF-days {i}/{total}")
        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        for sp in sig_pos:
            tr = simulate_trade(day, int(sp), p)
            if tr:
                tr["scope"] = scope
                for k, v in asdict(p).items():
                    tr[f"param_{k}"] = v
                trades.append(tr)

    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, p: ComboParams) -> pd.DataFrame:
    if trades.empty:
        return trades

    # 优先交易：预期空间大、relative volume 高、不过度追高
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


def split_trades(trades: pd.DataFrame, dev_start: str, dev_end: str, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    t = trades.copy()
    t["trade_date"] = t["trade_date"].astype(str).map(norm_date)
    dev = t[(t["trade_date"] >= dev_start) & (t["trade_date"] <= dev_end)].copy()
    test = t[(t["trade_date"] >= test_start) & (t["trade_date"] <= test_end)].copy()
    return dev, test


def date_range(panel: pd.DataFrame, start: str, end: str) -> List[str]:
    dates = sorted(panel["trade_date"].astype(str).map(norm_date).unique())
    return [d for d in dates if start <= d <= end]


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str], p: ComboParams) -> Tuple[Dict, pd.DataFrame]:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = p.position_weight * t["net_ret"]
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


def objective(perf: Dict) -> float:
    n = perf.get("trade_count", 0)
    if n < 20:
        return -1e9
    avg_net_bp = perf.get("avg_net_ret", np.nan) * 10000.0
    pf = perf.get("profit_factor", np.nan)
    sharpe = perf.get("sharpe", np.nan)
    mdd = abs(perf.get("max_drawdown", 0.0))
    if not np.isfinite(avg_net_bp) or not np.isfinite(pf) or not np.isfinite(sharpe):
        return -1e9
    return avg_net_bp + 0.40 * min(pf, 3.0) + 0.05 * min(sharpe, 5.0) - 10.0 * mdd


def category_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["scope", "t0_category_cn"], dropna=False)
        .agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            win_rate=("gross_ret", lambda s: (s > 0).mean()),
            profit_factor=("gross_ret", profit_factor),
            avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
        .sort_values(["scope", "avg_gross_ret"], ascending=[True, False])
    )


def etf_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["scope", "ts_code", "name", "t0_category_cn"], dropna=False)
        .agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            win_rate=("gross_ret", lambda s: (s > 0).mean()),
            profit_factor=("gross_ret", profit_factor),
            avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
        .sort_values(["scope", "avg_gross_ret"], ascending=[True, False])
    )


def main() -> int:
    args = parse_args()
    dev_start, dev_end = norm_date(args.dev_start), norm_date(args.dev_end)
    test_start, test_end = norm_date(args.test_start), norm_date(args.test_end)
    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]
    scopes = [s.strip() for s in str(args.scopes).split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Combined ORB + Noise + Relative Value strategy backtest")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"dev        : {dev_start} -> {dev_end}")
    print(f"test       : {test_start} -> {test_end}")
    print(f"scopes     : {scopes}")
    print(f"run_grid   : {args.run_grid}")
    print("=" * 100)

    panel = read_panel(Path(args.panel_file))
    dev_dates = date_range(panel, dev_start, dev_end)
    test_dates = date_range(panel, test_start, test_end)

    params_list = make_params(args)

    # 缓存特征，避免重复计算
    cache: Dict[Tuple, pd.DataFrame] = {}

    best = None
    best_score = -1e18
    grid_rows = []

    for idx, p in enumerate(params_list, 1):
        key = (
            p.opening_end,
            p.relvol_lookback_days,
            p.rel_open_amount_min,
            p.top_relvol_n,
            p.noise_lookback_days,
            p.noise_k,
            p.breakout_buffer,
        )
        if key not in cache:
            print(f"Precomputing features key={key}")
            d = add_orb_features(panel, p)
            d = add_noise_features(d, p)
            d = add_expected_edge(d, p)
            cache[key] = d
        else:
            d = cache[key]

        for scope in scopes:
            print(f"\n[{idx}/{len(params_list)}] scope={scope}, params={asdict(p)}")
            raw = generate_trades(d, p, scope)
            trades = apply_portfolio_constraints(raw, p)
            dev_tr, test_tr = split_trades(trades, dev_start, dev_end, test_start, test_end)

            dev_perf, _ = performance(dev_tr, args.main_cost_bp, dev_dates, p)
            score = objective(dev_perf)

            row = asdict(p)
            row["param_index"] = idx
            row["scope"] = scope
            row["dev_objective"] = score
            row["raw_trade_count"] = len(raw)
            row["accepted_trade_count"] = len(trades)
            row["dev_trade_count"] = len(dev_tr)
            row["test_trade_count"] = len(test_tr)
            for k2, v2 in dev_perf.items():
                row[f"dev_main_{k2}"] = v2
            grid_rows.append(row)

            print(f"  raw={len(raw):,}, accepted={len(trades):,}, dev={len(dev_tr):,}, test={len(test_tr):,}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best = {
                    "scope": scope,
                    "params": p,
                    "raw": raw,
                    "trades": trades,
                    "dev_trades": dev_tr,
                    "test_trades": test_tr,
                    "dev_perf": dev_perf,
                    "score": score,
                }

    if best is None:
        raise RuntimeError("没有有效结果。")

    p = best["params"]
    best_scope = best["scope"]
    trades = best["trades"]
    dev_trades = best["dev_trades"]
    test_trades = best["test_trades"]

    grid_df = pd.DataFrame(grid_rows).sort_values("dev_objective", ascending=False)
    grid_df.to_csv(out_dir / "combo_param_grid_summary.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    nav_parts = []
    for period, tr, dates in [
        ("dev", dev_trades, dev_dates),
        ("test", test_trades, test_dates),
    ]:
        for c in cost_bps:
            s, daily = performance(tr, c, dates, p)
            s["period"] = period
            s["scope"] = best_scope
            summary_rows.append(s)
            daily["period"] = period
            daily["cost_bp"] = c
            daily["scope"] = best_scope
            nav_parts.append(daily)

    summary = pd.DataFrame(summary_rows)
    nav = pd.concat(nav_parts, ignore_index=True)

    cat_sum = category_summary(trades)
    etf_sum = etf_summary(trades)

    trades.to_csv(out_dir / "combo_trades.csv", index=False, encoding="utf-8-sig")
    dev_trades.to_csv(out_dir / "combo_dev_trades.csv", index=False, encoding="utf-8-sig")
    test_trades.to_csv(out_dir / "combo_test_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "combo_summary_by_period_cost.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(out_dir / "combo_daily_nav_by_period_cost.csv", index=False, encoding="utf-8-sig")
    if not cat_sum.empty:
        cat_sum.to_csv(out_dir / "combo_category_summary.csv", index=False, encoding="utf-8-sig")
    if not etf_sum.empty:
        etf_sum.to_csv(out_dir / "combo_etf_summary.csv", index=False, encoding="utf-8-sig")

    best_info = {
        "best_scope": best_scope,
        "best_params": asdict(p),
        "dev_objective": best_score,
        "dev_start": dev_start,
        "dev_end": dev_end,
        "test_start": test_start,
        "test_end": test_end,
        "main_cost_bp": args.main_cost_bp,
        "run_grid": bool(args.run_grid),
    }
    with open(out_dir / "combo_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# 组合策略回测报告：ORB + Noise + Relative Value\n")
    lines.append("## 1. 策略逻辑\n")
    lines.append("本策略不是单独 ORB，也不是单独噪声边界，而是组合过滤：当日活跃 ETF、开盘区间突破、噪声边界突破、VWAP 确认、类别同步、相对价值不过热、收益空间覆盖成本后，下一根 1min bar 开多。退出采用 hard stop、VWAP stop、OR_high failure stop、trailing stop、take profit、max hold 和收盘前清仓。")
    lines.append("")
    lines.append("## 2. 样本切分\n")
    lines.append(f"- dev: {dev_start} ~ {dev_end}")
    lines.append(f"- test: {test_start} ~ {test_end}")
    lines.append(f"- main selection cost: {args.main_cost_bp}bp 单边")
    lines.append("")
    lines.append("## 3. 最优 scope 与参数\n")
    lines.append(f"- best scope: **{best_scope}**")
    lines.append(pd.DataFrame([asdict(p)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. dev/test 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not cat_sum.empty:
        lines.append("## 5. 类别归因\n")
        lines.append(cat_sum.to_markdown(index=False))
        lines.append("")
    if not etf_sum.empty:
        lines.append("## 6. ETF 归因\n")
        lines.append(etf_sum.head(30).to_markdown(index=False))
        lines.append("")
    lines.append("## 7. 判断标准\n")
    lines.append("- 如果 test 期 2bp 后仍为正，说明组合过滤比单一策略更有价值。")
    lines.append("- 如果 test 期 3bp 后仍为正，说明有继续开发价值。")
    lines.append("- 如果仍只在 0bp/1bp 有效，则说明当前 long-only 日内策略边际仍偏薄。")
    lines.append("- 如果交易数太少，即使收益好，也不能直接视为稳定策略。")
    (out_dir / "combo_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Finished")
    print(f"best scope : {best_scope}")
    print(f"summary    : {(out_dir / 'combo_summary_by_period_cost.csv').resolve()}")
    print(f"trades     : {(out_dir / 'combo_trades.csv').resolve()}")
    print(f"best       : {(out_dir / 'combo_best_params.json').resolve()}")
    print(f"report     : {(out_dir / 'combo_report.md').resolve()}")
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
