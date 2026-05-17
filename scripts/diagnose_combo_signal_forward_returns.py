#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
diagnose_combo_signal_forward_returns.py

用途：
    对当前组合策略的“裸信号”做预测力诊断。

核心问题：
    当前策略收益差，究竟是：
        1. 信号本身没有 alpha；
        2. 退出规则/仓位约束把 alpha 吃掉；
        3. 只有特定 ETF / 特定市场状态有效。

本脚本不做止盈、止损、仓位约束、动态退出。
只做：
    信号出现后，下一根 1min open 买入；
    然后观察未来 1/3/5/10/15/30/60min 以及尾盘的原始价格变化。

输入：
    data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet

输出：
    signal_forward_return_summary.csv
    signal_forward_return_by_etf.csv
    signal_forward_return_by_month.csv
    signal_forward_return_by_time.csv
    signal_forward_return_by_state.csv
    signal_forward_return_raw.csv
    signal_diagnostics_report.md

运行：
    python ".\diagnose_combo_signal_forward_returns.py" `
      --panel-file ".\data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet" `
      --out-dir ".\data_t0_2022_2024\diagnostics_combo_signal"

只看某一年：
    python ".\diagnose_combo_signal_forward_returns.py" `
      --panel-file ".\data_t0_2022_2024\processed\t0_intraday_bar_panel.parquet" `
      --out-dir ".\data_t0_2022_2024\diagnostics_combo_signal_2024" `
      --start-date 20240101 `
      --end-date 20241231
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


# ============================================================
# 1. 参数
# ============================================================

@dataclass
class SignalParams:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose raw forward returns after combo strategy signals.")
    p.add_argument("--panel-file", type=str, default="./data_t0_2022_2024/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0_2022_2024/diagnostics_combo_signal")

    p.add_argument("--scope", type=str, default="commodity_focus",
                   choices=["commodity_focus", "cross_commodity", "no_bond_money", "all"])
    p.add_argument("--start-date", type=str, default="")
    p.add_argument("--end-date", type=str, default="")
    p.add_argument("--horizons", type=str, default="1,3,5,10,15,30,60,eod",
                   help="逗号分隔。数字代表 entry 后 N 根 1min bar，eod 代表尾盘 force_flat 前后。")

    # 覆盖默认信号参数
    p.add_argument("--opening-end", type=str, default="09:45")
    p.add_argument("--rel-open-amount-min", type=float, default=1.30)
    p.add_argument("--noise-k", type=float, default=1.25)
    p.add_argument("--rv-z-max", type=float, default=1.50)
    p.add_argument("--rv-rank-pct-max", type=float, default=0.80)
    p.add_argument("--category-strength-min", type=float, default=0.55)
    p.add_argument("--expected-edge-mult", type=float, default=1.50)
    p.add_argument("--assumed-roundtrip-cost-bp", type=float, default=4.0)

    return p.parse_args()


def norm_date(x) -> str:
    if x is None or str(x).strip() == "":
        return ""
    return str(x).replace("-", "")[:8]


def next_minute(hhmm: str) -> str:
    hh, mm = hhmm.split(":")
    total = int(hh) * 60 + int(mm) + 1
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_horizons(s: str) -> List[str]:
    out = []
    for x in str(s).split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x == "eod":
            out.append("eod")
        else:
            int(x)
            out.append(x)
    return out


# ============================================================
# 2. 读取与补充字段
# ============================================================

def read_panel(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount", "vol",
        "bar_index", "day_open", "daily_close", "daily_ret",
        "intraday_move", "abs_intraday_move", "intraday_vwap",
        "ret_5m", "amount_z_20m",
        "valid_entry_bar", "force_flat_bar",
        "daily_adj_factor_change", "daily_is_extreme_return_day",
        "cat_positive_ratio", "cat_ret5_positive_ratio",
        "rv_z", "rv_rank_pct",
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

    if start_date:
        d = d[d["trade_date"] >= start_date].copy()
    if end_date:
        d = d[d["trade_date"] <= end_date].copy()

    for c in ["open", "high", "low", "close", "amount", "vol",
              "day_open", "daily_close", "daily_ret",
              "intraday_move", "abs_intraday_move", "intraday_vwap",
              "ret_5m", "amount_z_20m", "cat_positive_ratio",
              "cat_ret5_positive_ratio", "rv_z", "rv_rank_pct"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype("Int64")

    if d["day_open"].isna().all():
        d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")
    if d["intraday_move"].isna().all():
        d["intraday_move"] = d["close"] / d["day_open"] - 1.0
    if d["abs_intraday_move"].isna().all():
        d["abs_intraday_move"] = d["intraday_move"].abs()

    # 兜底计算类别同步与 relative value
    if d["cat_positive_ratio"].isna().all() or d["cat_ret5_positive_ratio"].isna().all():
        d["positive_intraday"] = (d["intraday_move"] > 0).astype(int)
        d["positive_5m"] = (d["ret_5m"] > 0).astype(int)
        d["cat_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
        d["cat_ret5_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")

    if d["rv_z"].isna().all() or d["rv_rank_pct"].isna().all():
        g = d.groupby(["trade_time", "t0_category"])
        mean = g["intraday_move"].transform("mean")
        std = g["intraday_move"].transform("std")
        d["rv_z"] = (d["intraday_move"] - mean) / std.replace(0, np.nan)
        d["rv_rank_pct"] = g["intraday_move"].rank(pct=True)

    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return d


# ============================================================
# 3. 信号特征
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


def add_orb_features(panel: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
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


def add_noise_features(panel: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
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


def add_expected_edge(d: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
    x = d.copy()
    x["or_range_bp"] = x["or_range"].fillna(0) * 10000.0
    x["noise_bp"] = x["noise"].fillna(0) * p.noise_k * 10000.0
    x["breakout_extra_bp"] = x[["price_to_or_high", "price_to_noise_upper"]].max(axis=1).fillna(0).clip(lower=0) * 10000.0
    x["expected_edge_bp"] = np.maximum(x["or_range_bp"], x["noise_bp"]) + 0.5 * x["breakout_extra_bp"]
    x["edge_ok"] = x["expected_edge_bp"] >= p.expected_edge_mult * p.assumed_roundtrip_cost_bp
    return x


def build_signal_mask(d: pd.DataFrame, p: SignalParams, scope: str) -> pd.Series:
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


# ============================================================
# 4. 信号 forward return
# ============================================================

def time_bucket(clock_time: str) -> str:
    if clock_time < "10:00":
        return "09:46-10:00"
    if clock_time < "10:30":
        return "10:00-10:30"
    if clock_time < "11:30":
        return "10:30-11:30"
    if clock_time < "13:30":
        return "13:00-13:30"
    if clock_time < "14:15":
        return "13:30-14:15"
    return "14:15+"


def add_state_bins(signal: pd.DataFrame) -> pd.DataFrame:
    x = signal.copy()

    x["month"] = x["trade_date"].astype(str).str[:6]
    x["signal_time_bucket"] = x["signal_clock_time"].map(time_bucket)

    # 分箱用 qcut，遇到重复值自动降级。
    def qbin(s: pd.Series, labels: List[str]) -> pd.Series:
        try:
            return pd.qcut(s, q=len(labels), labels=labels, duplicates="drop")
        except Exception:
            return pd.Series(["unknown"] * len(s), index=s.index)

    x["rel_open_amount_bin"] = qbin(x["rel_open_amount"].replace([np.inf, -np.inf], np.nan), ["low", "mid", "high"])
    x["or_range_bin"] = qbin(x["or_range"].replace([np.inf, -np.inf], np.nan), ["low", "mid", "high"])
    x["expected_edge_bin"] = qbin(x["expected_edge_bp"].replace([np.inf, -np.inf], np.nan), ["low", "mid", "high"])

    # 事后状态：用于解释，不用于真实开仓。
    # 这可以帮助判断信号到底依赖趋势日还是震荡日。
    x["abs_daily_ret"] = x["daily_ret"].abs()
    x["daily_state"] = np.where(
        x["abs_daily_ret"] >= x["abs_daily_ret"].quantile(0.67),
        "trend_day_expost",
        np.where(x["abs_daily_ret"] <= x["abs_daily_ret"].quantile(0.33), "quiet_day_expost", "normal_day_expost")
    )
    x["daily_direction"] = np.where(x["daily_ret"] > 0, "up_day", np.where(x["daily_ret"] < 0, "down_day", "flat_day"))

    return x


def extract_signal_forward_returns(d: pd.DataFrame, p: SignalParams, scope: str, horizons: List[str]) -> pd.DataFrame:
    x = d.copy()
    x["_signal"] = build_signal_mask(x, p, scope).astype(int)

    rows = []

    groups = x.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups

    for i, (_, day) in enumerate(groups, 1):
        if i % 3000 == 0:
            print(f"  extracting signal returns: ETF-days {i}/{total}")

        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        if len(sig_pos) == 0:
            continue

        # eod 用最后一个 force_flat_bar 前后可成交 bar 的 close；若没有，则用最后 close。
        if "force_flat_bar" in day.columns and (day["force_flat_bar"] == 1).any():
            eod_idx = int(np.flatnonzero(day["force_flat_bar"].to_numpy() == 1)[0])
        else:
            eod_idx = len(day) - 1

        for sp in sig_pos:
            entry_pos = int(sp) + 1
            if entry_pos >= len(day):
                continue

            sig = day.iloc[int(sp)]
            ent = day.iloc[entry_pos]

            if int(ent.get("force_flat_bar", 0)) == 1:
                continue

            entry_price = float(ent["open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            base = {
                "ts_code": sig["ts_code"],
                "name": sig.get("name", ""),
                "t0_category": sig.get("t0_category", ""),
                "t0_category_cn": sig.get("t0_category_cn", ""),
                "trade_date": sig["trade_date"],
                "signal_time": sig["trade_time"],
                "signal_clock_time": sig["clock_time"],
                "entry_time": ent["trade_time"],
                "entry_clock_time": ent["clock_time"],
                "signal_close": float(sig["close"]),
                "entry_open": entry_price,
                "daily_ret": float(sig["daily_ret"]) if np.isfinite(sig["daily_ret"]) else np.nan,
                "or_range": float(sig["or_range"]) if np.isfinite(sig["or_range"]) else np.nan,
                "or_return": float(sig["or_return"]) if np.isfinite(sig["or_return"]) else np.nan,
                "rel_open_amount": float(sig["rel_open_amount"]) if np.isfinite(sig["rel_open_amount"]) else np.nan,
                "rel_open_amount_rank": float(sig["rel_open_amount_rank"]) if np.isfinite(sig["rel_open_amount_rank"]) else np.nan,
                "noise": float(sig["noise"]) if np.isfinite(sig["noise"]) else np.nan,
                "price_to_or_high": float(sig["price_to_or_high"]) if np.isfinite(sig["price_to_or_high"]) else np.nan,
                "price_to_noise_upper": float(sig["price_to_noise_upper"]) if np.isfinite(sig["price_to_noise_upper"]) else np.nan,
                "rv_z": float(sig["rv_z"]) if np.isfinite(sig["rv_z"]) else np.nan,
                "rv_rank_pct": float(sig["rv_rank_pct"]) if np.isfinite(sig["rv_rank_pct"]) else np.nan,
                "cat_positive_ratio": float(sig["cat_positive_ratio"]) if np.isfinite(sig["cat_positive_ratio"]) else np.nan,
                "cat_ret5_positive_ratio": float(sig["cat_ret5_positive_ratio"]) if np.isfinite(sig["cat_ret5_positive_ratio"]) else np.nan,
                "expected_edge_bp": float(sig["expected_edge_bp"]) if np.isfinite(sig["expected_edge_bp"]) else np.nan,
            }

            for h in horizons:
                if h == "eod":
                    fpos = eod_idx
                    if fpos <= entry_pos:
                        continue
                    exit_price = float(day.iloc[fpos]["close"])
                    horizon_label = "eod"
                    holding_bars = int(fpos - entry_pos)
                else:
                    n = int(h)
                    fpos = entry_pos + n
                    if fpos >= len(day):
                        continue
                    if fpos >= eod_idx:
                        continue
                    exit_price = float(day.iloc[fpos]["close"])
                    horizon_label = f"{n}m"
                    holding_bars = n

                if not np.isfinite(exit_price) or exit_price <= 0:
                    continue

                r = base.copy()
                r["horizon"] = horizon_label
                r["holding_bars"] = holding_bars
                r["future_price"] = exit_price
                r["fwd_ret"] = exit_price / entry_price - 1.0
                r["fwd_ret_bp"] = r["fwd_ret"] * 10000.0
                rows.append(r)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_state_bins(out)
    return out


# ============================================================
# 5. 汇总
# ============================================================

def summarise(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def q25(s):
        return s.quantile(0.25)

    def q75(s):
        return s.quantile(0.75)

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            signal_count=("fwd_ret_bp", "size"),
            mean_bp=("fwd_ret_bp", "mean"),
            median_bp=("fwd_ret_bp", "median"),
            win_rate=("fwd_ret_bp", lambda s: (s > 0).mean()),
            p25_bp=("fwd_ret_bp", q25),
            p75_bp=("fwd_ret_bp", q75),
            std_bp=("fwd_ret_bp", "std"),
            avg_expected_edge_bp=("expected_edge_bp", "mean"),
            avg_rel_open_amount=("rel_open_amount", "mean"),
            avg_or_range_bp=("or_range", lambda s: s.mean() * 10000.0),
        )
        .reset_index()
    )

    return out


def write_report(out_dir: Path, params: SignalParams, scope: str, raw: pd.DataFrame,
                 summary: pd.DataFrame, by_etf: pd.DataFrame, by_state: pd.DataFrame,
                 by_month: pd.DataFrame, by_time: pd.DataFrame) -> None:
    lines = []
    lines.append("# 组合策略裸信号 forward return 诊断报告\n")
    lines.append("## 1. 诊断目的\n")
    lines.append("本报告不看止盈、止损、仓位约束和动态退出，只检查组合策略入场信号出现后，下一根 1min open 买入，在不同 horizon 上的原始价格变化。")
    lines.append("")
    lines.append("## 2. 信号参数\n")
    lines.append(f"- scope: **{scope}**")
    lines.append(pd.DataFrame([asdict(params)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 总体 forward return\n")
    if not summary.empty:
        lines.append(summary.to_markdown(index=False))
    else:
        lines.append("无信号。")
    lines.append("")
    lines.append("## 4. 关键解释\n")
    lines.append("- 如果 5m/10m/15m 的 mean_bp 明显小于 5bp，说明信号本身不厚，单靠止盈止损很难救。")
    lines.append("- 如果某个 horizon 的 mean_bp 较高，但完整回测收益低，说明退出或仓位约束可能在损耗收益。")
    lines.append("- 如果收益集中在少数 ETF、月份或状态，后续应做状态过滤，而不是全样本硬调。")
    lines.append("")
    if not by_etf.empty:
        lines.append("## 5. ETF 归因：按 10m horizon 排序")
        show = by_etf[by_etf["horizon"] == "10m"].sort_values("mean_bp", ascending=False).head(30)
        lines.append(show.to_markdown(index=False))
        lines.append("")
    if not by_time.empty:
        lines.append("## 6. 日内时间段归因")
        lines.append(by_time.to_markdown(index=False))
        lines.append("")
    if not by_state.empty:
        lines.append("## 7. 市场状态归因")
        lines.append(by_state.to_markdown(index=False))
        lines.append("")
    if not by_month.empty:
        lines.append("## 8. 月度稳定性")
        show = by_month[by_month["horizon"].isin(["5m", "10m", "15m"])]
        lines.append(show.head(200).to_markdown(index=False))
        lines.append("")

    (out_dir / "signal_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 6. 主流程
# ============================================================

def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = norm_date(args.start_date)
    end_date = norm_date(args.end_date)

    p = SignalParams(
        opening_end=args.opening_end,
        entry_start=next_minute(args.opening_end),
        rel_open_amount_min=args.rel_open_amount_min,
        noise_k=args.noise_k,
        rv_z_max=args.rv_z_max,
        rv_rank_pct_max=args.rv_rank_pct_max,
        category_strength_min=args.category_strength_min,
        expected_edge_mult=args.expected_edge_mult,
        assumed_roundtrip_cost_bp=args.assumed_roundtrip_cost_bp,
    )

    horizons = parse_horizons(args.horizons)

    print("=" * 100)
    print("Diagnose combo signal forward returns")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"scope      : {args.scope}")
    print(f"date range : {start_date or 'ALL'} -> {end_date or 'ALL'}")
    print(f"horizons   : {horizons}")
    print("=" * 100)

    panel = read_panel(Path(args.panel_file), start_date, end_date)

    print("Adding ORB features...")
    d = add_orb_features(panel, p)

    print("Adding noise features...")
    d = add_noise_features(d, p)

    print("Adding expected edge...")
    d = add_expected_edge(d, p)

    print("Extracting signal forward returns...")
    raw = extract_signal_forward_returns(d, p, args.scope, horizons)

    if raw.empty:
        print("[WARN] 没有提取到任何信号。")
    else:
        print(f"signals × horizons rows: {len(raw):,}")
        print(f"unique signals: {raw[['ts_code', 'trade_date', 'signal_time']].drop_duplicates().shape[0]:,}")

    raw.to_csv(out_dir / "signal_forward_return_raw.csv", index=False, encoding="utf-8-sig")

    summary = summarise(raw, ["horizon"])
    by_etf = summarise(raw, ["horizon", "ts_code", "name", "t0_category_cn"])
    by_month = summarise(raw, ["horizon", "month"])
    by_time = summarise(raw, ["horizon", "signal_time_bucket"])
    by_state = summarise(raw, ["horizon", "daily_state", "daily_direction", "rel_open_amount_bin", "or_range_bin", "expected_edge_bin"])

    summary.to_csv(out_dir / "signal_forward_return_summary.csv", index=False, encoding="utf-8-sig")
    by_etf.to_csv(out_dir / "signal_forward_return_by_etf.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(out_dir / "signal_forward_return_by_month.csv", index=False, encoding="utf-8-sig")
    by_time.to_csv(out_dir / "signal_forward_return_by_time.csv", index=False, encoding="utf-8-sig")
    by_state.to_csv(out_dir / "signal_forward_return_by_state.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel_file": str(Path(args.panel_file).resolve()),
        "out_dir": str(out_dir.resolve()),
        "scope": args.scope,
        "start_date": start_date,
        "end_date": end_date,
        "horizons": horizons,
        "signal_params": asdict(p),
        "raw_rows": int(len(raw)),
        "unique_signal_count": int(raw[["ts_code", "trade_date", "signal_time"]].drop_duplicates().shape[0]) if not raw.empty else 0,
    }
    (out_dir / "signal_diagnostics_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(out_dir, p, args.scope, raw, summary, by_etf, by_state, by_month, by_time)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"summary : {(out_dir / 'signal_forward_return_summary.csv').resolve()}")
    print(f"by_etf  : {(out_dir / 'signal_forward_return_by_etf.csv').resolve()}")
    print(f"by_state: {(out_dir / 'signal_forward_return_by_state.csv').resolve()}")
    print(f"report  : {(out_dir / 'signal_diagnostics_report.md').resolve()}")
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
