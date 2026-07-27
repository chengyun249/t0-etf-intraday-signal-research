#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_t0_intraday_bar_panel.py

用途：
    将 30 只 T+0 ETF 的 1min 原始行情，构造成日内交易策略使用的 bar 级别面板。

输入：
    1. data_t0/raw_1min/freq=1min/*.parquet
    2. data_t0/config/t0_etf_selected_30_detail.csv 或 t0_etf_codes.csv
    3. data_t0/auxiliary/daily_regime_features.csv
    4. data_t0/auxiliary/index_regime_features.csv，可选

输出：
    data_t0/processed/t0_intraday_bar_panel.parquet
    data_t0/processed/t0_feature_manifest.json
    data_t0/processed/t0_intraday_panel_audit_summary.csv
    data_t0/processed/t0_intraday_bad_days.csv
    data_t0/processed/t0_intraday_panel_report.md

核心原则：
    - 样本单位：ETF × 1min bar；
    - 所有特征只使用当前 bar 及以前信息；
    - 标签是未来 5 / 15 / 30 / 60 分钟收益；
    - 不隔夜计算分钟收益或未来标签；
    - 复权因子变化日、低流动性日作为过滤标记保留；
    - 第一版特征服务 VWAP 均值回归和日内突破动量，不做复杂机器学习堆砌。

运行示例：
    python ".\\build_t0_intraday_bar_panel.py" `
      --raw-dir ".\\data_t0\\raw_1min\\freq=1min" `
      --universe-file ".\\data_t0\\config\\t0_etf_selected_30_detail.csv" `
      --daily-regime-file ".\\data_t0\\auxiliary\\daily_regime_features.csv" `
      --out-dir ".\\data_t0\\processed"

如果你的 universe 文件只有 ts_code 列：
    python ".\\build_t0_intraday_bar_panel.py" `
      --raw-dir ".\\data_t0\\raw_1min\\freq=1min" `
      --universe-file ".\\data_t0\\config\\t0_etf_codes.csv" `
      --daily-regime-file ".\\data_t0\\auxiliary\\daily_regime_features.csv" `
      --out-dir ".\\data_t0\\processed"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build T+0 ETF intraday 1min bar panel.")
    p.add_argument("--raw-dir", type=str, default="./data_t0/raw_1min/freq=1min", help="1min parquet 目录。")
    p.add_argument("--universe-file", type=str, default="./data_t0/config/t0_etf_selected_30_detail.csv", help="ETF 池文件。")
    p.add_argument("--daily-regime-file", type=str, default="./data_t0/auxiliary/daily_regime_features.csv", help="ETF 日频状态特征。")
    p.add_argument("--index-regime-file", type=str, default="./data_t0/auxiliary/index_regime_features.csv", help="指数日频状态特征，可选。")
    p.add_argument("--out-dir", type=str, default="./data_t0/processed", help="输出目录。")
    p.add_argument("--freq", type=str, default="1min", help="频率标签。")
    p.add_argument("--min-bars-per-day", type=int, default=235, help="日内最少 bar 数，低于该值标记为不完整。")
    p.add_argument("--entry-start-time", type=str, default="09:40", help="允许开仓开始时间。")
    p.add_argument("--entry-end-time", type=str, default="14:30", help="允许新开仓截止时间。")
    p.add_argument("--force-flat-time", type=str, default="14:50", help="强制平仓参考时间。")
    p.add_argument("--cost-bp", type=float, default=2.0, help="单边成本 bp；v2统一基准为2bp。")
    return p.parse_args()


# =============================================================================
# IO
# =============================================================================

def read_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"universe-file 不存在：{path}")

    df = pd.read_csv(path)
    if "ts_code" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "ts_code"})
        else:
            raise ValueError("universe-file 必须包含 ts_code 列。")

    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df = df[df["ts_code"].notna() & (df["ts_code"] != "")].drop_duplicates("ts_code").copy()

    # 字段兜底
    if "t0_category" not in df.columns:
        df["t0_category"] = "unknown"
    if "t0_category_cn" not in df.columns:
        if "category_cn" in df.columns:
            df["t0_category_cn"] = df["category_cn"]
        else:
            df["t0_category_cn"] = df["t0_category"]
    if "name" not in df.columns:
        df["name"] = ""

    return df[["ts_code", "name", "t0_category", "t0_category_cn"]].copy()


def load_minute_data(raw_dir: Path, codes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir 不存在：{raw_dir}")

    parts = []
    file_rows = []
    for code in codes:
        pattern = code.replace(".", "_") + "_*.parquet"
        files = sorted(raw_dir.glob(pattern))
        if not files:
            file_rows.append({"ts_code": code, "file": "", "rows": 0, "status": "missing"})
            continue
        for f in files:
            try:
                d = pd.read_parquet(f)
                d["source_file"] = str(f)
                parts.append(d)
                file_rows.append({"ts_code": code, "file": str(f), "rows": len(d), "status": "ok"})
            except Exception as exc:
                file_rows.append({"ts_code": code, "file": str(f), "rows": 0, "status": f"error: {type(exc).__name__}: {exc}"})

    if not parts:
        raise RuntimeError("没有读到任何 1min parquet。")

    df = pd.concat(parts, ignore_index=True, sort=False)
    return df, pd.DataFrame(file_rows)


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "trade_time" not in d.columns:
        for c in ["datetime", "time", "trade_date"]:
            if c in d.columns:
                d = d.rename(columns={c: "trade_time"})
                break
    if "trade_time" not in d.columns:
        raise ValueError("原始分钟数据缺少 trade_time。")

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_time"].dt.strftime("%Y%m%d")
    d["clock_time"] = d["trade_time"].dt.strftime("%H:%M")
    d["date"] = d["trade_time"].dt.date

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if "ts_code" not in d.columns:
        raise ValueError("原始分钟数据缺少 ts_code。")

    d["ts_code"] = d["ts_code"].astype(str).str.strip()
    d = d.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    return d


# =============================================================================
# 审计
# =============================================================================

def in_regular_session(clock: pd.Series) -> pd.Series:
    # A股 ETF 常规时段：09:30-11:30，13:00-15:00。保守保留闭区间。
    return ((clock >= "09:30") & (clock <= "11:30")) | ((clock >= "13:00") & (clock <= "15:00"))


def audit_raw(d: pd.DataFrame, min_bars_per_day: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    bad_ohlc = (
        (d["high"] < d[["open", "close", "low"]].max(axis=1))
        | (d["low"] > d[["open", "close", "high"]].min(axis=1))
    )
    non_pos = (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
    neg_va = (d[["vol", "amount"]] < 0).any(axis=1)
    out_session = ~in_regular_session(d["clock_time"])

    daily = d.groupby(["ts_code", "trade_date"], as_index=False).agg(
        rows=("trade_time", "count"),
        start_time=("clock_time", "min"),
        end_time=("clock_time", "max"),
        first_close=("close", "first"),
        last_close=("close", "last"),
        day_amount=("amount", "sum"),
        day_vol=("vol", "sum"),
        max_abs_1m_ret=("close", lambda s: s.pct_change().abs().max()),
    )
    daily["is_incomplete_day"] = (daily["rows"] < min_bars_per_day).astype(int)
    daily["is_zero_amount_day"] = (daily["day_amount"].fillna(0) <= 0).astype(int)
    daily["is_suspect_day"] = ((daily["is_incomplete_day"] == 1) | (daily["is_zero_amount_day"] == 1)).astype(int)

    overall = {
        "rows": int(len(d)),
        "etf_count": int(d["ts_code"].nunique()),
        "trading_days": int(d["trade_date"].nunique()),
        "etf_day_count": int(daily.shape[0]),
        "duplicate_rows": int(d.duplicated(["ts_code", "trade_time"]).sum()),
        "out_of_session_rows": int(out_session.sum()),
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "non_positive_price_rows": int(non_pos.sum()),
        "negative_vol_amount_rows": int(neg_va.sum()),
        "incomplete_etf_days": int(daily["is_incomplete_day"].sum()),
        "zero_amount_etf_days": int(daily["is_zero_amount_day"].sum()),
        "median_rows_per_day": float(daily["rows"].median()) if len(daily) else np.nan,
        "min_rows_per_day": int(daily["rows"].min()) if len(daily) else 0,
        "max_rows_per_day": int(daily["rows"].max()) if len(daily) else 0,
    }

    bad_days = daily[daily["is_suspect_day"] == 1].copy()
    return daily, bad_days, overall


# =============================================================================
# 特征构造
# =============================================================================

def add_intraday_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    单 ETF 单日组内特征，全部只使用当前及历史 bar。
    """
    x = g.sort_values("trade_time").copy()
    x["bar_index"] = np.arange(len(x))
    x["n_bars_day"] = len(x)

    # 当前 bar 收盘相对过去 close 的收益，不跨日。
    x["ret_1m"] = x["close"].pct_change(1)
    for n in [3, 5, 10, 20, 30, 60]:
        x[f"ret_{n}m"] = x["close"].pct_change(n)

    # 成交额/量滚动状态
    for n in [5, 10, 20, 30, 60]:
        x[f"amount_ma_{n}m"] = x["amount"].rolling(n, min_periods=max(3, n // 2)).mean()
        x[f"amount_std_{n}m"] = x["amount"].rolling(n, min_periods=max(3, n // 2)).std()
        x[f"amount_z_{n}m"] = (x["amount"] - x[f"amount_ma_{n}m"]) / x[f"amount_std_{n}m"].replace(0, np.nan)

    # 成交量加权日内 VWAP：用 close × vol 构造，避免 amount 单位问题。
    vol_pos = x["vol"].clip(lower=0).fillna(0)
    cum_vol = vol_pos.cumsum()
    cum_pv = (x["close"] * vol_pos).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    # 成交量为0时，用 expanding close mean 兜底
    vwap_fallback = x["close"].expanding(min_periods=1).mean()
    x["intraday_vwap"] = vwap.fillna(vwap_fallback)
    x["price_to_vwap"] = x["close"] / x["intraday_vwap"] - 1.0

    # price_to_vwap 的日内 expanding z-score
    exp_mean = x["price_to_vwap"].expanding(min_periods=10).mean()
    exp_std = x["price_to_vwap"].expanding(min_periods=10).std()
    x["price_to_vwap_z"] = (x["price_to_vwap"] - exp_mean) / exp_std.replace(0, np.nan)

    # 日内高低点位置，只用当前之前的 running high/low
    x["running_high"] = x["high"].cummax()
    x["running_low"] = x["low"].cummin()
    rng = (x["running_high"] - x["running_low"]).replace(0, np.nan)
    x["position_in_day_range"] = (x["close"] - x["running_low"]) / rng
    x["distance_to_intraday_high"] = x["close"] / x["running_high"] - 1.0
    x["distance_to_intraday_low"] = x["close"] / x["running_low"] - 1.0

    # 过去窗口突破/跌破：用 shift(1) 避免把当前 high/low 自己纳入突破阈值
    for n in [5, 10, 20, 30]:
        past_high = x["high"].rolling(n, min_periods=max(3, n // 2)).max().shift(1)
        past_low = x["low"].rolling(n, min_periods=max(3, n // 2)).min().shift(1)
        x[f"breakout_high_{n}m"] = (x["close"] > past_high).astype(int)
        x[f"breakdown_low_{n}m"] = (x["close"] < past_low).astype(int)

    # 短期波动
    for n in [5, 10, 20, 30]:
        x[f"realized_vol_{n}m"] = x["ret_1m"].rolling(n, min_periods=max(3, n // 2)).std()
        downside = x["ret_1m"].where(x["ret_1m"] < 0, 0.0)
        x[f"downside_vol_{n}m"] = downside.rolling(n, min_periods=max(3, n // 2)).std()

    # 连续上涨/下跌 bar 数
    pos = (x["ret_1m"] > 0).astype(int)
    neg = (x["ret_1m"] < 0).astype(int)
    x["up_bar_streak"] = pos.groupby((pos != pos.shift()).cumsum()).cumcount() + 1
    x.loc[pos == 0, "up_bar_streak"] = 0
    x["down_bar_streak"] = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
    x.loc[neg == 0, "down_bar_streak"] = 0

    # 未来收益标签，不跨日
    for n in [1, 3, 5, 10, 15, 30, 60]:
        x[f"future_ret_{n}m"] = x["close"].shift(-n) / x["close"] - 1.0

    return x


def add_time_features(d: pd.DataFrame, entry_start: str, entry_end: str, force_flat: str) -> pd.DataFrame:
    x = d.copy()
    x["is_open_10m"] = ((x["clock_time"] >= "09:30") & (x["clock_time"] < "09:40")).astype(int)
    x["is_open_30m"] = ((x["clock_time"] >= "09:30") & (x["clock_time"] < "10:00")).astype(int)
    x["is_morning"] = ((x["clock_time"] >= "09:30") & (x["clock_time"] <= "11:30")).astype(int)
    x["is_afternoon"] = ((x["clock_time"] >= "13:00") & (x["clock_time"] <= "15:00")).astype(int)
    x["is_tail_30m"] = ((x["clock_time"] >= "14:30") & (x["clock_time"] <= "15:00")).astype(int)

    # 简单计算距离收盘分钟数：上午按 15:00 粗略不连续无所谓，主要用于过滤/特征。
    t = pd.to_datetime(x["clock_time"], format="%H:%M", errors="coerce")
    close_t = pd.Timestamp("1900-01-01 15:00")
    x["minutes_to_close"] = ((close_t - t).dt.total_seconds() / 60.0).clip(lower=0)

    x["entry_allowed_time"] = ((x["clock_time"] >= entry_start) & (x["clock_time"] <= entry_end)).astype(int)
    x["force_flat_bar"] = (x["clock_time"] >= force_flat).astype(int)
    return x


def add_cross_section_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        std = s.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    # 全池同一分钟横截面
    for col in ["ret_5m", "ret_10m", "ret_20m", "price_to_vwap", "amount_z_20m", "realized_vol_20m"]:
        if col in x.columns:
            x[f"cs_z_{col}"] = x.groupby("trade_time")[col].transform(zscore)
            x[f"cs_rank_{col}"] = x.groupby("trade_time")[col].rank(pct=True)

    # 类别内同一分钟横截面
    if "t0_category" in x.columns:
        for col in ["ret_5m", "ret_10m", "price_to_vwap", "amount_z_20m"]:
            if col in x.columns:
                x[f"cat_z_{col}"] = x.groupby(["trade_time", "t0_category"])[col].transform(zscore)
                x[f"cat_rank_{col}"] = x.groupby(["trade_time", "t0_category"])[col].rank(pct=True)

    return x


def merge_daily_regime(panel: pd.DataFrame, daily_file: Path) -> pd.DataFrame:
    if not daily_file.exists():
        print(f"[WARN] daily_regime_file 不存在，跳过：{daily_file}")
        return panel

    daily = pd.read_csv(daily_file)
    if daily.empty:
        return panel

    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)

    keep_cols = [
        "ts_code", "trade_date",
        "ret_1d", "ret_3d", "ret_5d", "ret_20d",
        "amount_ma_5", "amount_ma_20", "amount_ratio_5_20",
        "vol_5d", "vol_20d", "abs_ret_20d",
        "is_low_liquidity_day", "is_extreme_return_day", "is_high_vol_regime",
        "adj_factor", "adj_factor_change",
    ]
    keep_cols = [c for c in keep_cols if c in daily.columns]
    daily = daily[keep_cols].drop_duplicates(["ts_code", "trade_date"])

    # 为避免和分钟字段冲突，加前缀
    rename = {c: f"daily_{c}" for c in keep_cols if c not in ["ts_code", "trade_date"]}
    daily = daily.rename(columns=rename)

    return panel.merge(daily, on=["ts_code", "trade_date"], how="left")


def build_panel(raw: pd.DataFrame, universe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    d = normalize_raw(raw)
    d = d.merge(universe, on="ts_code", how="left")
    d["t0_category"] = d["t0_category"].fillna("unknown")
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"])

    # 只保留常规交易时段
    d = d[in_regular_session(d["clock_time"])].copy()

    # 时间特征
    d = add_time_features(d, args.entry_start_time, args.entry_end_time, args.force_flat_time)

    # ETF × 日内特征
    print("  building ETF-day rolling intraday features...")
    d = (
        d.groupby(["ts_code", "trade_date"], group_keys=False)
        .apply(add_intraday_features, include_groups=True)
        .reset_index(drop=True)
    )

    # 合并日频过滤器
    d = merge_daily_regime(d, Path(args.daily_regime_file))

    # 交易有效性过滤标记
    d["valid_price_bar"] = (
        d[["open", "high", "low", "close"]].notna().all(axis=1)
        & (d[["open", "high", "low", "close"]] > 0).all(axis=1)
    ).astype(int)

    d["valid_entry_bar"] = (
        (d["entry_allowed_time"] == 1)
        & (d["valid_price_bar"] == 1)
        & (d.get("daily_adj_factor_change", 0).fillna(0).astype(int) == 0)
        & (d.get("daily_is_extreme_return_day", 0).fillna(0).astype(int) == 0)
    ).astype(int)

    # 横截面特征放在最后，需要所有 ETF 同时存在
    print("  building cross-sectional features...")
    d = add_cross_section_features(d)

    # 成本覆盖标签
    one_way_cost = args.cost_bp / 10000.0
    roundtrip_cost = 2.0 * one_way_cost
    for horizon in [5, 15, 30, 60]:
        col = f"future_ret_{horizon}m"
        if col in d.columns:
            d[f"label_long_{horizon}m_gt_1x_cost"] = (d[col] > roundtrip_cost).astype(int)
            d[f"label_long_{horizon}m_gt_2x_cost"] = (d[col] > 2.0 * roundtrip_cost).astype(int)

    return d


def feature_manifest(panel: pd.DataFrame) -> Dict:
    exclude = {
        "ts_code", "name", "trade_time", "trade_date", "clock_time", "date", "source_file",
        "open", "high", "low", "close", "vol", "amount",
        "t0_category", "t0_category_cn",
    }
    label_cols = [c for c in panel.columns if c.startswith("future_ret_") or c.startswith("label_")]
    audit_cols = [
        "valid_price_bar", "valid_entry_bar", "entry_allowed_time", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_is_extreme_return_day", "daily_adj_factor_change",
    ]
    exclude.update(label_cols)
    exclude.update(audit_cols)

    feature_cols = []
    for c in panel.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            feature_cols.append(c)

    groups = {
        "short_momentum": [c for c in feature_cols if c.startswith("ret_") and c.endswith("m")],
        "vwap_deviation": [c for c in feature_cols if "vwap" in c],
        "intraday_position": [c for c in feature_cols if "intraday" in c or "day_range" in c or "distance_to" in c],
        "breakout_breakdown": [c for c in feature_cols if "breakout" in c or "breakdown" in c],
        "amount_state": [c for c in feature_cols if "amount" in c],
        "volatility_state": [c for c in feature_cols if "vol" in c and "daily_" not in c],
        "time_features": [c for c in feature_cols if c.startswith("is_") or c in ["bar_index", "n_bars_day", "minutes_to_close"]],
        "cross_sectional": [c for c in feature_cols if c.startswith("cs_") or c.startswith("cat_")],
        "daily_regime": [c for c in feature_cols if c.startswith("daily_")],
    }

    return {
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "audit_cols": audit_cols,
        "feature_groups": groups,
    }


def write_report(
    out_dir: Path,
    panel: pd.DataFrame,
    file_summary: pd.DataFrame,
    daily_quality: pd.DataFrame,
    bad_days: pd.DataFrame,
    overall: Dict,
    manifest: Dict,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# T+0 ETF 1min Bar 面板构建报告\n")
    lines.append("## 1. 输入\n")
    lines.append(f"- raw_dir: `{args.raw_dir}`")
    lines.append(f"- universe_file: `{args.universe_file}`")
    lines.append(f"- daily_regime_file: `{args.daily_regime_file}`")
    lines.append("")

    lines.append("## 2. 原始数据审计\n")
    for k, v in overall.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")

    lines.append("## 3. 输出面板规模\n")
    lines.append(f"- bar rows: **{len(panel):,}**")
    lines.append(f"- ETF 数量: **{panel['ts_code'].nunique()}**")
    lines.append(f"- 交易日数量: **{panel['trade_date'].nunique()}**")
    lines.append(f"- 特征数量: **{manifest['feature_count']}**")
    lines.append(f"- 标签数量: **{len(manifest['label_cols'])}**")
    lines.append(f"- valid_entry_bar 数量: **{int(panel['valid_entry_bar'].sum()) if 'valid_entry_bar' in panel.columns else 0}**")
    lines.append("")

    lines.append("## 4. 标签覆盖率\n")
    label_stats = []
    for c in manifest["label_cols"]:
        if c.startswith("future_ret_"):
            label_stats.append({
                "label": c,
                "non_missing_ratio": float(panel[c].notna().mean()),
                "mean": float(panel[c].mean(skipna=True)),
                "std": float(panel[c].std(skipna=True)),
            })
    if label_stats:
        lines.append(pd.DataFrame(label_stats).to_markdown(index=False))
    lines.append("")

    lines.append("## 5. 特征组\n")
    for g, feats in manifest["feature_groups"].items():
        lines.append(f"- {g}: {len(feats)}")
    lines.append("")

    lines.append("## 6. 可疑交易日\n")
    if bad_days.empty:
        lines.append("- 未发现低于最少 bar 数或零成交的 ETF-day。")
    else:
        lines.append(bad_days.head(30).to_markdown(index=False))
    lines.append("")

    lines.append("## 7. 后续使用原则\n")
    lines.append("- `valid_entry_bar=1` 才允许策略开仓。")
    lines.append("- 复权因子变化日、极端收益日不作为开仓日。")
    lines.append("- 均值回归和突破策略都应使用下一根 bar 成交，不能使用信号 bar 的 close 直接成交。")
    lines.append("- 第一版主标签建议看 `future_ret_30m`，对应未来 30 分钟收益。")

    (out_dir / "t0_intraday_panel_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = read_universe(Path(args.universe_file))
    codes = universe["ts_code"].tolist()

    print("=" * 100)
    print("Build T+0 ETF intraday bar panel")
    print("=" * 100)
    print(f"raw_dir          : {Path(args.raw_dir).resolve()}")
    print(f"universe         : {Path(args.universe_file).resolve()}")
    print(f"daily regime     : {Path(args.daily_regime_file).resolve()}")
    print(f"out_dir          : {out_dir.resolve()}")
    print(f"ETF count        : {len(codes)}")
    print("=" * 100)

    raw, file_summary = load_minute_data(Path(args.raw_dir), codes)
    raw = normalize_raw(raw)

    daily_quality, bad_days, overall = audit_raw(raw, args.min_bars_per_day)

    print("Audit summary:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    panel = build_panel(raw, universe, args)
    manifest = feature_manifest(panel)

    # 保存
    panel_path = out_dir / "t0_intraday_bar_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    file_summary.to_csv(out_dir / "t0_raw_file_summary.csv", index=False, encoding="utf-8-sig")
    daily_quality.to_csv(out_dir / "t0_intraday_daily_quality.csv", index=False, encoding="utf-8-sig")
    bad_days.to_csv(out_dir / "t0_intraday_bad_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([overall]).to_csv(out_dir / "t0_intraday_panel_audit_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "t0_feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_report(out_dir, panel, file_summary, daily_quality, bad_days, overall, manifest, args)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"panel        : {panel_path.resolve()}")
    print(f"manifest     : {(out_dir / 't0_feature_manifest.json').resolve()}")
    print(f"report       : {(out_dir / 't0_intraday_panel_report.md').resolve()}")
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
