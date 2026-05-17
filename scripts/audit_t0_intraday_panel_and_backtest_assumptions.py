#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_t0_intraday_panel_and_backtest_assumptions.py

用途：
    对 T+0 ETF 1min 面板和回测假设做“硬审计”，用于排查：
    1. 数据是否缺 bar / 重复 / 非交易时段 / OHLC 异常；
    2. 零成交、stale price、停牌式价格是否污染标签和交易；
    3. future_ret 标签是否跨日期、跨尾盘、午休处理是否合理；
    4. entry_time 是否错误地写成 signal_time + 1min，而不是下一根真实 bar；
    5. 商品池分类是否过粗；
    6. 回测 summary 是否被“0 笔交易 final_nav=1”误导；
    7. 若提供 trades 文件，检查交易入场/退出时间、成交额、价格匹配情况。

默认运行：
    python ".\\audit_t0_intraday_panel_and_backtest_assumptions.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --scope all

也可以只审计商品池：
    python ".\\audit_t0_intraday_panel_and_backtest_assumptions.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --scope commodity_focus `
      --out-dir ".\\data_t0_2022_2024\\audit_intraday_panel_commodity"

如果要附带审计某次模型输出：
    python ".\\audit_t0_intraday_panel_and_backtest_assumptions.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --scope commodity_focus `
      --backtest-summary-file ".\\data_t0_2022_2024\\ml_intraday_return_commodity_focus_h60_noexit\\return_model_backtest_summary.csv" `
      --trades-file ".\\data_t0_2022_2024\\ml_intraday_return_commodity_focus_h60_noexit\\return_model_best_trades.csv"

输出：
    audit_report.md
    audit_summary.json
    etf_day_quality.csv
    category_summary.csv
    horizon_label_audit.csv
    stale_price_summary.csv
    extreme_future_returns_h*.csv
    backtest_summary_audit.csv
    trade_audit.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 参数与工具
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit T+0 ETF intraday panel and backtest assumptions.")

    p.add_argument("--root-dir", type=str, default="./data_t0_2022_2024")
    p.add_argument("--panel-file", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")

    p.add_argument("--scope", type=str, default="all",
                   choices=["all", "commodity_focus", "cross_commodity", "no_bond_money"])

    p.add_argument("--horizons", type=str, default="15,30,60",
                   help="需要审计的 holding horizon bars，例如 15,30,60。")
    p.add_argument("--expected-bars-per-day", type=int, default=241)
    p.add_argument("--force-flat-time", type=str, default="14:55")
    p.add_argument("--entry-start", type=str, default="09:40")
    p.add_argument("--entry-end", type=str, default="14:15")
    p.add_argument("--extreme-ret-bp", type=float, default=300.0,
                   help="future_ret 绝对值超过该 bp 时输出为异常样本。")
    p.add_argument("--stale-run-bars", type=int, default=10,
                   help="连续零成交或连续价格不变超过该 bar 数则标记。")

    p.add_argument("--backtest-summary-file", type=str, default="")
    p.add_argument("--trades-file", type=str, default="")
    p.add_argument("--prediction-file", type=str, default="",
                   help="可选，模型 compact predictions 文件。用于检查测试输出自身。")

    return p.parse_args()


def norm_date(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("-", "")[:8]


def parse_int_list(s: str) -> List[int]:
    return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.root_dir)
    panel_file = Path(args.panel_file) if args.panel_file else root / "processed" / "t0_intraday_bar_panel.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else root / f"audit_intraday_panel_{args.scope}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return panel_file, out_dir


def to_hhmm(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce")
    return dt.dt.strftime("%H:%M")


def clock_to_minutes(clock: pd.Series) -> pd.Series:
    s = clock.astype(str).str.slice(0, 5)
    hh = pd.to_numeric(s.str.slice(0, 2), errors="coerce")
    mm = pd.to_numeric(s.str.slice(3, 5), errors="coerce")
    return hh * 60 + mm


def safe_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    x = df.copy()
    if max_rows is not None:
        x = x.head(max_rows)
    try:
        return x.to_markdown(index=index)
    except Exception:
        return x.to_string(index=index)


def scope_mask(d: pd.DataFrame, scope: str) -> pd.Series:
    cat = d.get("t0_category", pd.Series("", index=d.index)).fillna("").astype(str)
    cat_cn = d.get("t0_category_cn", pd.Series("", index=d.index)).fillna("").astype(str)

    if scope == "all":
        return pd.Series(True, index=d.index)
    if scope == "no_bond_money":
        return ~(cat.isin(["bond", "money_market"]) | cat_cn.str.contains("债券|货币", regex=True, na=False))
    if scope == "commodity_focus":
        return cat.isin(["gold_commodity"]) | cat_cn.str.contains("黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    if scope == "cross_commodity":
        return cat.isin(["cross_border", "gold_commodity"]) | cat_cn.str.contains("跨境|黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    raise ValueError(scope)


def max_consecutive_true(arr: np.ndarray) -> int:
    max_run = run = 0
    for v in arr.astype(bool):
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)


# ============================================================
# 数据读取
# ============================================================

def read_panel(path: Path, scope: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    d = pd.read_parquet(path)
    required = ["ts_code", "trade_time", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"面板缺少必要字段：{missing}")

    d["trade_time"] = pd.to_datetime(d["trade_time"], errors="coerce")
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)

    if "clock_time" not in d.columns:
        d["clock_time"] = to_hhmm(d["trade_time"])
    else:
        # 统一成 HH:MM，避免 HH:MM:SS 和 HH:MM 混用导致字符串比较混乱。
        ct = d["clock_time"].astype(str).str.slice(0, 5)
        fallback = to_hhmm(d["trade_time"])
        d["clock_time"] = np.where(ct.str.match(r"^\d{2}:\d{2}$", na=False), ct, fallback)

    for c in ["name", "t0_category", "t0_category_cn"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)

    if "bar_index" not in d.columns:
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce")

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_ret", "intraday_vwap", "day_open", "daily_close"]:
        if c not in d.columns:
            d[c] = np.nan

    d = d[scope_mask(d, scope)].copy()
    d = d.sort_values(["ts_code", "trade_date", "trade_time"]).reset_index(drop=True)
    return d


# ============================================================
# 审计模块
# ============================================================

def audit_basic_panel(d: pd.DataFrame, expected_bars: int, stale_run_bars: int) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary: Dict = {}

    summary["rows"] = int(len(d))
    summary["etf_count"] = int(d["ts_code"].nunique())
    summary["trading_days"] = int(d["trade_date"].nunique())
    summary["etf_day_count"] = int(d.groupby(["ts_code", "trade_date"]).ngroups)

    dup = d.duplicated(["ts_code", "trade_time"]).sum()
    summary["duplicate_ts_code_trade_time_rows"] = int(dup)

    # session audit
    m = clock_to_minutes(d["clock_time"])
    morning = (m >= 9 * 60 + 30) & (m <= 11 * 60 + 30)
    afternoon = (m >= 13 * 60) & (m <= 15 * 60)
    session_ok = morning | afternoon
    summary["out_of_session_rows"] = int((~session_ok).sum())

    # OHLC sanity
    bad_ohlc = (
        (d["open"] <= 0) | (d["high"] <= 0) | (d["low"] <= 0) | (d["close"] <= 0)
        | (d["high"] < d[["open", "close", "low"]].max(axis=1))
        | (d["low"] > d[["open", "close", "high"]].min(axis=1))
    )
    summary["bad_ohlc_rows"] = int(bad_ohlc.sum())
    summary["negative_vol_amount_rows"] = int(((d["vol"] < 0) | (d["amount"] < 0)).sum())
    summary["zero_amount_rows"] = int((d["amount"].fillna(0) == 0).sum())
    summary["zero_vol_rows"] = int((d["vol"].fillna(0) == 0).sum())

    # ETF-day quality
    g = d.groupby(["ts_code", "name", "t0_category", "t0_category_cn", "trade_date"], dropna=False)
    etf_day = g.agg(
        rows=("trade_time", "size"),
        first_time=("clock_time", "first"),
        last_time=("clock_time", "last"),
        day_amount=("amount", "sum"),
        day_vol=("vol", "sum"),
        min_price=("low", "min"),
        max_price=("high", "max"),
        close_first=("close", "first"),
        close_last=("close", "last"),
        zero_amount_bars=("amount", lambda s: int((s.fillna(0) == 0).sum())),
        zero_vol_bars=("vol", lambda s: int((s.fillna(0) == 0).sum())),
    ).reset_index()

    etf_day["day_ret_raw"] = etf_day["close_last"] / etf_day["close_first"] - 1.0
    etf_day["day_range"] = etf_day["max_price"] / etf_day["min_price"].replace(0, np.nan) - 1.0
    etf_day["incomplete_day"] = etf_day["rows"] != expected_bars
    etf_day["zero_amount_day"] = etf_day["day_amount"].fillna(0) <= 0
    etf_day["extreme_day_range"] = etf_day["day_range"].abs() > 0.10
    etf_day["extreme_day_ret"] = etf_day["day_ret_raw"].abs() > 0.08

    summary["incomplete_etf_days"] = int(etf_day["incomplete_day"].sum())
    summary["zero_amount_etf_days"] = int(etf_day["zero_amount_day"].sum())
    summary["extreme_day_range_gt_10pct"] = int(etf_day["extreme_day_range"].sum())
    summary["extreme_day_ret_gt_8pct"] = int(etf_day["extreme_day_ret"].sum())
    summary["median_rows_per_etf_day"] = float(etf_day["rows"].median())
    summary["min_rows_per_etf_day"] = int(etf_day["rows"].min())
    summary["max_rows_per_etf_day"] = int(etf_day["rows"].max())

    # stale price / zero amount runs
    stale_rows = []
    for keys, day in d.groupby(["ts_code", "name", "t0_category_cn", "trade_date"], sort=False):
        day = day.sort_values("trade_time")
        zero_amount_run = max_consecutive_true((day["amount"].fillna(0) <= 0).to_numpy())
        zero_vol_run = max_consecutive_true((day["vol"].fillna(0) <= 0).to_numpy())
        same_close_run = max_consecutive_true(day["close"].diff().fillna(1).eq(0).to_numpy())
        stale_rows.append({
            "ts_code": keys[0],
            "name": keys[1],
            "t0_category_cn": keys[2],
            "trade_date": keys[3],
            "max_zero_amount_run": zero_amount_run,
            "max_zero_vol_run": zero_vol_run,
            "max_same_close_run": same_close_run,
            "flag_long_zero_amount_run": zero_amount_run >= stale_run_bars,
            "flag_long_same_close_run": same_close_run >= stale_run_bars,
        })

    stale = pd.DataFrame(stale_rows)
    summary["etf_days_long_zero_amount_run"] = int(stale["flag_long_zero_amount_run"].sum()) if not stale.empty else 0
    summary["etf_days_long_same_close_run"] = int(stale["flag_long_same_close_run"].sum()) if not stale.empty else 0

    category = (
        d.groupby(["ts_code", "name", "t0_category", "t0_category_cn"], dropna=False)
        .agg(rows=("trade_time", "size"), days=("trade_date", "nunique"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values(["t0_category_cn", "total_amount"], ascending=[True, False])
    )

    return summary, etf_day, stale, category


def audit_horizon_labels(d: pd.DataFrame, horizons: List[int], force_flat_time: str,
                         entry_start: str, entry_end: str, extreme_ret_bp: float,
                         out_dir: Path) -> pd.DataFrame:
    x = d.sort_values(["ts_code", "trade_date", "trade_time"]).copy()
    gb = x.groupby(["ts_code", "trade_date"], group_keys=False)

    x["next_trade_time"] = gb["trade_time"].shift(-1)
    x["next_clock_time"] = gb["clock_time"].shift(-1)
    x["next_open"] = gb["open"].shift(-1)
    x["next_amount"] = gb["amount"].shift(-1)
    x["next_vol"] = gb["vol"].shift(-1)
    x["signal_plus_1min"] = x["trade_time"] + pd.Timedelta(minutes=1)

    base_entry = (
        (x["clock_time"] >= entry_start)
        & (x["clock_time"] <= entry_end)
        & (x["amount"].fillna(0) > 0)
        & (x["next_open"].fillna(0) > 0)
    )

    rows = []
    for h in horizons:
        y = x.copy()
        y["future_time_h"] = gb["trade_time"].shift(-(h + 1))
        y["future_clock_h"] = gb["clock_time"].shift(-(h + 1))
        y["future_close_h"] = gb["close"].shift(-(h + 1))
        y["future_amount_h"] = gb["amount"].shift(-(h + 1))
        y["future_vol_h"] = gb["vol"].shift(-(h + 1))
        y["future_trade_date_h"] = gb["trade_date"].shift(-(h + 1))

        y["future_ret_bp"] = (y["future_close_h"] / y["next_open"] - 1.0) * 10000.0
        y["next_time_mismatch_plus_1min"] = y["next_trade_time"] != y["signal_plus_1min"]
        y["cross_lunch_next"] = (y["clock_time"] <= "11:30") & (y["next_clock_time"] >= "13:00")
        y["cross_lunch_horizon"] = (y["clock_time"] <= "11:30") & (y["future_clock_h"] >= "13:00")
        y["future_valid"] = (
            base_entry
            & y["future_close_h"].notna()
            & (y["future_trade_date_h"] == y["trade_date"])
            & (y["future_clock_h"].astype(str) < force_flat_time)
            & (y["next_amount"].fillna(0) > 0)
            & (y["future_amount_h"].fillna(0) > 0)
        )

        valid = y[y["future_valid"]].copy()
        extreme = valid[valid["future_ret_bp"].abs() >= extreme_ret_bp].copy()
        if not extreme.empty:
            cols = [
                "ts_code", "name", "t0_category_cn", "trade_date", "trade_time", "clock_time",
                "next_trade_time", "next_clock_time", "future_time_h", "future_clock_h",
                "next_open", "future_close_h", "future_ret_bp",
                "amount", "next_amount", "future_amount_h",
            ]
            extreme[cols].sort_values("future_ret_bp", key=lambda s: s.abs(), ascending=False).head(1000).to_csv(
                out_dir / f"extreme_future_returns_h{h}.csv", index=False, encoding="utf-8-sig"
            )

        rows.append({
            "horizon_bars": h,
            "candidate_rows_by_time_amount": int(base_entry.sum()),
            "valid_future_rows": int(y["future_valid"].sum()),
            "valid_ratio": float(y["future_valid"].sum() / max(base_entry.sum(), 1)),
            "next_time_mismatch_plus_1min_rows": int((base_entry & y["next_time_mismatch_plus_1min"]).sum()),
            "cross_lunch_next_rows": int((base_entry & y["cross_lunch_next"]).sum()),
            "cross_lunch_horizon_valid_rows": int((y["future_valid"] & y["cross_lunch_horizon"]).sum()),
            "next_amount_zero_rows": int((base_entry & (y["next_amount"].fillna(0) <= 0)).sum()),
            "future_amount_zero_rows": int((base_entry & (y["future_amount_h"].fillna(0) <= 0)).sum()),
            "future_cross_date_rows": int((base_entry & y["future_trade_date_h"].notna() & (y["future_trade_date_h"] != y["trade_date"])).sum()),
            "future_after_force_flat_rows": int((base_entry & (y["future_clock_h"].astype(str) >= force_flat_time)).sum()),
            "future_ret_mean_bp": float(valid["future_ret_bp"].mean()) if not valid.empty else np.nan,
            "future_ret_median_bp": float(valid["future_ret_bp"].median()) if not valid.empty else np.nan,
            "future_ret_p25_bp": float(valid["future_ret_bp"].quantile(0.25)) if not valid.empty else np.nan,
            "future_ret_p75_bp": float(valid["future_ret_bp"].quantile(0.75)) if not valid.empty else np.nan,
            "future_ret_extreme_abs_rows": int(len(extreme)),
            "future_ret_gt_0_rate": float((valid["future_ret_bp"] > 0).mean()) if not valid.empty else np.nan,
            "future_ret_gt_4bp_rate": float((valid["future_ret_bp"] > 4).mean()) if not valid.empty else np.nan,
            "future_ret_lt_minus4bp_rate": float((valid["future_ret_bp"] < -4).mean()) if not valid.empty else np.nan,
        })

    return pd.DataFrame(rows)


def audit_backtest_summary(path: Path, out_dir: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()

    s = pd.read_csv(path)
    rows = []
    for cost, g in s.groupby("cost_bp"):
        g = g.copy()
        rows.append({
            "cost_bp": cost,
            "view": "best_including_zero_trade",
            **g.sort_values("final_nav", ascending=False).iloc[0].to_dict()
        })
        nz = g[g["trade_count"].fillna(0) > 0]
        if not nz.empty:
            rows.append({
                "cost_bp": cost,
                "view": "best_trade_count_gt_0",
                **nz.sort_values("final_nav", ascending=False).iloc[0].to_dict()
            })
        n20 = g[g["trade_count"].fillna(0) >= 20]
        if not n20.empty:
            rows.append({
                "cost_bp": cost,
                "view": "best_trade_count_ge_20",
                **n20.sort_values("final_nav", ascending=False).iloc[0].to_dict()
            })
        n50 = g[g["trade_count"].fillna(0) >= 50]
        if not n50.empty:
            rows.append({
                "cost_bp": cost,
                "view": "best_trade_count_ge_50",
                **n50.sort_values("final_nav", ascending=False).iloc[0].to_dict()
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "backtest_summary_audit.csv", index=False, encoding="utf-8-sig")
    return out


def audit_prediction_file(path: Path, out_dir: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()

    p = pd.read_csv(path)
    rows = []
    rows.append({
        "rows": int(len(p)),
        "ts_code_count": int(p["ts_code"].nunique()) if "ts_code" in p else np.nan,
        "trade_date_count": int(p["trade_date"].nunique()) if "trade_date" in p else np.nan,
        "mean_future_ret_bp": float(p["future_ret_bp"].mean()) if "future_ret_bp" in p else np.nan,
        "median_future_ret_bp": float(p["future_ret_bp"].median()) if "future_ret_bp" in p else np.nan,
        "extreme_future_ret_abs_gt_300bp": int((p["future_ret_bp"].abs() > 300).sum()) if "future_ret_bp" in p else np.nan,
        "candidate_event_rate": float(p["candidate_event"].mean()) if "candidate_event" in p else np.nan,
    })

    # optional decile checks
    for col in ["pred_ret_bp", "downside_prob", "pred_up_proba"]:
        if col in p.columns:
            try:
                p[f"{col}_decile"] = pd.qcut(p[col], 10, duplicates="drop")
                tab = p.groupby(f"{col}_decile", dropna=False).agg(
                    rows=("future_ret_bp", "size"),
                    mean_score=(col, "mean"),
                    mean_future_ret_bp=("future_ret_bp", "mean"),
                    median_future_ret_bp=("future_ret_bp", "median"),
                ).reset_index()
                tab.to_csv(out_dir / f"prediction_decile_{col}.csv", index=False, encoding="utf-8-sig")
            except Exception:
                pass

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "prediction_file_audit.csv", index=False, encoding="utf-8-sig")
    return out


def audit_trades_file(path: Path, panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()

    # Some strategy runs legitimately produce an empty trade file.
    # pandas.read_csv raises EmptyDataError when the file has zero bytes or no header.
    try:
        t = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        out = pd.DataFrame([{
            "trades_rows": 0,
            "note": "trades file is empty; this usually means the selected best strategy had zero trades."
        }])
        out.to_csv(out_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
        return out

    if t.empty:
        out = pd.DataFrame([{
            "trades_rows": 0,
            "note": "trades file has header but no rows."
        }])
        out.to_csv(out_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
        return out

    for c in ["signal_time", "entry_time", "exit_time"]:
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")
    if "trade_date" in t.columns:
        t["trade_date"] = t["trade_date"].astype(str).map(norm_date)

    p = panel.sort_values(["ts_code", "trade_date", "trade_time"]).copy()
    gb = p.groupby(["ts_code", "trade_date"], group_keys=False)
    p["next_trade_time"] = gb["trade_time"].shift(-1)
    p["next_open"] = gb["open"].shift(-1)
    p["next_amount"] = gb["amount"].shift(-1)
    p["next_vol"] = gb["vol"].shift(-1)

    signal_map = p[["ts_code", "trade_date", "trade_time", "next_trade_time", "next_open", "next_amount", "next_vol"]].rename(
        columns={"trade_time": "signal_time"}
    )
    merged = t.merge(signal_map, on=["ts_code", "trade_date", "signal_time"], how="left")

    entry_panel = p[["ts_code", "trade_date", "trade_time", "open", "close", "amount", "vol"]].rename(
        columns={
            "trade_time": "entry_time",
            "open": "panel_entry_open",
            "close": "panel_entry_close",
            "amount": "panel_entry_amount",
            "vol": "panel_entry_vol",
        }
    )
    merged = merged.merge(entry_panel, on=["ts_code", "trade_date", "entry_time"], how="left")

    exit_panel = p[["ts_code", "trade_date", "trade_time", "open", "close", "amount", "vol"]].rename(
        columns={
            "trade_time": "exit_time",
            "open": "panel_exit_open",
            "close": "panel_exit_close",
            "amount": "panel_exit_amount",
            "vol": "panel_exit_vol",
        }
    )
    merged = merged.merge(exit_panel, on=["ts_code", "trade_date", "exit_time"], how="left")

    merged["entry_time_should_be_next_bar"] = merged["entry_time"] == merged["next_trade_time"]
    if "entry_price" in merged.columns:
        merged["entry_price_diff"] = merged["entry_price"] - merged["next_open"]
        merged["entry_price_matches_next_open"] = merged["entry_price_diff"].abs() < 1e-8
    else:
        merged["entry_price_matches_next_open"] = np.nan

    if "exit_price" in merged.columns:
        # If path stop/take profit is used, exit_price may intentionally differ from panel close.
        merged["exit_price_diff_vs_panel_close"] = merged["exit_price"] - merged["panel_exit_close"]
    else:
        merged["exit_price_diff_vs_panel_close"] = np.nan

    summary = pd.DataFrame([{
        "trades_rows": int(len(merged)),
        "missing_signal_bar": int(merged["next_trade_time"].isna().sum()),
        "entry_time_not_next_bar": int((~merged["entry_time_should_be_next_bar"].fillna(False)).sum()),
        "entry_price_not_next_open": int((~merged["entry_price_matches_next_open"].fillna(False)).sum()) if "entry_price" in merged.columns else np.nan,
        "entry_bar_zero_amount": int((merged["panel_entry_amount"].fillna(0) <= 0).sum()),
        "exit_bar_zero_amount": int((merged["panel_exit_amount"].fillna(0) <= 0).sum()),
        "missing_entry_bar": int(merged["panel_entry_open"].isna().sum()),
        "missing_exit_bar": int(merged["panel_exit_close"].isna().sum()),
        "mean_gross_ret_bp": float(merged["gross_ret"].mean() * 10000.0) if "gross_ret" in merged else np.nan,
        "median_gross_ret_bp": float(merged["gross_ret"].median() * 10000.0) if "gross_ret" in merged else np.nan,
    }])

    merged.to_csv(out_dir / "trade_audit_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    return summary


def write_report(
    out_dir: Path,
    config: Dict,
    basic_summary: Dict,
    etf_day: pd.DataFrame,
    stale: pd.DataFrame,
    category: pd.DataFrame,
    horizon_audit: pd.DataFrame,
    bt_audit: pd.DataFrame,
    pred_audit: pd.DataFrame,
    trade_audit: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# T+0 ETF Intraday Panel & Backtest Assumption Audit\n")

    lines.append("## 1. 运行配置\n")
    lines.append(safe_to_markdown(pd.DataFrame([config])))
    lines.append("")

    lines.append("## 2. 面板基础审计\n")
    lines.append(safe_to_markdown(pd.DataFrame([basic_summary])))
    lines.append("")

    lines.append("## 3. ETF-day 质量问题 Top 样本\n")
    problem = etf_day[
        etf_day["incomplete_day"] | etf_day["zero_amount_day"] | etf_day["extreme_day_range"] | etf_day["extreme_day_ret"]
    ].copy()
    if problem.empty:
        lines.append("未发现 incomplete / zero_amount / extreme range / extreme ret 的 ETF-day。")
    else:
        lines.append(safe_to_markdown(problem.sort_values(["incomplete_day", "zero_amount_day", "day_range"], ascending=False), max_rows=50))
    lines.append("")

    lines.append("## 4. stale price / 零成交连续段\n")
    stale_problem = stale[stale["flag_long_zero_amount_run"] | stale["flag_long_same_close_run"]].copy()
    if stale_problem.empty:
        lines.append("未发现超过阈值的连续零成交或连续价格不变 ETF-day。")
    else:
        lines.append(safe_to_markdown(stale_problem.sort_values(["max_zero_amount_run", "max_same_close_run"], ascending=False), max_rows=50))
    lines.append("")

    lines.append("## 5. 分类与标的池审计\n")
    lines.append(safe_to_markdown(category, max_rows=80))
    lines.append("")
    lines.append("提示：如果 commodity_focus 内部的 t0_category 过粗，而 t0_category_cn 实际包含黄金、豆粕、有色、能源等不同驱动，应避免只按 t0_category 做相对强弱。")
    lines.append("")

    lines.append("## 6. horizon / future_ret 标签审计\n")
    lines.append(safe_to_markdown(horizon_audit))
    lines.append("")
    lines.append("解释：`next_time_mismatch_plus_1min_rows` 不一定是错误，午休会导致下一根真实 bar 不是 signal_time + 1min；但如果回测用 signal_time+1min 做持仓重叠，就会污染组合约束。")
    lines.append("")

    if not bt_audit.empty:
        lines.append("## 7. 回测 summary 审计\n")
        show_cols = [
            "cost_bp", "view", "final_nav", "trade_count", "avg_gross_ret", "avg_net_ret",
            "win_rate", "profit_factor", "score_col", "score_quantile", "pred_ret_min",
            "downside_max", "use_event_filter"
        ]
        show_cols = [c for c in show_cols if c in bt_audit.columns]
        lines.append(safe_to_markdown(bt_audit[show_cols], max_rows=80))
        lines.append("")
        lines.append("提示：不要把 `trade_count=0, final_nav=1` 当成最优策略。应优先看 `best_trade_count_gt_0`、`best_trade_count_ge_20`。")
        lines.append("")

    if not pred_audit.empty:
        lines.append("## 8. prediction 文件审计\n")
        lines.append(safe_to_markdown(pred_audit))
        lines.append("")

    if not trade_audit.empty:
        lines.append("## 9. trades 文件审计\n")
        lines.append(safe_to_markdown(trade_audit))
        lines.append("")

    lines.append("## 10. 结论判读")
    lines.append("- 如果面板审计出现大量缺 bar、零成交、异常跳变，先修数据，不要继续调模型。")
    lines.append("- 如果标签审计显示 entry/exit bar 大量零成交，应在训练和回测里强制过滤 entry/exit amount > 0。")
    lines.append("- 如果 trade audit 显示 entry_time 不是下一根真实 bar，组合约束代码需要修正。")
    lines.append("- 如果数据层面基本干净，但最高分层收益仍只有 1—2bp 以下，则问题主要不是工程 bug，而是可交易 alpha 太薄。")

    (out_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    panel_file, out_dir = resolve_paths(args)
    horizons = parse_int_list(args.horizons)

    print("=" * 100)
    print("Audit T+0 ETF Intraday Panel and Backtest Assumptions")
    print("=" * 100)
    print(f"panel_file : {panel_file.resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"scope      : {args.scope}")
    print(f"horizons   : {horizons}")
    print("=" * 100)

    panel = read_panel(panel_file, args.scope)

    print("Auditing basic panel quality...")
    basic_summary, etf_day, stale, category = audit_basic_panel(
        panel, expected_bars=args.expected_bars_per_day, stale_run_bars=args.stale_run_bars
    )
    etf_day.to_csv(out_dir / "etf_day_quality.csv", index=False, encoding="utf-8-sig")
    stale.to_csv(out_dir / "stale_price_summary.csv", index=False, encoding="utf-8-sig")
    category.to_csv(out_dir / "category_summary.csv", index=False, encoding="utf-8-sig")

    print("Auditing future label construction...")
    horizon_audit = audit_horizon_labels(
        panel, horizons=horizons, force_flat_time=args.force_flat_time,
        entry_start=args.entry_start, entry_end=args.entry_end,
        extreme_ret_bp=args.extreme_ret_bp, out_dir=out_dir
    )
    horizon_audit.to_csv(out_dir / "horizon_label_audit.csv", index=False, encoding="utf-8-sig")

    print("Auditing optional backtest summary / predictions / trades...")
    bt_audit = audit_backtest_summary(Path(args.backtest_summary_file), out_dir) if args.backtest_summary_file else pd.DataFrame()
    pred_audit = audit_prediction_file(Path(args.prediction_file), out_dir) if args.prediction_file else pd.DataFrame()
    trade_audit = audit_trades_file(Path(args.trades_file), panel, out_dir) if args.trades_file else pd.DataFrame()

    config = {
        "panel_file": str(panel_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "scope": args.scope,
        "horizons": horizons,
        "expected_bars_per_day": args.expected_bars_per_day,
        "force_flat_time": args.force_flat_time,
        "entry_start": args.entry_start,
        "entry_end": args.entry_end,
        "extreme_ret_bp": args.extreme_ret_bp,
        "stale_run_bars": args.stale_run_bars,
        "backtest_summary_file": args.backtest_summary_file,
        "prediction_file": args.prediction_file,
        "trades_file": args.trades_file,
    }

    summary_json = {
        "config": config,
        "basic_summary": basic_summary,
        "horizon_audit": horizon_audit.to_dict(orient="records"),
        "backtest_audit": bt_audit.to_dict(orient="records") if not bt_audit.empty else [],
        "prediction_audit": pred_audit.to_dict(orient="records") if not pred_audit.empty else [],
        "trade_audit": trade_audit.to_dict(orient="records") if not trade_audit.empty else [],
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    write_report(out_dir, config, basic_summary, etf_day, stale, category, horizon_audit, bt_audit, pred_audit, trade_audit)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"report : {(out_dir / 'audit_report.md').resolve()}")
    print(f"json   : {(out_dir / 'audit_summary.json').resolve()}")
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
