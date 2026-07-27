#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
build_t0_intraday_bar_panel_2022_2024.py

用途：
    将 2022-2024 下载好的 T+0 ETF 1min raw parquet 合并成策略可直接使用的
    t0_intraday_bar_panel.parquet。

默认输入：
    .\data_t0_2022_2024\raw_1min\freq=1min\
    .\data_t0_2022_2024\config\t0_etf_selected_2022_2024_detail.csv

默认输出：
    .\data_t0_2022_2024\processed\
      ├─ t0_intraday_bar_panel.parquet
      ├─ t0_feature_manifest.json
      ├─ t0_intraday_panel_audit_summary.csv
      ├─ t0_intraday_bad_days.csv
      └─ t0_intraday_panel_report.md

说明：
    1. 这个脚本不依赖日频辅助数据。daily regime flag 会从分钟数据内生构造：
       - daily_is_low_liquidity_day
       - daily_is_extreme_return_day
       - daily_adj_factor_change 暂设为 0
    2. 如果后面需要更精细地处理复权/大盘状态，可以再合并 fund_adj 或 index_daily。
    3. 当前组合策略需要的核心字段都会生成：
       intraday_vwap, bar_index, ret_5m, ret_10m, ret_20m, amount_z_20m,
       breakout_high_20m, valid_entry_bar, force_flat_bar 等。

运行：
    python ".\build_t0_intraday_bar_panel_2022_2024.py" `
      --raw-dir ".\data_t0_2022_2024\raw_1min\freq=1min" `
      --universe-file ".\data_t0_2022_2024\config\t0_etf_selected_2022_2024_detail.csv" `
      --out-dir ".\data_t0_2022_2024\processed"

如果想把 selected_for_main=0 的短样本/债券/货币也合并进去：
    python ".\build_t0_intraday_bar_panel_2022_2024.py" `
      --include-non-main
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 2022-2024 T+0 ETF intraday bar panel.")
    p.add_argument("--raw-dir", type=str, default="./data_t0/raw_1min/freq=1min")
    p.add_argument("--universe-file", type=str, default="./data_t0/config/t0_etf_selected_30_detail_final.csv")
    p.add_argument("--out-dir", type=str, default="./data_t0/processed")
    p.add_argument("--include-non-main", action="store_true", help="默认只合并 selected_for_main=1；加此参数则合并 universe 中所有有文件的 ETF。")
    p.add_argument("--allow-missing", action="store_true", help="默认 selected 主池缺 raw 文件会报错；加此参数则记录并跳过。")
    p.add_argument("--force-flat-time", type=str, default="14:55")
    p.add_argument("--valid-entry-start", type=str, default="09:31")
    p.add_argument("--valid-entry-end", type=str, default="14:50")
    return p.parse_args()


def norm_date(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "-" in s:
        return pd.Timestamp(s).strftime("%Y%m%d")
    return s[:8]


def safe_code(ts_code: str) -> str:
    return str(ts_code).replace(".", "_")


def load_universe(path: Path, include_non_main: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"universe-file 不存在：{path}")

    u = pd.read_csv(path)
    if "ts_code" not in u.columns:
        raise ValueError("universe-file 必须包含 ts_code 列。")

    for c in ["name_manual", "name", "t0_category", "t0_category_cn", "selected_for_main"]:
        if c not in u.columns:
            if c == "selected_for_main":
                u[c] = 1
            else:
                u[c] = ""

    u["ts_code"] = u["ts_code"].astype(str).str.strip()
    u = u[u["ts_code"] != ""].drop_duplicates("ts_code").reset_index(drop=True)

    if not include_non_main:
        u = u[pd.to_numeric(u["selected_for_main"], errors="coerce").fillna(0).astype(int) == 1].copy()

    # name 统一
    if "name_manual" in u.columns:
        u["name"] = u["name_manual"].fillna("").astype(str)
    else:
        u["name"] = u["name"].fillna("").astype(str)

    u["t0_category"] = u["t0_category"].fillna("unknown").astype(str)
    u["t0_category_cn"] = u["t0_category_cn"].fillna(u["t0_category"]).astype(str)

    return u


def find_raw_file(raw_dir: Path, ts_code: str) -> Optional[Path]:
    pattern = f"{safe_code(ts_code)}_*.parquet"
    matches = sorted(raw_dir.glob(pattern))
    if not matches:
        return None
    # 若有多个，取最大的/最新的那个
    matches = sorted(matches, key=lambda p: (p.stat().st_size, p.name), reverse=True)
    return matches[0]


def read_one_raw(path: Path, meta: pd.Series) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d.columns = [str(c).lower() for c in d.columns]

    if "trade_time" not in d.columns:
        for alt in ["datetime", "time", "trade_date"]:
            if alt in d.columns:
                d = d.rename(columns={alt: "trade_time"})
                break

    required = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
    for c in required:
        if c not in d.columns:
            if c == "ts_code":
                d[c] = meta["ts_code"]
            else:
                d[c] = np.nan

    d["ts_code"] = d["ts_code"].astype(str)
    # 有些接口返回的 ts_code 可能缺失或混乱，以 universe 为准
    d["ts_code"] = str(meta["ts_code"])

    d["trade_time"] = pd.to_datetime(d["trade_time"], errors="coerce")
    d = d.dropna(subset=["trade_time"]).copy()
    d["trade_date"] = d["trade_time"].dt.strftime("%Y%m%d")
    d["clock_time"] = d["trade_time"].dt.strftime("%H:%M")

    # 只保留正常交易时段
    d = d[
        ((d["clock_time"] >= "09:30") & (d["clock_time"] <= "11:30"))
        | ((d["clock_time"] >= "13:00") & (d["clock_time"] <= "15:00"))
    ].copy()

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["name"] = str(meta.get("name", ""))
    d["t0_category"] = str(meta.get("t0_category", "unknown"))
    d["t0_category_cn"] = str(meta.get("t0_category_cn", meta.get("t0_category", "unknown")))

    d = d.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    keep = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "vol", "amount",
    ]
    return d[keep]


def add_intraday_features(d: pd.DataFrame, force_flat_time: str, valid_entry_start: str, valid_entry_end: str) -> pd.DataFrame:
    x = d.sort_values(["ts_code", "trade_date", "trade_time"]).copy()

    # bar index
    x["bar_index"] = x.groupby(["ts_code", "trade_date"]).cumcount().astype(int)

    # day open / intraday move
    x["day_open"] = x.groupby(["ts_code", "trade_date"])["open"].transform("first")
    x["day_close"] = x.groupby(["ts_code", "trade_date"])["close"].transform("last")
    x["intraday_move"] = x["close"] / x["day_open"] - 1.0
    x["abs_intraday_move"] = x["intraday_move"].abs()

    # intraday VWAP：优先用 close*vol 的成交量加权均价。vol 为手时常数会抵消。
    # 若 vol 无效，则退化为 close 的日内均值。
    x["vol_nonneg"] = x["vol"].clip(lower=0)
    x["cum_vol"] = x.groupby(["ts_code", "trade_date"])["vol_nonneg"].cumsum()
    x["cum_close_vol"] = (x["close"] * x["vol_nonneg"]).groupby([x["ts_code"], x["trade_date"]]).cumsum()
    x["intraday_vwap"] = x["cum_close_vol"] / x["cum_vol"].replace(0, np.nan)
    x["intraday_vwap"] = x["intraday_vwap"].fillna(
        x.groupby(["ts_code", "trade_date"])["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )

    # returns within day
    for n in [1, 3, 5, 10, 20, 30, 60]:
        x[f"ret_{n}m"] = x.groupby(["ts_code", "trade_date"])["close"].pct_change(n)

    # rolling volatility within day
    for n in [5, 10, 20, 60]:
        x[f"ret_1m_vol_{n}m"] = (
            x.groupby(["ts_code", "trade_date"])["ret_1m"]
            .transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).std())
        )

    # amount z-score within day
    def rolling_z(s: pd.Series, win: int = 20) -> pd.Series:
        m = s.rolling(win, min_periods=max(5, win // 2)).mean()
        sd = s.rolling(win, min_periods=max(5, win // 2)).std()
        return (s - m) / sd.replace(0, np.nan)

    x["amount_z_20m"] = (
        x.groupby(["ts_code", "trade_date"])["amount"]
        .transform(lambda s: rolling_z(s.fillna(0), 20))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # previous high/low breakout
    for n in [10, 20, 30, 60]:
        prev_high = (
            x.groupby(["ts_code", "trade_date"])["high"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=max(3, n // 2)).max())
        )
        prev_low = (
            x.groupby(["ts_code", "trade_date"])["low"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=max(3, n // 2)).min())
        )
        x[f"prev_high_{n}m"] = prev_high
        x[f"prev_low_{n}m"] = prev_low
        x[f"breakout_high_{n}m"] = (x["close"] > prev_high).fillna(False).astype(int)
        x[f"breakdown_low_{n}m"] = (x["close"] < prev_low).fillna(False).astype(int)

    # active / execution flags
    price_ok = (x[["open", "high", "low", "close"]] > 0).all(axis=1)
    amount_ok = x["amount"].fillna(0) >= 0
    vol_ok = x["vol"].fillna(0) >= 0

    x["valid_entry_bar"] = (
        price_ok
        & amount_ok
        & vol_ok
        & (x["clock_time"] >= valid_entry_start)
        & (x["clock_time"] <= valid_entry_end)
    ).astype(int)

    x["force_flat_bar"] = (x["clock_time"] >= force_flat_time).astype(int)

    # daily metrics
    daily = x.groupby(["ts_code", "trade_date"], as_index=False).agg(
        daily_open=("open", "first"),
        daily_high=("high", "max"),
        daily_low=("low", "min"),
        daily_close=("close", "last"),
        day_amount=("amount", "sum"),
        day_vol=("vol", "sum"),
        rows_per_day=("trade_time", "count"),
        zero_amount_bars=("amount", lambda s: int((s.fillna(0) <= 0).sum())),
    )
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["daily_ret"] = daily.groupby("ts_code")["daily_close"].pct_change()
    daily["amount_ma20"] = daily.groupby("ts_code")["day_amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    daily["daily_is_low_liquidity_day"] = (
        (daily["day_amount"].fillna(0) <= 0)
        | ((daily["amount_ma20"].notna()) & (daily["day_amount"] < 0.2 * daily["amount_ma20"]))
    ).astype(int)
    daily["daily_is_extreme_return_day"] = (daily["daily_ret"].abs() >= 0.08).fillna(False).astype(int)
    # 没有 fund_adj 时先置 0。后续若提供复权因子，可单独合并。
    daily["daily_adj_factor_change"] = 0

    merge_cols = [
        "ts_code", "trade_date",
        "daily_open", "daily_high", "daily_low", "daily_close", "daily_ret",
        "day_amount", "day_vol", "rows_per_day", "zero_amount_bars",
        "daily_is_low_liquidity_day", "daily_is_extreme_return_day", "daily_adj_factor_change",
    ]
    x = x.merge(daily[merge_cols], on=["ts_code", "trade_date"], how="left")

    # cross-sectional/category breadth at each timestamp
    x["positive_intraday"] = (x["intraday_move"] > 0).astype(int)
    x["positive_5m"] = (x["ret_5m"] > 0).astype(int)
    x["cat_positive_ratio"] = x.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
    x["cat_ret5_positive_ratio"] = x.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")

    # relative value / cluster z
    g = x.groupby(["trade_time", "t0_category"])
    x["cluster_move_mean"] = g["intraday_move"].transform("mean")
    x["cluster_move_std"] = g["intraday_move"].transform("std")
    x["rv_z"] = (x["intraday_move"] - x["cluster_move_mean"]) / x["cluster_move_std"].replace(0, np.nan)
    x["rv_rank_pct"] = g["intraday_move"].rank(pct=True)

    # 清理中间列但保留策略可用字段
    drop_cols = ["vol_nonneg", "cum_vol", "cum_close_vol"]
    x = x.drop(columns=[c for c in drop_cols if c in x.columns])

    return x.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def audit_panel(panel: pd.DataFrame, universe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    p = panel.copy()

    bad_ohlc = (
        (p["high"] < p[["open", "close", "low"]].max(axis=1))
        | (p["low"] > p[["open", "close", "high"]].min(axis=1))
    )
    non_positive = (p[["open", "high", "low", "close"]] <= 0).any(axis=1)
    neg_va = (p[["vol", "amount"]] < 0).any(axis=1)
    dup = p.duplicated(["ts_code", "trade_time"])

    etf_day = p.groupby(["ts_code", "name", "t0_category_cn", "trade_date"], as_index=False).agg(
        rows=("trade_time", "count"),
        amount=("amount", "sum"),
        vol=("vol", "sum"),
        first_time=("trade_time", "min"),
        last_time=("trade_time", "max"),
        valid_entries=("valid_entry_bar", "sum"),
        force_flat_bars=("force_flat_bar", "sum"),
    )
    bad_days = etf_day[(etf_day["rows"] != 241) | (etf_day["amount"] <= 0)].copy()

    summary = {
        "rows": int(len(p)),
        "etf_count": int(p["ts_code"].nunique()),
        "universe_count": int(universe["ts_code"].nunique()),
        "trading_days": int(p["trade_date"].nunique()),
        "etf_day_count": int(p.groupby(["ts_code", "trade_date"]).ngroups),
        "duplicate_rows": int(dup.sum()),
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "non_positive_price_rows": int(non_positive.sum()),
        "negative_vol_amount_rows": int(neg_va.sum()),
        "median_rows_per_etf_day": float(etf_day["rows"].median()) if not etf_day.empty else np.nan,
        "min_rows_per_etf_day": int(etf_day["rows"].min()) if not etf_day.empty else 0,
        "max_rows_per_etf_day": int(etf_day["rows"].max()) if not etf_day.empty else 0,
        "zero_amount_etf_days": int((etf_day["amount"] <= 0).sum()),
        "incomplete_etf_days": int((etf_day["rows"] != 241).sum()),
        "start_time": str(p["trade_time"].min()) if len(p) else "",
        "end_time": str(p["trade_time"].max()) if len(p) else "",
    }

    return etf_day, bad_days, summary


def write_manifest(out_dir: Path, panel: pd.DataFrame, universe: pd.DataFrame, summary: Dict) -> None:
    feature_groups = {
        "identity": ["ts_code", "name", "t0_category", "t0_category_cn"],
        "time": ["trade_time", "trade_date", "clock_time", "bar_index"],
        "raw_bar": ["open", "high", "low", "close", "vol", "amount"],
        "intraday_core": ["day_open", "intraday_vwap", "intraday_move", "abs_intraday_move"],
        "returns": [c for c in panel.columns if c.startswith("ret_")],
        "breakout": [c for c in panel.columns if "breakout" in c or "breakdown" in c or c.startswith("prev_")],
        "amount": ["amount_z_20m", "day_amount", "day_vol"],
        "daily_flags": ["daily_is_low_liquidity_day", "daily_is_extreme_return_day", "daily_adj_factor_change"],
        "category_breadth": ["cat_positive_ratio", "cat_ret5_positive_ratio"],
        "relative_value": ["cluster_move_mean", "cluster_move_std", "rv_z", "rv_rank_pct"],
        "execution_flags": ["valid_entry_bar", "force_flat_bar"],
    }

    manifest = {
        "panel_file": str((out_dir / "t0_intraday_bar_panel.parquet").resolve()),
        "row_count": int(len(panel)),
        "etf_count": int(panel["ts_code"].nunique()),
        "trade_date_count": int(panel["trade_date"].nunique()),
        "columns": list(panel.columns),
        "feature_groups": feature_groups,
        "audit_summary": summary,
        "universe_codes": universe["ts_code"].tolist(),
    }

    (out_dir / "t0_feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(out_dir: Path, universe: pd.DataFrame, summary: Dict, bad_days: pd.DataFrame) -> None:
    lines = []
    lines.append("# 2022-2024 T+0 ETF 1min 面板构建报告\n")
    lines.append("## 1. 构建结果\n")
    lines.append(pd.DataFrame([summary]).to_markdown(index=False))
    lines.append("")
    lines.append("## 2. 主策略池\n")
    show_cols = [c for c in ["ts_code", "name", "t0_category_cn"] if c in universe.columns]
    lines.append(universe[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 数据说明\n")
    lines.append("- 本面板由 1min raw parquet 直接构建，不强依赖外部日频辅助数据。")
    lines.append("- intraday_vwap 使用 close × vol 的日内成交量加权均价近似；vol 的单位常数不影响 VWAP。")
    lines.append("- daily_is_low_liquidity_day 与 daily_is_extreme_return_day 由分钟聚合生成。")
    lines.append("- daily_adj_factor_change 暂设为 0；如后续需要严格处理复权，可再合并 fund_adj。")
    lines.append("- force_flat_bar 默认从 14:55 开始。")
    lines.append("")
    if bad_days.empty:
        lines.append("## 4. 异常 ETF-day\n")
        lines.append("未发现 rows != 241 或 day_amount <= 0 的 ETF-day。")
    else:
        lines.append("## 4. 异常 ETF-day 样例\n")
        lines.append(bad_days.head(50).to_markdown(index=False))
    lines.append("")
    lines.append("## 5. 后续使用\n")
    lines.append("下一步可直接将本面板输入组合策略脚本，测试 ORB + Noise Boundary + Relative Value Filter + Category Breadth + Dynamic Exit。")

    (out_dir / "t0_intraday_panel_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    universe_file = Path(args.universe_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Build 2022-2024 T+0 ETF intraday bar panel")
    print("=" * 100)
    print(f"raw_dir       : {raw_dir.resolve()}")
    print(f"universe_file : {universe_file.resolve()}")
    print(f"out_dir       : {out_dir.resolve()}")
    print("=" * 100)

    universe = load_universe(universe_file, args.include_non_main)
    if universe.empty:
        raise RuntimeError("universe 为空。")

    print(f"ETF count in universe: {len(universe)}")

    parts = []
    missing = []

    for i, (_, row) in enumerate(universe.iterrows(), 1):
        code = row["ts_code"]
        f = find_raw_file(raw_dir, code)
        print(f"[{i}/{len(universe)}] {code} {row.get('name', '')}")
        if f is None:
            msg = f"missing raw parquet for {code}"
            print(f"  [MISSING] {msg}")
            missing.append({"ts_code": code, "reason": msg})
            if not args.allow_missing:
                continue
            else:
                continue

        d = read_one_raw(f, row)
        print(f"  rows={len(d):,}, days={d['trade_date'].nunique()}, file={f.name}")
        parts.append(d)

    if missing and not args.allow_missing:
        # 这里在读完整个 universe 后再报错，方便一次性看到所有缺失。
        miss = pd.DataFrame(missing)
        miss.to_csv(out_dir / "missing_raw_files.csv", index=False, encoding="utf-8-sig")
        raise FileNotFoundError(f"存在 {len(missing)} 个 universe ETF 找不到 raw 文件。已写出 missing_raw_files.csv。")

    if not parts:
        raise RuntimeError("没有读取到任何 raw 数据。")

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    print("\nAdding intraday features...")
    panel = add_intraday_features(raw, args.force_flat_time, args.valid_entry_start, args.valid_entry_end)

    # 基础审计
    etf_day, bad_days, summary = audit_panel(panel, universe)

    # 输出
    panel_file = out_dir / "t0_intraday_bar_panel.parquet"
    panel.to_parquet(panel_file, index=False)

    pd.DataFrame([summary]).to_csv(out_dir / "t0_intraday_panel_audit_summary.csv", index=False, encoding="utf-8-sig")
    etf_day.to_csv(out_dir / "t0_intraday_etf_day_summary.csv", index=False, encoding="utf-8-sig")
    bad_days.to_csv(out_dir / "t0_intraday_bad_days.csv", index=False, encoding="utf-8-sig")

    write_manifest(out_dir, panel, universe, summary)
    write_report(out_dir, universe, summary, bad_days)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"panel    : {panel_file.resolve()}")
    print(f"manifest : {(out_dir / 't0_feature_manifest.json').resolve()}")
    print(f"audit    : {(out_dir / 't0_intraday_panel_audit_summary.csv').resolve()}")
    print(f"report   : {(out_dir / 't0_intraday_panel_report.md').resolve()}")
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
