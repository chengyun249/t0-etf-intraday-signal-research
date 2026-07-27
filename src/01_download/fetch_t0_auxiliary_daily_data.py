#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_t0_auxiliary_daily_data.py

用途：
    为 T+0 ETF 1min 日内交易项目补充轻量日频辅助数据。
    这些数据不替代 1min 主行情，而是用于：
        1. 数据审计：分钟收盘价 vs 日线收盘价、复权因子变化日、异常波动日；
        2. 交易过滤：低流动性日、高波动日、异常跳变日不交易或降权；
        3. 市场状态解释：A 股风险偏好、宽基指数状态、跨市场环境代理；
        4. 后续报告：说明策略在不同市场状态下的表现。

核心下载内容：
    1. fund_daily：30 只 T+0 ETF 日线行情；
    2. fund_adj：30 只 T+0 ETF 基金复权因子；
    3. index_daily：A 股核心指数日线；
    4. trade_cal：交易日历；
    5. fund_basic_selected：所选 ETF 基础信息；
    6. daily_regime_features：由日线数据生成的辅助状态特征。

运行示例：
    $env:TUSHARE_TOKEN="你的token"

    python ".\\fetch_t0_auxiliary_daily_data.py" `
      --codes-file ".\\data_t0\\config\\t0_etf_codes.csv" `
      --start-date 20250101 `
      --end-date 20251231 `
      --out-dir ".\\data_t0\\auxiliary" `
      --sleep-sec 0.60

说明：
    - 默认 base_url 为 http://tsy.xiaodefa.cn。
    - 如果某个接口权限不足，脚本会记录 error，不会影响其他数据下载。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_INDEX_CODES = [
    "000001.SH",  # 上证指数
    "399001.SZ",  # 深证成指
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000852.SH",  # 中证1000
    "000016.SH",  # 上证50
    "399006.SZ",  # 创业板指
    "000688.SH",  # 科创50
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch auxiliary daily data for T+0 ETF intraday project.")
    p.add_argument("--token", type=str, default=None, help="Tushare token。也可用环境变量 TUSHARE_TOKEN。")
    p.add_argument("--base-url", type=str, default="http://tsy.xiaodefa.cn", help="Tushare API 地址。")
    p.add_argument("--codes-file", type=str, default="./data_t0/config/t0_etf_codes.csv", help="ETF 代码文件，包含 ts_code 列。")
    p.add_argument("--start-date", type=str, default="20250101", help="开始日期，YYYYMMDD。")
    p.add_argument("--end-date", type=str, default="20251231", help="结束日期，YYYYMMDD。")
    p.add_argument("--out-dir", type=str, default="./data_t0/auxiliary", help="输出目录。")
    p.add_argument("--sleep-sec", type=float, default=0.60, help="请求间隔。")
    p.add_argument("--max-retries", type=int, default=3, help="单个请求失败重试次数。")
    p.add_argument("--index-codes", type=str, default=",".join(DEFAULT_INDEX_CODES), help="逗号分隔的指数代码列表。")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有输出。")
    return p.parse_args()


def init_tushare(token: Optional[str], base_url: Optional[str]):
    try:
        import tushare as ts
    except Exception as exc:
        raise ImportError("未安装 tushare，请先 pip install tushare。") from exc

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("未找到 token。请传 --token 或设置环境变量 TUSHARE_TOKEN。")

    ts.set_token(token)
    pro = ts.pro_api(token)

    if base_url:
        try:
            pro._DataApi__http_url = base_url
        except Exception:
            pass

    return pro


def safe_call(func, sleep_sec: float, max_retries: int = 3, **kwargs) -> pd.DataFrame:
    last_err = None
    for i in range(max_retries):
        try:
            df = func(**kwargs)
            time.sleep(sleep_sec)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            last_err = exc
            wait = 3 * (i + 1)
            print(f"    [WARN] request failed {i+1}/{max_retries}: {type(exc).__name__}: {exc}; wait {wait}s")
            time.sleep(wait)
    raise last_err


def read_codes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"codes-file 不存在：{path}")

    df = pd.read_csv(path)
    if "ts_code" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "ts_code"})
        else:
            raise ValueError("codes-file 必须包含 ts_code 列。")

    codes = (
        df["ts_code"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not codes:
        raise ValueError("codes-file 中没有有效 ts_code。")
    return codes


def normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "trade_date" in d.columns:
        d["trade_date"] = d["trade_date"].astype(str)
        d["trade_date_dt"] = pd.to_datetime(d["trade_date"], format="%Y%m%d", errors="coerce")
    return d


def save_df(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        print(f"[WARN] parquet save failed for {path_base}: {exc}")


def fetch_fund_daily(pro, codes: List[str], args: argparse.Namespace, log_rows: List[Dict]) -> pd.DataFrame:
    parts = []
    print("\n========== Fetch fund_daily ==========")
    for i, code in enumerate(codes, 1):
        print(f"[fund_daily {i}/{len(codes)}] {code}")
        try:
            df = safe_call(
                pro.fund_daily,
                sleep_sec=args.sleep_sec,
                max_retries=args.max_retries,
                ts_code=code,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            if df.empty:
                log_rows.append({"dataset": "fund_daily", "ts_code": code, "rows": 0, "status": "empty", "error": ""})
                continue
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = normalize_trade_date(df)
            parts.append(df)
            log_rows.append({"dataset": "fund_daily", "ts_code": code, "rows": len(df), "status": "ok", "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}")
            log_rows.append({"dataset": "fund_daily", "ts_code": code, "rows": 0, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def fetch_fund_adj(pro, codes: List[str], args: argparse.Namespace, log_rows: List[Dict]) -> pd.DataFrame:
    parts = []
    print("\n========== Fetch fund_adj ==========")
    for i, code in enumerate(codes, 1):
        print(f"[fund_adj {i}/{len(codes)}] {code}")
        try:
            df = safe_call(
                pro.fund_adj,
                sleep_sec=args.sleep_sec,
                max_retries=args.max_retries,
                ts_code=code,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            if df.empty:
                log_rows.append({"dataset": "fund_adj", "ts_code": code, "rows": 0, "status": "empty", "error": ""})
                continue
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = normalize_trade_date(df)
            parts.append(df)
            log_rows.append({"dataset": "fund_adj", "ts_code": code, "rows": len(df), "status": "ok", "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}")
            log_rows.append({"dataset": "fund_adj", "ts_code": code, "rows": 0, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def fetch_index_daily(pro, index_codes: List[str], args: argparse.Namespace, log_rows: List[Dict]) -> pd.DataFrame:
    parts = []
    print("\n========== Fetch index_daily ==========")
    for i, code in enumerate(index_codes, 1):
        print(f"[index_daily {i}/{len(index_codes)}] {code}")
        try:
            df = safe_call(
                pro.index_daily,
                sleep_sec=args.sleep_sec,
                max_retries=args.max_retries,
                ts_code=code,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            if df.empty:
                log_rows.append({"dataset": "index_daily", "ts_code": code, "rows": 0, "status": "empty", "error": ""})
                continue
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = normalize_trade_date(df)
            parts.append(df)
            log_rows.append({"dataset": "index_daily", "ts_code": code, "rows": len(df), "status": "ok", "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}")
            log_rows.append({"dataset": "index_daily", "ts_code": code, "rows": 0, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def fetch_trade_calendar(pro, args: argparse.Namespace, log_rows: List[Dict]) -> pd.DataFrame:
    print("\n========== Fetch trade_cal ==========")
    try:
        df = safe_call(
            pro.trade_cal,
            sleep_sec=args.sleep_sec,
            max_retries=args.max_retries,
            exchange="SSE",
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if df.empty:
            log_rows.append({"dataset": "trade_cal", "ts_code": "SSE", "rows": 0, "status": "empty", "error": ""})
            return pd.DataFrame()
        if "cal_date" in df.columns:
            df["cal_date_dt"] = pd.to_datetime(df["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
        log_rows.append({"dataset": "trade_cal", "ts_code": "SSE", "rows": len(df), "status": "ok", "error": ""})
        return df
    except Exception as exc:
        print(f"  ERROR {type(exc).__name__}: {exc}")
        log_rows.append({"dataset": "trade_cal", "ts_code": "SSE", "rows": 0, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return pd.DataFrame()


def fetch_fund_basic_selected(pro, codes: List[str], args: argparse.Namespace, log_rows: List[Dict]) -> pd.DataFrame:
    print("\n========== Fetch fund_basic selected ==========")
    try:
        df = safe_call(
            pro.fund_basic,
            sleep_sec=args.sleep_sec,
            max_retries=args.max_retries,
            market="E",
        )
        if df.empty:
            log_rows.append({"dataset": "fund_basic", "ts_code": "ALL", "rows": 0, "status": "empty", "error": ""})
            return pd.DataFrame()
        if "ts_code" not in df.columns:
            log_rows.append({"dataset": "fund_basic", "ts_code": "ALL", "rows": len(df), "status": "error", "error": "missing ts_code"})
            return pd.DataFrame()
        out = df[df["ts_code"].isin(codes)].copy()
        log_rows.append({"dataset": "fund_basic", "ts_code": "selected", "rows": len(out), "status": "ok", "error": ""})
        return out
    except Exception as exc:
        print(f"  ERROR {type(exc).__name__}: {exc}")
        log_rows.append({"dataset": "fund_basic", "ts_code": "ALL", "rows": 0, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return pd.DataFrame()


def build_etf_daily_features(fund_daily: pd.DataFrame, fund_adj: pd.DataFrame) -> pd.DataFrame:
    if fund_daily.empty:
        return pd.DataFrame()

    d = fund_daily.copy()
    d = normalize_trade_date(d)
    d = d.sort_values(["ts_code", "trade_date_dt"]).reset_index(drop=True)

    for c in ["open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # 统一收益
    d["ret_1d"] = d["close"] / d["pre_close"] - 1.0
    d["ret_1d"] = d["ret_1d"].where(d["ret_1d"].notna(), d["pct_chg"] / 100.0)

    # 日内振幅与成交额
    d["hl_range"] = d["high"] / d["low"] - 1.0
    d["amount_mil"] = d["amount"] / 1000.0  # 多数 Tushare amount 单位近似千元，转百万元供排序参考
    d["log_amount"] = np.log1p(d["amount"].clip(lower=0))

    g = d.groupby("ts_code", group_keys=False)

    d["ret_3d"] = g["close"].pct_change(3)
    d["ret_5d"] = g["close"].pct_change(5)
    d["ret_20d"] = g["close"].pct_change(20)

    d["amount_ma_5"] = g["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["amount_ma_20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["amount_ratio_5_20"] = d["amount_ma_5"] / d["amount_ma_20"]

    d["vol_5d"] = g["ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).std())
    d["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    d["abs_ret_20d"] = g["ret_1d"].transform(lambda s: s.abs().rolling(20, min_periods=10).mean())

    d["is_low_liquidity_day"] = (
        d["amount_ma_20"].notna()
        & (d["amount"] < 0.3 * d["amount_ma_20"])
    ).astype(int)

    d["is_extreme_return_day"] = (
        d["ret_1d"].abs() > (5.0 * d["vol_20d"])
    ).fillna(False).astype(int)

    d["is_high_vol_regime"] = (
        d["vol_20d"] > g["vol_20d"].transform(lambda s: s.rolling(60, min_periods=30).median())
    ).fillna(False).astype(int)

    # 复权因子变化日
    d["adj_factor"] = np.nan
    d["adj_factor_change"] = 0

    if not fund_adj.empty and "adj_factor" in fund_adj.columns:
        a = fund_adj.copy()
        a = normalize_trade_date(a)
        a["adj_factor"] = pd.to_numeric(a["adj_factor"], errors="coerce")
        a = a[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
        d = d.merge(a, on=["ts_code", "trade_date"], how="left", suffixes=("", "_new"))
        if "adj_factor_new" in d.columns:
            d["adj_factor"] = d["adj_factor_new"]
            d = d.drop(columns=["adj_factor_new"])
        d["adj_factor_change"] = (
            d.groupby("ts_code")["adj_factor"].pct_change().abs() > 1e-8
        ).fillna(False).astype(int)

    return d


def build_index_features(index_daily: pd.DataFrame) -> pd.DataFrame:
    if index_daily.empty:
        return pd.DataFrame()

    d = index_daily.copy()
    d = normalize_trade_date(d)
    d = d.sort_values(["ts_code", "trade_date_dt"]).reset_index(drop=True)

    for c in ["open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["index_ret_1d"] = d["close"] / d["pre_close"] - 1.0
    d["index_ret_1d"] = d["index_ret_1d"].where(d["index_ret_1d"].notna(), d["pct_chg"] / 100.0)

    g = d.groupby("ts_code", group_keys=False)
    d["index_ret_5d"] = g["close"].pct_change(5)
    d["index_ret_20d"] = g["close"].pct_change(20)
    d["index_vol_20d"] = g["index_ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    d["index_ma_20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["index_above_ma20"] = (d["close"] > d["index_ma_20"]).astype(int)
    return d


def write_report(
    out_dir: Path,
    codes: List[str],
    fund_daily: pd.DataFrame,
    fund_adj: pd.DataFrame,
    index_daily: pd.DataFrame,
    trade_cal: pd.DataFrame,
    etf_features: pd.DataFrame,
    log_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# T+0 ETF 辅助日频数据下载报告\n")
    lines.append("## 1. 下载范围\n")
    lines.append(f"- ETF 数量：{len(codes)}")
    lines.append(f"- 日期范围：{args.start_date} 至 {args.end_date}")
    lines.append(f"- 输出目录：`{out_dir}`")
    lines.append("")

    lines.append("## 2. 数据集概况\n")
    lines.append(f"- fund_daily 行数：{len(fund_daily):,}")
    lines.append(f"- fund_adj 行数：{len(fund_adj):,}")
    lines.append(f"- index_daily 行数：{len(index_daily):,}")
    lines.append(f"- trade_cal 行数：{len(trade_cal):,}")
    lines.append(f"- daily_regime_features 行数：{len(etf_features):,}")
    lines.append("")

    if not fund_daily.empty:
        rows_by_code = fund_daily.groupby("ts_code").size().sort_values()
        lines.append("## 3. ETF 日线行数检查\n")
        lines.append(f"- 最少行数：{int(rows_by_code.min())}")
        lines.append(f"- 最多行数：{int(rows_by_code.max())}")
        lines.append(f"- 中位数：{float(rows_by_code.median()):.1f}")
        lines.append("")

    if not etf_features.empty:
        adj_events = etf_features.groupby("ts_code")["adj_factor_change"].sum().sort_values(ascending=False)
        extreme = etf_features.groupby("ts_code")["is_extreme_return_day"].sum().sort_values(ascending=False)
        lowliq = etf_features.groupby("ts_code")["is_low_liquidity_day"].sum().sort_values(ascending=False)

        lines.append("## 4. 风险标记摘要\n")
        lines.append(f"- 复权因子变化事件总数：{int(etf_features['adj_factor_change'].sum())}")
        lines.append(f"- 极端收益日总数：{int(etf_features['is_extreme_return_day'].sum())}")
        lines.append(f"- 低流动性日总数：{int(etf_features['is_low_liquidity_day'].sum())}")
        lines.append("")
        lines.append("复权因子变化较多的 ETF：")
        lines.append(adj_events.head(10).to_string())
        lines.append("")
        lines.append("极端收益日较多的 ETF：")
        lines.append(extreme.head(10).to_string())
        lines.append("")
        lines.append("低流动性日较多的 ETF：")
        lines.append(lowliq.head(10).to_string())
        lines.append("")

    if not log_df.empty:
        err = log_df[log_df["status"] == "error"]
        lines.append("## 5. 接口错误\n")
        if err.empty:
            lines.append("- 无接口错误。")
        else:
            lines.append(err.to_markdown(index=False))
        lines.append("")

    lines.append("## 6. 后续用途\n")
    lines.append("- `fund_daily`：校验分钟数据、计算日频流动性/波动 regime。")
    lines.append("- `fund_adj`：识别复权因子变化日，防止异常价格断点污染回测。")
    lines.append("- `index_daily`：构建 A 股市场状态变量，用于过滤和报告解释。")
    lines.append("- `daily_regime_features`：后续会并入 1min bar 面板，作为交易过滤器，而不是第一版核心 alpha。")

    (out_dir / "auxiliary_data_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes_file = Path(args.codes_file)
    codes = read_codes(codes_file)
    index_codes = [x.strip() for x in str(args.index_codes).split(",") if x.strip()]

    pro = init_tushare(args.token, args.base_url)

    print("=" * 100)
    print("T+0 ETF auxiliary daily data downloader")
    print("=" * 100)
    print(f"codes-file       : {codes_file.resolve()}")
    print(f"ETF codes        : {len(codes)}")
    print(f"date range       : {args.start_date} -> {args.end_date}")
    print(f"index codes      : {index_codes}")
    print(f"out_dir          : {out_dir.resolve()}")
    print(f"base_url         : {args.base_url}")
    print(f"sleep_sec        : {args.sleep_sec}")
    print("=" * 100)

    log_rows: List[Dict] = []

    fund_daily_path = out_dir / "fund_daily"
    fund_adj_path = out_dir / "fund_adj"
    index_daily_path = out_dir / "index_daily"
    trade_cal_path = out_dir / "trade_calendar"
    fund_basic_path = out_dir / "fund_basic_selected"
    etf_features_path = out_dir / "daily_regime_features"
    index_features_path = out_dir / "index_regime_features"

    if args.overwrite or not fund_daily_path.with_suffix(".csv").exists():
        fund_daily = fetch_fund_daily(pro, codes, args, log_rows)
        save_df(fund_daily, fund_daily_path)
    else:
        print(f"[SKIP] {fund_daily_path.with_suffix('.csv')} exists")
        fund_daily = pd.read_csv(fund_daily_path.with_suffix(".csv"))

    if args.overwrite or not fund_adj_path.with_suffix(".csv").exists():
        fund_adj = fetch_fund_adj(pro, codes, args, log_rows)
        save_df(fund_adj, fund_adj_path)
    else:
        print(f"[SKIP] {fund_adj_path.with_suffix('.csv')} exists")
        fund_adj = pd.read_csv(fund_adj_path.with_suffix(".csv"))

    if args.overwrite or not index_daily_path.with_suffix(".csv").exists():
        index_daily = fetch_index_daily(pro, index_codes, args, log_rows)
        save_df(index_daily, index_daily_path)
    else:
        print(f"[SKIP] {index_daily_path.with_suffix('.csv')} exists")
        index_daily = pd.read_csv(index_daily_path.with_suffix(".csv"))

    if args.overwrite or not trade_cal_path.with_suffix(".csv").exists():
        trade_cal = fetch_trade_calendar(pro, args, log_rows)
        save_df(trade_cal, trade_cal_path)
    else:
        print(f"[SKIP] {trade_cal_path.with_suffix('.csv')} exists")
        trade_cal = pd.read_csv(trade_cal_path.with_suffix(".csv"))

    if args.overwrite or not fund_basic_path.with_suffix(".csv").exists():
        fund_basic = fetch_fund_basic_selected(pro, codes, args, log_rows)
        save_df(fund_basic, fund_basic_path)
    else:
        print(f"[SKIP] {fund_basic_path.with_suffix('.csv')} exists")
        fund_basic = pd.read_csv(fund_basic_path.with_suffix(".csv"))

    print("\n========== Build daily regime features ==========")
    etf_features = build_etf_daily_features(fund_daily, fund_adj)
    save_df(etf_features, etf_features_path)

    index_features = build_index_features(index_daily)
    save_df(index_features, index_features_path)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_dir / "auxiliary_download_log.csv", index=False, encoding="utf-8-sig")

    write_report(
        out_dir=out_dir,
        codes=codes,
        fund_daily=fund_daily,
        fund_adj=fund_adj,
        index_daily=index_daily,
        trade_cal=trade_cal,
        etf_features=etf_features,
        log_df=log_df,
        args=args,
    )

    print("\n" + "=" * 100)
    print("Finished auxiliary daily data download")
    print("=" * 100)
    print(f"fund_daily              : {(out_dir / 'fund_daily.csv').resolve()}")
    print(f"fund_adj                : {(out_dir / 'fund_adj.csv').resolve()}")
    print(f"index_daily             : {(out_dir / 'index_daily.csv').resolve()}")
    print(f"trade_calendar          : {(out_dir / 'trade_calendar.csv').resolve()}")
    print(f"daily_regime_features   : {(out_dir / 'daily_regime_features.csv').resolve()}")
    print(f"index_regime_features   : {(out_dir / 'index_regime_features.csv').resolve()}")
    print(f"report                  : {(out_dir / 'auxiliary_data_report.md').resolve()}")
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
