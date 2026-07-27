#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_t0_etf_1min_tushare.py

用途：
    下载 T+0 ETF 池的一年 1min 分钟行情数据，用于后续日内交易策略项目。

输入：
    data_t0/config/t0_etf_codes.csv
    至少包含一列：ts_code

输出：
    data_t0/raw_1min/freq=1min/*.parquet
    data_t0/raw_1min/_logs/download_log.csv
    data_t0/raw_1min/_logs/validation_summary.csv

运行示例：
    $env:TUSHARE_TOKEN="你的token"

    python ".\\fetch_t0_etf_1min_tushare.py" `
      --codes-file ".\\data_t0\\config\\t0_etf_codes.csv" `
      --start-date 2025-01-01 `
      --end-date 2025-12-31 `
      --freq 1min `
      --out-dir ".\\data_t0\\raw_1min" `
      --chunk-days 10 `
      --sleep-sec 0.65

说明：
    - 默认 base_url 为 http://tsy.xiaodefa.cn。
    - 默认不覆盖已存在文件。如果中断后重跑，会跳过已经下载完成的 ETF。
    - 若要覆盖重下，添加 --overwrite。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download T+0 ETF 1min data from Tushare.")
    p.add_argument("--token", type=str, default=None, help="Tushare token。也可用环境变量 TUSHARE_TOKEN。")
    p.add_argument("--base-url", type=str, default="http://tsy.xiaodefa.cn", help="Tushare API 地址。")
    p.add_argument("--codes-file", type=str, default="./data_t0/config/t0_etf_codes.csv", help="ETF 代码文件，包含 ts_code 列。")
    p.add_argument("--start-date", type=str, default="2025-01-01", help="开始日期，YYYY-MM-DD。")
    p.add_argument("--end-date", type=str, default="2025-12-31", help="结束日期，YYYY-MM-DD。")
    p.add_argument("--freq", type=str, default="1min", choices=["1min", "5min", "15min", "30min", "60min"], help="分钟频率。")
    p.add_argument("--out-dir", type=str, default="./data_t0/raw_1min", help="输出目录。")
    p.add_argument("--chunk-days", type=int, default=10, help="每次请求的自然日切块长度。1min 建议 7—15。")
    p.add_argument("--sleep-sec", type=float, default=0.65, help="请求间隔。120次/分钟理论最小0.50秒，建议0.60—0.80。")
    p.add_argument("--limit-codes", type=int, default=None, help="调试用：只下载前 N 只 ETF。")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在 parquet。")
    p.add_argument("--max-retries", type=int, default=3, help="单个请求失败重试次数。")
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


def read_codes(path: Path, limit: Optional[int] = None) -> List[str]:
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
    if limit is not None:
        codes = codes[:limit]
    if not codes:
        raise ValueError("codes-file 中没有有效 ts_code。")
    return codes


def make_chunks(start_date: str, end_date: str, chunk_days: int):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def fmt_dt_start(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d") + " 09:00:00"


def fmt_dt_end(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d") + " 15:30:00"


def safe_stk_mins(pro, ts_code: str, start_dt: str, end_dt: str, freq: str, sleep_sec: float, max_retries: int):
    last_err = None
    for i in range(max_retries):
        try:
            df = pro.stk_mins(
                ts_code=ts_code,
                freq=freq,
                start_date=start_dt,
                end_date=end_dt,
            )
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


def normalize_minute_df(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if "trade_time" not in d.columns:
        for c in ["trade_date", "datetime", "time"]:
            if c in d.columns:
                d = d.rename(columns={c: "trade_time"})
                break

    if "ts_code" not in d.columns:
        d["ts_code"] = ts_code

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c not in d.columns:
            d[c] = np.nan

    if "trade_time" not in d.columns:
        raise ValueError(f"{ts_code} 返回数据缺少 trade_time/trade_date/datetime 字段。columns={list(d.columns)}")

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_time"].dt.strftime("%Y-%m-%d")

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    keep = ["ts_code", "trade_time", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    extra = [c for c in d.columns if c not in keep]
    d = d[keep + extra]

    d = d.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return d


def validate_one(df: pd.DataFrame, ts_code: str) -> Dict:
    if df.empty:
        return {
            "ts_code": ts_code,
            "rows": 0,
            "start_time": "",
            "end_time": "",
            "trading_days": 0,
            "bad_ohlc_rows": 0,
            "non_positive_price_rows": 0,
            "negative_vol_amount_rows": 0,
            "duplicate_time_rows": 0,
            "max_rows_per_day": 0,
            "median_rows_per_day": 0,
        }

    d = df.copy()
    bad_ohlc = (
        (d["high"] < d[["open", "close", "low"]].max(axis=1))
        | (d["low"] > d[["open", "close", "high"]].min(axis=1))
    )
    non_pos = (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
    neg_va = (d[["vol", "amount"]] < 0).any(axis=1)
    counts = d.groupby("trade_date").size()

    return {
        "ts_code": ts_code,
        "rows": int(len(d)),
        "start_time": str(d["trade_time"].min()),
        "end_time": str(d["trade_time"].max()),
        "trading_days": int(d["trade_date"].nunique()),
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "non_positive_price_rows": int(non_pos.sum()),
        "negative_vol_amount_rows": int(neg_va.sum()),
        "duplicate_time_rows": int(d.duplicated(["ts_code", "trade_time"]).sum()),
        "max_rows_per_day": int(counts.max()) if len(counts) else 0,
        "median_rows_per_day": float(counts.median()) if len(counts) else 0,
    }


def safe_filename(ts_code: str, freq: str, start_date: str, end_date: str) -> str:
    code = ts_code.replace(".", "_")
    return f"{code}_{freq}_{start_date}_{end_date}.parquet"


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    data_dir = out_dir / f"freq={args.freq}"
    log_dir = out_dir / "_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    codes_file = Path(args.codes_file)
    codes = read_codes(codes_file, args.limit_codes)
    chunks = make_chunks(args.start_date, args.end_date, args.chunk_days)
    pro = init_tushare(args.token, args.base_url)

    print("=" * 100)
    print("T+0 ETF minute downloader")
    print("=" * 100)
    print(f"codes           : {len(codes)}")
    print(f"date range      : {args.start_date} -> {args.end_date}")
    print(f"freq            : {args.freq}")
    print(f"chunk_days      : {args.chunk_days}")
    print(f"out_dir         : {out_dir.resolve()}")
    print(f"base_url        : {args.base_url}")
    print(f"sleep_sec       : {args.sleep_sec}")
    print(f"overwrite       : {args.overwrite}")
    print("=" * 100)

    log_rows = []
    val_rows = []

    for i, code in enumerate(codes, 1):
        out_file = data_dir / safe_filename(code, args.freq, args.start_date, args.end_date)

        print(f"\n[{i}/{len(codes)}] {code}")

        if out_file.exists() and not args.overwrite:
            print(f"  exists, skipped: {out_file}")
            try:
                existing = pd.read_parquet(out_file)
                val_rows.append(validate_one(existing, code))
            except Exception as exc:
                print(f"  [WARN] failed to validate existing file: {exc}")
            continue

        parts = []

        for cs, ce in chunks:
            start_dt = fmt_dt_start(cs)
            end_dt = fmt_dt_end(ce)

            try:
                raw = safe_stk_mins(
                    pro=pro,
                    ts_code=code,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    freq=args.freq,
                    sleep_sec=args.sleep_sec,
                    max_retries=args.max_retries,
                )
                d = normalize_minute_df(raw, code)
                n = len(d)
                if n > 0:
                    parts.append(d)
                print(f"  {start_dt} -> {end_dt}: {n:6d} rows")
                log_rows.append({
                    "ts_code": code,
                    "chunk_start": start_dt,
                    "chunk_end": end_dt,
                    "rows": n,
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:
                print(f"  {start_dt} -> {end_dt}: ERROR {type(exc).__name__}: {exc}")
                log_rows.append({
                    "ts_code": code,
                    "chunk_start": start_dt,
                    "chunk_end": end_dt,
                    "rows": 0,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        if parts:
            full = pd.concat(parts, ignore_index=True)
            full = full.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
            full.to_parquet(out_file, index=False)
            val_rows.append(validate_one(full, code))
            print(f"  saved: {out_file} ({len(full)} rows)")
        else:
            val_rows.append(validate_one(pd.DataFrame(), code))
            print("  no rows, skipped saving empty file")

        pd.DataFrame(log_rows).to_csv(log_dir / "download_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_rows).to_csv(log_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(log_rows).to_csv(log_dir / "download_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_rows).to_csv(log_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("Finished")
    print(f"minute parquet dir     : {data_dir.resolve()}")
    print(f"download log           : {(log_dir / 'download_log.csv').resolve()}")
    print(f"validation summary     : {(log_dir / 'validation_summary.csv').resolve()}")
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
