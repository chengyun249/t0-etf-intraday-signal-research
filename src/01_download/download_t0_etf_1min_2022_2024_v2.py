#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_t0_etf_1min_2022_2024_v2.py

修复点：
    v1 使用 tushare.pro_bar / 官方接口，部分环境会提示 token 错误或权限不匹配。
    v2 改回你之前能用的下载方式：
        - 默认 base_url = http://tsy.xiaodefa.cn
        - 使用 pro.stk_mins(...)
        - 分块下载分钟数据
    这和你之前的 fetch_t0_etf_1min_tushare.py 的接口方式一致。

用途：
    下载 2022-01-01 至 2024-12-31 的 T+0 ETF 1min 数据。
    默认内置更适合日内策略的 ETF 候选池：
        黄金/商品/油气
        高波动跨境
        港股科技/互联网/医药
        美股/日经/德国跨境 ETF
    债券/货币默认放在候选里，但 selected 主池默认剔除。

输出：
    data_t0_2022_2024/
      ├─ config/
      │   ├─ t0_etf_candidate_2022_2024_detail.csv
      │   └─ t0_etf_selected_2022_2024_detail.csv
      ├─ raw_1min/
      │   ├─ freq=1min/*.parquet
      │   └─ _logs/
      │       ├─ download_log.csv
      │       └─ validation_summary.csv
      └─ reports/
          ├─ download_report.md
          └─ selected_universe_summary.csv

运行示例：
    python ".\\download_t0_etf_1min_2022_2024_v2.py" `
      --token "你的token" `
      --root-dir ".\\data_t0_2022_2024" `
      --start-date 2022-01-01 `
      --end-date 2024-12-31 `
      --freq 1min `
      --chunk-days 10 `
      --sleep-sec 0.65

如果你已有 codes-file，也可以：
    python ".\\download_t0_etf_1min_2022_2024_v2.py" `
      --token "你的token" `
      --codes-file ".\\data_t0\\config\\t0_etf_codes.csv" `
      --root-dir ".\\data_t0_2022_2024"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = [
    # 黄金 / 商品 / 油气
    ("518880.SH", "黄金ETF华安", "gold_commodity", "黄金商品", "黄金核心"),
    ("159934.SZ", "黄金ETF易方达", "gold_commodity", "黄金商品", "黄金核心"),
    ("159937.SZ", "黄金ETF博时", "gold_commodity", "黄金商品", "黄金核心"),
    ("518800.SH", "黄金ETF国泰", "gold_commodity", "黄金商品", "黄金核心，若接口无数据会自动剔除"),
    ("159985.SZ", "豆粕ETF华夏", "gold_commodity", "黄金商品", "商品期货类"),
    ("159980.SZ", "有色ETF", "gold_commodity", "黄金商品", "商品期货类，自动校验"),
    ("159981.SZ", "能源化工ETF", "gold_commodity", "黄金商品", "商品期货类，自动校验"),
    ("159518.SZ", "标普油气ETF嘉实", "gold_commodity", "黄金商品", "油气，可能 2022 不完整"),
    ("513350.SH", "标普油气ETF富国", "gold_commodity", "黄金商品", "油气，可能 2022 不完整"),

    # 美股 / 日经 / 德国
    ("513050.SH", "中概互联网ETF易方达", "cross_border", "跨境", "高波动跨境"),
    ("513100.SH", "纳指ETF国泰", "cross_border", "跨境", "纳指"),
    ("159941.SZ", "纳指ETF广发", "cross_border", "跨境", "纳指"),
    ("513500.SH", "标普500ETF博时", "cross_border", "跨境", "标普500"),
    ("159612.SZ", "标普500ETF国泰", "cross_border", "跨境", "可能 2022 不完整"),
    ("513030.SH", "德国ETF华安", "cross_border", "跨境", "德国市场"),
    ("513520.SH", "日经ETF华夏", "cross_border", "跨境", "日经"),
    ("513000.SH", "日经225ETF", "cross_border", "跨境", "日经，自动校验"),

    # 港股科技 / 互联网 / 医药
    ("513180.SH", "恒生科技ETF华夏", "cross_border", "跨境", "港股科技"),
    ("513330.SH", "恒生互联网ETF华夏", "cross_border", "跨境", "港股互联网"),
    ("159792.SZ", "港股通互联网ETF富国", "cross_border", "跨境", "港股互联网"),
    ("513060.SH", "恒生医疗ETF博时", "cross_border", "跨境", "港股医疗"),
    ("513120.SH", "港股创新药ETF广发", "cross_border", "跨境", "可能 2022 不完整"),
    ("159570.SZ", "港股通创新药ETF", "cross_border", "跨境", "可能 2022 不完整"),
    ("513090.SH", "香港证券ETF易方达", "cross_border", "跨境", "港股金融"),
    ("510900.SH", "H股ETF易方达", "cross_border", "跨境", "H股"),
    ("159920.SZ", "恒生ETF华夏", "cross_border", "跨境", "恒生指数"),

    # 参考池：默认不进入主策略池
    ("511880.SH", "银华日利ETF", "money_market", "货币", "参考，不建议主动日内策略"),
    ("511990.SH", "华宝添益ETF", "money_market", "货币", "参考，不建议主动日内策略"),
    ("511010.SH", "国债ETF", "bond", "债券", "参考，不建议主动日内策略"),
    ("511260.SH", "十年国债ETF", "bond", "债券", "参考，不建议主动日内策略"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download T+0 ETF 1min data 2022-2024 using pro.stk_mins.")
    p.add_argument("--token", type=str, default=None, help="Tushare token。也可用环境变量 TUSHARE_TOKEN。")
    p.add_argument("--base-url", type=str, default="http://tsy.xiaodefa.cn", help="Tushare API 地址。沿用之前可用脚本的默认地址。")
    p.add_argument("--root-dir", type=str, default="./data_t0")
    p.add_argument("--codes-file", type=str, default="", help="可选。若提供，则以该文件为下载代码池；否则使用内置候选池。")
    p.add_argument("--start-date", type=str, default="2022-01-01", help="YYYY-MM-DD 或 YYYYMMDD")
    p.add_argument("--end-date", type=str, default="2024-12-31", help="YYYY-MM-DD 或 YYYYMMDD")
    p.add_argument("--freq", type=str, default="1min", choices=["1min", "5min", "15min", "30min", "60min"])
    p.add_argument("--chunk-days", type=int, default=10)
    p.add_argument("--sleep-sec", type=float, default=0.65)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-codes", type=int, default=None)
    p.add_argument("--min-trading-days", type=int, default=300)
    p.add_argument("--include-bond-money", action="store_true", help="默认主策略池剔除债券/货币；加此参数则保留。")
    return p.parse_args()


def normalize_date_input(s: str) -> str:
    s = str(s).strip()
    if "-" in s:
        return pd.Timestamp(s).strftime("%Y-%m-%d")
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def safe_filename(ts_code: str, freq: str, start_date: str, end_date: str) -> str:
    code = ts_code.replace(".", "_")
    s = pd.Timestamp(start_date).strftime("%Y%m%d")
    e = pd.Timestamp(end_date).strftime("%Y%m%d")
    return f"{code}_{freq}_{s}_{e}.parquet"


def init_tushare(token: Optional[str], base_url: Optional[str]):
    try:
        import tushare as ts
    except Exception as exc:
        raise ImportError("未安装 tushare，请先 pip install tushare。") from exc

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("未找到 token。请传 --token 或设置环境变量 TUSHARE_TOKEN。")

    token = str(token).strip().strip('"').strip("'")
    ts.set_token(token)
    pro = ts.pro_api(token)

    if base_url:
        try:
            pro._DataApi__http_url = base_url
        except Exception:
            pass

    return pro


def make_candidate_file(config_dir: Path, codes_file: str, limit_codes: Optional[int]) -> pd.DataFrame:
    config_dir.mkdir(parents=True, exist_ok=True)

    if codes_file:
        path = Path(codes_file)
        if not path.exists():
            raise FileNotFoundError(f"codes-file 不存在：{path}")
        df = pd.read_csv(path)
        if "ts_code" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "ts_code"})
            else:
                raise ValueError("codes-file 必须包含 ts_code 列。")
        if "name_manual" not in df.columns:
            df["name_manual"] = ""
        if "t0_category" not in df.columns:
            df["t0_category"] = "unknown"
        if "t0_category_cn" not in df.columns:
            df["t0_category_cn"] = df["t0_category"]
        if "reason" not in df.columns:
            df["reason"] = "from codes-file"
        df = df[["ts_code", "name_manual", "t0_category", "t0_category_cn", "reason"]].copy()
    else:
        df = pd.DataFrame(DEFAULT_CANDIDATES, columns=["ts_code", "name_manual", "t0_category", "t0_category_cn", "reason"])

    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df = df[df["ts_code"] != ""].drop_duplicates("ts_code").reset_index(drop=True)

    if limit_codes is not None:
        df = df.head(limit_codes).copy()

    df.to_csv(config_dir / "t0_etf_candidate_2022_2024_detail.csv", index=False, encoding="utf-8-sig")
    return df


def make_chunks(start_date: str, end_date: str, chunk_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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
    d.columns = [str(c).lower() for c in d.columns]

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

    d["trade_time"] = pd.to_datetime(d["trade_time"], errors="coerce")
    d = d.dropna(subset=["trade_time"])
    d["trade_date"] = d["trade_time"].dt.strftime("%Y-%m-%d")
    d["clock_time"] = d["trade_time"].dt.strftime("%H:%M")

    # 正常交易时段
    d = d[
        ((d["clock_time"] >= "09:30") & (d["clock_time"] <= "11:30"))
        | ((d["clock_time"] >= "13:00") & (d["clock_time"] <= "15:00"))
    ].copy()

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    keep = ["ts_code", "trade_time", "trade_date", "clock_time", "open", "high", "low", "close", "vol", "amount"]
    extra = [c for c in d.columns if c not in keep]
    d = d[keep + extra]

    d = d.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return d


def validate_one(df: pd.DataFrame, ts_code: str) -> Dict:
    if df is None or df.empty:
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
            "total_amount": 0.0,
            "status": "empty",
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
        "total_amount": float(d["amount"].fillna(0).sum()),
        "status": "ok",
    }


def download_all(pro, candidates: pd.DataFrame, raw_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    data_dir = raw_root / f"freq={args.freq}"
    log_dir = raw_root / "_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    chunks = make_chunks(args.start_date, args.end_date, args.chunk_days)
    log_rows = []
    val_rows = []

    for i, row in candidates.iterrows():
        code = row["ts_code"]
        out_file = data_dir / safe_filename(code, args.freq, args.start_date, args.end_date)
        print(f"\n[{i+1}/{len(candidates)}] {code} {row.get('name_manual', '')}")

        if out_file.exists() and not args.overwrite:
            print(f"  exists, skipped: {out_file}")
            try:
                existing = pd.read_parquet(out_file)
                val = validate_one(existing, code)
                val["status"] = "exists"
                val_rows.append(val)
            except Exception as exc:
                print(f"  [WARN] validate existing failed: {exc}")
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
            print(f"  saved: {out_file} ({len(full):,} rows)")
        else:
            val_rows.append(validate_one(pd.DataFrame(), code))
            print("  no rows, skipped saving empty file")

        pd.DataFrame(log_rows).to_csv(log_dir / "download_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_rows).to_csv(log_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(log_rows)
    val_df = pd.DataFrame(val_rows)

    log_df.to_csv(log_dir / "download_log.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(log_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")
    return val_df


def write_selected_and_report(candidates: pd.DataFrame, val: pd.DataFrame, config_dir: Path, reports_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    x = candidates.merge(val, on="ts_code", how="left")
    x["rows"] = x["rows"].fillna(0).astype(int)
    x["trading_days"] = x["trading_days"].fillna(0).astype(int)
    # 覆盖率只用于审计，不能据此事后剔除中途上市/退市标的。
    x["has_full_sample_coverage"] = (
        (x["trading_days"] >= args.min_trading_days) & (x["rows"] > 0)
    ).astype(int)
    x["eligible_for_dynamic_universe"] = (x["rows"] > 0).astype(int)

    if args.include_bond_money:
        x["selected_for_main"] = x["eligible_for_dynamic_universe"]
    else:
        x["selected_for_main"] = (
            (x["eligible_for_dynamic_universe"] == 1)
            & (~x["t0_category"].isin(["bond", "money_market"]))
        ).astype(int)

    selected_path = config_dir / "t0_etf_selected_2022_2024_detail.csv"
    x.to_csv(selected_path, index=False, encoding="utf-8-sig")
    x.to_csv(reports_dir / "selected_universe_summary.csv", index=False, encoding="utf-8-sig")

    selected = x[x["selected_for_main"] == 1].copy()

    lines = []
    lines.append("# 2022-2024 T+0 ETF 分钟数据下载报告\n")
    lines.append("## 1. 说明")
    lines.append("本脚本使用 pro.stk_mins 与 base_url=http://tsy.xiaodefa.cn，沿用之前可用的分钟数据接口方式。")
    lines.append("")
    lines.append("## 2. 下载范围")
    lines.append(f"- start_date: {args.start_date}")
    lines.append(f"- end_date: {args.end_date}")
    lines.append(f"- freq: {args.freq}")
    lines.append(f"- candidate_count: {len(candidates)}")
    lines.append(f"- dynamic-universe candidates: {len(selected)}")
    lines.append("")
    lines.append("## 3. 动态标的池候选（最终每日资格由历史数据决定）")
    if not selected.empty:
        lines.append(selected[["ts_code", "name_manual", "t0_category_cn", "trading_days", "rows", "status"]].to_markdown(index=False))
    else:
        lines.append("无。请检查 token/base_url/接口权限或降低 --min-trading-days。")
    lines.append("")
    lines.append("## 4. 策略建议")
    lines.append("2022-2024 建议用于开发和参数选择，2025 保留为最终样本外检验。债券/货币默认不进入主策略池，除非后续单独做低波动/低风险模块。")
    (reports_dir / "download_report.md").write_text("\n".join(lines), encoding="utf-8")

    return x


def main() -> int:
    args = parse_args()
    args.start_date = normalize_date_input(args.start_date)
    args.end_date = normalize_date_input(args.end_date)

    root = Path(args.root_dir)
    config_dir = root / "config"
    raw_root = root / "raw_1min"
    reports_dir = root / "reports"

    for p in [config_dir, raw_root, reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Download T+0 ETF 1min 2022-2024 v2")
    print("=" * 100)
    print(f"root_dir    : {root.resolve()}")
    print(f"date range  : {args.start_date} -> {args.end_date}")
    print(f"freq        : {args.freq}")
    print(f"base_url    : {args.base_url}")
    print(f"chunk_days  : {args.chunk_days}")
    print("=" * 100)

    candidates = make_candidate_file(config_dir, args.codes_file, args.limit_codes)
    print(f"candidate count: {len(candidates)}")

    pro = init_tushare(args.token, args.base_url)
    val = download_all(pro, candidates, raw_root, args)
    selected = write_selected_and_report(candidates, val, config_dir, reports_dir, args)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"candidate file : {(config_dir / 't0_etf_candidate_2022_2024_detail.csv').resolve()}")
    print(f"selected file  : {(config_dir / 't0_etf_selected_2022_2024_detail.csv').resolve()}")
    print(f"raw data dir   : {(raw_root / ('freq=' + args.freq)).resolve()}")
    print(f"validation     : {(raw_root / '_logs' / 'validation_summary.csv').resolve()}")
    print(f"report         : {(reports_dir / 'download_report.md').resolve()}")
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
