#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_t0_etf_1min_2022_2024.py

用途：
    下载 2022-01-01 至 2024-12-31 的 T+0 ETF 1min 数据，并同时下载日频辅助数据。
    目标不是“全市场覆盖”，而是围绕更适合日内策略的 ETF：黄金/商品/油气/高波动跨境。

推荐研究口径：
    2022-2024：开发期 / 参数选择 / walk-forward
    2025：最终样本外检验，不要参与调参

输出目录默认：
    data_t0_2022_2024/
      ├─ config/
      │   ├─ t0_etf_candidate_2022_2024_detail.csv
      │   └─ t0_etf_selected_2022_2024_detail.csv
      ├─ raw_1min/freq=1min/
      │   └─ 513050_SH_1min_20220101_20241231.parquet
      ├─ auxiliary/
      │   ├─ fund_daily.csv
      │   ├─ fund_adj.csv
      │   ├─ index_daily.csv
      │   ├─ trade_cal.csv
      │   └─ daily_regime_features.csv
      └─ reports/
          ├─ download_1min_summary.csv
          └─ download_report.md

运行：
    set TUSHARE_TOKEN=你的token
    python ".\\download_t0_etf_1min_2022_2024.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --start-date 20220101 `
      --end-date 20241231

如果要覆盖重下：
    python ".\\download_t0_etf_1min_2022_2024.py" --overwrite

注意：
    1. 默认会用 fund_basic 校验 ETF 是否在 2022-2024 期间上市；
    2. 如果 fund_basic 权限不可用，会退回到手工候选池；
    3. pro_bar 分月下载，失败会记录，不会静默吞掉；
    4. Tushare 分钟 amount 的单位后续要审计，脚本不假设它是元还是千元。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. ETF 候选池
# ============================================================

# 设计原则：
# - 优先选择 2022 年前后已经存在、适合 2022-2024 回测的 T+0 ETF；
# - 商品/黄金/油气优先；
# - 高波动跨境作为第二层；
# - 债券/货币不放入主策略池，只在需要时另做基准。
#
# 注意：部分代码如果已更名、未上市、接口无数据，脚本会自动在下载阶段剔除。
DEFAULT_CANDIDATES = [
    # ---------------- 黄金 / 商品 / 油气：主力候选 ----------------
    ("518880.SH", "黄金ETF华安", "gold_commodity", "黄金商品", "核心：黄金"),
    ("159934.SZ", "黄金ETF易方达", "gold_commodity", "黄金商品", "核心：黄金"),
    ("159937.SZ", "黄金ETF博时", "gold_commodity", "黄金商品", "核心：黄金"),
    ("518800.SH", "黄金ETF国泰", "gold_commodity", "黄金商品", "核心：黄金，若接口有数据则保留"),
    ("159985.SZ", "豆粕ETF华夏", "gold_commodity", "黄金商品", "商品期货类"),
    ("159980.SZ", "有色ETF", "gold_commodity", "黄金商品", "商品期货类，脚本校验有效性"),
    ("159981.SZ", "能源化工ETF", "gold_commodity", "黄金商品", "商品期货类，脚本校验有效性"),
    ("159518.SZ", "标普油气ETF嘉实", "gold_commodity", "黄金商品", "若 2022 不完整则自动标记"),
    ("513350.SH", "标普油气ETF富国", "gold_commodity", "黄金商品", "若 2022 不完整则自动标记"),

    # ---------------- 美股 / 日经 / 德国：高波动跨境 ----------------
    ("513050.SH", "中概互联网ETF易方达", "cross_border", "跨境", "高波动跨境"),
    ("513100.SH", "纳指ETF国泰", "cross_border", "跨境", "纳指"),
    ("159941.SZ", "纳指ETF广发", "cross_border", "跨境", "纳指"),
    ("513500.SH", "标普500ETF博时", "cross_border", "跨境", "标普500"),
    ("159612.SZ", "标普500ETF国泰", "cross_border", "跨境", "若 2022 不完整则自动标记"),
    ("513030.SH", "德国ETF华安", "cross_border", "跨境", "德国市场"),
    ("513520.SH", "日经ETF华夏", "cross_border", "跨境", "日经"),
    ("513000.SH", "日经225ETF", "cross_border", "跨境", "日经，脚本校验有效性"),

    # ---------------- 港股科技 / 互联网 / 医药 ----------------
    ("513180.SH", "恒生科技ETF华夏", "cross_border", "跨境", "港股科技"),
    ("513330.SH", "恒生互联网ETF华夏", "cross_border", "跨境", "港股互联网"),
    ("159792.SZ", "港股通互联网ETF富国", "cross_border", "跨境", "港股互联网"),
    ("513060.SH", "恒生医疗ETF博时", "cross_border", "跨境", "港股医疗"),
    ("513120.SH", "港股创新药ETF广发", "cross_border", "跨境", "若 2022 不完整则自动标记"),
    ("159570.SZ", "港股通创新药ETF", "cross_border", "跨境", "若 2022 不完整则自动标记"),
    ("513090.SH", "香港证券ETF易方达", "cross_border", "跨境", "港股金融"),
    ("510900.SH", "H股ETF易方达", "cross_border", "跨境", "H股"),
    ("159920.SZ", "恒生ETF华夏", "cross_border", "跨境", "恒生指数"),

    # ---------------- 可选基准，不建议主策略交易 ----------------
    ("511880.SH", "银华日利ETF", "money_market", "货币", "只作参考，不建议主动日内策略"),
    ("511990.SH", "华宝添益ETF", "money_market", "货币", "只作参考，不建议主动日内策略"),
    ("511010.SH", "国债ETF", "bond", "债券", "只作参考，不建议主动日内策略"),
    ("511260.SH", "十年国债ETF", "bond", "债券", "只作参考，不建议主动日内策略"),
]


# ============================================================
# 2. 参数
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download T+0 ETF 1min and auxiliary data for 2022-2024.")
    p.add_argument("--token", type=str, default=os.environ.get("TUSHARE_TOKEN", ""), help="Tushare token，也可用环境变量 TUSHARE_TOKEN。")
    p.add_argument("--root-dir", type=str, default="./data_t0_2022_2024")
    p.add_argument("--start-date", type=str, default="20220101")
    p.add_argument("--end-date", type=str, default="20241231")
    p.add_argument("--freq", type=str, default="1min")
    p.add_argument("--sleep", type=float, default=0.25, help="每次接口调用后的暂停秒数。")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在 parquet。")
    p.add_argument("--min-trading-days", type=int, default=300, help="最终 selected 池要求的最低有效交易日数。")
    p.add_argument("--include-bond-money", action="store_true", help="默认 selected 主池剔除债券/货币；加此参数则保留。")
    return p.parse_args()


# ============================================================
# 3. 工具函数
# ============================================================

def safe_code(ts_code: str) -> str:
    return str(ts_code).replace(".", "_")


def month_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    cur = pd.Timestamp(start_ts.year, start_ts.month, 1)
    chunks = []
    while cur <= end_ts:
        nxt = cur + pd.offsets.MonthBegin(1)
        s = max(cur, start_ts)
        e = min(nxt - pd.Timedelta(days=1), end_ts)
        chunks.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
        cur = nxt
    return chunks


def normalize_minute_df(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]

    # pro_bar 常见列：ts_code, trade_time, open, high, low, close, vol, amount
    if "ts_code" not in x.columns:
        x["ts_code"] = ts_code
    x["ts_code"] = x["ts_code"].astype(str)

    if "trade_time" not in x.columns:
        # 极少数接口可能叫 datetime/time
        for alt in ["datetime", "time", "trade_date"]:
            if alt in x.columns:
                x["trade_time"] = x[alt]
                break
    if "trade_time" not in x.columns:
        raise ValueError(f"{ts_code} minute data 缺少 trade_time。columns={list(x.columns)}")

    x["trade_time"] = pd.to_datetime(x["trade_time"], errors="coerce")
    x = x.dropna(subset=["trade_time"])

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x["trade_date"] = x["trade_time"].dt.strftime("%Y%m%d")
    x["clock_time"] = x["trade_time"].dt.strftime("%H:%M")

    # 保留 A 股 ETF 正常交易时间。午休没有数据也没关系。
    x = x[
        ((x["clock_time"] >= "09:30") & (x["clock_time"] <= "11:30"))
        | ((x["clock_time"] >= "13:00") & (x["clock_time"] <= "15:00"))
    ].copy()

    x = x.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    return x[[
        "ts_code", "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "vol", "amount"
    ]]


def write_markdown_report(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 4. Tushare 下载函数
# ============================================================

def init_tushare(token: str):
    if not token:
        raise RuntimeError("缺少 Tushare token。请设置环境变量 TUSHARE_TOKEN 或传入 --token。")
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api(token)
    return ts, pro


def build_candidate_universe(pro, start_date: str, end_date: str, out_config: Path) -> pd.DataFrame:
    base = pd.DataFrame(DEFAULT_CANDIDATES, columns=[
        "ts_code", "name_manual", "t0_category", "t0_category_cn", "reason"
    ])

    fund_basic = pd.DataFrame()
    try:
        fund_basic = pro.fund_basic(market="E")
        if fund_basic is not None and not fund_basic.empty:
            fund_basic.columns = [str(c).lower() for c in fund_basic.columns]
    except Exception as e:
        print(f"[WARN] fund_basic 获取失败，使用手工候选池继续：{e}")

    if fund_basic is not None and not fund_basic.empty and "ts_code" in fund_basic.columns:
        keep_cols = [c for c in ["ts_code", "name", "fund_type", "list_date", "delist_date", "market"] if c in fund_basic.columns]
        uni = base.merge(fund_basic[keep_cols], on="ts_code", how="left")
    else:
        uni = base.copy()
        uni["name"] = uni["name_manual"]
        uni["list_date"] = ""
        uni["delist_date"] = ""

    # 上市日期校验：不直接删除，但标记。
    uni["list_date"] = uni.get("list_date", "").fillna("").astype(str)
    uni["delist_date"] = uni.get("delist_date", "").fillna("").astype(str)

    uni["listed_before_start"] = uni["list_date"].apply(lambda x: bool(x) and x <= start_date)
    uni["listed_before_end"] = uni["list_date"].apply(lambda x: (not bool(x)) or x <= end_date)
    uni["delisted_before_end"] = uni["delist_date"].apply(lambda x: bool(x) and x <= end_date)
    uni["candidate_status"] = np.where(
        uni["delisted_before_end"],
        "delisted_before_end",
        np.where(uni["listed_before_end"], "candidate", "listed_after_end")
    )

    out_config.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(out_config, index=False, encoding="utf-8-sig")
    return uni


def download_one_etf_minute(ts, ts_code: str, start_date: str, end_date: str, freq: str, sleep: float) -> Tuple[pd.DataFrame, List[Dict]]:
    parts = []
    logs = []
    for s, e in month_chunks(start_date, end_date):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=s,
                end_date=e,
                asset="FD",
                freq=freq,
                adj=None,
            )
            time.sleep(sleep)
            if df is None or df.empty:
                logs.append({"ts_code": ts_code, "chunk_start": s, "chunk_end": e, "rows": 0, "status": "empty"})
                continue

            x = normalize_minute_df(df, ts_code)
            parts.append(x)
            logs.append({"ts_code": ts_code, "chunk_start": s, "chunk_end": e, "rows": len(x), "status": "ok"})
        except Exception as ex:
            logs.append({"ts_code": ts_code, "chunk_start": s, "chunk_end": e, "rows": 0, "status": f"error: {ex}"})
            time.sleep(max(sleep, 1.0))

    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(["ts_code", "trade_time"]).sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    return out, logs


def download_minute_all(ts, candidates: pd.DataFrame, raw_dir: Path, reports_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    chunk_logs_all = []

    for i, row in candidates.iterrows():
        code = row["ts_code"]
        outfile = raw_dir / f"{safe_code(code)}_{args.freq}_{args.start_date}_{args.end_date}.parquet"

        print(f"\n[{i+1}/{len(candidates)}] {code} {row.get('name_manual', '')}")
        if outfile.exists() and not args.overwrite:
            try:
                old = pd.read_parquet(outfile, columns=["ts_code", "trade_date", "trade_time", "amount"])
                summary_rows.append({
                    "ts_code": code,
                    "file": str(outfile),
                    "rows": len(old),
                    "trading_days": old["trade_date"].nunique(),
                    "first_time": str(pd.to_datetime(old["trade_time"]).min()),
                    "last_time": str(pd.to_datetime(old["trade_time"]).max()),
                    "total_amount": float(pd.to_numeric(old["amount"], errors="coerce").fillna(0).sum()),
                    "status": "exists",
                })
                print(f"  exists rows={len(old):,}, days={old['trade_date'].nunique()}")
                continue
            except Exception:
                pass

        df, logs = download_one_etf_minute(ts, code, args.start_date, args.end_date, args.freq, args.sleep)
        chunk_logs_all.extend(logs)

        if df.empty:
            summary_rows.append({
                "ts_code": code,
                "file": str(outfile),
                "rows": 0,
                "trading_days": 0,
                "first_time": "",
                "last_time": "",
                "total_amount": 0.0,
                "status": "empty_or_failed",
            })
            print("  empty_or_failed")
            continue

        df.to_parquet(outfile, index=False)
        summary_rows.append({
            "ts_code": code,
            "file": str(outfile),
            "rows": len(df),
            "trading_days": df["trade_date"].nunique(),
            "first_time": str(df["trade_time"].min()),
            "last_time": str(df["trade_time"].max()),
            "total_amount": float(df["amount"].fillna(0).sum()),
            "status": "ok",
        })
        print(f"  saved rows={len(df):,}, days={df['trade_date'].nunique()}, file={outfile}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports_dir / "download_1min_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chunk_logs_all).to_csv(reports_dir / "download_1min_chunk_logs.csv", index=False, encoding="utf-8-sig")
    return summary


def download_auxiliary(pro, selected_codes: List[str], aux_dir: Path, args: argparse.Namespace) -> None:
    aux_dir.mkdir(parents=True, exist_ok=True)

    # 交易日历
    try:
        cal = pro.trade_cal(exchange="SSE", start_date=args.start_date, end_date=args.end_date)
        cal.to_csv(aux_dir / "trade_cal.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] trade_cal 下载失败：{e}")

    # ETF 日线
    daily_parts = []
    for i, code in enumerate(selected_codes, 1):
        try:
            df = pro.fund_daily(ts_code=code, start_date=args.start_date, end_date=args.end_date)
            time.sleep(args.sleep)
            if df is not None and not df.empty:
                daily_parts.append(df)
            print(f"  fund_daily {i}/{len(selected_codes)} {code}: {0 if df is None else len(df)}")
        except Exception as e:
            print(f"[WARN] fund_daily {code} 失败：{e}")

    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True)
        daily.columns = [str(c).lower() for c in daily.columns]
        daily = daily.drop_duplicates(["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])
        daily.to_csv(aux_dir / "fund_daily.csv", index=False, encoding="utf-8-sig")
    else:
        daily = pd.DataFrame()

    # 复权因子
    adj_parts = []
    for i, code in enumerate(selected_codes, 1):
        try:
            df = pro.fund_adj(ts_code=code)
            time.sleep(args.sleep)
            if df is not None and not df.empty:
                df.columns = [str(c).lower() for c in df.columns]
                df = df[(df["trade_date"] >= args.start_date) & (df["trade_date"] <= args.end_date)]
                adj_parts.append(df)
            print(f"  fund_adj {i}/{len(selected_codes)} {code}: {0 if df is None else len(df)}")
        except Exception as e:
            print(f"[WARN] fund_adj {code} 失败：{e}")

    if adj_parts:
        adj = pd.concat(adj_parts, ignore_index=True)
        adj = adj.drop_duplicates(["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])
        adj.to_csv(aux_dir / "fund_adj.csv", index=False, encoding="utf-8-sig")
    else:
        adj = pd.DataFrame()

    # 指数日线：主要用于后续 regime，不强依赖。
    index_codes = ["000001.SH", "000300.SH", "000905.SH", "000852.SH", "399006.SZ"]
    idx_parts = []
    for code in index_codes:
        try:
            df = pro.index_daily(ts_code=code, start_date=args.start_date, end_date=args.end_date)
            time.sleep(args.sleep)
            if df is not None and not df.empty:
                idx_parts.append(df)
            print(f"  index_daily {code}: {0 if df is None else len(df)}")
        except Exception as e:
            print(f"[WARN] index_daily {code} 失败：{e}")

    if idx_parts:
        idx = pd.concat(idx_parts, ignore_index=True)
        idx.columns = [str(c).lower() for c in idx.columns]
        idx.to_csv(aux_dir / "index_daily.csv", index=False, encoding="utf-8-sig")

    # daily regime features
    build_daily_regime_features(daily, adj, aux_dir / "daily_regime_features.csv")


def build_daily_regime_features(daily: pd.DataFrame, adj: pd.DataFrame, outfile: Path) -> None:
    if daily is None or daily.empty:
        print("[WARN] fund_daily 为空，无法生成 daily_regime_features.csv")
        return

    x = daily.copy()
    x.columns = [str(c).lower() for c in x.columns]

    for c in ["open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if "pct_chg" in x.columns and x["pct_chg"].notna().any():
        x["daily_ret"] = x["pct_chg"] / 100.0
    else:
        x["daily_ret"] = x.groupby("ts_code")["close"].pct_change()

    x["daily_amount"] = x["amount"]
    x["daily_vol"] = x["vol"]

    # 低流动性日：只做标记，不删数据。
    x["amount_ma20"] = x.groupby("ts_code")["daily_amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    x["daily_is_low_liquidity_day"] = ((x["daily_amount"].fillna(0) <= 0) | (x["daily_amount"] < 0.2 * x["amount_ma20"])).astype(int)

    # 极端收益日
    x["daily_is_extreme_return_day"] = (x["daily_ret"].abs() >= 0.08).astype(int)

    if adj is not None and not adj.empty:
        a = adj.copy()
        a.columns = [str(c).lower() for c in a.columns]
        if "adj_factor" in a.columns:
            a["adj_factor"] = pd.to_numeric(a["adj_factor"], errors="coerce")
            a = a.sort_values(["ts_code", "trade_date"])
            a["adj_factor_prev"] = a.groupby("ts_code")["adj_factor"].shift(1)
            a["daily_adj_factor_change"] = (
                a["adj_factor"].notna()
                & a["adj_factor_prev"].notna()
                & (a["adj_factor"] != a["adj_factor_prev"])
            ).astype(int)
            x = x.merge(a[["ts_code", "trade_date", "adj_factor", "daily_adj_factor_change"]], on=["ts_code", "trade_date"], how="left")
        else:
            x["daily_adj_factor_change"] = 0
    else:
        x["daily_adj_factor_change"] = 0

    x["daily_adj_factor_change"] = x["daily_adj_factor_change"].fillna(0).astype(int)

    out_cols = [
        "ts_code", "trade_date",
        "open", "high", "low", "close", "daily_ret",
        "daily_amount", "daily_vol",
        "daily_is_low_liquidity_day",
        "daily_adj_factor_change",
        "daily_is_extreme_return_day",
    ]
    out_cols = [c for c in out_cols if c in x.columns]
    x[out_cols].to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"  daily_regime_features saved: {outfile}, rows={len(x):,}")


# ============================================================
# 5. 主流程
# ============================================================

def main() -> int:
    args = parse_args()

    root = Path(args.root_dir)
    config_dir = root / "config"
    raw_dir = root / "raw_1min" / f"freq={args.freq}"
    aux_dir = root / "auxiliary"
    reports_dir = root / "reports"

    for p in [config_dir, raw_dir, aux_dir, reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Download T+0 ETF 1min data: 2022-2024")
    print("=" * 100)
    print(f"root_dir   : {root.resolve()}")
    print(f"start_date : {args.start_date}")
    print(f"end_date   : {args.end_date}")
    print(f"freq       : {args.freq}")
    print("=" * 100)

    ts, pro = init_tushare(args.token)

    candidate_path = config_dir / "t0_etf_candidate_2022_2024_detail.csv"
    candidates = build_candidate_universe(pro, args.start_date, args.end_date, candidate_path)

    # 不下载明确在 end_date 后才上市或已退市的。
    candidates_to_download = candidates[candidates["candidate_status"] == "candidate"].copy()
    print(f"candidate count for download: {len(candidates_to_download)}")

    summary = download_minute_all(ts, candidates_to_download, raw_dir, reports_dir, args)

    # 合并候选信息，筛 selected。
    selected = candidates.merge(summary[["ts_code", "rows", "trading_days", "first_time", "last_time", "status"]], on="ts_code", how="left")
    selected["rows"] = selected["rows"].fillna(0).astype(int)
    selected["trading_days"] = selected["trading_days"].fillna(0).astype(int)

    selected["has_enough_data"] = (selected["trading_days"] >= args.min_trading_days) & (selected["rows"] > 0)

    if not args.include_bond_money:
        selected["selected_for_main"] = (
            selected["has_enough_data"]
            & (~selected["t0_category"].isin(["bond", "money_market"]))
        ).astype(int)
    else:
        selected["selected_for_main"] = selected["has_enough_data"].astype(int)

    selected_path = config_dir / "t0_etf_selected_2022_2024_detail.csv"
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    selected_codes = selected.loc[selected["selected_for_main"] == 1, "ts_code"].tolist()
    print(f"\nselected_for_main count: {len(selected_codes)}")
    print(selected.loc[selected["selected_for_main"] == 1, ["ts_code", "name_manual", "t0_category_cn", "trading_days"]].to_string(index=False))

    # 下载日频辅助数据
    print("\nDownloading auxiliary daily data...")
    download_auxiliary(pro, selected_codes, aux_dir, args)

    # 报告
    lines = []
    lines.append("# 2022-2024 T+0 ETF 数据下载报告\n")
    lines.append("## 1. 目标")
    lines.append("本次数据用于重新评估 T+0 ETF 日内策略。2022-2024 建议作为开发/训练/参数选择样本，2025 保留为最终样本外测试。")
    lines.append("")
    lines.append("## 2. 选池原则")
    lines.append("- 优先选择黄金、商品、油气、高波动跨境 ETF。")
    lines.append("- 债券、货币 ETF 不作为主策略池，因为前期测试显示其日内突破/动量收益边际较弱。")
    lines.append("- 对 2022-2024 数据覆盖不足的 ETF 自动标记，不进入主策略池。")
    lines.append("")
    lines.append("## 3. 下载结果")
    lines.append(f"- 候选 ETF 数：{len(candidates)}")
    lines.append(f"- 实际尝试下载：{len(candidates_to_download)}")
    lines.append(f"- selected_for_main：{len(selected_codes)}")
    lines.append("")
    if selected_codes:
        lines.append("## 4. 主策略池")
        lines.append(selected.loc[selected["selected_for_main"] == 1, ["ts_code", "name_manual", "t0_category_cn", "trading_days", "rows"]].to_markdown(index=False))
    lines.append("")
    lines.append("## 5. 重要说明")
    lines.append("旧样本不一定更容易赚钱；更合理的做法是用 2022-2024 做开发，2025 做最终检验。策略如果只在旧样本有效、2025 失效，说明它可能已经被套利、市场结构变化，或只是过拟合。")
    write_markdown_report(reports_dir / "download_report.md", lines)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"candidate universe : {candidate_path.resolve()}")
    print(f"selected universe  : {selected_path.resolve()}")
    print(f"raw 1min dir       : {raw_dir.resolve()}")
    print(f"auxiliary dir      : {aux_dir.resolve()}")
    print(f"download report    : {(reports_dir / 'download_report.md').resolve()}")
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
