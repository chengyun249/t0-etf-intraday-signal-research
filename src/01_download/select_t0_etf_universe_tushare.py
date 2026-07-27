#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_t0_etf_universe_tushare.py

用途：
    从 Tushare ETF 基础信息中筛选适合 1min / 5min 日内交易研究的 T+0 ETF 池。
    目标是生成约 30 只高流动性、可日内回转交易的 ETF 清单。

核心选择逻辑：
    1. 只纳入理论上支持 T+0 的 ETF 类型：跨境 ETF、债券 ETF、黄金/商品期货 ETF、货币 ETF。
    2. 排除 A 股股票型 ETF、行业 ETF、宽基 ETF、联接基金等 T+1 或非交易目标。
    3. 用日频成交额、活跃交易日、波动率代理指标做流动性和可交易性评分。
    4. 控制类别比例，避免 30 只全是跨境或全是债券。
    5. 输出候选全集、最终 30 只清单、代码文件和类别汇总。

注意：
    Tushare 基础表通常没有直接的“是否 T+0”字段，本脚本使用 ETF 名称和基金类型关键词进行规则分类。
    最终结果必须人工快速复核一遍，尤其是“沪港深”“港股通”等名称边界品种。

运行示例：
    $env:TUSHARE_TOKEN="你的token"

    python ".\\select_t0_etf_universe_tushare.py" `
      --out-dir ".\\data_t0\\config" `
      --target-size 30 `
      --liquidity-start 20250101 `
      --liquidity-end 20251231 `
      --sleep-sec 0.60

如果使用代理 Tushare 地址：
    python ".\\select_t0_etf_universe_tushare.py" `
      --base-url "http://tsy.xiaodefa.cn" `
      --out-dir ".\\data_t0\\config"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select about 30 T+0 ETF candidates for intraday trading project.")
    p.add_argument("--token", type=str, default=None, help="Tushare token。也可用环境变量 TUSHARE_TOKEN。")
    p.add_argument("--base-url", type=str, default="http://tsy.xiaodefa.cn", help="Tushare API 地址。默认使用代理地址。")
    p.add_argument("--out-dir", type=str, default="./data_t0/config", help="输出目录。")
    p.add_argument("--target-size", type=int, default=30, help="最终 ETF 数量，默认 30。")
    p.add_argument("--liquidity-start", type=str, default="20250101", help="流动性统计开始日期，YYYYMMDD。")
    p.add_argument("--liquidity-end", type=str, default="20251231", help="流动性统计结束日期，YYYYMMDD。")
    p.add_argument("--sleep-sec", type=float, default=0.60, help="接口请求间隔，避免限速。")
    p.add_argument("--min-active-days", type=int, default=120, help="流动性窗口内最低有效交易日。")
    p.add_argument("--min-median-amount-mil", type=float, default=5.0, help="最低日成交额中位数，单位：百万元。")
    p.add_argument("--max-per-family", type=int, default=3, help="同一主题/系列最多保留数量，用于同质 ETF 降重。")
    p.add_argument("--dry-run", action="store_true", help="只输出基础候选，不拉取日行情。")
    return p.parse_args()


def init_tushare(token: Optional[str], base_url: Optional[str]):
    try:
        import tushare as ts
    except Exception as exc:
        raise ImportError("未安装 tushare，请先 pip install tushare。") from exc
    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("没有 token。请传 --token 或设置环境变量 TUSHARE_TOKEN。")
    ts.set_token(token)
    pro = ts.pro_api(token)
    if base_url:
        try:
            pro._DataApi__http_url = base_url
        except Exception:
            pass
    return pro


def safe_call(func, sleep_sec: float, retries: int = 3, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            out = func(**kwargs)
            time.sleep(sleep_sec)
            return out
        except Exception as exc:
            last_err = exc
            wait = (i + 1) * 2
            print(f"[WARN] API call failed {i+1}/{retries}: {exc}; wait {wait}s")
            time.sleep(wait)
    raise last_err


# T+0 正向类别关键词。
RE_CROSS_BORDER = re.compile(
    r"(跨境|QDII|恒生|恒指|恒生科技|港股|港股通|H股|中概|中概互联|互联网|"
    r"纳指|纳斯达克|标普|标普500|S&P|道琼斯|日经|日本|德国|法国|印度|"
    r"东南亚|亚太|海外|全球|MSCI|美元|美股|港美|香港)",
    re.I,
)
RE_BOND = re.compile(
    r"(债|国债|政金债|金融债|信用债|公司债|企业债|城投债|地方债|利率债|"
    r"可转债|转债|短融|中票|AAA|5年|10年|30年)",
    re.I,
)
RE_GOLD_COMMODITY = re.compile(
    r"(黄金|商品|期货|豆粕|能源化工|有色期货|铜|原油|油气|白银)",
    re.I,
)
RE_MONEY = re.compile(
    r"(货币|现金|添益|快线|日利|保证金|场内货币|收益快线|现金宝)",
    re.I,
)


def classify_t0_category(name: str, fund_type: str = "") -> Tuple[str, str, int]:
    s = f"{name or ''} {fund_type or ''}"
    if re.search(r"(联接|LOF|FOF|REIT|REITS)", s, re.I):
        return "exclude", "剔除", 0
    if RE_GOLD_COMMODITY.search(s):
        # 防止“有色金属ETF”等 A股行业误判为商品。
        if re.search(r"(有色金属|稀土|钢铁|煤炭)", s) and not re.search(r"(期货|商品|黄金|原油|豆粕|能源化工)", s):
            return "exclude", "剔除", 0
        return "gold_commodity", "黄金商品", 90
    if RE_BOND.search(s):
        return "bond", "债券", 80
    if RE_MONEY.search(s):
        return "money_market", "货币", 60
    if RE_CROSS_BORDER.search(s):
        return "cross_border", "跨境", 100
    return "exclude", "剔除", 0


def family_key(name: str) -> str:
    s = str(name)
    s = re.sub(r"(ETF|交易型开放式指数证券投资基金|交易型开放式指数基金|指数证券投资基金|指数基金)", "", s, flags=re.I)
    s = re.sub(r"(华夏|易方达|华泰柏瑞|南方|嘉实|广发|博时|富国|招商|国泰|汇添富|银华|鹏华|工银瑞信|天弘|景顺长城|建信|平安|华安|国联安|大成|华宝|海富通|摩根|万家|永赢|华富|申万菱信|浦银安盛)", "", s)
    s = re.sub(r"[\s\-\(\)（）A-Za-z0-9]+", "", s)
    return s[:12] if s else str(name)[:12]


def fetch_fund_basic(pro, sleep_sec: float) -> pd.DataFrame:
    try:
        df = safe_call(
            pro.fund_basic,
            sleep_sec=sleep_sec,
            market="E",
            fields="ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,issue_date,delist_date,exchange",
        )
    except Exception:
        df = safe_call(pro.fund_basic, sleep_sec=sleep_sec, market="E")
    if df is None or df.empty:
        raise RuntimeError("fund_basic 返回为空。")
    for c in ["ts_code", "name", "fund_type", "list_date", "delist_date", "exchange"]:
        if c not in df.columns:
            df[c] = ""
    return df.drop_duplicates("ts_code").copy()


def fetch_daily_liquidity_for_code(pro, ts_code: str, start_date: str, end_date: str, sleep_sec: float) -> Dict:
    try:
        d = safe_call(pro.fund_daily, sleep_sec=sleep_sec, ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as exc:
        return {
            "ts_code": ts_code,
            "active_days": 0,
            "avg_amount_raw": np.nan,
            "median_amount_raw": np.nan,
            "avg_amount_mil": np.nan,
            "median_amount_mil": np.nan,
            "daily_vol": np.nan,
            "abs_ret_mean": np.nan,
            "liquidity_error": f"{type(exc).__name__}: {exc}",
        }
    if d is None or d.empty:
        return {
            "ts_code": ts_code,
            "active_days": 0,
            "avg_amount_raw": np.nan,
            "median_amount_raw": np.nan,
            "avg_amount_mil": np.nan,
            "median_amount_mil": np.nan,
            "daily_vol": np.nan,
            "abs_ret_mean": np.nan,
            "liquidity_error": "empty",
        }
    d = d.copy()
    for c in ["amount", "close", "pre_close", "pct_chg"]:
        if c not in d.columns:
            d[c] = np.nan
    amount = pd.to_numeric(d["amount"], errors="coerce")
    close = pd.to_numeric(d["close"], errors="coerce")
    pre_close = pd.to_numeric(d["pre_close"], errors="coerce")
    ret = close / pre_close - 1.0
    pct = pd.to_numeric(d["pct_chg"], errors="coerce") / 100.0
    ret = ret.where(ret.notna(), pct)
    active = (amount.fillna(0) > 0) & close.notna()
    return {
        "ts_code": ts_code,
        "active_days": int(active.sum()),
        "avg_amount_raw": float(amount[active].mean()) if active.any() else np.nan,
        "median_amount_raw": float(amount[active].median()) if active.any() else np.nan,
        # 多数 Tushare 行情 amount 单位为千元，除以 1000 近似为百万元；若你的接口单位不同，不影响排序主体。
        "avg_amount_mil": float(amount[active].mean() / 1000.0) if active.any() else np.nan,
        "median_amount_mil": float(amount[active].median() / 1000.0) if active.any() else np.nan,
        "daily_vol": float(ret[active].std()) if active.sum() >= 20 else np.nan,
        "abs_ret_mean": float(ret[active].abs().mean()) if active.sum() >= 20 else np.nan,
        "liquidity_error": "",
    }


def add_liquidity(pro, candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    total = len(candidates)
    for i, code in enumerate(candidates["ts_code"], 1):
        print(f"[{i}/{total}] liquidity {code}")
        rows.append(fetch_daily_liquidity_for_code(pro, code, args.liquidity_start, args.liquidity_end, args.sleep_sec))
    return candidates.merge(pd.DataFrame(rows), on="ts_code", how="left")


def pct_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True).fillna(0.0)


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rank_avg_amount"] = d.groupby("t0_category")["avg_amount_mil"].transform(pct_rank)
    d["rank_median_amount"] = d.groupby("t0_category")["median_amount_mil"].transform(pct_rank)
    d["rank_active_days"] = d.groupby("t0_category")["active_days"].transform(pct_rank)
    d["rank_abs_ret"] = d.groupby("t0_category")["abs_ret_mean"].transform(pct_rank)
    d["selection_score"] = (
        0.40 * d["rank_avg_amount"]
        + 0.25 * d["rank_median_amount"]
        + 0.20 * d["rank_active_days"]
        + 0.15 * d["rank_abs_ret"]
    )
    return d


def quota_by_category(target_size: int) -> Dict[str, int]:
    base = {"cross_border": 14, "bond": 7, "gold_commodity": 6, "money_market": 3}
    if target_size == 30:
        return base
    total_base = sum(base.values())
    q = {k: max(1, int(round(v / total_base * target_size))) for k, v in base.items()}
    diff = target_size - sum(q.values())
    order = ["cross_border", "gold_commodity", "bond", "money_market"]
    j = 0
    while diff != 0:
        k = order[j % len(order)]
        if diff > 0:
            q[k] += 1
            diff -= 1
        else:
            if q[k] > 1:
                q[k] -= 1
                diff += 1
        j += 1
    return q


def select_final(df: pd.DataFrame, target_size: int, max_per_family: int, min_active_days: int, min_median_amount_mil: float) -> pd.DataFrame:
    d = df[(df["active_days"] >= min_active_days) & (df["median_amount_mil"] >= min_median_amount_mil)].copy()
    if d.empty:
        raise RuntimeError("硬过滤后没有候选。请降低 min-active-days 或 min-median-amount-mil。")
    q = quota_by_category(target_size)
    selected_parts = []
    selected_codes = set()
    for cat, n in q.items():
        sub = d[d["t0_category"] == cat].sort_values("selection_score", ascending=False).copy()
        fam_count: Dict[str, int] = {}
        picked = []
        for _, row in sub.iterrows():
            code = row["ts_code"]
            fam = row["family_key"]
            if code in selected_codes:
                continue
            if fam_count.get(fam, 0) >= max_per_family:
                continue
            picked.append(row)
            selected_codes.add(code)
            fam_count[fam] = fam_count.get(fam, 0) + 1
            if len(picked) >= n:
                break
        if picked:
            selected_parts.append(pd.DataFrame(picked))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if len(selected) < target_size:
        remaining = d[~d["ts_code"].isin(selected_codes)].sort_values("selection_score", ascending=False)
        fam_count_global = selected["family_key"].value_counts().to_dict() if not selected.empty else {}
        add_rows = []
        for _, row in remaining.iterrows():
            fam = row["family_key"]
            if fam_count_global.get(fam, 0) >= max_per_family:
                continue
            add_rows.append(row)
            selected_codes.add(row["ts_code"])
            fam_count_global[fam] = fam_count_global.get(fam, 0) + 1
            if len(selected) + len(add_rows) >= target_size:
                break
        if add_rows:
            selected = pd.concat([selected, pd.DataFrame(add_rows)], ignore_index=True)
    selected = selected.sort_values(["t0_category", "selection_score"], ascending=[True, False]).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    return selected


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("T+0 ETF universe selector")
    print("=" * 100)
    print(f"out_dir          : {out_dir.resolve()}")
    print(f"target_size      : {args.target_size}")
    print(f"liquidity window : {args.liquidity_start} -> {args.liquidity_end}")
    print(f"base_url         : {args.base_url}")
    print("=" * 100)
    pro = init_tushare(args.token, args.base_url)
    basic = fetch_fund_basic(pro, args.sleep_sec)
    basic["name"] = basic["name"].fillna("")
    basic["fund_type"] = basic["fund_type"].fillna("")
    basic["family_key"] = basic["name"].map(family_key)
    cls = basic.apply(lambda r: classify_t0_category(r["name"], r.get("fund_type", "")), axis=1)
    basic["t0_category"] = [x[0] for x in cls]
    basic["t0_category_cn"] = [x[1] for x in cls]
    basic["t0_priority"] = [x[2] for x in cls]
    basic["list_date_num"] = pd.to_numeric(basic["list_date"], errors="coerce")
    start_num = int(args.liquidity_start)
    candidates = basic[
        (basic["t0_category"] != "exclude")
        & (basic["list_date_num"].isna() | (basic["list_date_num"] <= start_num))
        & (basic["delist_date"].isna() | (basic["delist_date"].astype(str).str.len() == 0))
    ].copy()
    basic.to_csv(out_dir / "t0_etf_basic_classified.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out_dir / "t0_etf_candidates_pre_liquidity.csv", index=False, encoding="utf-8-sig")
    print(f"ETF basic rows      : {len(basic)}")
    print(f"T+0 candidate rows  : {len(candidates)}")
    if not candidates.empty:
        print(candidates["t0_category_cn"].value_counts(dropna=False).to_string())
    if args.dry_run:
        print("[DRY RUN] stop before liquidity fetching.")
        return 0
    candidates = add_liquidity(pro, candidates, args)
    scored = score_candidates(candidates)
    scored.to_csv(out_dir / "t0_etf_candidates_scored.csv", index=False, encoding="utf-8-sig")
    selected = select_final(scored, args.target_size, args.max_per_family, args.min_active_days, args.min_median_amount_mil)
    selected[["ts_code"]].to_csv(out_dir / "t0_etf_codes.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "t0_etf_selected_30_detail.csv", index=False, encoding="utf-8-sig")
    summary = selected.groupby(["t0_category", "t0_category_cn"], as_index=False).agg(
        n=("ts_code", "count"),
        avg_amount_mil=("avg_amount_mil", "mean"),
        median_amount_mil=("median_amount_mil", "median"),
        active_days=("active_days", "mean"),
        avg_abs_ret=("abs_ret_mean", "mean"),
    )
    summary.to_csv(out_dir / "t0_etf_selected_category_summary.csv", index=False, encoding="utf-8-sig")
    print("\n选择结果：")
    print(summary.to_string(index=False))
    print("\n输出文件：")
    print(f"  1. {out_dir / 't0_etf_selected_30_detail.csv'}")
    print(f"  2. {out_dir / 't0_etf_codes.csv'}")
    print(f"  3. {out_dir / 't0_etf_selected_category_summary.csv'}")
    print(f"  4. {out_dir / 't0_etf_candidates_scored.csv'}")
    print("\n请人工复核 t0_etf_selected_30_detail.csv：")
    print("  - 是否有非 T+0 股票型 ETF 误入；")
    print("  - 是否有过多同质跨境 ETF；")
    print("  - 货币 ETF 是否只作为现金/对照，不作为主要 alpha 标的。")
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
