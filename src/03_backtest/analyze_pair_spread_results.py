#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_pair_spread_results.py

用途：
    对 T+0 ETF 配对价差回归策略结果做 pair-level 归因、成本存活分析和可选滑点/容量诊断。

核心回答三个问题：
    1. 当前策略是不是被固定 bp 成本打穿？
    2. 哪些 pair 有真实毛收益边际，哪些只是噪声？
    3. 在给定资金规模下，交易相对于 1min 成交额是否过大，是否需要动态滑点/容量限制？

输入：
    data_t0/backtest_pair_spread/pair_spread_trades.csv
    data_t0/backtest_pair_spread/pair_spread_summary_by_cost.csv
    可选：data_t0/processed/t0_intraday_bar_panel.parquet

输出：
    data_t0/analysis_pair_spread/
      ├─ pair_level_summary.csv
      ├─ pair_group_summary.csv
      ├─ pair_cost_survival.csv
      ├─ exit_reason_summary.csv
      ├─ monthly_summary.csv
      ├─ trade_cost_sensitivity_expanded.csv
      ├─ slippage_capacity_diagnostics.csv  # 如果提供 panel-file
      ├─ pair_recommendation.csv
      └─ pair_analysis_report.md

运行示例：
    python ".\\analyze_pair_spread_results.py" `
      --trades-file ".\\data_t0\\backtest_pair_spread\\pair_spread_trades.csv" `
      --summary-file ".\\data_t0\\backtest_pair_spread\\pair_spread_summary_by_cost.csv" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\analysis_pair_spread"

如果只做结果归因，不做滑点/容量诊断：
    python ".\\analyze_pair_spread_results.py" `
      --trades-file ".\\data_t0\\backtest_pair_spread\\pair_spread_trades.csv" `
      --out-dir ".\\data_t0\\analysis_pair_spread"
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
    p = argparse.ArgumentParser(description="Analyze pair spread strategy results.")
    p.add_argument("--trades-file", type=str, default="./data_t0/backtest_pair_spread/pair_spread_trades.csv")
    p.add_argument("--summary-file", type=str, default="./data_t0/backtest_pair_spread/pair_spread_summary_by_cost.csv")
    p.add_argument("--panel-file", type=str, default="", help="可选：1min panel parquet，用于滑点/容量诊断。")
    p.add_argument("--out-dir", type=str, default="./data_t0/analysis_pair_spread")

    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10,20", help="单边成本 bp，逗号分隔。")
    p.add_argument("--position-weight", type=float, default=0.20, help="单笔组合仓位权重。")
    p.add_argument("--portfolio-capital", type=float, default=1_000_000.0, help="用于容量诊断的假设资金规模，单位元。")

    # 滑点模型参数。注意：这是诊断用 proxy，不是真实盘口模型。
    p.add_argument("--amount-multiplier", type=float, default=1.0, help="将 panel amount 转成元的乘数。若 amount 单位为千元，可设 1000。")
    p.add_argument("--base-oneway-cost-bp", type=float, default=1.0, help="动态成本模型基础单边成本 bp。")
    p.add_argument("--spread-proxy-bp", type=float, default=2.0, help="买卖价差代理 bp。")
    p.add_argument("--impact-bp-per-1pct-participation", type=float, default=1.0, help="参与率每增加 1% 增加多少 bp 冲击。")
    p.add_argument("--open-tail-penalty-bp", type=float, default=2.0, help="开盘/尾盘惩罚 bp。")
    return p.parse_args()


def read_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trades-file 不存在：{path}")

    t = pd.read_csv(path)
    if t.empty:
        raise ValueError("trades-file 为空。")

    for c in ["entry_time", "exit_time", "signal_time"]:
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")

    if "trade_date" not in t.columns:
        t["trade_date"] = t["entry_time"].dt.strftime("%Y%m%d")
    t["trade_date"] = t["trade_date"].astype(str)

    for c in ["gross_ret", "holding_bars", "z_signal", "z_exit"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    for c in ["pair_id", "pair_group", "exit_reason", "mode", "long_code", "short_code"]:
        if c not in t.columns:
            t[c] = ""

    return t


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def summarize_by(trades: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        gross = pd.to_numeric(g["gross_ret"], errors="coerce")
        n = len(g)
        out = {
            "trade_count": n,
            "avg_gross_ret": gross.mean(),
            "median_gross_ret": gross.median(),
            "sum_gross_ret": gross.sum(),
            "gross_win_rate": (gross > 0).mean(),
            "gross_profit_factor": profit_factor(gross),
            "gross_ret_std": gross.std(ddof=1),
            "avg_holding_bars": g["holding_bars"].mean() if "holding_bars" in g.columns else np.nan,
            "spread_revert_rate": (g["exit_reason"] == "spread_revert").mean(),
            "stop_loss_rate": (g["exit_reason"] == "stop_loss").mean(),
            "take_profit_rate": (g["exit_reason"] == "take_profit").mean(),
            "max_hold_rate": (g["exit_reason"] == "max_hold").mean(),
            "force_flat_rate": (g["exit_reason"] == "force_flat").mean(),
            "avg_abs_z_signal": g["z_signal"].abs().mean() if "z_signal" in g.columns else np.nan,
            "avg_abs_z_exit": g["z_exit"].abs().mean() if "z_exit" in g.columns else np.nan,
            # 单边 breakeven 成本：gross - 2*cost = 0 => cost_bp = avg_gross_ret*10000/2
            "breakeven_oneway_cost_bp_avg": gross.mean() * 10000.0 / 2.0,
            "breakeven_oneway_cost_bp_median": gross.median() * 10000.0 / 2.0,
        }
        return pd.Series(out)

    res = trades.groupby(group_cols, dropna=False).apply(_agg, include_groups=False).reset_index()
    return res.sort_values(["breakeven_oneway_cost_bp_avg", "gross_profit_factor", "trade_count"], ascending=[False, False, False])


def cost_survival(trades: pd.DataFrame, cost_bps: List[float], group_col: str = "pair_id") -> pd.DataFrame:
    rows = []
    for key, g in trades.groupby(group_col, dropna=False):
        gross = pd.to_numeric(g["gross_ret"], errors="coerce")
        for c in cost_bps:
            net = gross - 2.0 * c / 10000.0
            rows.append({
                group_col: key,
                "cost_bp": c,
                "trade_count": len(g),
                "avg_net_ret": float(net.mean()),
                "median_net_ret": float(net.median()),
                "net_win_rate": float((net > 0).mean()),
                "net_profit_factor": profit_factor(net),
                "sum_net_ret": float(net.sum()),
                "survives_avg_net_positive": int(net.mean() > 0),
                "survives_pf_gt_1": int(profit_factor(net) > 1),
            })
    return pd.DataFrame(rows)


def expanded_trade_cost_sensitivity(trades: pd.DataFrame, cost_bps: List[float]) -> pd.DataFrame:
    parts = []
    for c in cost_bps:
        x = trades.copy()
        x["cost_bp"] = c
        x["net_trade_ret"] = x["gross_ret"] - 2.0 * c / 10000.0
        x["net_win"] = (x["net_trade_ret"] > 0).astype(int)
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def monthly_summary(trades: pd.DataFrame, cost_bps: List[float], position_weight: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["month"] = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m")
    rows = []
    for c in cost_bps:
        x = t.copy()
        x["net_trade_ret"] = x["gross_ret"] - 2.0 * c / 10000.0
        x["weighted_pnl"] = position_weight * x["net_trade_ret"]
        g = x.groupby("month")
        rows.append(g.agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            avg_net_ret=("net_trade_ret", "mean"),
            gross_win_rate=("gross_ret", lambda s: (s > 0).mean()),
            net_win_rate=("net_trade_ret", lambda s: (s > 0).mean()),
            weighted_pnl=("weighted_pnl", "sum"),
        ).reset_index().assign(cost_bp=c))
    return pd.concat(rows, ignore_index=True)


def exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    return (
        trades.groupby(["pair_id", "exit_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .assign(rate=lambda d: d["count"] / d.groupby("pair_id")["count"].transform("sum"))
        .sort_values(["pair_id", "count"], ascending=[True, False])
    )


def recommendation(pair_summary: pd.DataFrame, cost_surv: pd.DataFrame) -> pd.DataFrame:
    if pair_summary.empty:
        return pd.DataFrame()

    p = pair_summary.copy()
    # 取 1bp/2bp/5bp 是否能活
    pivot = cost_surv.pivot_table(
        index="pair_id",
        columns="cost_bp",
        values="avg_net_ret",
        aggfunc="first"
    )
    pivot.columns = [f"avg_net_ret_at_{c:g}bp" for c in pivot.columns]
    p = p.merge(pivot.reset_index(), on="pair_id", how="left")

    recs = []
    reasons = []
    for _, r in p.iterrows():
        n = r.get("trade_count", 0)
        avg_bp = r.get("avg_gross_ret", np.nan) * 10000
        be = r.get("breakeven_oneway_cost_bp_avg", np.nan)
        pf = r.get("gross_profit_factor", np.nan)
        sl = r.get("stop_loss_rate", np.nan)
        rev = r.get("spread_revert_rate", np.nan)

        reason = []
        if n < 50:
            rec = "drop_or_retest"
            reason.append("交易数过少")
        elif not np.isfinite(avg_bp) or avg_bp <= 0:
            rec = "drop"
            reason.append("0bp 毛收益均值不为正")
        elif be < 1.0:
            rec = "drop"
            reason.append("可承受单边成本低于1bp，边际过薄")
        elif be < 3.0:
            rec = "thin_edge_review"
            reason.append("有毛边际但成本承受能力弱")
        elif pf < 1.10:
            rec = "review"
            reason.append("gross profit factor 偏低")
        elif sl > 0.35:
            rec = "review"
            reason.append("止损率偏高")
        elif rev < 0.40:
            rec = "review"
            reason.append("价差回归退出占比偏低")
        else:
            rec = "keep_for_next_stage"
            reason.append("毛边际和回归结构尚可")

        recs.append(rec)
        reasons.append("; ".join(reason))

    p["recommendation"] = recs
    p["reason"] = reasons
    rank_cols = ["recommendation", "breakeven_oneway_cost_bp_avg", "gross_profit_factor", "trade_count"]
    return p.sort_values(rank_cols, ascending=[True, False, False, False])


def load_panel_for_slippage(panel_file: Path, trades: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not panel_file or str(panel_file) == "." or not panel_file.exists():
        return None

    codes = sorted(set(trades["long_code"].dropna().astype(str)) | set(trades["short_code"].dropna().astype(str)))
    codes = [c for c in codes if c and c.lower() != "nan"]

    needed = ["ts_code", "trade_time", "open", "close", "amount", "clock_time"]
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(panel_file).names
    cols = [c for c in needed if c in schema_cols]
    panel = pd.read_parquet(panel_file, columns=cols)
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel = panel[panel["ts_code"].isin(codes)].copy()
    panel["trade_time"] = pd.to_datetime(panel["trade_time"], errors="coerce")
    for c in ["open", "close", "amount"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    return panel


def add_slippage_capacity_diagnostics(
    trades: pd.DataFrame,
    panel: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if panel is None or panel.empty or trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])

    # long leg entry / exit amount
    entry_panel = panel.rename(columns={
        "ts_code": "long_code",
        "trade_time": "entry_time",
        "amount": "entry_amount_raw",
        "clock_time": "entry_clock_time",
    })[["long_code", "entry_time", "entry_amount_raw", "entry_clock_time"]]

    exit_panel = panel.rename(columns={
        "ts_code": "long_code",
        "trade_time": "exit_time",
        "amount": "exit_amount_raw",
        "clock_time": "exit_clock_time",
    })[["long_code", "exit_time", "exit_amount_raw", "exit_clock_time"]]

    t = t.merge(entry_panel, on=["long_code", "entry_time"], how="left")
    t = t.merge(exit_panel, on=["long_code", "exit_time"], how="left")

    leg_notional = args.portfolio_capital * args.position_weight
    if "mode" in t.columns:
        # long_short 里每条腿按 0.5 gross exposure
        ls = t["mode"].astype(str).str.lower().eq("long_short")
        t["assumed_leg_notional"] = leg_notional
        t.loc[ls, "assumed_leg_notional"] = leg_notional * 0.5
    else:
        t["assumed_leg_notional"] = leg_notional

    t["entry_amount_yuan"] = t["entry_amount_raw"] * args.amount_multiplier
    t["exit_amount_yuan"] = t["exit_amount_raw"] * args.amount_multiplier

    t["entry_participation"] = t["assumed_leg_notional"] / t["entry_amount_yuan"].replace(0, np.nan)
    t["exit_participation"] = t["assumed_leg_notional"] / t["exit_amount_yuan"].replace(0, np.nan)
    t["max_participation"] = t[["entry_participation", "exit_participation"]].max(axis=1)

    # 动态单边成本 proxy：基础 + spread代理 + 冲击 + 开尾盘惩罚
    def time_penalty(clock):
        if pd.isna(clock):
            return 0.0
        s = str(clock)
        if ("09:30" <= s < "09:45") or ("14:30" <= s <= "15:00"):
            return args.open_tail_penalty_bp
        return 0.0

    t["entry_time_penalty_bp"] = t["entry_clock_time"].map(time_penalty)
    t["exit_time_penalty_bp"] = t["exit_clock_time"].map(time_penalty)
    t["entry_impact_bp"] = args.impact_bp_per_1pct_participation * (t["entry_participation"] * 100.0)
    t["exit_impact_bp"] = args.impact_bp_per_1pct_participation * (t["exit_participation"] * 100.0)

    t["estimated_entry_oneway_cost_bp"] = (
        args.base_oneway_cost_bp
        + args.spread_proxy_bp
        + t["entry_impact_bp"].clip(lower=0)
        + t["entry_time_penalty_bp"]
    )
    t["estimated_exit_oneway_cost_bp"] = (
        args.base_oneway_cost_bp
        + args.spread_proxy_bp
        + t["exit_impact_bp"].clip(lower=0)
        + t["exit_time_penalty_bp"]
    )
    t["estimated_roundtrip_cost_bp"] = t["estimated_entry_oneway_cost_bp"] + t["estimated_exit_oneway_cost_bp"]
    t["net_ret_dynamic_cost_proxy"] = t["gross_ret"] - t["estimated_roundtrip_cost_bp"] / 10000.0

    return t


def slippage_summary(slip: pd.DataFrame) -> pd.DataFrame:
    if slip.empty:
        return pd.DataFrame()
    def q(s, p):
        return s.quantile(p)
    return slip.groupby("pair_id").agg(
        trade_count=("gross_ret", "size"),
        median_entry_participation=("entry_participation", "median"),
        p90_entry_participation=("entry_participation", lambda s: q(s.dropna(), 0.90) if s.notna().any() else np.nan),
        p95_entry_participation=("entry_participation", lambda s: q(s.dropna(), 0.95) if s.notna().any() else np.nan),
        median_roundtrip_cost_bp=("estimated_roundtrip_cost_bp", "median"),
        p90_roundtrip_cost_bp=("estimated_roundtrip_cost_bp", lambda s: q(s.dropna(), 0.90) if s.notna().any() else np.nan),
        avg_dynamic_net_ret=("net_ret_dynamic_cost_proxy", "mean"),
        dynamic_net_win_rate=("net_ret_dynamic_cost_proxy", lambda s: (s > 0).mean()),
    ).reset_index().sort_values("avg_dynamic_net_ret", ascending=False)


def write_report(
    out_dir: Path,
    trades: pd.DataFrame,
    summary_file_df: Optional[pd.DataFrame],
    pair_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    cost_surv: pd.DataFrame,
    rec: pd.DataFrame,
    slip_diag: pd.DataFrame,
    slip_sum: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# 配对价差回归结果诊断报告\n")

    lines.append("## 1. 总体判断\n")
    n = len(trades)
    avg_gross = trades["gross_ret"].mean()
    be = avg_gross * 10000.0 / 2.0
    lines.append(f"- 交易数：**{n:,}**")
    lines.append(f"- 平均单笔毛收益：**{avg_gross * 10000:.3f} bp**")
    lines.append(f"- 由平均毛收益推导的理论单边 breakeven 成本：**{be:.3f} bp**")
    lines.append("")
    if be < 1:
        lines.append("> 结论：整体边际极薄，普通固定 5bp/10bp 成本会直接打穿策略。")
    elif be < 3:
        lines.append("> 结论：整体存在毛边际，但只适合极低成本环境；需要严格筛 pair 和动态成本过滤。")
    else:
        lines.append("> 结论：整体毛边际尚可，但仍需样本外验证和动态滑点分析。")
    lines.append("")

    if summary_file_df is not None and not summary_file_df.empty:
        lines.append("## 2. 原始成本敏感性摘要\n")
        lines.append(summary_file_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 3. Pair-level 毛收益诊断\n")
    show_cols = [
        "pair_id", "trade_count", "avg_gross_ret", "gross_win_rate", "gross_profit_factor",
        "spread_revert_rate", "stop_loss_rate", "avg_holding_bars",
        "breakeven_oneway_cost_bp_avg"
    ]
    show_cols = [c for c in show_cols if c in pair_summary.columns]
    lines.append(pair_summary[show_cols].head(20).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Group-level 诊断\n")
    gcols = [
        "pair_group", "trade_count", "avg_gross_ret", "gross_win_rate", "gross_profit_factor",
        "spread_revert_rate", "stop_loss_rate", "breakeven_oneway_cost_bp_avg"
    ]
    gcols = [c for c in gcols if c in group_summary.columns]
    lines.append(group_summary[gcols].to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Pair 建议\n")
    rcols = [
        "pair_id", "pair_group", "trade_count", "avg_gross_ret", "gross_profit_factor",
        "breakeven_oneway_cost_bp_avg", "recommendation", "reason"
    ]
    rcols = [c for c in rcols if c in rec.columns]
    lines.append(rec[rcols].to_markdown(index=False))
    lines.append("")

    if not slip_sum.empty:
        lines.append("## 6. 滑点/容量 proxy 诊断\n")
        lines.append(f"- 假设资金规模：{args.portfolio_capital:,.0f} 元")
        lines.append(f"- 单笔仓位：{args.position_weight:.2%}")
        lines.append(f"- amount multiplier：{args.amount_multiplier}")
        lines.append("")
        lines.append(slip_sum.to_markdown(index=False))
        lines.append("")
        lines.append("注意：该动态成本只是 proxy，不是盘口级真实滑点。真实商用还需要买一卖一、盘口深度、订单簿或至少更可靠的成交额单位确认。")
        lines.append("")

    lines.append("## 7. 下一步建议\n")
    lines.append("1. 不要只用固定 5/10/20bp 粗暴判断；先看每个 pair 的 breakeven 单边成本。")
    lines.append("2. 如果某 pair 平均毛收益只能承受 1bp 以下单边成本，直接删除或仅作为做市环境研究。")
    lines.append("3. 对保留 pair 加入动态成本过滤：预计回归空间必须大于动态 roundtrip cost 的 1.5—2 倍。")
    lines.append("4. 再做样本外参数选择：开发期选 pair/阈值，测试期只验证。")
    lines.append("5. 若要商用，必须补完整分钟级权益曲线、容量限制和更真实的滑点模型。")

    (out_dir / "pair_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_file = Path(args.trades_file)
    summary_file = Path(args.summary_file)
    panel_file = Path(args.panel_file) if args.panel_file else None
    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]

    print("=" * 100)
    print("Analyze pair spread results")
    print("=" * 100)
    print(f"trades_file : {trades_file.resolve()}")
    print(f"summary_file: {summary_file.resolve() if summary_file.exists() else 'not found'}")
    print(f"panel_file  : {panel_file.resolve() if panel_file and panel_file.exists() else 'not used'}")
    print(f"out_dir     : {out_dir.resolve()}")
    print(f"cost_bps    : {cost_bps}")
    print("=" * 100)

    trades = read_trades(trades_file)

    summary_df = pd.read_csv(summary_file) if summary_file.exists() else None

    pair_summary = summarize_by(trades, ["pair_id", "pair_group"])
    group_summary = summarize_by(trades, ["pair_group"])
    cost_surv = cost_survival(trades, cost_bps, "pair_id")
    expanded = expanded_trade_cost_sensitivity(trades, cost_bps)
    month = monthly_summary(trades, cost_bps, args.position_weight)
    exit_sum = exit_reason_summary(trades)
    rec = recommendation(pair_summary, cost_surv)

    # 滑点/容量诊断
    slip_diag = pd.DataFrame()
    slip_sum = pd.DataFrame()
    if panel_file and panel_file.exists():
        panel = load_panel_for_slippage(panel_file, trades)
        slip_diag = add_slippage_capacity_diagnostics(trades, panel, args)
        if not slip_diag.empty:
            slip_sum = slippage_summary(slip_diag)

    pair_summary.to_csv(out_dir / "pair_level_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "pair_group_summary.csv", index=False, encoding="utf-8-sig")
    cost_surv.to_csv(out_dir / "pair_cost_survival.csv", index=False, encoding="utf-8-sig")
    expanded.to_csv(out_dir / "trade_cost_sensitivity_expanded.csv", index=False, encoding="utf-8-sig")
    month.to_csv(out_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    exit_sum.to_csv(out_dir / "exit_reason_summary.csv", index=False, encoding="utf-8-sig")
    rec.to_csv(out_dir / "pair_recommendation.csv", index=False, encoding="utf-8-sig")

    if not slip_diag.empty:
        slip_diag.to_csv(out_dir / "slippage_capacity_diagnostics.csv", index=False, encoding="utf-8-sig")
        slip_sum.to_csv(out_dir / "slippage_capacity_summary_by_pair.csv", index=False, encoding="utf-8-sig")

    write_report(
        out_dir=out_dir,
        trades=trades,
        summary_file_df=summary_df,
        pair_summary=pair_summary,
        group_summary=group_summary,
        cost_surv=cost_surv,
        rec=rec,
        slip_diag=slip_diag,
        slip_sum=slip_sum,
        args=args,
    )

    print("\n" + "=" * 100)
    print("Finished")
    print(f"pair summary     : {(out_dir / 'pair_level_summary.csv').resolve()}")
    print(f"recommendation   : {(out_dir / 'pair_recommendation.csv').resolve()}")
    print(f"cost survival    : {(out_dir / 'pair_cost_survival.csv').resolve()}")
    if not slip_diag.empty:
        print(f"slippage diag    : {(out_dir / 'slippage_capacity_diagnostics.csv').resolve()}")
    print(f"report           : {(out_dir / 'pair_analysis_report.md').resolve()}")
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
