#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_vwap_mean_reversion.py

用途：
    在 T+0 ETF 1min bar 面板上回测 VWAP 均值回归日内交易策略。

策略逻辑：
    1. 每分钟检查当前 ETF 是否明显低于日内 VWAP；
    2. 若短期出现下跌，但没有继续放量/破位杀跌，则在下一根 bar 开多；
    3. 开仓后按止盈、止损、回到 VWAP、最长持仓、收盘前强平退出；
    4. 不隔夜，所有仓位日内完成；
    5. 默认最多同时持有 3 只 ETF，单笔仓位 20%。

输入：
    data_t0/processed/t0_intraday_bar_panel.parquet

输出：
    data_t0/backtest_vwap_mr/
      ├─ vwap_mr_trades.csv
      ├─ vwap_mr_daily_nav_by_cost.csv
      ├─ vwap_mr_summary_by_cost.csv
      ├─ vwap_mr_param_grid_summary.csv  # 如果 --run-grid
      └─ vwap_mr_report.md

运行示例：
    python ".\\backtest_vwap_mean_reversion.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_vwap_mr"

参数网格：
    python ".\\backtest_vwap_mean_reversion.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_vwap_mr_grid" `
      --run-grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    z_entry: float = -1.5
    ret5m_entry: float = -0.0010
    amount_z_max: float = 2.5
    take_profit: float = 0.0030
    stop_loss: float = 0.0020
    max_hold_bars: int = 30
    vwap_exit_band: float = 0.0001
    max_positions: int = 3
    position_weight: float = 0.20
    cooldown_bars: int = 3
    avoid_breakdown_20m: bool = True
    exclude_low_liquidity_day: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest VWAP mean reversion strategy on T+0 ETF 1min panel.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_vwap_mr")

    p.add_argument("--z-entry", type=float, default=-1.5, help="price_to_vwap_z 开仓阈值，越负越极端。")
    p.add_argument("--ret5m-entry", type=float, default=-0.0010, help="过去5分钟收益需低于该值。")
    p.add_argument("--amount-z-max", type=float, default=2.5, help="避免极端放量下跌。")
    p.add_argument("--take-profit", type=float, default=0.0030, help="止盈比例。")
    p.add_argument("--stop-loss", type=float, default=0.0020, help="止损比例。")
    p.add_argument("--max-hold-bars", type=int, default=30, help="最长持仓 bar 数，1min 下 30=30分钟。")
    p.add_argument("--vwap-exit-band", type=float, default=0.0001, help="接近 VWAP 平仓阈值。")
    p.add_argument("--max-positions", type=int, default=3, help="最多同时持仓数。")
    p.add_argument("--position-weight", type=float, default=0.20, help="单笔仓位。")
    p.add_argument("--cooldown-bars", type=int, default=3, help="同一 ETF 平仓后冷却 bar 数。")
    p.add_argument("--allow-breakdown", action="store_true", help="允许跌破过去20分钟低点仍做均值回归。默认不允许。")
    p.add_argument("--exclude-low-liquidity-day", action="store_true", help="排除 daily_is_low_liquidity_day=1 的交易日。")

    p.add_argument("--cost-bps", type=str, default="0,5,10,20", help="成本敏感性，逗号分隔，单位 bp，单边成本。")
    p.add_argument("--run-grid", action="store_true", help="运行小参数网格。")
    return p.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "intraday_vwap", "price_to_vwap", "price_to_vwap_z",
        "ret_5m", "amount_z_20m", "realized_vol_20m",
        "breakdown_low_20m",
        "valid_entry_bar", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]

    # parquet 列读取容错
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
    d["trade_date"] = d["trade_date"].astype(str)
    d["ts_code"] = d["ts_code"].astype(str)
    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    # 必要数值列
    for c in ["open", "high", "low", "close", "amount", "intraday_vwap", "price_to_vwap", "price_to_vwap_z", "ret_5m", "amount_z_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day", "breakdown_low_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    return d


def make_param_list(args: argparse.Namespace) -> List[StrategyParams]:
    if not args.run_grid:
        return [StrategyParams(
            z_entry=args.z_entry,
            ret5m_entry=args.ret5m_entry,
            amount_z_max=args.amount_z_max,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            max_hold_bars=args.max_hold_bars,
            vwap_exit_band=args.vwap_exit_band,
            max_positions=args.max_positions,
            position_weight=args.position_weight,
            cooldown_bars=args.cooldown_bars,
            avoid_breakdown_20m=not args.allow_breakdown,
            exclude_low_liquidity_day=args.exclude_low_liquidity_day,
        )]

    grid = {
        "z_entry": [-1.0, -1.5, -2.0],
        "ret5m_entry": [-0.0005, -0.0010, -0.0020],
        "take_profit": [0.0020, 0.0030, 0.0050],
        "stop_loss": [0.0015, 0.0020, 0.0030],
        "max_hold_bars": [15, 30],
    }
    params = []
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, values))
        params.append(StrategyParams(
            **kw,
            amount_z_max=args.amount_z_max,
            vwap_exit_band=args.vwap_exit_band,
            max_positions=args.max_positions,
            position_weight=args.position_weight,
            cooldown_bars=args.cooldown_bars,
            avoid_breakdown_20m=not args.allow_breakdown,
            exclude_low_liquidity_day=args.exclude_low_liquidity_day,
        ))
    return params


def signal_mask(d: pd.DataFrame, p: StrategyParams) -> pd.Series:
    m = (
        (d["valid_entry_bar"] == 1)
        & (d["amount"].fillna(0) > 0)
        & d["open"].notna()
        & d["close"].notna()
        & d["intraday_vwap"].notna()
        & d["price_to_vwap_z"].notna()
        & d["ret_5m"].notna()
        & (d["price_to_vwap_z"] <= p.z_entry)
        & (d["price_to_vwap"] < 0)
        & (d["ret_5m"] <= p.ret5m_entry)
        & (d["amount_z_20m"].fillna(0) <= p.amount_z_max)
    )
    if p.avoid_breakdown_20m and "breakdown_low_20m" in d.columns:
        m &= (d["breakdown_low_20m"].fillna(0).astype(int) == 0)
    if p.exclude_low_liquidity_day and "daily_is_low_liquidity_day" in d.columns:
        m &= (d["daily_is_low_liquidity_day"].fillna(0).astype(int) == 0)
    return m


def simulate_trade_from_entry(day: pd.DataFrame, entry_pos: int, signal_pos: int, p: StrategyParams) -> Dict:
    """
    entry_pos 是实际成交 bar 位置，使用该 bar open 成交。
    从 entry_pos 到 max_hold 范围内检查退出。
    同一 bar 同时触发止盈止损时保守按止损。
    """
    n = len(day)
    entry_row = day.iloc[entry_pos]
    signal_row = day.iloc[signal_pos]

    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    max_exit_pos = min(n - 1, entry_pos + p.max_hold_bars)
    tp_price = entry_price * (1.0 + p.take_profit)
    sl_price = entry_price * (1.0 - p.stop_loss)

    exit_pos = max_exit_pos
    exit_price = float(day.iloc[exit_pos]["close"])
    exit_reason = "max_hold"

    for j in range(entry_pos, max_exit_pos + 1):
        row = day.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        vwap = float(row["intraday_vwap"]) if np.isfinite(row["intraday_vwap"]) else np.nan

        if int(row.get("force_flat_bar", 0)) == 1:
            exit_pos = j
            exit_price = close
            exit_reason = "force_flat"
            break

        # 保守：同一 bar 同时碰到止盈止损，先算止损。
        if np.isfinite(low) and low <= sl_price:
            exit_pos = j
            exit_price = sl_price
            exit_reason = "stop_loss"
            break

        if np.isfinite(high) and high >= tp_price:
            exit_pos = j
            exit_price = tp_price
            exit_reason = "take_profit"
            break

        # 回到 VWAP 附近
        if np.isfinite(vwap) and close >= vwap * (1.0 - p.vwap_exit_band):
            exit_pos = j
            exit_price = close
            exit_reason = "vwap_revert"
            break

    if not np.isfinite(exit_price) or exit_price <= 0:
        return {}

    exit_row = day.iloc[exit_pos]
    gross_ret = exit_price / entry_price - 1.0

    return {
        "ts_code": signal_row["ts_code"],
        "name": signal_row.get("name", ""),
        "t0_category": signal_row.get("t0_category", ""),
        "t0_category_cn": signal_row.get("t0_category_cn", ""),
        "trade_date": signal_row["trade_date"],
        "signal_time": signal_row["trade_time"],
        "entry_time": entry_row["trade_time"],
        "exit_time": exit_row["trade_time"],
        "signal_close": float(signal_row["close"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "signal_price_to_vwap": float(signal_row["price_to_vwap"]),
        "signal_price_to_vwap_z": float(signal_row["price_to_vwap_z"]),
        "signal_ret_5m": float(signal_row["ret_5m"]),
        "signal_amount_z_20m": float(signal_row["amount_z_20m"]) if np.isfinite(signal_row["amount_z_20m"]) else np.nan,
        "param_z_entry": p.z_entry,
        "param_ret5m_entry": p.ret5m_entry,
        "param_take_profit": p.take_profit,
        "param_stop_loss": p.stop_loss,
        "param_max_hold_bars": p.max_hold_bars,
    }


def generate_candidate_trades(d: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    candidates = []
    sig = signal_mask(d, p)
    d = d.copy()
    d["_signal"] = sig.astype(int)

    total_days = d.groupby(["ts_code", "trade_date"]).ngroups
    k = 0

    for (code, date), day in d.groupby(["ts_code", "trade_date"], sort=False):
        k += 1
        if k % 500 == 0:
            print(f"    simulated ETF-days: {k}/{total_days}")

        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)

        for sp in sig_pos:
            ep = sp + 1
            if ep >= len(day):
                continue
            # 下一根 bar 如果已强平或不能成交，跳过
            if int(day.iloc[ep].get("force_flat_bar", 0)) == 1:
                continue
            trade = simulate_trade_from_entry(day, ep, sp, p)
            if trade:
                candidates.append(trade)

    return pd.DataFrame(candidates)


def apply_portfolio_constraints(candidates: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    c = candidates.sort_values(["entry_time", "signal_price_to_vwap_z"]).reset_index(drop=True).copy()
    open_positions: List[Dict] = []
    cooldown_until: Dict[str, pd.Timestamp] = {}
    accepted = []

    for _, row in c.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        exit_time = pd.Timestamp(row["exit_time"])
        code = row["ts_code"]

        # 移除已结束持仓
        open_positions = [pos for pos in open_positions if pos["exit_time"] > entry_time]

        # 同一 ETF 冷却
        if code in cooldown_until and entry_time < cooldown_until[code]:
            continue

        # 同一 ETF 不重复持仓
        if any(pos["ts_code"] == code for pos in open_positions):
            continue

        # 最大同时持仓
        if len(open_positions) >= p.max_positions:
            continue

        accepted.append(row.to_dict())
        open_positions.append({"ts_code": code, "exit_time": exit_time})

        # 冷却到 exit 后若干分钟
        cooldown_until[code] = exit_time + pd.Timedelta(minutes=p.cooldown_bars)

    out = pd.DataFrame(accepted)
    if out.empty:
        return out

    out = out.sort_values(["entry_time", "ts_code"]).reset_index(drop=True)
    return out


def performance_summary(trades: pd.DataFrame, cost_bp: float, p: StrategyParams, all_dates: List[str]) -> Dict:
    one_way = cost_bp / 10000.0
    roundtrip = 2.0 * one_way
    if trades.empty:
        daily = pd.DataFrame({"trade_date": all_dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_trade_ret"] = t["gross_ret"] - roundtrip
        t["portfolio_pnl"] = p.position_weight * t["net_trade_ret"]
        daily = t.groupby("trade_date", as_index=False)["portfolio_pnl"].sum().rename(columns={"portfolio_pnl": "daily_ret"})
        daily = pd.DataFrame({"trade_date": all_dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = (1.0 + daily["daily_ret"]).cumprod()
    n_days = len(daily)
    nav_final = float(daily["nav"].iloc[-1]) if n_days else 1.0
    ann_ret = nav_final ** (252.0 / n_days) - 1.0 if n_days and nav_final > 0 else np.nan
    ann_vol = float(daily["daily_ret"].std(ddof=1) * np.sqrt(252)) if n_days > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
    roll_max = daily["nav"].cummax()
    dd = daily["nav"] / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    if trades.empty:
        win_rate = np.nan
        avg_gross = np.nan
        avg_net = np.nan
        profit_factor = np.nan
        avg_holding = np.nan
        exit_counts = {}
    else:
        net_trade = trades["gross_ret"] - roundtrip
        win_rate = float((net_trade > 0).mean())
        avg_gross = float(trades["gross_ret"].mean())
        avg_net = float(net_trade.mean())
        gains = net_trade[net_trade > 0].sum()
        losses = -net_trade[net_trade < 0].sum()
        profit_factor = float(gains / losses) if losses > 0 else np.inf
        avg_holding = float(trades["holding_bars"].mean())
        exit_counts = trades["exit_reason"].value_counts().to_dict()

    return {
        "cost_bp": cost_bp,
        "trade_count": int(len(trades)),
        "trading_days": int(n_days),
        "avg_trades_per_day": float(len(trades) / n_days) if n_days else np.nan,
        "final_nav": nav_final,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_gross_trade_ret": avg_gross,
        "avg_net_trade_ret": avg_net,
        "profit_factor": profit_factor,
        "avg_holding_bars": avg_holding,
        "exit_reason_counts": json.dumps(exit_counts, ensure_ascii=False),
    }, daily


def run_one(d: pd.DataFrame, p: StrategyParams, cost_bps: List[float], all_dates: List[str]) -> Dict:
    print(f"\nRunning params: {asdict(p)}")
    cand = generate_candidate_trades(d, p)
    print(f"  candidate trades: {len(cand):,}")
    trades = apply_portfolio_constraints(cand, p)
    print(f"  accepted trades : {len(trades):,}")

    summaries = []
    nav_parts = []
    for cost in cost_bps:
        s, daily = performance_summary(trades, cost, p, all_dates)
        summaries.append(s)
        daily["cost_bp"] = cost
        nav_parts.append(daily)

    return {
        "params": p,
        "candidate_trades": cand,
        "trades": trades,
        "summary": pd.DataFrame(summaries),
        "nav": pd.concat(nav_parts, ignore_index=True),
    }


def write_report(out_dir: Path, params: StrategyParams, summary: pd.DataFrame, trades: pd.DataFrame, run_grid: bool) -> None:
    lines = []
    lines.append("# VWAP 均值回归策略回测报告\n")
    lines.append("## 1. 策略逻辑\n")
    lines.append("当 ETF 价格显著低于当日 VWAP，且过去 5 分钟出现下跌但没有极端放量/破位时，下一根 1min bar 开多；随后按止盈、止损、回到 VWAP、最长持仓或收盘前强平退出。")
    lines.append("")
    lines.append("## 2. 主参数\n")
    lines.append(pd.DataFrame([asdict(params)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 成本敏感性结果\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not trades.empty:
        lines.append("## 4. 交易概况\n")
        lines.append(f"- 交易数：{len(trades):,}")
        lines.append(f"- ETF 数量：{trades['ts_code'].nunique()}")
        lines.append(f"- 交易日数量：{trades['trade_date'].nunique()}")
        lines.append("")
        lines.append("退出原因：")
        lines.append(trades["exit_reason"].value_counts().to_markdown())
        lines.append("")
        lines.append("类别交易数：")
        lines.append(trades["t0_category_cn"].value_counts().to_markdown())
        lines.append("")
    if run_grid:
        lines.append("## 5. 参数网格\n")
        lines.append("本报告主表只展示默认参数。完整参数网格见 `vwap_mr_param_grid_summary.csv`。")
    (out_dir / "vwap_mr_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip() != ""]

    print("=" * 100)
    print("VWAP mean reversion backtest")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"cost_bps   : {cost_bps}")
    print(f"run_grid   : {args.run_grid}")
    print("=" * 100)

    d = load_panel(Path(args.panel_file))
    all_dates = sorted(d["trade_date"].unique().tolist())

    params_list = make_param_list(args)

    # 默认主参数先跑，保存明细。
    primary = params_list[0]
    result = run_one(d, primary, cost_bps, all_dates)

    result["trades"].to_csv(out_dir / "vwap_mr_trades.csv", index=False, encoding="utf-8-sig")
    result["summary"].to_csv(out_dir / "vwap_mr_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    result["nav"].to_csv(out_dir / "vwap_mr_daily_nav_by_cost.csv", index=False, encoding="utf-8-sig")

    grid_rows = []
    if args.run_grid:
        print("\n========== Parameter grid ==========")
        for idx, p in enumerate(params_list, 1):
            if idx == 1:
                # 已经跑过
                summ = result["summary"].copy()
            else:
                r = run_one(d, p, cost_bps, all_dates)
                summ = r["summary"].copy()
            row = asdict(p)
            # 主要按 10bp 取结果
            main_cost = 10.0 if 10.0 in cost_bps else cost_bps[0]
            s10 = summ[summ["cost_bp"] == main_cost].iloc[0].to_dict()
            for k, v in s10.items():
                row[f"metric_{k}"] = v
            grid_rows.append(row)

        grid = pd.DataFrame(grid_rows)
        grid = grid.sort_values(["metric_sharpe", "metric_ann_return"], ascending=False)
        grid.to_csv(out_dir / "vwap_mr_param_grid_summary.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, primary, result["summary"], result["trades"], args.run_grid)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"summary : {(out_dir / 'vwap_mr_summary_by_cost.csv').resolve()}")
    print(f"trades  : {(out_dir / 'vwap_mr_trades.csv').resolve()}")
    print(f"nav     : {(out_dir / 'vwap_mr_daily_nav_by_cost.csv').resolve()}")
    print(f"report  : {(out_dir / 'vwap_mr_report.md').resolve()}")
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
