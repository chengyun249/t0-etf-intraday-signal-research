#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_pair_spread_reversion.py

用途：
    在 T+0 ETF 1min bar 面板上回测“配对价差回归 / 相对价值轮换”策略。

策略思想：
    对经济含义相近的 ETF pair，例如：
        黄金ETF华安 vs 黄金ETF易方达
        标普500ETF博时 vs 标普500ETF国泰
        恒生科技ETF vs 港股科技ETF
    计算二者价格价差：
        spread_t = log(P_A,t) - beta * log(P_B,t)

    再计算滚动 z-score：
        z_t = (spread_t - rolling_mean(spread)) / rolling_std(spread)

    如果 z_t 显著偏离，并且开始向 0 回落，则认为相对价差可能回归：
        z_t < -entry_z 且 z_t - z_{t-1} > 0:
            A 相对 B 便宜
        z_t >  entry_z 且 z_t - z_{t-1} < 0:
            B 相对 A 便宜

    默认使用 long-only 轮换：
        买相对便宜腿，不做空昂贵腿。
    可选 --mode long_short：
        做多便宜腿，做空昂贵腿。注意：真实账户是否能做空 ETF 要另行确认。

输入：
    data_t0/processed/t0_intraday_bar_panel.parquet
    data_t0/config/t0_pair_config.csv

输出：
    data_t0/backtest_pair_spread/
      pair_spread_trades.csv
      pair_spread_summary_by_cost.csv
      pair_spread_daily_nav_by_cost.csv
      pair_spread_report.md

运行示例：
    python ".\\backtest_pair_spread_reversion.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --pair-file ".\\data_t0\\config\\t0_pair_config.csv" `
      --out-dir ".\\data_t0\\backtest_pair_spread"

可选 long-short：
    python ".\\backtest_pair_spread_reversion.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --pair-file ".\\data_t0\\config\\t0_pair_config.csv" `
      --out-dir ".\\data_t0\\backtest_pair_spread_ls" `
      --mode long_short
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class PairStrategyParams:
    mode: str = "long_only"              # long_only / long_short
    rolling_window: int = 60             # z-score 估计窗口，1min 下为 60 分钟
    min_rolling_obs: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5
    confirm_reversion: bool = True       # 偏离后开始回落再进
    min_abs_delta_z: float = 0.05        # z 的回落/变化幅度过滤
    take_profit: float = 0.0030
    stop_loss: float = 0.0020
    max_hold_bars: int = 30
    max_open_pairs: int = 3
    pair_daily_trade_limit: int = 3
    cooldown_minutes: int = 15
    position_weight: float = 0.20
    same_group_max_open: int = 1
    beta: float = 1.0                    # 第一版固定 beta=1；后续可做 rolling beta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest T+0 ETF pair spread reversion strategy.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--pair-file", type=str, default="./data_t0/config/t0_pair_config.csv")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_pair_spread")

    p.add_argument("--mode", type=str, default="long_only", choices=["long_only", "long_short"])
    p.add_argument("--rolling-window", type=int, default=60)
    p.add_argument("--min-rolling-obs", type=int, default=30)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--min-abs-delta-z", type=float, default=0.05)
    p.add_argument("--no-confirm-reversion", action="store_true", help="不要求 z 开始回落，偏离即开仓。")
    p.add_argument("--take-profit", type=float, default=0.0030)
    p.add_argument("--stop-loss", type=float, default=0.0020)
    p.add_argument("--max-hold-bars", type=int, default=30)
    p.add_argument("--max-open-pairs", type=int, default=3)
    p.add_argument("--pair-daily-trade-limit", type=int, default=3)
    p.add_argument("--cooldown-minutes", type=int, default=15)
    p.add_argument("--position-weight", type=float, default=0.20)
    p.add_argument("--same-group-max-open", type=int, default=1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--cost-bps", type=str, default="0,5,10,20", help="单边成本 bp，逗号分隔。")
    return p.parse_args()


def load_pair_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"pair-file 不存在：{path}")
    df = pd.read_csv(path)
    required = ["pair_id", "leg_a", "leg_b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pair-file 缺少列：{missing}")

    if "pair_group" not in df.columns:
        df["pair_group"] = "unknown"
    if "pair_name" not in df.columns:
        df["pair_name"] = df["pair_id"]

    for c in ["pair_id", "leg_a", "leg_b", "pair_group", "pair_name"]:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates("pair_id").reset_index(drop=True)
    return df


def load_panel(path: Path, codes: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "valid_entry_bar", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]

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

    d = d[d["ts_code"].astype(str).isin(codes)].copy()
    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str)
    d["ts_code"] = d["ts_code"].astype(str)

    for c in ["open", "high", "low", "close", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return d


def make_pair_panel(panel: pd.DataFrame, pair_row: pd.Series, params: PairStrategyParams) -> pd.DataFrame:
    a = pair_row["leg_a"]
    b = pair_row["leg_b"]
    pair_id = pair_row["pair_id"]

    pa = panel[panel["ts_code"] == a].copy()
    pb = panel[panel["ts_code"] == b].copy()

    if pa.empty or pb.empty:
        return pd.DataFrame()

    # 只保留需要列并加前缀
    base_cols = [
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "valid_entry_bar", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]
    pa = pa[base_cols].rename(columns={c: f"a_{c}" for c in base_cols if c not in ["trade_time", "trade_date", "clock_time"]})
    pb = pb[base_cols].rename(columns={c: f"b_{c}" for c in base_cols if c not in ["trade_time", "trade_date", "clock_time"]})

    m = pa.merge(pb, on=["trade_time", "trade_date", "clock_time"], how="inner")
    if m.empty:
        return m

    m["pair_id"] = pair_id
    m["leg_a"] = a
    m["leg_b"] = b
    m["pair_group"] = pair_row.get("pair_group", "unknown")
    m["pair_name"] = pair_row.get("pair_name", pair_id)

    # 价差：log(A) - beta log(B)
    m["spread"] = np.log(m["a_close"]) - params.beta * np.log(m["b_close"])

    # 每天内部 rolling，避免隔夜
    parts = []
    for date, g in m.groupby("trade_date", sort=False):
        x = g.sort_values("trade_time").copy()
        roll_mean = x["spread"].rolling(params.rolling_window, min_periods=params.min_rolling_obs).mean()
        roll_std = x["spread"].rolling(params.rolling_window, min_periods=params.min_rolling_obs).std()
        x["spread_mean"] = roll_mean
        x["spread_std"] = roll_std
        x["z"] = (x["spread"] - x["spread_mean"]) / x["spread_std"].replace(0, np.nan)
        x["delta_z_1m"] = x["z"].diff()
        x["delta_z_3m"] = x["z"].diff(3)
        x["bar_index_pair_day"] = np.arange(len(x))
        parts.append(x)
    m = pd.concat(parts, ignore_index=True)
    return m


def pair_signal_mask(pp: pd.DataFrame, params: PairStrategyParams) -> pd.Series:
    both_valid = (
        (pp["a_valid_entry_bar"] == 1)
        & (pp["b_valid_entry_bar"] == 1)
        & (pp["a_amount"].fillna(0) > 0)
        & (pp["b_amount"].fillna(0) > 0)
        & pp["z"].notna()
        & pp["delta_z_1m"].notna()
    )

    # 避免复权、极端日、零流动性日
    both_valid &= (
        (pp["a_daily_adj_factor_change"] == 0)
        & (pp["b_daily_adj_factor_change"] == 0)
        & (pp["a_daily_is_extreme_return_day"] == 0)
        & (pp["b_daily_is_extreme_return_day"] == 0)
    )

    if params.confirm_reversion:
        cheap_a = (
            (pp["z"] <= -params.entry_z)
            & (pp["delta_z_1m"] >= params.min_abs_delta_z)
        )
        cheap_b = (
            (pp["z"] >= params.entry_z)
            & (pp["delta_z_1m"] <= -params.min_abs_delta_z)
        )
    else:
        cheap_a = pp["z"] <= -params.entry_z
        cheap_b = pp["z"] >= params.entry_z

    return both_valid & (cheap_a | cheap_b)


def simulate_pair_trade(pp: pd.DataFrame, signal_pos: int, params: PairStrategyParams) -> Dict:
    entry_pos = signal_pos + 1
    if entry_pos >= len(pp):
        return {}

    signal = pp.iloc[signal_pos]
    entry = pp.iloc[entry_pos]

    if int(entry.get("a_force_flat_bar", 0)) == 1 or int(entry.get("b_force_flat_bar", 0)) == 1:
        return {}

    z0 = float(signal["z"])
    if not np.isfinite(z0):
        return {}

    if z0 <= -params.entry_z:
        cheap_leg = "A"
        long_code = signal["leg_a"]
        short_code = signal["leg_b"]
    elif z0 >= params.entry_z:
        cheap_leg = "B"
        long_code = signal["leg_b"]
        short_code = signal["leg_a"]
    else:
        return {}

    # 成交价：下一根 bar open
    a_entry = float(entry["a_open"])
    b_entry = float(entry["b_open"])
    if not (np.isfinite(a_entry) and np.isfinite(b_entry) and a_entry > 0 and b_entry > 0):
        return {}

    max_exit_pos = min(len(pp) - 1, entry_pos + params.max_hold_bars)

    exit_pos = max_exit_pos
    exit_reason = "max_hold"

    # 逐 bar 检查退出；用 close 作为退出价。
    for j in range(entry_pos, max_exit_pos + 1):
        row = pp.iloc[j]

        a_close = float(row["a_close"])
        b_close = float(row["b_close"])
        if not (np.isfinite(a_close) and np.isfinite(b_close) and a_close > 0 and b_close > 0):
            continue

        # 当前 pair 毛收益
        if params.mode == "long_only":
            if cheap_leg == "A":
                gross = a_close / a_entry - 1.0
            else:
                gross = b_close / b_entry - 1.0
        else:
            # gross exposure 归一：0.5 long cheap + 0.5 short expensive
            if cheap_leg == "A":
                long_ret = a_close / a_entry - 1.0
                short_ret = -(b_close / b_entry - 1.0)
            else:
                long_ret = b_close / b_entry - 1.0
                short_ret = -(a_close / a_entry - 1.0)
            gross = 0.5 * long_ret + 0.5 * short_ret

        z_now = float(row["z"]) if np.isfinite(row["z"]) else np.nan

        if int(row.get("a_force_flat_bar", 0)) == 1 or int(row.get("b_force_flat_bar", 0)) == 1:
            exit_pos = j
            exit_reason = "force_flat"
            break

        if gross <= -params.stop_loss:
            exit_pos = j
            exit_reason = "stop_loss"
            break

        if gross >= params.take_profit:
            exit_pos = j
            exit_reason = "take_profit"
            break

        if np.isfinite(z_now) and abs(z_now) <= params.exit_z:
            exit_pos = j
            exit_reason = "spread_revert"
            break

    exit_row = pp.iloc[exit_pos]
    a_exit = float(exit_row["a_close"])
    b_exit = float(exit_row["b_close"])
    if not (np.isfinite(a_exit) and np.isfinite(b_exit) and a_exit > 0 and b_exit > 0):
        return {}

    if params.mode == "long_only":
        if cheap_leg == "A":
            gross_ret = a_exit / a_entry - 1.0
            entry_price = a_entry
            exit_price = a_exit
        else:
            gross_ret = b_exit / b_entry - 1.0
            entry_price = b_entry
            exit_price = b_exit
    else:
        if cheap_leg == "A":
            long_ret = a_exit / a_entry - 1.0
            short_ret = -(b_exit / b_entry - 1.0)
        else:
            long_ret = b_exit / b_entry - 1.0
            short_ret = -(a_exit / a_entry - 1.0)
        gross_ret = 0.5 * long_ret + 0.5 * short_ret
        entry_price = np.nan
        exit_price = np.nan

    return {
        "pair_id": signal["pair_id"],
        "pair_name": signal["pair_name"],
        "pair_group": signal["pair_group"],
        "leg_a": signal["leg_a"],
        "leg_b": signal["leg_b"],
        "trade_date": signal["trade_date"],
        "signal_time": signal["trade_time"],
        "entry_time": entry["trade_time"],
        "exit_time": exit_row["trade_time"],
        "mode": params.mode,
        "cheap_leg": cheap_leg,
        "long_code": long_code,
        "short_code": short_code if params.mode == "long_short" else "",
        "z_signal": z0,
        "delta_z_1m_signal": float(signal["delta_z_1m"]),
        "delta_z_3m_signal": float(signal["delta_z_3m"]) if np.isfinite(signal["delta_z_3m"]) else np.nan,
        "z_exit": float(exit_row["z"]) if np.isfinite(exit_row["z"]) else np.nan,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "a_entry": a_entry,
        "b_entry": b_entry,
        "a_exit": a_exit,
        "b_exit": b_exit,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "param_entry_z": params.entry_z,
        "param_exit_z": params.exit_z,
        "param_rolling_window": params.rolling_window,
        "param_take_profit": params.take_profit,
        "param_stop_loss": params.stop_loss,
        "param_max_hold_bars": params.max_hold_bars,
    }


def generate_candidate_trades(panel: pd.DataFrame, pairs: pd.DataFrame, params: PairStrategyParams) -> pd.DataFrame:
    trades = []

    for i, row in pairs.iterrows():
        print(f"[{i+1}/{len(pairs)}] pair {row['pair_id']}: {row['leg_a']} - {row['leg_b']}")
        pp = make_pair_panel(panel, row, params)
        if pp.empty:
            print("  empty pair panel, skipped")
            continue

        sig = pair_signal_mask(pp, params)
        pp = pp.copy()
        pp["_signal"] = sig.astype(int)

        for date, day in pp.groupby("trade_date", sort=False):
            day = day.sort_values("trade_time").reset_index(drop=True)
            sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
            for sp in sig_pos:
                tr = simulate_pair_trade(day, int(sp), params)
                if tr:
                    trades.append(tr)

        print(f"  cumulative candidates: {len(trades):,}")

    return pd.DataFrame(trades)


def apply_portfolio_constraints(candidates: pd.DataFrame, params: PairStrategyParams) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    # z 偏离越大优先，但同一时间排序要稳定
    c = candidates.copy()
    c["abs_z_signal"] = c["z_signal"].abs()
    c = c.sort_values(["entry_time", "abs_z_signal"], ascending=[True, False]).reset_index(drop=True)

    open_positions: List[Dict] = []
    cooldown_until: Dict[str, pd.Timestamp] = {}
    pair_day_count: Dict[Tuple[str, str], int] = {}
    accepted = []

    for _, row in c.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        exit_time = pd.Timestamp(row["exit_time"])
        pair_id = row["pair_id"]
        date = row["trade_date"]
        group = row["pair_group"]
        leg_a = row["leg_a"]
        leg_b = row["leg_b"]

        # 清理已经结束的持仓
        open_positions = [p for p in open_positions if p["exit_time"] > entry_time]

        # pair 日内次数限制
        key = (pair_id, date)
        if pair_day_count.get(key, 0) >= params.pair_daily_trade_limit:
            continue

        # pair 冷却
        if pair_id in cooldown_until and entry_time < cooldown_until[pair_id]:
            continue

        # 最多同时开几个 pair
        if len(open_positions) >= params.max_open_pairs:
            continue

        # 同一 group 同时开仓限制
        group_open = sum(1 for p in open_positions if p["pair_group"] == group)
        if group_open >= params.same_group_max_open:
            continue

        # 同一 ETF 不允许同时出现在多个 pair 持仓
        open_etfs = set()
        for p in open_positions:
            open_etfs.add(p["leg_a"])
            open_etfs.add(p["leg_b"])
        if leg_a in open_etfs or leg_b in open_etfs:
            continue

        accepted.append(row.to_dict())
        open_positions.append({
            "pair_id": pair_id,
            "pair_group": group,
            "leg_a": leg_a,
            "leg_b": leg_b,
            "exit_time": exit_time,
        })
        pair_day_count[key] = pair_day_count.get(key, 0) + 1
        cooldown_until[pair_id] = exit_time + pd.Timedelta(minutes=params.cooldown_minutes)

    return pd.DataFrame(accepted)


def performance_summary(trades: pd.DataFrame, cost_bp: float, params: PairStrategyParams, all_dates: List[str]) -> Tuple[Dict, pd.DataFrame]:
    one_way = cost_bp / 10000.0

    # 这里按“组合单位收益”扣成本。
    # long_only: 买入+卖出一个腿，成本约 2 * one_way。
    # long_short: 0.5 long + 0.5 short，两腿进出，总交易额为组合单位的 2 倍，成本也约 2 * one_way。
    roundtrip_cost = 2.0 * one_way

    if trades.empty:
        daily = pd.DataFrame({"trade_date": all_dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_trade_ret"] = t["gross_ret"] - roundtrip_cost
        t["portfolio_pnl"] = params.position_weight * t["net_trade_ret"]
        daily = t.groupby("trade_date", as_index=False)["portfolio_pnl"].sum().rename(columns={"portfolio_pnl": "daily_ret"})
        daily = pd.DataFrame({"trade_date": all_dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = (1.0 + daily["daily_ret"]).cumprod()

    n_days = len(daily)
    final_nav = float(daily["nav"].iloc[-1]) if n_days else 1.0
    ann_return = final_nav ** (252.0 / n_days) - 1.0 if n_days and final_nav > 0 else np.nan
    ann_vol = float(daily["daily_ret"].std(ddof=1) * np.sqrt(252)) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan

    roll_max = daily["nav"].cummax()
    drawdown = daily["nav"] / roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    if trades.empty:
        win_rate = np.nan
        avg_gross = np.nan
        avg_net = np.nan
        profit_factor = np.nan
        avg_holding = np.nan
        exit_counts = {}
    else:
        net = trades["gross_ret"] - roundtrip_cost
        win_rate = float((net > 0).mean())
        avg_gross = float(trades["gross_ret"].mean())
        avg_net = float(net.mean())
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()
        profit_factor = float(gains / losses) if losses > 0 else np.inf
        avg_holding = float(trades["holding_bars"].mean())
        exit_counts = trades["exit_reason"].value_counts().to_dict()

    summary = {
        "cost_bp": cost_bp,
        "mode": params.mode,
        "trade_count": int(len(trades)),
        "trading_days": int(n_days),
        "avg_trades_per_day": float(len(trades) / n_days) if n_days else np.nan,
        "final_nav": final_nav,
        "ann_return": ann_return,
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
    }
    return summary, daily


def write_report(out_dir: Path, params: PairStrategyParams, pairs: pd.DataFrame, summary: pd.DataFrame, trades: pd.DataFrame) -> None:
    lines = []
    lines.append("# T+0 ETF 配对价差回归策略回测报告\n")
    lines.append("## 1. 策略逻辑\n")
    lines.append("对经济含义相近的 ETF pair 计算 log price spread 及其滚动 z-score。当价差显著偏离且开始回落时，默认 long-only 买入相对便宜腿；可选 long-short 做多便宜腿、做空昂贵腿。随后按价差回归、止盈、止损、最长持仓和收盘前强平退出。")
    lines.append("")
    lines.append("## 2. 参数\n")
    lines.append(pd.DataFrame([asdict(params)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Pair 池\n")
    lines.append(pairs.to_markdown(index=False))
    lines.append("")
    lines.append("## 4. 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")

    if not trades.empty:
        lines.append("## 5. 交易概况\n")
        lines.append(f"- 交易数：{len(trades):,}")
        lines.append(f"- pair 数量：{trades['pair_id'].nunique()}")
        lines.append(f"- 交易日数量：{trades['trade_date'].nunique()}")
        lines.append("")
        lines.append("退出原因：")
        lines.append(trades["exit_reason"].value_counts().to_markdown())
        lines.append("")
        lines.append("Pair 交易数：")
        lines.append(trades["pair_id"].value_counts().to_markdown())
        lines.append("")
        lines.append("Pair group 交易数：")
        lines.append(trades["pair_group"].value_counts().to_markdown())
        lines.append("")

    lines.append("## 6. 解释限制\n")
    lines.append("- long-only 版本不是严格市场中性统计套利，它是相似 ETF 内部相对价值轮换。")
    lines.append("- long-short 版本需要实际做空能力，普通账户未必可执行。")
    lines.append("- 当前仍使用下一根 1min bar open/close 近似成交和平仓，没有盘口价差、订单簿和动态滑点。")
    lines.append("- 若 10bp/20bp 成本下仍有效，后续才值得加入动态滑点、容量约束和样本外参数选择。")

    (out_dir / "pair_spread_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip() != ""]

    params = PairStrategyParams(
        mode=args.mode,
        rolling_window=args.rolling_window,
        min_rolling_obs=args.min_rolling_obs,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        confirm_reversion=not args.no_confirm_reversion,
        min_abs_delta_z=args.min_abs_delta_z,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_hold_bars=args.max_hold_bars,
        max_open_pairs=args.max_open_pairs,
        pair_daily_trade_limit=args.pair_daily_trade_limit,
        cooldown_minutes=args.cooldown_minutes,
        position_weight=args.position_weight,
        same_group_max_open=args.same_group_max_open,
        beta=args.beta,
    )

    print("=" * 100)
    print("T+0 ETF pair spread reversion backtest")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"pair_file  : {Path(args.pair_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"cost_bps   : {cost_bps}")
    print(f"params     : {asdict(params)}")
    print("=" * 100)

    pairs = load_pair_config(Path(args.pair_file))
    codes = sorted(set(pairs["leg_a"]).union(set(pairs["leg_b"])))
    panel = load_panel(Path(args.panel_file), codes)

    missing = sorted(set(codes) - set(panel["ts_code"].unique()))
    if missing:
        raise RuntimeError(f"panel 中缺少这些 pair ETF 数据：{missing}")

    all_dates = sorted(panel["trade_date"].unique().tolist())

    candidates = generate_candidate_trades(panel, pairs, params)
    print(f"\nCandidate trades: {len(candidates):,}")

    trades = apply_portfolio_constraints(candidates, params)
    print(f"Accepted trades : {len(trades):,}")

    summaries = []
    navs = []
    for cost in cost_bps:
        s, daily = performance_summary(trades, cost, params, all_dates)
        summaries.append(s)
        daily["cost_bp"] = cost
        navs.append(daily)

    summary = pd.DataFrame(summaries)
    nav = pd.concat(navs, ignore_index=True)

    candidates.to_csv(out_dir / "pair_spread_candidate_trades.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out_dir / "pair_spread_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "pair_spread_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(out_dir / "pair_spread_daily_nav_by_cost.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, params, pairs, summary, trades)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"summary : {(out_dir / 'pair_spread_summary_by_cost.csv').resolve()}")
    print(f"trades  : {(out_dir / 'pair_spread_trades.csv').resolve()}")
    print(f"nav     : {(out_dir / 'pair_spread_daily_nav_by_cost.csv').resolve()}")
    print(f"report  : {(out_dir / 'pair_spread_report.md').resolve()}")
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
