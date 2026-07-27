#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_pair_spread_filtered_oos.py

用途：
    对 T+0 ETF 配对价差回归策略做“样本外验证版”回测。

设计目标：
    避免看完整 2025 年结果后反复调参导致过拟合。
    本脚本采用：
        1. 开发期 dev：只用于选择参数和筛选 pair；
        2. 测试期 test：参数和 pair 固定后，只验证，不再调参；
        3. 动态成本过滤：开仓前要求预计价差回归空间足以覆盖估计成本；
        4. 输出 dev/test 分期结果，防止只看全样本净值。

核心逻辑：
    spread_t = log(P_A,t) - beta * log(P_B,t)
    z_t = (spread_t - rolling_mean) / rolling_std
    expected_edge_bp = max(abs(z_t) - exit_z, 0) * spread_std * 10000

    只有当：
        abs(z_t) >= entry_z
        z_t 开始向 0 回落
        expected_edge_bp >= edge_cost_mult * estimated_roundtrip_cost_bp
    才允许开仓。

注意：
    默认仍为 long-only：只买相对便宜的一边。
    long-short 仅做研究对照，真实账户是否能做空 ETF 需要另行确认。

运行示例：
    python ".\\backtest_pair_spread_filtered_oos.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --pair-file ".\\data_t0\\config\\t0_pair_config.csv" `
      --out-dir ".\\data_t0\\backtest_pair_spread_filtered_oos" `
      --amount-multiplier 1000

建议先用默认切分：
    dev  = 20250101 ~ 20250831
    test = 20250901 ~ 20251231
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    mode: str = "long_only"
    rolling_window: int = 60
    min_rolling_obs: int = 30
    entry_z: float = 2.5
    exit_z: float = 0.5
    min_abs_delta_z: float = 0.05
    edge_cost_mult: float = 1.5
    take_profit: float = 0.003
    stop_loss: float = 0.002
    max_hold_bars: int = 30
    max_open_pairs: int = 3
    pair_daily_trade_limit: int = 2
    cooldown_minutes: int = 15
    position_weight: float = 0.20
    same_group_max_open: int = 1
    beta: float = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filtered OOS backtest for T+0 ETF pair spread reversion.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--pair-file", type=str, default="./data_t0/config/t0_pair_config.csv")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_pair_spread_filtered_oos")

    p.add_argument("--dev-start", type=str, default="20250101")
    p.add_argument("--dev-end", type=str, default="20250831")
    p.add_argument("--test-start", type=str, default="20250901")
    p.add_argument("--test-end", type=str, default="20251231")

    p.add_argument("--mode", type=str, default="long_only", choices=["long_only", "long_short"])
    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10", help="单边成本 bp 列表。")
    p.add_argument("--main-cost-bp", type=float, default=1.0, help="开发期选择参数时使用的主成本。")

    # 动态成本 proxy
    p.add_argument("--portfolio-capital", type=float, default=1_000_000.0)
    p.add_argument("--position-weight", type=float, default=0.20)
    p.add_argument("--amount-multiplier", type=float, default=1000.0, help="Tushare amount 若为千元，设 1000。")
    p.add_argument("--base-oneway-cost-bp", type=float, default=1.0)
    p.add_argument("--spread-proxy-bp", type=float, default=2.0)
    p.add_argument("--impact-bp-per-1pct-participation", type=float, default=1.0)
    p.add_argument("--open-tail-penalty-bp", type=float, default=2.0)

    # pair 筛选，只在 dev 期上做
    p.add_argument("--min-dev-trades-per-pair", type=int, default=30)
    p.add_argument("--min-dev-breakeven-bp", type=float, default=1.0)
    p.add_argument("--min-dev-profit-factor", type=float, default=1.05)

    # 是否跑小网格
    p.add_argument("--run-grid", action="store_true", help="运行小参数网格；否则只跑默认参数。")
    return p.parse_args()


def normalize_date_str(s) -> str:
    s = str(s).replace("-", "")
    return s[:8]


def read_pair_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"pair-file 不存在：{path}")
    df = pd.read_csv(path)
    for c in ["pair_id", "leg_a", "leg_b"]:
        if c not in df.columns:
            raise ValueError(f"pair-file 缺少列：{c}")
    if "pair_group" not in df.columns:
        df["pair_group"] = "unknown"
    if "pair_name" not in df.columns:
        df["pair_name"] = df["pair_id"]
    for c in ["pair_id", "leg_a", "leg_b", "pair_group", "pair_name"]:
        df[c] = df[c].astype(str).str.strip()
    return df.drop_duplicates("pair_id").reset_index(drop=True)


def read_panel(path: Path, codes: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    cols = [
        "ts_code", "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "valid_entry_bar", "force_flat_bar",
        "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]

    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(path).names
    use_cols = [c for c in cols if c in schema_cols]
    d = pd.read_parquet(path, columns=use_cols)

    for c in cols:
        if c not in d.columns:
            d[c] = np.nan

    d["ts_code"] = d["ts_code"].astype(str)
    d = d[d["ts_code"].isin(codes)].copy()
    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(normalize_date_str)

    for c in ["open", "high", "low", "close", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def make_param_grid(args: argparse.Namespace) -> List[StrategyParams]:
    if not args.run_grid:
        return [StrategyParams(
            mode=args.mode,
            position_weight=args.position_weight,
        )]

    grid = {
        "rolling_window": [60, 120],
        "entry_z": [2.0, 2.5, 3.0],
        "exit_z": [0.3, 0.5],
        "min_abs_delta_z": [0.05, 0.10],
        "edge_cost_mult": [1.5, 2.0],
        "max_hold_bars": [15, 30],
        "pair_daily_trade_limit": [1, 2],
    }
    keys = list(grid.keys())
    params = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        params.append(StrategyParams(
            mode=args.mode,
            position_weight=args.position_weight,
            **kw,
        ))
    return params


def estimate_oneway_cost_bp(
    amount_raw: float,
    clock_time: str,
    args: argparse.Namespace,
    params: StrategyParams,
) -> float:
    amount_yuan = amount_raw * args.amount_multiplier if np.isfinite(amount_raw) else np.nan
    leg_notional = args.portfolio_capital * params.position_weight
    if params.mode == "long_short":
        leg_notional *= 0.5

    if not np.isfinite(amount_yuan) or amount_yuan <= 0:
        participation = np.inf
    else:
        participation = leg_notional / amount_yuan

    impact_bp = args.impact_bp_per_1pct_participation * (participation * 100.0)

    penalty = 0.0
    s = str(clock_time)
    if ("09:30" <= s < "09:45") or ("14:30" <= s <= "15:00"):
        penalty = args.open_tail_penalty_bp

    return args.base_oneway_cost_bp + args.spread_proxy_bp + max(impact_bp, 0.0) + penalty


def make_pair_panel(panel: pd.DataFrame, pair_row: pd.Series, params: StrategyParams, args: argparse.Namespace) -> pd.DataFrame:
    a = pair_row["leg_a"]
    b = pair_row["leg_b"]
    pa = panel[panel["ts_code"] == a].copy()
    pb = panel[panel["ts_code"] == b].copy()
    if pa.empty or pb.empty:
        return pd.DataFrame()

    base_cols = [
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "valid_entry_bar", "force_flat_bar",
        "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]
    pa = pa[base_cols].rename(columns={c: f"a_{c}" for c in base_cols if c not in ["trade_time", "trade_date", "clock_time"]})
    pb = pb[base_cols].rename(columns={c: f"b_{c}" for c in base_cols if c not in ["trade_time", "trade_date", "clock_time"]})
    m = pa.merge(pb, on=["trade_time", "trade_date", "clock_time"], how="inner")
    if m.empty:
        return m

    m["pair_id"] = pair_row["pair_id"]
    m["pair_group"] = pair_row.get("pair_group", "unknown")
    m["pair_name"] = pair_row.get("pair_name", pair_row["pair_id"])
    m["leg_a"] = a
    m["leg_b"] = b

    m["spread"] = np.log(m["a_close"]) - params.beta * np.log(m["b_close"])

    parts = []
    for date, g in m.groupby("trade_date", sort=False):
        x = g.sort_values("trade_time").copy()
        roll_mean = x["spread"].rolling(params.rolling_window, min_periods=params.min_rolling_obs).mean()
        roll_std = x["spread"].rolling(params.rolling_window, min_periods=params.min_rolling_obs).std()
        x["spread_mean"] = roll_mean
        x["spread_std"] = roll_std
        x["z"] = (x["spread"] - roll_mean) / roll_std.replace(0, np.nan)
        x["delta_z_1m"] = x["z"].diff()
        x["expected_edge_bp"] = ((x["z"].abs() - params.exit_z).clip(lower=0) * x["spread_std"] * 10000.0)

        # 只能用信号 bar 已知成交额估计入场成本。A/B 取便宜腿那一侧的 amount。
        cheap_a = x["z"] <= -params.entry_z
        cheap_b = x["z"] >= params.entry_z
        signal_amount = np.where(cheap_a, x["a_amount"], np.where(cheap_b, x["b_amount"], np.nan))
        x["signal_leg_amount"] = signal_amount
        x["estimated_entry_oneway_cost_bp"] = [
            estimate_oneway_cost_bp(a0, ct, args, params)
            for a0, ct in zip(x["signal_leg_amount"], x["clock_time"])
        ]
        # 出场成本无法预知，用入场估计的同等成本代理。
        x["estimated_roundtrip_cost_bp"] = 2.0 * x["estimated_entry_oneway_cost_bp"]
        parts.append(x)

    return pd.concat(parts, ignore_index=True)


def signal_mask(pp: pd.DataFrame, params: StrategyParams) -> pd.Series:
    valid = (
        (pp["a_valid_entry_bar"] == 1)
        & (pp["b_valid_entry_bar"] == 1)
        & (pp["a_amount"].fillna(0) > 0)
        & (pp["b_amount"].fillna(0) > 0)
        & (pp["a_daily_adj_factor_change"] == 0)
        & (pp["b_daily_adj_factor_change"] == 0)
        & (pp["a_daily_is_extreme_return_day"] == 0)
        & (pp["b_daily_is_extreme_return_day"] == 0)
        & pp["z"].notna()
        & pp["delta_z_1m"].notna()
        & pp["spread_std"].notna()
        & pp["expected_edge_bp"].notna()
        & pp["estimated_roundtrip_cost_bp"].notna()
    )

    cheap_a_reverting = (
        (pp["z"] <= -params.entry_z)
        & (pp["delta_z_1m"] >= params.min_abs_delta_z)
    )
    cheap_b_reverting = (
        (pp["z"] >= params.entry_z)
        & (pp["delta_z_1m"] <= -params.min_abs_delta_z)
    )
    enough_edge = pp["expected_edge_bp"] >= params.edge_cost_mult * pp["estimated_roundtrip_cost_bp"]

    return valid & (cheap_a_reverting | cheap_b_reverting) & enough_edge


def simulate_trade(day: pd.DataFrame, signal_pos: int, params: StrategyParams) -> Dict:
    entry_pos = signal_pos + 1
    if entry_pos >= len(day):
        return {}

    sig = day.iloc[signal_pos]
    ent = day.iloc[entry_pos]

    if int(ent.get("a_force_flat_bar", 0)) == 1 or int(ent.get("b_force_flat_bar", 0)) == 1:
        return {}

    z0 = float(sig["z"])
    if z0 <= -params.entry_z:
        cheap_leg = "A"
        long_code = sig["leg_a"]
        short_code = sig["leg_b"]
        entry_price = float(ent["a_open"])
        other_entry = float(ent["b_open"])
    elif z0 >= params.entry_z:
        cheap_leg = "B"
        long_code = sig["leg_b"]
        short_code = sig["leg_a"]
        entry_price = float(ent["b_open"])
        other_entry = float(ent["a_open"])
    else:
        return {}

    if not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(other_entry) and other_entry > 0):
        return {}

    max_exit_pos = min(len(day) - 1, entry_pos + params.max_hold_bars)
    exit_pos = max_exit_pos
    exit_reason = "max_hold"

    for j in range(entry_pos, max_exit_pos + 1):
        row = day.iloc[j]
        if int(row.get("a_force_flat_bar", 0)) == 1 or int(row.get("b_force_flat_bar", 0)) == 1:
            exit_pos = j
            exit_reason = "force_flat"
            break

        a_close = float(row["a_close"])
        b_close = float(row["b_close"])
        if not (np.isfinite(a_close) and np.isfinite(b_close) and a_close > 0 and b_close > 0):
            continue

        if params.mode == "long_only":
            if cheap_leg == "A":
                gross = a_close / entry_price - 1.0
            else:
                gross = b_close / entry_price - 1.0
        else:
            if cheap_leg == "A":
                long_ret = a_close / entry_price - 1.0
                short_ret = -(b_close / other_entry - 1.0)
            else:
                long_ret = b_close / entry_price - 1.0
                short_ret = -(a_close / other_entry - 1.0)
            gross = 0.5 * long_ret + 0.5 * short_ret

        z_now = float(row["z"]) if np.isfinite(row["z"]) else np.nan

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

    ex = day.iloc[exit_pos]
    a_exit = float(ex["a_close"])
    b_exit = float(ex["b_close"])
    if not (np.isfinite(a_exit) and np.isfinite(b_exit) and a_exit > 0 and b_exit > 0):
        return {}

    if params.mode == "long_only":
        if cheap_leg == "A":
            gross_ret = a_exit / entry_price - 1.0
            exit_price = a_exit
        else:
            gross_ret = b_exit / entry_price - 1.0
            exit_price = b_exit
    else:
        if cheap_leg == "A":
            long_ret = a_exit / entry_price - 1.0
            short_ret = -(b_exit / other_entry - 1.0)
        else:
            long_ret = b_exit / entry_price - 1.0
            short_ret = -(a_exit / other_entry - 1.0)
        gross_ret = 0.5 * long_ret + 0.5 * short_ret
        exit_price = np.nan

    return {
        "pair_id": sig["pair_id"],
        "pair_group": sig["pair_group"],
        "pair_name": sig["pair_name"],
        "leg_a": sig["leg_a"],
        "leg_b": sig["leg_b"],
        "trade_date": sig["trade_date"],
        "signal_time": sig["trade_time"],
        "entry_time": ent["trade_time"],
        "exit_time": ex["trade_time"],
        "mode": params.mode,
        "cheap_leg": cheap_leg,
        "long_code": long_code,
        "short_code": short_code if params.mode == "long_short" else "",
        "z_signal": z0,
        "z_exit": float(ex["z"]) if np.isfinite(ex["z"]) else np.nan,
        "expected_edge_bp": float(sig["expected_edge_bp"]),
        "estimated_roundtrip_cost_bp": float(sig["estimated_roundtrip_cost_bp"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
    }


def generate_trades_for_params(panel: pd.DataFrame, pairs: pd.DataFrame, params: StrategyParams, args: argparse.Namespace) -> pd.DataFrame:
    trades = []
    for _, row in pairs.iterrows():
        pp = make_pair_panel(panel, row, params, args)
        if pp.empty:
            continue
        pp["_signal"] = signal_mask(pp, params).astype(int)

        for _, day in pp.groupby("trade_date", sort=False):
            day = day.sort_values("trade_time").reset_index(drop=True)
            sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
            for sp in sig_pos:
                tr = simulate_trade(day, int(sp), params)
                if tr:
                    # 记录参数
                    for k, v in asdict(params).items():
                        tr[f"param_{k}"] = v
                    trades.append(tr)
    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.copy()
    c["abs_z"] = c["z_signal"].abs()
    c = c.sort_values(["entry_time", "abs_z"], ascending=[True, False]).reset_index(drop=True)

    open_pos = []
    cooldown_until = {}
    pair_day_count = {}
    accepted = []

    for _, row in c.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_t = pd.Timestamp(row["exit_time"])
        pair_id = row["pair_id"]
        date = row["trade_date"]
        group = row["pair_group"]

        open_pos = [p for p in open_pos if p["exit_time"] > entry]

        key = (pair_id, date)
        if pair_day_count.get(key, 0) >= params.pair_daily_trade_limit:
            continue
        if pair_id in cooldown_until and entry < cooldown_until[pair_id]:
            continue
        if len(open_pos) >= params.max_open_pairs:
            continue
        if sum(1 for p in open_pos if p["pair_group"] == group) >= params.same_group_max_open:
            continue

        # 同一 ETF 不重复占用
        used = set()
        for p in open_pos:
            used.add(p["leg_a"])
            used.add(p["leg_b"])
        if row["leg_a"] in used or row["leg_b"] in used:
            continue

        accepted.append(row.to_dict())
        open_pos.append({
            "pair_id": pair_id,
            "pair_group": group,
            "leg_a": row["leg_a"],
            "leg_b": row["leg_b"],
            "exit_time": exit_t,
        })
        pair_day_count[key] = pair_day_count.get(key, 0) + 1
        cooldown_until[pair_id] = exit_t + pd.Timedelta(minutes=params.cooldown_minutes)

    return pd.DataFrame(accepted)


def split_trades(trades: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    t = trades.copy()
    t["trade_date"] = t["trade_date"].astype(str).map(normalize_date_str)
    dev = t[(t["trade_date"] >= args.dev_start) & (t["trade_date"] <= args.dev_end)].copy()
    test = t[(t["trade_date"] >= args.test_start) & (t["trade_date"] <= args.test_end)].copy()
    return dev, test


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def pair_dev_filter(dev_trades: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
    if dev_trades.empty:
        return pd.DataFrame(), []

    rows = []
    for pair_id, g in dev_trades.groupby("pair_id"):
        gross = g["gross_ret"]
        avg = gross.mean()
        be = avg * 10000.0 / 2.0
        pf = profit_factor(gross)
        rows.append({
            "pair_id": pair_id,
            "dev_trade_count": len(g),
            "dev_avg_gross_ret": avg,
            "dev_breakeven_oneway_cost_bp": be,
            "dev_profit_factor": pf,
            "dev_win_rate": (gross > 0).mean(),
            "dev_spread_revert_rate": (g["exit_reason"] == "spread_revert").mean(),
            "selected": int(
                (len(g) >= args.min_dev_trades_per_pair)
                and (be >= args.min_dev_breakeven_bp)
                and (pf >= args.min_dev_profit_factor)
            ),
        })
    df = pd.DataFrame(rows).sort_values(["selected", "dev_breakeven_oneway_cost_bp"], ascending=[False, False])
    selected = df.loc[df["selected"] == 1, "pair_id"].tolist()
    return df, selected


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str], params: StrategyParams) -> Dict:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = params.position_weight * t["net_ret"]
        daily = t.groupby("trade_date", as_index=False)["pnl"].sum().rename(columns={"pnl": "daily_ret"})
        daily = pd.DataFrame({"trade_date": dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})

    daily = daily.sort_values("trade_date")
    daily["nav"] = (1 + daily["daily_ret"]).cumprod()
    n = len(daily)
    final_nav = float(daily["nav"].iloc[-1]) if n else 1.0
    ann_return = final_nav ** (252 / n) - 1 if n and final_nav > 0 else np.nan
    ann_vol = daily["daily_ret"].std(ddof=1) * np.sqrt(252) if n > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
    dd = daily["nav"] / daily["nav"].cummax() - 1
    max_dd = float(dd.min()) if n else 0.0

    if trades.empty:
        return {
            "cost_bp": cost_bp,
            "trade_count": 0,
            "final_nav": final_nav,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": np.nan,
            "avg_gross_ret": np.nan,
            "avg_net_ret": np.nan,
            "profit_factor": np.nan,
            "avg_holding_bars": np.nan,
        }

    net = trades["gross_ret"] - 2.0 * cost_bp / 10000.0
    return {
        "cost_bp": cost_bp,
        "trade_count": len(trades),
        "final_nav": final_nav,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float((net > 0).mean()),
        "avg_gross_ret": float(trades["gross_ret"].mean()),
        "avg_net_ret": float(net.mean()),
        "profit_factor": profit_factor(net),
        "avg_holding_bars": float(trades["holding_bars"].mean()),
    }


def dev_objective(row: Dict) -> float:
    # 目标不是单纯最大收益，而是偏稳健：
    # 平均净收益、profit factor、回撤共同考虑。
    avg_net_bp = row.get("avg_net_ret", np.nan) * 10000.0
    pf = row.get("profit_factor", np.nan)
    mdd = abs(row.get("max_drawdown", 0.0))
    n = row.get("trade_count", 0)
    if not np.isfinite(avg_net_bp) or not np.isfinite(pf) or n < 30:
        return -1e9
    return avg_net_bp + 0.5 * min(pf, 3.0) - 10.0 * mdd


def date_range_from_panel(panel: pd.DataFrame, start: str, end: str) -> List[str]:
    dates = sorted(panel["trade_date"].astype(str).map(normalize_date_str).unique())
    return [d for d in dates if start <= d <= end]


def main() -> int:
    args = parse_args()
    args.dev_start = normalize_date_str(args.dev_start)
    args.dev_end = normalize_date_str(args.dev_end)
    args.test_start = normalize_date_str(args.test_start)
    args.test_end = normalize_date_str(args.test_end)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]
    params_list = make_param_grid(args)

    print("=" * 100)
    print("Filtered OOS pair spread backtest")
    print("=" * 100)
    print(f"panel_file   : {Path(args.panel_file).resolve()}")
    print(f"pair_file    : {Path(args.pair_file).resolve()}")
    print(f"out_dir      : {out_dir.resolve()}")
    print(f"dev          : {args.dev_start} -> {args.dev_end}")
    print(f"test         : {args.test_start} -> {args.test_end}")
    print(f"params count : {len(params_list)}")
    print(f"cost_bps     : {cost_bps}; main={args.main_cost_bp}")
    print("=" * 100)

    pairs = read_pair_config(Path(args.pair_file))
    codes = sorted(set(pairs["leg_a"]).union(set(pairs["leg_b"])))
    panel = read_panel(Path(args.panel_file), codes)

    dev_dates = date_range_from_panel(panel, args.dev_start, args.dev_end)
    test_dates = date_range_from_panel(panel, args.test_start, args.test_end)

    grid_rows = []
    all_results = []

    best_score = -1e18
    best_pack = None

    for idx, params in enumerate(params_list, 1):
        print(f"\n[{idx}/{len(params_list)}] params={asdict(params)}")
        cand = generate_trades_for_params(panel, pairs, params, args)
        cand = apply_portfolio_constraints(cand, params)
        dev_tr, test_tr = split_trades(cand, args)

        pair_filter_df, selected_pairs = pair_dev_filter(dev_tr, args)
        filtered_dev = dev_tr[dev_tr["pair_id"].isin(selected_pairs)].copy()
        filtered_test = test_tr[test_tr["pair_id"].isin(selected_pairs)].copy()

        dev_perf_main = performance(filtered_dev, args.main_cost_bp, dev_dates, params)
        score = dev_objective(dev_perf_main)

        row = asdict(params)
        row.update({
            "param_index": idx,
            "selected_pair_count": len(selected_pairs),
            "selected_pairs": ",".join(selected_pairs),
            "dev_objective": score,
        })
        for k, v in dev_perf_main.items():
            row[f"dev_main_{k}"] = v
        grid_rows.append(row)

        if score > best_score:
            best_score = score
            best_pack = {
                "params": params,
                "candidates": cand,
                "dev_trades": filtered_dev,
                "test_trades": filtered_test,
                "pair_filter": pair_filter_df,
                "selected_pairs": selected_pairs,
                "dev_perf_main": dev_perf_main,
                "score": score,
            }

    grid_df = pd.DataFrame(grid_rows).sort_values("dev_objective", ascending=False)
    grid_df.to_csv(out_dir / "filtered_oos_grid_summary.csv", index=False, encoding="utf-8-sig")

    if best_pack is None:
        raise RuntimeError("没有得到有效参数结果。")

    best_params = best_pack["params"]
    selected_pairs = best_pack["selected_pairs"]
    dev_trades = best_pack["dev_trades"]
    test_trades = best_pack["test_trades"]
    pair_filter = best_pack["pair_filter"]

    # 成本敏感性：dev/test 分开输出
    rows = []
    nav_rows = []
    for period, tr, dates in [
        ("dev", dev_trades, dev_dates),
        ("test", test_trades, test_dates),
    ]:
        for c in cost_bps:
            perf = performance(tr, c, dates, best_params)
            perf["period"] = period
            rows.append(perf)

            # 重新构建 daily nav 明细
            if tr.empty:
                daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
            else:
                x = tr.copy()
                x["net_ret"] = x["gross_ret"] - 2.0 * c / 10000.0
                x["pnl"] = best_params.position_weight * x["net_ret"]
                daily = x.groupby("trade_date", as_index=False)["pnl"].sum().rename(columns={"pnl": "daily_ret"})
                daily = pd.DataFrame({"trade_date": dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})
            daily["nav"] = (1 + daily["daily_ret"]).cumprod()
            daily["period"] = period
            daily["cost_bp"] = c
            nav_rows.append(daily)

    summary = pd.DataFrame(rows)
    nav = pd.concat(nav_rows, ignore_index=True)

    best_info = {
        "best_params": asdict(best_params),
        "dev_objective": best_pack["score"],
        "selected_pairs": selected_pairs,
        "dev_start": args.dev_start,
        "dev_end": args.dev_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "main_cost_bp": args.main_cost_bp,
        "selection_rules": {
            "min_dev_trades_per_pair": args.min_dev_trades_per_pair,
            "min_dev_breakeven_bp": args.min_dev_breakeven_bp,
            "min_dev_profit_factor": args.min_dev_profit_factor,
        },
    }

    # 输出
    with open(out_dir / "filtered_oos_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    pair_filter.to_csv(out_dir / "filtered_oos_pair_selection_on_dev.csv", index=False, encoding="utf-8-sig")
    dev_trades.to_csv(out_dir / "filtered_oos_dev_trades.csv", index=False, encoding="utf-8-sig")
    test_trades.to_csv(out_dir / "filtered_oos_test_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "filtered_oos_summary_by_period_cost.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(out_dir / "filtered_oos_daily_nav_by_period_cost.csv", index=False, encoding="utf-8-sig")

    # 报告
    lines = []
    lines.append("# 配对价差回归：强过滤 + 样本外验证报告\n")
    lines.append("## 1. 设计原则\n")
    lines.append("本次不是看完整 2025 年结果后调参数，而是在开发期选择参数和 pair，测试期只验证。动态成本过滤要求预计价差回归空间必须覆盖估计往返成本的若干倍。")
    lines.append("")
    lines.append("## 2. 样本切分\n")
    lines.append(f"- dev: {args.dev_start} ~ {args.dev_end}")
    lines.append(f"- test: {args.test_start} ~ {args.test_end}")
    lines.append(f"- main selection cost: {args.main_cost_bp}bp 单边")
    lines.append("")
    lines.append("## 3. 最优参数（仅由 dev 选择）\n")
    lines.append(pd.DataFrame([asdict(best_params)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. dev 期选出的 pair\n")
    lines.append(", ".join(selected_pairs) if selected_pairs else "无")
    lines.append("")
    lines.append("## 5. dev/test 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. pair 选择明细\n")
    lines.append(pair_filter.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. 解释\n")
    lines.append("- 如果 dev 好、test 差，说明当前强过滤规则仍可能过拟合。")
    lines.append("- 如果 test 在 1bp/2bp 下仍能保持正期望，但 5bp 后不行，说明策略属于极低成本相对价值策略。")
    lines.append("- 如果 test 的交易数过少，不能证明策略可交易，只能说明过滤过严。")
    lines.append("- 若 test 表现通过，下一步才值得补 rolling beta、完整分钟级权益曲线和更真实滑点模型。")
    (out_dir / "filtered_oos_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Finished")
    print(f"best params : {(out_dir / 'filtered_oos_best_params.json').resolve()}")
    print(f"summary     : {(out_dir / 'filtered_oos_summary_by_period_cost.csv').resolve()}")
    print(f"report      : {(out_dir / 'filtered_oos_report.md').resolve()}")
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
