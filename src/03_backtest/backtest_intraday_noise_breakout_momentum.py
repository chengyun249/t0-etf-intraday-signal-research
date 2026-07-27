#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_intraday_noise_breakout_momentum.py

用途：
    参考“日内噪声边界突破 + 趋势跟随”的 ETF 日内动量思路，
    在当前 T+0 ETF 1min 面板上测试 long-only 可执行策略。

核心思想：
    对每只 ETF、每个日内 bar_index，使用过去 N 个交易日同一分钟的
    |close_t / open_day - 1| 的均值作为“正常日内噪声”。

    当今天价格突破：
        upper_band_t = open_today * (1 + k * noise_t)
    且价格站上 VWAP、成交额放大、同类 ETF 同步走强时，
    下一根 1min bar 开多。

    退出：
        trailing stop / VWAP stop / hard stop / take profit / max hold / force flat。

为什么做这个：
    配对交易 long-short 理论版有边际，但现实做空 ETF 不稳定；
    本脚本测试不依赖做空的 long-only 日内突破动量策略。

输入：
    data_t0/processed/t0_intraday_bar_panel.parquet

输出：
    data_t0/backtest_intraday_noise_breakout/
      ├─ noise_breakout_trades.csv
      ├─ noise_breakout_summary_by_period_cost.csv
      ├─ noise_breakout_daily_nav_by_period_cost.csv
      ├─ noise_breakout_param_grid_summary.csv        # 如果 --run-grid
      ├─ noise_breakout_best_params.json
      └─ noise_breakout_report.md

运行默认参数：
    python ".\\backtest_intraday_noise_breakout_momentum.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_intraday_noise_breakout"

运行快速参数网格：
    python ".\\backtest_intraday_noise_breakout_momentum.py" `
      --panel-file ".\\data_t0\\processed\\t0_intraday_bar_panel.parquet" `
      --out-dir ".\\data_t0\\backtest_intraday_noise_breakout_grid" `
      --run-grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from t0_research.execution import simulate_long_exit


@dataclass
class StrategyParams:
    lookback_days: int = 14
    min_noise_obs: int = 7
    band_k: float = 1.25
    amount_z_min: float = 0.50
    category_strength_min: float = 0.55
    require_breakout_high_20m: int = 1
    signal_rising_edge: int = 1

    take_profit: float = 0.0060
    hard_stop_loss: float = 0.0030
    trailing_stop: float = 0.0035
    vwap_stop_band: float = 0.0005
    max_hold_bars: int = 60
    stop_slippage_bp: float = 1.0

    max_positions: int = 3
    same_category_max_open: int = 1
    position_weight: float = 0.20
    cooldown_minutes: int = 15
    etf_daily_trade_limit: int = 2

    entry_start_time: str = "09:45"
    entry_end_time: str = "14:15"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest intraday noise-boundary breakout momentum strategy.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_intraday_noise_breakout")

    p.add_argument("--dev-start", type=str, default="20250101")
    p.add_argument("--dev-end", type=str, default="20250831")
    p.add_argument("--test-start", type=str, default="20250901")
    p.add_argument("--test-end", type=str, default="20251231")

    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10", help="单边成本 bp 列表。")
    p.add_argument("--main-cost-bp", type=float, default=2.0, help="run-grid 时开发期主选择成本。")

    p.add_argument("--run-grid", action="store_true", help="运行快速参数网格；否则只跑默认参数。")

    # 单组参数，也可手动覆盖
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--band-k", type=float, default=1.25)
    p.add_argument("--amount-z-min", type=float, default=0.50)
    p.add_argument("--category-strength-min", type=float, default=0.55)
    p.add_argument("--take-profit", type=float, default=0.0060)
    p.add_argument("--hard-stop-loss", type=float, default=0.0030)
    p.add_argument("--trailing-stop", type=float, default=0.0035)
    p.add_argument("--max-hold-bars", type=int, default=60)
    return p.parse_args()


def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "intraday_vwap", "bar_index", "ret_5m", "ret_10m", "ret_20m",
        "amount_z_20m", "breakout_high_20m",
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

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)

    for c in ["open", "high", "low", "close", "amount", "intraday_vwap", "ret_5m", "ret_10m", "ret_20m", "amount_z_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["breakout_high_20m", "valid_entry_bar", "force_flat_bar", "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    # bar_index 兜底
    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype("Int64")

    # 当日开盘价
    d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")
    d["intraday_move"] = d["close"] / d["day_open"] - 1.0
    d["abs_intraday_move"] = d["intraday_move"].abs()

    # 同类 ETF 同步强度：同一时刻同一类别中，站上当日开盘且短期动量为正的比例。
    d["positive_intraday"] = (d["intraday_move"] > 0).astype(int)
    d["positive_5m"] = (d["ret_5m"] > 0).astype(int)
    d["cat_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
    d["cat_ret5_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")

    return d


def make_params(args: argparse.Namespace) -> List[StrategyParams]:
    if not args.run_grid:
        return [StrategyParams(
            lookback_days=args.lookback_days,
            min_noise_obs=max(5, args.lookback_days // 2),
            band_k=args.band_k,
            amount_z_min=args.amount_z_min,
            category_strength_min=args.category_strength_min,
            take_profit=args.take_profit,
            hard_stop_loss=args.hard_stop_loss,
            trailing_stop=args.trailing_stop,
            max_hold_bars=args.max_hold_bars,
        )]

    # 快速网格：先控制规模，避免跑太久。
    grid = {
        "lookback_days": [10, 14, 20],
        "band_k": [1.0, 1.25, 1.5],
        "amount_z_min": [0.0, 0.5],
        "category_strength_min": [0.50, 0.60],
        "trailing_stop": [0.0035],
        "take_profit": [0.0060],
        "max_hold_bars": [30, 60],
    }
    keys = list(grid.keys())
    params = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        params.append(StrategyParams(
            **kw,
            min_noise_obs=max(5, kw["lookback_days"] // 2),
        ))
    return params


def add_noise_features(panel: pd.DataFrame, lookback_days: int, min_obs: int) -> pd.DataFrame:
    d = panel.copy()
    # 对每只 ETF 的同一 bar_index，用过去 N 天同一分钟的 abs move 均值做噪声。
    d = d.sort_values(["ts_code", "bar_index", "trade_date"]).reset_index(drop=True)
    noise = (
        d.groupby(["ts_code", "bar_index"], group_keys=False)["abs_intraday_move"]
        .apply(lambda s: s.shift(1).rolling(lookback_days, min_periods=min_obs).mean())
    )
    d[f"noise_{lookback_days}d"] = noise.to_numpy()
    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def build_signal_mask(d: pd.DataFrame, p: StrategyParams) -> pd.Series:
    noise_col = f"noise_{p.lookback_days}d"
    upper_band = d["day_open"] * (1.0 + p.band_k * d[noise_col])
    d["_upper_band_tmp"] = upper_band

    m = (
        (d["valid_entry_bar"] == 1)
        & (d["force_flat_bar"] == 0)
        & (d["clock_time"] >= p.entry_start_time)
        & (d["clock_time"] <= p.entry_end_time)
        & d[noise_col].notna()
        & (d[noise_col] > 0)
        & (d["amount"].fillna(0) > 0)
        & (d["close"] > upper_band)
        & (d["close"] > d["intraday_vwap"])
        & (d["intraday_move"] > 0)
        & (d["ret_5m"].fillna(0) > 0)
        & (d["amount_z_20m"].fillna(0) >= p.amount_z_min)
        & (d["cat_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["cat_ret5_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["daily_adj_factor_change"].fillna(0).astype(int) == 0)
        & (d["daily_is_extreme_return_day"].fillna(0).astype(int) == 0)
    )

    if p.require_breakout_high_20m:
        m &= (d["breakout_high_20m"].fillna(0).astype(int) == 1)

    if p.signal_rising_edge:
        # 同一 ETF-day 内只在信号从 0 变 1 的地方触发，防止同一趋势连续每分钟刷单。
        prev = m.groupby([d["ts_code"], d["trade_date"]]).shift(1).fillna(False)
        m = m & (~prev)

    return m


def simulate_trade(day: pd.DataFrame, signal_pos: int, p: StrategyParams) -> Dict:
    day = day.copy()
    day["known_vwap"] = day["intraday_vwap"].shift(1)
    entry_pos = signal_pos + 1
    if entry_pos >= len(day):
        return {}

    sig = day.iloc[signal_pos]
    ent = day.iloc[entry_pos]

    if int(ent.get("force_flat_bar", 0)) == 1:
        return {}

    entry_price = float(ent["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    max_exit_pos = min(len(day) - 1, entry_pos + p.max_hold_bars)
    tp_price = entry_price * (1.0 + p.take_profit)
    hard_stop_price = entry_price * (1.0 - p.hard_stop_loss)
    result = simulate_long_exit(
        day, entry_pos=entry_pos, max_exit_pos=max_exit_pos, entry_price=entry_price,
        take_profit_price=tp_price, hard_stop_price=hard_stop_price,
        trailing_stop=p.trailing_stop, vwap_stop_band=p.vwap_stop_band,
        opening_range_stop_price=-np.inf, stop_slippage_bp=p.stop_slippage_bp,
    )
    exit_pos, exit_price, exit_reason = result.bar_position, result.price, result.reason

    if not np.isfinite(exit_price) or exit_price <= 0:
        return {}

    gross_ret = exit_price / entry_price - 1.0
    ex = day.iloc[exit_pos]

    return {
        "ts_code": sig["ts_code"],
        "name": sig.get("name", ""),
        "t0_category": sig.get("t0_category", ""),
        "t0_category_cn": sig.get("t0_category_cn", ""),
        "trade_date": sig["trade_date"],
        "signal_time": sig["trade_time"],
        "entry_time": ent["trade_time"],
        "exit_time": ex["trade_time"],
        "signal_close": float(sig["close"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "signal_intraday_move": float(sig["intraday_move"]),
        "signal_noise": float(sig[f"noise_{p.lookback_days}d"]),
        "signal_upper_band": float(sig["_upper_band_tmp"]),
        "signal_price_to_upper": float(sig["close"] / sig["_upper_band_tmp"] - 1.0),
        "signal_amount_z_20m": float(sig["amount_z_20m"]) if np.isfinite(sig["amount_z_20m"]) else np.nan,
        "signal_cat_positive_ratio": float(sig["cat_positive_ratio"]),
        "signal_cat_ret5_positive_ratio": float(sig["cat_ret5_positive_ratio"]),
    }


def generate_trades(panel_with_noise: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    d = panel_with_noise.copy()
    d["_signal"] = build_signal_mask(d, p).astype(int)

    trades = []
    groups = d.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups
    for i, (_, day) in enumerate(groups, 1):
        if i % 1000 == 0:
            print(f"    simulated ETF-days {i}/{total}")
        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        for sp in sig_pos:
            tr = simulate_trade(day, int(sp), p)
            if tr:
                for k, v in asdict(p).items():
                    tr[f"param_{k}"] = v
                trades.append(tr)

    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.sort_values(["entry_time", "signal_price_to_upper"], ascending=[True, False]).reset_index(drop=True).copy()

    open_pos = []
    cooldown_until = {}
    etf_day_count = {}
    accepted = []

    for _, row in c.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_t = pd.Timestamp(row["exit_time"])
        code = row["ts_code"]
        date = row["trade_date"]
        cat = row["t0_category"]

        open_pos = [pos for pos in open_pos if pos["exit_time"] > entry]

        key = (code, date)
        if etf_day_count.get(key, 0) >= p.etf_daily_trade_limit:
            continue

        if code in cooldown_until and entry < cooldown_until[code]:
            continue

        if any(pos["ts_code"] == code for pos in open_pos):
            continue

        if len(open_pos) >= p.max_positions:
            continue

        if sum(1 for pos in open_pos if pos["t0_category"] == cat) >= p.same_category_max_open:
            continue

        accepted.append(row.to_dict())
        open_pos.append({
            "ts_code": code,
            "t0_category": cat,
            "exit_time": exit_t,
        })
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=p.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1

    return pd.DataFrame(accepted)


def date_range(panel: pd.DataFrame, start: str, end: str) -> List[str]:
    dates = sorted(panel["trade_date"].astype(str).unique())
    return [d for d in dates if start <= d <= end]


def split_trades(trades: pd.DataFrame, dev_start: str, dev_end: str, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    t = trades.copy()
    t["trade_date"] = t["trade_date"].astype(str).map(norm_date)
    dev = t[(t["trade_date"] >= dev_start) & (t["trade_date"] <= dev_end)].copy()
    test = t[(t["trade_date"] >= test_start) & (t["trade_date"] <= test_end)].copy()
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


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str], p: StrategyParams) -> Tuple[Dict, pd.DataFrame]:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = p.position_weight * t["net_ret"]
        daily = t.groupby("trade_date", as_index=False)["pnl"].sum().rename(columns={"pnl": "daily_ret"})
        daily = pd.DataFrame({"trade_date": dates}).merge(daily, on="trade_date", how="left").fillna({"daily_ret": 0.0})

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = (1.0 + daily["daily_ret"]).cumprod()

    n = len(daily)
    final_nav = float(daily["nav"].iloc[-1]) if n else 1.0
    ann_return = final_nav ** (252.0 / n) - 1.0 if n and final_nav > 0 else np.nan
    ann_vol = float(daily["daily_ret"].std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
    dd = daily["nav"] / daily["nav"].cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    if trades.empty:
        s = {
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
    else:
        net = trades["gross_ret"] - 2.0 * cost_bp / 10000.0
        s = {
            "cost_bp": cost_bp,
            "trade_count": int(len(trades)),
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
    return s, daily


def objective_from_perf(perf: Dict) -> float:
    n = perf.get("trade_count", 0)
    if n < 30:
        return -1e9
    avg_net_bp = perf.get("avg_net_ret", np.nan) * 10000.0
    pf = perf.get("profit_factor", np.nan)
    mdd = abs(perf.get("max_drawdown", 0.0))
    sharpe = perf.get("sharpe", np.nan)
    if not np.isfinite(avg_net_bp) or not np.isfinite(pf) or not np.isfinite(sharpe):
        return -1e9
    return avg_net_bp + 0.30 * min(pf, 3.0) + 0.05 * min(sharpe, 5.0) - 10.0 * mdd


def main() -> int:
    args = parse_args()
    dev_start, dev_end = norm_date(args.dev_start), norm_date(args.dev_end)
    test_start, test_end = norm_date(args.test_start), norm_date(args.test_end)
    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Intraday noise-boundary breakout momentum backtest")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"dev        : {dev_start} -> {dev_end}")
    print(f"test       : {test_start} -> {test_end}")
    print(f"run_grid   : {args.run_grid}")
    print("=" * 100)

    panel = read_panel(Path(args.panel_file))
    dev_dates = date_range(panel, dev_start, dev_end)
    test_dates = date_range(panel, test_start, test_end)

    params_list = make_params(args)
    lookbacks = sorted(set(p.lookback_days for p in params_list))
    panels_by_lb: Dict[int, pd.DataFrame] = {}

    for lb in lookbacks:
        min_obs = max(5, lb // 2)
        print(f"Precomputing noise for lookback_days={lb}, min_obs={min_obs} ...")
        panels_by_lb[lb] = add_noise_features(panel, lb, min_obs)

    best = None
    best_score = -1e18
    grid_rows = []

    for idx, p in enumerate(params_list, 1):
        print(f"\n[{idx}/{len(params_list)}] params={asdict(p)}")
        d = panels_by_lb[p.lookback_days]
        raw_trades = generate_trades(d, p)
        trades = apply_portfolio_constraints(raw_trades, p)
        dev_trades, test_trades = split_trades(trades, dev_start, dev_end, test_start, test_end)

        dev_perf, _ = performance(dev_trades, args.main_cost_bp, dev_dates, p)
        score = objective_from_perf(dev_perf)

        row = asdict(p)
        row["param_index"] = idx
        row["dev_objective"] = score
        row["raw_trade_count"] = len(raw_trades)
        row["accepted_trade_count"] = len(trades)
        for k, v in dev_perf.items():
            row[f"dev_main_{k}"] = v
        grid_rows.append(row)

        print(f"  raw trades={len(raw_trades):,}, accepted={len(trades):,}, dev trades={len(dev_trades):,}, test trades={len(test_trades):,}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best = {
                "params": p,
                "raw_trades": raw_trades,
                "trades": trades,
                "dev_trades": dev_trades,
                "test_trades": test_trades,
                "dev_perf": dev_perf,
                "score": score,
            }

    if best is None:
        raise RuntimeError("没有得到有效结果。")

    p = best["params"]
    dev_trades = best["dev_trades"]
    test_trades = best["test_trades"]
    all_trades = best["trades"]

    grid_df = pd.DataFrame(grid_rows).sort_values("dev_objective", ascending=False)
    grid_df.to_csv(out_dir / "noise_breakout_param_grid_summary.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    nav_parts = []
    for period, tr, dates in [
        ("dev", dev_trades, dev_dates),
        ("test", test_trades, test_dates),
    ]:
        for c in cost_bps:
            s, daily = performance(tr, c, dates, p)
            s["period"] = period
            summary_rows.append(s)
            daily["period"] = period
            daily["cost_bp"] = c
            nav_parts.append(daily)

    summary = pd.DataFrame(summary_rows)
    nav = pd.concat(nav_parts, ignore_index=True)

    all_trades.to_csv(out_dir / "noise_breakout_trades.csv", index=False, encoding="utf-8-sig")
    dev_trades.to_csv(out_dir / "noise_breakout_dev_trades.csv", index=False, encoding="utf-8-sig")
    test_trades.to_csv(out_dir / "noise_breakout_test_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "noise_breakout_summary_by_period_cost.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(out_dir / "noise_breakout_daily_nav_by_period_cost.csv", index=False, encoding="utf-8-sig")

    best_info = {
        "best_params": asdict(p),
        "dev_objective": best_score,
        "dev_start": dev_start,
        "dev_end": dev_end,
        "test_start": test_start,
        "test_end": test_end,
        "main_cost_bp": args.main_cost_bp,
        "run_grid": bool(args.run_grid),
    }
    with open(out_dir / "noise_breakout_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    # 简单归因
    category_stats = pd.DataFrame()
    if not all_trades.empty:
        category_stats = (
            all_trades.groupby(["t0_category_cn"], dropna=False)
            .agg(
                trade_count=("gross_ret", "size"),
                avg_gross_ret=("gross_ret", "mean"),
                win_rate=("gross_ret", lambda s: (s > 0).mean()),
                profit_factor=("gross_ret", profit_factor),
                avg_holding_bars=("holding_bars", "mean"),
            )
            .reset_index()
            .sort_values("avg_gross_ret", ascending=False)
        )
        category_stats.to_csv(out_dir / "noise_breakout_category_summary.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# T+0 ETF 日内噪声边界突破动量策略报告\n")
    lines.append("## 1. 策略逻辑\n")
    lines.append("使用过去 N 个交易日同一分钟相对开盘价的平均绝对波动作为噪声边界。当价格突破上边界、站上 VWAP、成交额放大、同类 ETF 同步走强时，下一根 1min bar 开多；随后用 trailing stop、VWAP stop、hard stop、take profit、max hold 和收盘前强平退出。")
    lines.append("")
    lines.append("## 2. 样本切分\n")
    lines.append(f"- dev: {dev_start} ~ {dev_end}")
    lines.append(f"- test: {test_start} ~ {test_end}")
    lines.append(f"- main selection cost: {args.main_cost_bp}bp 单边")
    lines.append("")
    lines.append("## 3. 最优参数\n")
    lines.append(pd.DataFrame([asdict(p)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. dev/test 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not category_stats.empty:
        lines.append("## 5. 类别归因\n")
        lines.append(category_stats.to_markdown(index=False))
        lines.append("")
    lines.append("## 6. 判断标准\n")
    lines.append("- 如果 test 期 2bp/3bp 仍为正，说明 long-only 动量比配对 long-only 更有现实价值。")
    lines.append("- 如果只有 0bp 正、1bp 后失效，则说明信号仍太薄，应继续改结构或换方向。")
    lines.append("- 如果 dev 好、test 差，说明参数或信号过拟合。")
    (out_dir / "noise_breakout_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Finished")
    print(f"summary : {(out_dir / 'noise_breakout_summary_by_period_cost.csv').resolve()}")
    print(f"trades  : {(out_dir / 'noise_breakout_trades.csv').resolve()}")
    print(f"best    : {(out_dir / 'noise_breakout_best_params.json').resolve()}")
    print(f"report  : {(out_dir / 'noise_breakout_report.md').resolve()}")
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
