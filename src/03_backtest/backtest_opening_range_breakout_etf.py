#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
backtest_opening_range_breakout_etf.py

ETF Opening Range Breakout (ORB) long-only intraday strategy.

Idea:
    1. Define opening range, e.g. 09:30-09:45 or 09:30-10:00.
    2. Compute OR_high, OR_low, opening-range amount.
    3. Compute relative opening volume/amount vs previous N trading days.
    4. After opening range, buy if price breaks OR_high, stands above VWAP,
       relative opening amount is high, and peer category strength is positive.
    5. Exit by take profit, hard stop, trailing stop, VWAP stop, max hold, force flat.

Outputs:
    orb_trades.csv
    orb_dev_trades.csv
    orb_test_trades.csv
    orb_summary_by_scope_period_cost.csv
    orb_daily_nav_by_scope_period_cost.csv
    orb_param_grid_summary.csv
    orb_best_params.json
    orb_category_summary.csv
    orb_report.md

Run default:
    python .\backtest_opening_range_breakout_etf.py `
      --panel-file .\data_t0\processed\t0_intraday_bar_panel.parquet `
      --out-dir .\data_t0\backtest_orb_etf

Run grid:
    python .\backtest_opening_range_breakout_etf.py `
      --panel-file .\data_t0\processed\t0_intraday_bar_panel.parquet `
      --out-dir .\data_t0\backtest_orb_etf_grid `
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
class ORBParams:
    opening_start: str = "09:30"
    opening_end: str = "09:45"
    entry_start: str = "09:46"
    entry_end: str = "14:15"
    lookback_days: int = 10
    min_relvol_obs: int = 5
    rel_open_amount_min: float = 1.30
    top_relvol_n: int = 10
    breakout_buffer: float = 0.0002
    amount_z_min: float = 0.00
    category_strength_min: float = 0.50
    require_ret5_positive: int = 1
    require_above_vwap: int = 1
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest ETF Opening Range Breakout strategy.")
    p.add_argument("--panel-file", type=str, default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    p.add_argument("--out-dir", type=str, default="./data_t0/backtest_orb_etf")
    p.add_argument("--dev-start", type=str, default="20250101")
    p.add_argument("--dev-end", type=str, default="20250831")
    p.add_argument("--test-start", type=str, default="20250901")
    p.add_argument("--test-end", type=str, default="20251231")
    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")
    p.add_argument("--main-cost-bp", type=float, default=2.0)
    p.add_argument("--run-grid", action="store_true")
    p.add_argument("--scopes", type=str, default="all,no_bond_money,commodity_focus,cross_commodity",
                   help="comma-separated: all,no_bond_money,commodity_focus,cross_commodity")
    p.add_argument("--opening-end", type=str, default="09:45")
    p.add_argument("--lookback-days", type=int, default=10)
    p.add_argument("--rel-open-amount-min", type=float, default=1.30)
    p.add_argument("--top-relvol-n", type=int, default=10)
    p.add_argument("--breakout-buffer", type=float, default=0.0002)
    p.add_argument("--amount-z-min", type=float, default=0.00)
    p.add_argument("--category-strength-min", type=float, default=0.50)
    p.add_argument("--take-profit", type=float, default=0.0060)
    p.add_argument("--hard-stop-loss", type=float, default=0.0030)
    p.add_argument("--trailing-stop", type=float, default=0.0035)
    p.add_argument("--max-hold-bars", type=int, default=60)
    return p.parse_args()


def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def next_minute(hhmm: str) -> str:
    hh, mm = hhmm.split(":")
    total = int(hh) * 60 + int(mm) + 1
    return f"{total // 60:02d}:{total % 60:02d}"


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file does not exist: {path}")
    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time",
        "open", "high", "low", "close", "amount",
        "intraday_vwap", "ret_5m", "ret_10m", "amount_z_20m",
        "valid_entry_bar", "force_flat_bar",
        "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day",
    ]
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(path).names
    cols = [c for c in needed if c in schema_cols]
    d = pd.read_parquet(path, columns=cols)
    for c in needed:
        if c not in d.columns:
            d[c] = "" if c in ["name", "t0_category", "t0_category_cn"] else np.nan
    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)
    for c in ["open", "high", "low", "close", "amount", "intraday_vwap", "ret_5m", "ret_10m", "amount_z_20m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["valid_entry_bar", "force_flat_bar", "daily_is_low_liquidity_day", "daily_adj_factor_change", "daily_is_extreme_return_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    d = d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")
    d["intraday_move"] = d["close"] / d["day_open"] - 1.0
    d["positive_intraday"] = (d["intraday_move"] > 0).astype(int)
    d["positive_5m"] = (d["ret_5m"] > 0).astype(int)
    d["cat_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_intraday"].transform("mean")
    d["cat_ret5_positive_ratio"] = d.groupby(["trade_time", "t0_category"])["positive_5m"].transform("mean")
    return d


def make_params(args: argparse.Namespace) -> List[ORBParams]:
    if not args.run_grid:
        return [ORBParams(
            opening_end=args.opening_end,
            entry_start=next_minute(args.opening_end),
            lookback_days=args.lookback_days,
            min_relvol_obs=max(5, args.lookback_days // 2),
            rel_open_amount_min=args.rel_open_amount_min,
            top_relvol_n=args.top_relvol_n,
            breakout_buffer=args.breakout_buffer,
            amount_z_min=args.amount_z_min,
            category_strength_min=args.category_strength_min,
            take_profit=args.take_profit,
            hard_stop_loss=args.hard_stop_loss,
            trailing_stop=args.trailing_stop,
            max_hold_bars=args.max_hold_bars,
        )]
    grid = {
        "opening_end": ["09:40", "09:45", "10:00"],
        "lookback_days": [5, 10, 14],
        "rel_open_amount_min": [1.0, 1.3, 1.6],
        "top_relvol_n": [5, 10],
        "breakout_buffer": [0.0, 0.0002, 0.0005],
        "amount_z_min": [0.0, 0.5],
        "category_strength_min": [0.50, 0.60],
        "max_hold_bars": [30, 60],
    }
    keys = list(grid.keys())
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        out.append(ORBParams(
            **kw,
            entry_start=next_minute(kw["opening_end"]),
            min_relvol_obs=max(3, kw["lookback_days"] // 2),
            take_profit=args.take_profit,
            hard_stop_loss=args.hard_stop_loss,
            trailing_stop=args.trailing_stop,
        ))
    return out


def add_orb_features(panel: pd.DataFrame, p: ORBParams) -> pd.DataFrame:
    d = panel.copy()
    or_mask = (d["clock_time"] >= p.opening_start) & (d["clock_time"] <= p.opening_end)
    opening = d[or_mask].copy()
    daily = opening.groupby(["ts_code", "trade_date"], as_index=False).agg(
        or_high=("high", "max"),
        or_low=("low", "min"),
        or_open=("open", "first"),
        or_close=("close", "last"),
        or_amount=("amount", "sum"),
        or_bar_count=("clock_time", "count"),
    )
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["or_range"] = daily["or_high"] / daily["or_low"] - 1.0
    daily["or_return"] = daily["or_close"] / daily["or_open"] - 1.0
    daily["or_amount_ma"] = daily.groupby("ts_code")["or_amount"].transform(
        lambda s: s.shift(1).rolling(p.lookback_days, min_periods=p.min_relvol_obs).mean()
    )
    daily["rel_open_amount"] = daily["or_amount"] / daily["or_amount_ma"].replace(0, np.nan)
    daily["rel_open_amount_rank"] = daily.groupby("trade_date")["rel_open_amount"].rank(ascending=False, method="first")
    daily["is_top_relvol"] = (daily["rel_open_amount_rank"] <= p.top_relvol_n).astype(int)
    d = d.merge(daily, on=["ts_code", "trade_date"], how="left")
    d["orb_upper"] = d["or_high"] * (1.0 + p.breakout_buffer)
    d["orb_lower"] = d["or_low"] * (1.0 - p.breakout_buffer)
    d["price_to_or_high"] = d["close"] / d["orb_upper"] - 1.0
    return d


def scope_mask(d: pd.DataFrame, scope: str) -> pd.Series:
    cat = d["t0_category"].astype(str)
    cat_cn = d["t0_category_cn"].astype(str)
    if scope == "all":
        return pd.Series(True, index=d.index)
    if scope == "no_bond_money":
        return ~(cat.isin(["bond", "money_market"]) | cat_cn.str.contains("债券|货币", regex=True, na=False))
    if scope == "commodity_focus":
        return cat.isin(["gold_commodity"]) | cat_cn.str.contains("黄金|商品|油气|豆粕", regex=True, na=False)
    if scope == "cross_commodity":
        return cat.isin(["cross_border", "gold_commodity"]) | cat_cn.str.contains("跨境|黄金|商品|油气|豆粕", regex=True, na=False)
    raise ValueError(f"unknown scope: {scope}")


def build_signal_mask(d: pd.DataFrame, p: ORBParams, scope: str) -> pd.Series:
    m = (
        scope_mask(d, scope)
        & (d["valid_entry_bar"] == 1)
        & (d["force_flat_bar"] == 0)
        & (d["clock_time"] >= p.entry_start)
        & (d["clock_time"] <= p.entry_end)
        & d["or_high"].notna()
        & d["rel_open_amount"].notna()
        & (d["amount"].fillna(0) > 0)
        & (d["or_amount"].fillna(0) > 0)
        & (d["close"] > d["orb_upper"])
        & ((d["rel_open_amount"] >= p.rel_open_amount_min) | (d["is_top_relvol"] == 1))
        & (d["amount_z_20m"].fillna(0) >= p.amount_z_min)
        & (d["cat_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["cat_ret5_positive_ratio"].fillna(0) >= p.category_strength_min)
        & (d["daily_adj_factor_change"].fillna(0).astype(int) == 0)
        & (d["daily_is_extreme_return_day"].fillna(0).astype(int) == 0)
    )
    if p.require_above_vwap:
        m &= d["close"] > d["intraday_vwap"]
    if p.require_ret5_positive:
        m &= d["ret_5m"].fillna(0) > 0
    if p.signal_rising_edge:
        prev = m.groupby([d["ts_code"], d["trade_date"]]).shift(1).fillna(False)
        m = m & (~prev)
    return m


def simulate_trade(day: pd.DataFrame, signal_pos: int, p: ORBParams) -> Dict:
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
        "gross_ret": exit_price / entry_price - 1.0,
        "holding_bars": int(exit_pos - entry_pos + 1),
        "exit_reason": exit_reason,
        "signal_or_high": float(sig["or_high"]),
        "signal_or_low": float(sig["or_low"]),
        "signal_or_range": float(sig["or_range"]) if np.isfinite(sig["or_range"]) else np.nan,
        "signal_rel_open_amount": float(sig["rel_open_amount"]) if np.isfinite(sig["rel_open_amount"]) else np.nan,
        "signal_relvol_rank": float(sig["rel_open_amount_rank"]) if np.isfinite(sig["rel_open_amount_rank"]) else np.nan,
        "signal_price_to_or_high": float(sig["price_to_or_high"]),
        "signal_amount_z_20m": float(sig["amount_z_20m"]) if np.isfinite(sig["amount_z_20m"]) else np.nan,
        "signal_cat_positive_ratio": float(sig["cat_positive_ratio"]) if np.isfinite(sig["cat_positive_ratio"]) else np.nan,
    }


def generate_trades(d: pd.DataFrame, p: ORBParams, scope: str) -> pd.DataFrame:
    x = d.copy()
    x["_signal"] = build_signal_mask(x, p, scope).astype(int)
    trades = []
    groups = x.groupby(["ts_code", "trade_date"], sort=False)
    total = groups.ngroups
    for i, (_, day) in enumerate(groups, 1):
        if i % 1000 == 0:
            print(f"    simulated ETF-days {i}/{total}")
        day = day.sort_values("trade_time").reset_index(drop=True)
        sig_pos = np.flatnonzero(day["_signal"].to_numpy() == 1)
        for sp in sig_pos:
            tr = simulate_trade(day, int(sp), p)
            if tr:
                tr["scope"] = scope
                for k, v in asdict(p).items():
                    tr[f"param_{k}"] = v
                trades.append(tr)
    return pd.DataFrame(trades)


def apply_portfolio_constraints(trades: pd.DataFrame, p: ORBParams) -> pd.DataFrame:
    if trades.empty:
        return trades
    c = trades.sort_values(["entry_time", "signal_rel_open_amount", "signal_price_to_or_high"], ascending=[True, False, False]).reset_index(drop=True).copy()
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
        open_pos.append({"ts_code": code, "t0_category": cat, "exit_time": exit_t})
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=p.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1
    return pd.DataFrame(accepted)


def split_trades(trades: pd.DataFrame, dev_start: str, dev_end: str, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    t = trades.copy()
    t["trade_date"] = t["trade_date"].astype(str).map(norm_date)
    dev = t[(t["trade_date"] >= dev_start) & (t["trade_date"] <= dev_end)].copy()
    test = t[(t["trade_date"] >= test_start) & (t["trade_date"] <= test_end)].copy()
    return dev, test


def date_range(panel: pd.DataFrame, start: str, end: str) -> List[str]:
    dates = sorted(panel["trade_date"].astype(str).unique())
    return [d for d in dates if start <= d <= end]


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str], p: ORBParams) -> Tuple[Dict, pd.DataFrame]:
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
        s = {"cost_bp": cost_bp, "trade_count": 0, "final_nav": final_nav, "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd, "win_rate": np.nan, "avg_gross_ret": np.nan, "avg_net_ret": np.nan, "profit_factor": np.nan, "avg_holding_bars": np.nan}
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


def objective(perf: Dict) -> float:
    n = perf.get("trade_count", 0)
    if n < 20:
        return -1e9
    avg_net_bp = perf.get("avg_net_ret", np.nan) * 10000.0
    pf = perf.get("profit_factor", np.nan)
    sharpe = perf.get("sharpe", np.nan)
    mdd = abs(perf.get("max_drawdown", 0.0))
    if not np.isfinite(avg_net_bp) or not np.isfinite(pf) or not np.isfinite(sharpe):
        return -1e9
    return avg_net_bp + 0.30 * min(pf, 3.0) + 0.05 * min(sharpe, 5.0) - 10.0 * mdd


def category_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby(["scope", "t0_category_cn"], dropna=False).agg(
        trade_count=("gross_ret", "size"),
        avg_gross_ret=("gross_ret", "mean"),
        win_rate=("gross_ret", lambda s: (s > 0).mean()),
        profit_factor=("gross_ret", profit_factor),
        avg_holding_bars=("holding_bars", "mean"),
    ).reset_index().sort_values(["scope", "avg_gross_ret"], ascending=[True, False])


def main() -> int:
    args = parse_args()
    dev_start, dev_end = norm_date(args.dev_start), norm_date(args.dev_end)
    test_start, test_end = norm_date(args.test_start), norm_date(args.test_end)
    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]
    scopes = [s.strip() for s in str(args.scopes).split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("ETF Opening Range Breakout backtest")
    print("=" * 100)
    print(f"panel_file : {Path(args.panel_file).resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"dev        : {dev_start} -> {dev_end}")
    print(f"test       : {test_start} -> {test_end}")
    print(f"scopes     : {scopes}")
    print(f"run_grid   : {args.run_grid}")
    print("=" * 100)
    panel = read_panel(Path(args.panel_file))
    dev_dates = date_range(panel, dev_start, dev_end)
    test_dates = date_range(panel, test_start, test_end)
    params_list = make_params(args)
    cache: Dict[Tuple[str, int, float, int], pd.DataFrame] = {}
    best = None
    best_score = -1e18
    grid_rows = []
    for idx, p in enumerate(params_list, 1):
        key = (p.opening_end, p.lookback_days, p.rel_open_amount_min, p.top_relvol_n)
        if key not in cache:
            print(f"Precomputing ORB features for key={key}")
            cache[key] = add_orb_features(panel, p)
        d = cache[key]
        for scope in scopes:
            print(f"\n[{idx}/{len(params_list)}] scope={scope}, params={asdict(p)}")
            raw = generate_trades(d, p, scope)
            trades = apply_portfolio_constraints(raw, p)
            dev_tr, test_tr = split_trades(trades, dev_start, dev_end, test_start, test_end)
            dev_perf, _ = performance(dev_tr, args.main_cost_bp, dev_dates, p)
            score = objective(dev_perf)
            row = asdict(p)
            row.update({"param_index": idx, "scope": scope, "dev_objective": score, "raw_trade_count": len(raw), "accepted_trade_count": len(trades), "dev_trade_count": len(dev_tr), "test_trade_count": len(test_tr)})
            for k2, v2 in dev_perf.items():
                row[f"dev_main_{k2}"] = v2
            grid_rows.append(row)
            print(f"  raw={len(raw):,}, accepted={len(trades):,}, dev={len(dev_tr):,}, test={len(test_tr):,}, score={score:.4f}")
            if score > best_score:
                best_score = score
                best = {"scope": scope, "params": p, "raw": raw, "trades": trades, "dev_trades": dev_tr, "test_trades": test_tr, "score": score}
    if best is None:
        raise RuntimeError("No valid result.")
    p = best["params"]
    best_scope = best["scope"]
    trades = best["trades"]
    dev_trades = best["dev_trades"]
    test_trades = best["test_trades"]
    grid_df = pd.DataFrame(grid_rows).sort_values("dev_objective", ascending=False)
    grid_df.to_csv(out_dir / "orb_param_grid_summary.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    nav_parts = []
    for period, tr, dates in [("dev", dev_trades, dev_dates), ("test", test_trades, test_dates)]:
        for c in cost_bps:
            s, daily = performance(tr, c, dates, p)
            s["period"] = period
            s["scope"] = best_scope
            summary_rows.append(s)
            daily["period"] = period
            daily["cost_bp"] = c
            daily["scope"] = best_scope
            nav_parts.append(daily)
    summary = pd.DataFrame(summary_rows)
    nav = pd.concat(nav_parts, ignore_index=True)
    cat_sum = category_summary(trades)
    trades.to_csv(out_dir / "orb_trades.csv", index=False, encoding="utf-8-sig")
    dev_trades.to_csv(out_dir / "orb_dev_trades.csv", index=False, encoding="utf-8-sig")
    test_trades.to_csv(out_dir / "orb_test_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "orb_summary_by_scope_period_cost.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(out_dir / "orb_daily_nav_by_scope_period_cost.csv", index=False, encoding="utf-8-sig")
    if not cat_sum.empty:
        cat_sum.to_csv(out_dir / "orb_category_summary.csv", index=False, encoding="utf-8-sig")
    best_info = {"best_scope": best_scope, "best_params": asdict(p), "dev_objective": best_score, "dev_start": dev_start, "dev_end": dev_end, "test_start": test_start, "test_end": test_end, "main_cost_bp": args.main_cost_bp, "run_grid": bool(args.run_grid)}
    with open(out_dir / "orb_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    lines = []
    lines.append("# T+0 ETF Opening Range Breakout 回测报告\n")
    lines.append("## 1. 策略逻辑\n")
    lines.append("使用开盘区间高点作为突破边界，并用开盘区间相对成交额识别当日活跃 ETF。突破 OR_high、站上 VWAP、同类 ETF 同步走强后，下一根 1min bar 开多，并通过 trailing stop、VWAP stop、hard stop、take profit、max hold 和收盘前强平退出。")
    lines.append("")
    lines.append("## 2. 样本切分\n")
    lines.append(f"- dev: {dev_start} ~ {dev_end}")
    lines.append(f"- test: {test_start} ~ {test_end}")
    lines.append(f"- main selection cost: {args.main_cost_bp}bp 单边")
    lines.append("")
    lines.append("## 3. 最优 scope 与参数\n")
    lines.append(f"- best scope: **{best_scope}**")
    lines.append(pd.DataFrame([asdict(p)]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. dev/test 成本敏感性\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    if not cat_sum.empty:
        lines.append("## 5. 类别归因\n")
        lines.append(cat_sum.to_markdown(index=False))
        lines.append("")
    lines.append("## 6. 解释\n")
    lines.append("- 如果 test 期 2bp/3bp 仍为正，说明 ORB 在当前 ETF 池上有进一步研究价值。")
    lines.append("- 如果只在 0bp/1bp 为正，则策略边际仍偏薄。")
    lines.append("- 如果 best scope 收缩到 commodity_focus，说明策略适用范围应限定在更有日内波动和趋势延续的 ETF 类别。")
    (out_dir / "orb_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 100)
    print("Finished")
    print(f"best scope : {best_scope}")
    print(f"summary    : {(out_dir / 'orb_summary_by_scope_period_cost.csv').resolve()}")
    print(f"trades     : {(out_dir / 'orb_trades.csv').resolve()}")
    print(f"best       : {(out_dir / 'orb_best_params.json').resolve()}")
    print(f"report     : {(out_dir / 'orb_report.md').resolve()}")
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
