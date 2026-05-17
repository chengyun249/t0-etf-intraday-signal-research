#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_intraday_lgbm_direction_model.py

目标：
    不再手写 ORB/Noise/adaptive short_score 规则，而是用 1min OHLCV 派生特征训练
    分钟级短周期方向预测模型。

核心问题：
    当前 bar 已经结束后，只使用当前及之前可见的分钟信息，预测：
        下一根 1min open 买入后，未来 horizon 分钟收益是否超过阈值。

默认标签：
    y = 1 if future_ret_horizon_bp > +4bp else 0

默认交易：
    如果 P(up) >= 阈值，则下一根 1min open 买入；
    固定持有 horizon 分钟，用 horizon 位置 close 卖出；
    不做复杂止盈止损，先验证模型信号本身是否有交易价值。

默认路径：
    python ".\\train_intraday_lgbm_direction_model.py" `
      --root-dir ".\\data_t0_2022_2024" `
      --scope commodity_focus

建议先跑 commodity_focus；如果结果好，再跑 all。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    horizon_bars: int = 15
    label_threshold_bp: float = 4.0

    train_start: str = "20220101"
    train_end: str = "20231231"
    test_start: str = "20240101"
    test_end: str = "20241231"

    entry_start: str = "09:40"
    entry_end: str = "14:15"
    force_flat_time: str = "14:55"

    max_train_rows: int = 800000
    random_state: int = 42

    position_weight: float = 0.20
    max_positions: int = 3
    max_total_exposure: float = 1.00
    same_category_max_open: int = 2
    etf_daily_trade_limit: int = 2
    cooldown_minutes: int = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train intraday ML direction model on ETF 1min panel.")

    p.add_argument("--root-dir", type=str, default="./data_t0_2022_2024")
    p.add_argument("--panel-file", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")

    p.add_argument("--scope", type=str, default="commodity_focus",
                   choices=["commodity_focus", "cross_commodity", "no_bond_money", "all"])

    p.add_argument("--horizon-bars", type=int, default=15)
    p.add_argument("--label-threshold-bp", type=float, default=4.0)

    p.add_argument("--train-start", type=str, default="20220101")
    p.add_argument("--train-end", type=str, default="20231231")
    p.add_argument("--test-start", type=str, default="20240101")
    p.add_argument("--test-end", type=str, default="20241231")

    p.add_argument("--entry-start", type=str, default="09:40")
    p.add_argument("--entry-end", type=str, default="14:15")
    p.add_argument("--cost-bps", type=str, default="0,1,2,3,5,10")
    p.add_argument("--prob-thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")

    p.add_argument("--max-train-rows", type=int, default=800000)
    p.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "histgb"],
                   help="默认 lightgbm；若环境没有 lightgbm，可用 histgb。")

    p.add_argument("--position-weight", type=float, default=0.20)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--same-category-max-open", type=int, default=2)
    p.add_argument("--etf-daily-trade-limit", type=int, default=2)
    p.add_argument("--cooldown-minutes", type=int, default=10)

    return p.parse_args()


def norm_date(x) -> str:
    return str(x).replace("-", "")[:8]


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.root_dir)
    panel = Path(args.panel_file) if args.panel_file else root / "processed" / "t0_intraday_bar_panel.parquet"
    out = Path(args.out_dir) if args.out_dir else root / f"ml_intraday_direction_{args.scope}_h{args.horizon_bars}"
    return panel, out


def scope_mask(d: pd.DataFrame, scope: str) -> pd.Series:
    cat = d["t0_category"].astype(str)
    cat_cn = d["t0_category_cn"].astype(str)

    if scope == "all":
        return pd.Series(True, index=d.index)
    if scope == "no_bond_money":
        return ~(cat.isin(["bond", "money_market"]) | cat_cn.str.contains("债券|货币", regex=True, na=False))
    if scope == "commodity_focus":
        return cat.isin(["gold_commodity"]) | cat_cn.str.contains("黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    if scope == "cross_commodity":
        return cat.isin(["cross_border", "gold_commodity"]) | cat_cn.str.contains("跨境|黄金|商品|油气|豆粕|有色|能源", regex=True, na=False)
    raise ValueError(scope)


def read_panel(path: Path, scope: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"panel-file 不存在：{path}")

    needed = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_time", "trade_date", "clock_time", "bar_index",
        "open", "high", "low", "close", "vol", "amount",
        "intraday_vwap", "valid_entry_bar", "force_flat_bar",
        "day_open", "daily_ret",
    ]

    import pyarrow.parquet as pq
    cols = pq.read_schema(path).names
    read_cols = [c for c in needed if c in cols]
    d = pd.read_parquet(path, columns=read_cols)

    for c in needed:
        if c not in d.columns:
            if c in ["name", "t0_category", "t0_category_cn"]:
                d[c] = ""
            else:
                d[c] = np.nan

    d["trade_time"] = pd.to_datetime(d["trade_time"])
    d["trade_date"] = d["trade_date"].astype(str).map(norm_date)
    d["ts_code"] = d["ts_code"].astype(str)
    d["name"] = d["name"].fillna("").astype(str)
    d["t0_category"] = d["t0_category"].fillna("unknown").astype(str)
    d["t0_category_cn"] = d["t0_category_cn"].fillna(d["t0_category"]).astype(str)

    d = d[scope_mask(d, scope)].copy()

    for c in ["open", "high", "low", "close", "vol", "amount", "intraday_vwap", "day_open", "daily_ret"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["valid_entry_bar", "force_flat_bar"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    if d["bar_index"].isna().all():
        d["bar_index"] = d.groupby(["ts_code", "trade_date"]).cumcount()
    d["bar_index"] = pd.to_numeric(d["bar_index"], errors="coerce").astype(int)

    if d["day_open"].isna().all():
        d["day_open"] = d.groupby(["ts_code", "trade_date"])["open"].transform("first")

    if d["intraday_vwap"].isna().all():
        tmp_vol = d["vol"].clip(lower=0)
        cum_vol = tmp_vol.groupby([d["ts_code"], d["trade_date"]]).cumsum()
        cum_cv = (d["close"] * tmp_vol).groupby([d["ts_code"], d["trade_date"]]).cumsum()
        d["intraday_vwap"] = cum_cv / cum_vol.replace(0, np.nan)
        d["intraday_vwap"] = d["intraday_vwap"].fillna(
            d.groupby(["ts_code", "trade_date"])["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
        )

    return d.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def rolling_z(s: pd.Series, win: int, minp: int | None = None) -> pd.Series:
    if minp is None:
        minp = max(5, win // 2)
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd.replace(0, np.nan)


def add_daily_long_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["ret_1m_tmp"] = x.groupby(["ts_code", "trade_date"])["close"].pct_change()

    daily = x.groupby(["ts_code", "trade_date"], as_index=False).agg(
        day_amount=("amount", "sum"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_open=("open", "first"),
        day_close=("close", "last"),
        rv_1m_day=("ret_1m_tmp", "std"),
    )
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["day_ret"] = daily.groupby("ts_code")["day_close"].pct_change()
    daily["day_range"] = daily["day_high"] / daily["day_low"] - 1.0
    daily["abs_day_ret"] = daily["day_ret"].abs()
    daily["trendiness"] = daily["abs_day_ret"] / daily["day_range"].replace(0, np.nan)

    g = daily.groupby("ts_code", group_keys=False)
    for win in [5, 10, 20, 30]:
        daily[f"hist_amount_med_{win}d"] = g["day_amount"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_rv_med_{win}d"] = g["rv_1m_day"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_range_med_{win}d"] = g["day_range"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())
        daily[f"hist_trend_med_{win}d"] = g["trendiness"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 2)).median())

    keep = ["ts_code", "trade_date"] + [c for c in daily.columns if c.startswith("hist_")]
    x = x.merge(daily[keep], on=["ts_code", "trade_date"], how="left")
    return x.drop(columns=["ret_1m_tmp"])


def add_intraday_features(d: pd.DataFrame, cfg: ModelConfig) -> Tuple[pd.DataFrame, List[str]]:
    x = d.copy().sort_values(["ts_code", "trade_date", "trade_time"]).reset_index(drop=True)
    gb = x.groupby(["ts_code", "trade_date"], group_keys=False)

    x["log_close"] = np.log(x["close"].replace(0, np.nan))
    x["ret_1m"] = gb["close"].pct_change(1)

    ret_windows = [2, 3, 5, 10, 15, 20, 30]
    for n in ret_windows:
        x[f"ret_{n}m"] = gb["close"].pct_change(n)
        x[f"logret_{n}m"] = gb["log_close"].diff(n)

    # return acceleration / curvature
    x["acc_ret_1_3"] = x["ret_1m"] - x["ret_3m"] / 3.0
    x["acc_ret_3_10"] = x["ret_3m"] / 3.0 - x["ret_10m"] / 10.0
    x["acc_ret_5_20"] = x["ret_5m"] / 5.0 - x["ret_20m"] / 20.0
    x["slope_5"] = x["logret_5m"] / 5.0
    x["slope_10"] = x["logret_10m"] / 10.0
    x["slope_20"] = x["logret_20m"] / 20.0
    x["curvature_5_20"] = x["slope_5"] - x["slope_20"]

    # realised volatility and expansion
    for n in [5, 10, 20, 30]:
        x[f"rv_{n}m"] = gb["ret_1m"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).std())
        x[f"range_{n}m"] = (
            gb["high"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).max())
            / gb["low"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).min())
            - 1.0
        )

    x["rv_ratio_5_20"] = x["rv_5m"] / x["rv_20m"].replace(0, np.nan)
    x["rv_ratio_10_30"] = x["rv_10m"] / x["rv_30m"].replace(0, np.nan)
    x["range_ratio_5_20"] = x["range_5m"] / x["range_20m"].replace(0, np.nan)

    # amount / volume features
    for n in [5, 10, 20, 30]:
        x[f"amount_sum_{n}m"] = gb["amount"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).sum())
        x[f"amount_z_{n}m"] = gb["amount"].transform(lambda s, n=n: rolling_z(s.fillna(0), n))
    x["amount_acc_5_20"] = x["amount_z_5m"] - x["amount_z_20m"]
    x["amount_acc_10_30"] = x["amount_z_10m"] - x["amount_z_30m"]
    x["price_volume_confirm_5m"] = x["ret_5m"] * x["amount_z_10m"]
    x["price_volume_confirm_10m"] = x["ret_10m"] * x["amount_z_20m"]

    # VWAP features
    x["vwap_gap"] = x["close"] / x["intraday_vwap"] - 1.0
    for n in [1, 3, 5, 10, 20]:
        x[f"vwap_gap_chg_{n}m"] = gb["vwap_gap"].diff(n)
        x[f"vwap_slope_{n}m"] = gb["intraday_vwap"].pct_change(n)

    # price position / high-low
    for n in [10, 20, 30, 60]:
        roll_high = gb["high"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).max())
        roll_low = gb["low"].transform(lambda s, n=n: s.rolling(n, min_periods=max(3, n // 2)).min())
        x[f"pos_{n}m"] = ((x["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)).clip(0, 1)
        x[f"dist_high_{n}m"] = x["close"] / roll_high.replace(0, np.nan) - 1.0
        x[f"dist_low_{n}m"] = x["close"] / roll_low.replace(0, np.nan) - 1.0

    # candle structure
    bar_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = (x["close"] - x["open"]) / bar_range
    x["close_location"] = (x["close"] - x["low"]) / bar_range
    x["upper_shadow_ratio"] = (x["high"] - x[["open", "close"]].max(axis=1)) / bar_range
    x["lower_shadow_ratio"] = (x[["open", "close"]].min(axis=1) - x["low"]) / bar_range
    x["is_green"] = (x["close"] > x["open"]).astype(int)
    x["green_count_3"] = gb["is_green"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    x["green_count_5"] = gb["is_green"].transform(lambda s: s.rolling(5, min_periods=1).sum())

    # new-high counts
    for n in [5, 10, 20]:
        prev_high = gb["high"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=max(3, n // 2)).max())
        x[f"new_high_{n}m"] = (x["close"] > prev_high).astype(int)
        x[f"new_high_count_{n}m"] = gb[f"new_high_{n}m"].transform(lambda s, n=n: s.rolling(n, min_periods=1).sum())

    # intraday position and time
    x["intraday_ret_from_open"] = x["close"] / x["day_open"] - 1.0
    x["minutes"] = pd.to_datetime(x["trade_time"]).dt.hour * 60 + pd.to_datetime(x["trade_time"]).dt.minute
    # trading minutes since 09:30, adjusting lunch roughly
    x["minutes_since_open"] = np.where(
        x["clock_time"] <= "11:30",
        x["minutes"] - (9 * 60 + 30),
        x["minutes"] - (13 * 60) + 121,
    )
    x["time_sin"] = np.sin(2 * np.pi * x["minutes_since_open"] / 241.0)
    x["time_cos"] = np.cos(2 * np.pi * x["minutes_since_open"] / 241.0)

    x["ts_code_code"] = pd.Categorical(x["ts_code"]).codes
    x["category_code"] = pd.Categorical(x["t0_category"]).codes

    # label: signal at row t, entry at t+1 open, exit at t+1+horizon close
    h = cfg.horizon_bars
    x["entry_open_next"] = gb["open"].shift(-1)
    x["future_close_h"] = gb["close"].shift(-(h + 1))
    x["future_time_h"] = gb["trade_time"].shift(-(h + 1))
    x["future_clock_h"] = gb["clock_time"].shift(-(h + 1))

    x["future_ret"] = x["future_close_h"] / x["entry_open_next"] - 1.0
    x["future_ret_bp"] = x["future_ret"] * 10000.0
    x["label_up"] = (x["future_ret_bp"] > cfg.label_threshold_bp).astype(int)

    valid = (
        (x["valid_entry_bar"].fillna(0).astype(int) == 1)
        & (x["clock_time"] >= cfg.entry_start)
        & (x["clock_time"] <= cfg.entry_end)
        & (x["amount"].fillna(0) > 0)
        & (x["entry_open_next"].fillna(0) > 0)
        & (x["future_close_h"].fillna(0) > 0)
        & (x["future_clock_h"].astype(str) < cfg.force_flat_time)
    )
    x["ml_valid_row"] = valid.astype(int)

    feature_cols = [
        # ID / categorical numeric
        "ts_code_code", "category_code", "bar_index", "minutes_since_open", "time_sin", "time_cos",

        # returns
        "ret_1m", "ret_2m", "ret_3m", "ret_5m", "ret_10m", "ret_15m", "ret_20m", "ret_30m",
        "acc_ret_1_3", "acc_ret_3_10", "acc_ret_5_20", "slope_5", "slope_10", "slope_20", "curvature_5_20",

        # volatility
        "rv_5m", "rv_10m", "rv_20m", "rv_30m", "rv_ratio_5_20", "rv_ratio_10_30",
        "range_5m", "range_10m", "range_20m", "range_30m", "range_ratio_5_20",

        # amount
        "amount_z_5m", "amount_z_10m", "amount_z_20m", "amount_z_30m",
        "amount_acc_5_20", "amount_acc_10_30", "price_volume_confirm_5m", "price_volume_confirm_10m",

        # vwap
        "vwap_gap", "vwap_gap_chg_1m", "vwap_gap_chg_3m", "vwap_gap_chg_5m", "vwap_gap_chg_10m", "vwap_gap_chg_20m",
        "vwap_slope_1m", "vwap_slope_3m", "vwap_slope_5m", "vwap_slope_10m", "vwap_slope_20m",

        # price position
        "pos_10m", "pos_20m", "pos_30m", "pos_60m",
        "dist_high_10m", "dist_high_20m", "dist_high_30m", "dist_high_60m",
        "dist_low_10m", "dist_low_20m", "dist_low_30m", "dist_low_60m",

        # candle
        "body_ratio", "close_location", "upper_shadow_ratio", "lower_shadow_ratio",
        "green_count_3", "green_count_5",
        "new_high_count_5m", "new_high_count_10m", "new_high_count_20m",

        # intraday / long state
        "intraday_ret_from_open",
        "hist_amount_med_5d", "hist_amount_med_10d", "hist_amount_med_20d", "hist_amount_med_30d",
        "hist_rv_med_5d", "hist_rv_med_10d", "hist_rv_med_20d", "hist_rv_med_30d",
        "hist_range_med_5d", "hist_range_med_10d", "hist_range_med_20d", "hist_range_med_30d",
        "hist_trend_med_5d", "hist_trend_med_10d", "hist_trend_med_20d", "hist_trend_med_30d",
    ]

    feature_cols = [c for c in feature_cols if c in x.columns]
    return x, feature_cols


def clean_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # 用 0 填充主要是为了树模型可快速运行；长期版本可改为训练集 median。
    return X.fillna(0.0).astype("float32")


def sample_train_data(train: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train

    pos = train[train["label_up"] == 1]
    neg = train[train["label_up"] == 0]

    # 保留尽量多的正类，再抽负类，避免 positive 过少。
    max_pos = min(len(pos), max_rows // 2)
    pos_s = pos.sample(n=max_pos, random_state=random_state) if len(pos) > max_pos else pos
    remaining = max_rows - len(pos_s)
    neg_s = neg.sample(n=min(len(neg), remaining), random_state=random_state)

    out = pd.concat([pos_s, neg_s], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_model(train: pd.DataFrame, feature_cols: List[str], cfg: ModelConfig, model_name: str):
    train_s = sample_train_data(train, cfg.max_train_rows, cfg.random_state)

    X = clean_matrix(train_s, feature_cols)
    y = train_s["label_up"].astype(int)

    pos = int(y.sum())
    neg = int((1 - y).sum())
    scale_pos_weight = neg / max(pos, 1)

    if model_name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=31,
                max_depth=-1,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=5.0,
                min_child_samples=80,
                scale_pos_weight=scale_pos_weight,
                random_state=cfg.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X, y)
            used_model = "lightgbm"
            return model, used_model, train_s
        except Exception as e:
            print(f"[WARN] LightGBM 不可用或训练失败，改用 HistGradientBoostingClassifier。原因：{e}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    sample_weight = np.where(y == 1, scale_pos_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=cfg.random_state,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, "hist_gradient_boosting", train_s


def predict_proba(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = clean_matrix(df, feature_cols)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def model_metrics(df: pd.DataFrame, pred_col: str) -> Dict:
    out = {
        "rows": int(len(df)),
        "positive_rate": float(df["label_up"].mean()) if len(df) else np.nan,
        "mean_future_ret_bp": float(df["future_ret_bp"].mean()) if len(df) else np.nan,
    }
    if len(df) and df["label_up"].nunique() > 1:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            out["auc"] = float(roc_auc_score(df["label_up"], df[pred_col]))
            out["average_precision"] = float(average_precision_score(df["label_up"], df[pred_col]))
        except Exception:
            out["auc"] = np.nan
            out["average_precision"] = np.nan
    else:
        out["auc"] = np.nan
        out["average_precision"] = np.nan
    return out


def probability_quantile_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    x = df.copy()
    try:
        x["prob_bin"] = pd.qcut(x[pred_col], q=10, duplicates="drop")
    except Exception:
        x["prob_bin"] = "all"

    return (
        x.groupby("prob_bin", dropna=False)
        .agg(
            rows=("future_ret_bp", "size"),
            mean_prob=(pred_col, "mean"),
            mean_future_ret_bp=("future_ret_bp", "mean"),
            median_future_ret_bp=("future_ret_bp", "median"),
            win_rate=("future_ret_bp", lambda s: (s > 0).mean()),
            hit_label_rate=("label_up", "mean"),
            p25_bp=("future_ret_bp", lambda s: s.quantile(0.25)),
            p75_bp=("future_ret_bp", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )


def make_signal_trades(test_df: pd.DataFrame, prob_threshold: float, cfg: ModelConfig) -> pd.DataFrame:
    sig = test_df[test_df["pred_up_proba"] >= prob_threshold].copy()
    if sig.empty:
        return pd.DataFrame()

    sig = sig.sort_values(["trade_time", "pred_up_proba"], ascending=[True, False]).reset_index(drop=True)

    trades = pd.DataFrame({
        "ts_code": sig["ts_code"].values,
        "name": sig["name"].values,
        "t0_category": sig["t0_category"].values,
        "t0_category_cn": sig["t0_category_cn"].values,
        "trade_date": sig["trade_date"].values,
        "signal_time": sig["trade_time"].values,
        "signal_clock_time": sig["clock_time"].values,
        "entry_time": sig.groupby(["ts_code", "trade_date"])["trade_time"].shift(-1).values,
    })

    # entry_time 上面按 sig 子集 shift 是错的；改用原 df 已有下一行信息更稳。
    trades["entry_time"] = sig["trade_time"] + pd.Timedelta(minutes=1)
    # 注意：午休附近可能不是实际下一根；价格收益使用已经计算好的 entry_open_next / future_close_h。
    trades["entry_price"] = sig["entry_open_next"].values
    trades["exit_time"] = sig["future_time_h"].values
    trades["exit_price"] = sig["future_close_h"].values
    trades["gross_ret"] = sig["future_ret"].values
    trades["future_ret_bp"] = sig["future_ret_bp"].values
    trades["pred_up_proba"] = sig["pred_up_proba"].values
    trades["label_up"] = sig["label_up"].values
    trades["horizon_bars"] = cfg.horizon_bars
    trades["weight"] = cfg.position_weight
    trades["holding_bars"] = cfg.horizon_bars

    trades = trades.dropna(subset=["entry_price", "exit_price", "gross_ret", "exit_time"])
    trades = trades[(trades["entry_price"] > 0) & (trades["exit_price"] > 0)].copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    return apply_portfolio_constraints(trades, cfg)


def apply_portfolio_constraints(trades: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    if trades.empty:
        return trades

    c = trades.sort_values(["entry_time", "pred_up_proba"], ascending=[True, False]).reset_index(drop=True)

    open_pos = []
    cooldown_until: Dict[str, pd.Timestamp] = {}
    etf_day_count: Dict[Tuple[str, str], int] = {}
    accepted = []

    for _, row in c.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_t = pd.Timestamp(row["exit_time"])
        code = row["ts_code"]
        date = row["trade_date"]
        cat = row["t0_category"]
        weight = float(row["weight"])

        open_pos = [p for p in open_pos if p["exit_time"] > entry]
        current_exposure = sum(p["weight"] for p in open_pos)

        key = (code, date)
        if etf_day_count.get(key, 0) >= cfg.etf_daily_trade_limit:
            continue
        if code in cooldown_until and entry < cooldown_until[code]:
            continue
        if any(p["ts_code"] == code for p in open_pos):
            continue
        if len(open_pos) >= cfg.max_positions:
            continue
        if sum(1 for p in open_pos if p["t0_category"] == cat) >= cfg.same_category_max_open:
            continue
        if current_exposure + weight > cfg.max_total_exposure:
            continue

        accepted.append(row.to_dict())
        open_pos.append({"ts_code": code, "t0_category": cat, "exit_time": exit_t, "weight": weight})
        cooldown_until[code] = exit_t + pd.Timedelta(minutes=cfg.cooldown_minutes)
        etf_day_count[key] = etf_day_count.get(key, 0) + 1

    return pd.DataFrame(accepted)


def profit_factor(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def performance(trades: pd.DataFrame, cost_bp: float, dates: List[str]) -> Tuple[Dict, pd.DataFrame]:
    if trades.empty:
        daily = pd.DataFrame({"trade_date": dates, "daily_ret": 0.0})
    else:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10000.0
        t["pnl"] = t["weight"] * t["net_ret"]
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
            "avg_pred_up_proba": np.nan,
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
            "avg_pred_up_proba": float(trades["pred_up_proba"].mean()),
        }

    return s, daily


def group_summary(trades: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(group_cols, dropna=False)
        .agg(
            trade_count=("gross_ret", "size"),
            avg_gross_ret=("gross_ret", "mean"),
            win_rate=("gross_ret", lambda s: (s > 0).mean()),
            profit_factor=("gross_ret", profit_factor),
            avg_pred_up_proba=("pred_up_proba", "mean"),
        )
        .reset_index()
        .sort_values("avg_gross_ret", ascending=False)
    )


def write_report(out_dir: Path, config: Dict, model_metrics_df: pd.DataFrame,
                 quantile_table: pd.DataFrame, bt_summary: pd.DataFrame,
                 etf_summary: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    lines = []
    lines.append("# Intraday ML Direction Model Report\n")
    lines.append("## 1. 目的\n")
    lines.append("本脚本用分钟级短周期特征训练方向预测模型，预测下一根 open 买入后，未来固定 horizon 的收益是否超过阈值。它不再手写 short_score 规则，而是让模型学习动量、加速度、波动、成交额、VWAP、K线结构和时间状态的非线性关系。")
    lines.append("")
    lines.append("## 2. 配置\n")
    lines.append(pd.DataFrame([config]).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 预测能力\n")
    lines.append(model_metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 4. 测试集概率分层表现\n")
    lines.append(quantile_table.to_markdown(index=False))
    lines.append("")
    lines.append("## 5. 策略回测：阈值 x 成本敏感性\n")
    lines.append(bt_summary.to_markdown(index=False))
    lines.append("")
    if not etf_summary.empty:
        lines.append("## 6. ETF 归因")
        lines.append(etf_summary.head(50).to_markdown(index=False))
        lines.append("")
    if not feature_importance.empty:
        lines.append("## 7. 特征重要性 Top 50")
        lines.append(feature_importance.head(50).to_markdown(index=False))
        lines.append("")
    lines.append("## 8. 读数标准")
    lines.append("- 如果高概率分层的 mean_future_ret_bp 明显高于全样本，说明模型能识别短期优势状态。")
    lines.append("- 如果 P 阈值提高后 avg_gross_ret 上升但交易数下降，说明模型有排序能力。")
    lines.append("- 如果 2bp 单边成本后仍为正，才说明有继续细化执行和风控的价值。")
    (out_dir / "ml_intraday_direction_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    panel_file, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        horizon_bars=args.horizon_bars,
        label_threshold_bp=args.label_threshold_bp,
        train_start=norm_date(args.train_start),
        train_end=norm_date(args.train_end),
        test_start=norm_date(args.test_start),
        test_end=norm_date(args.test_end),
        entry_start=args.entry_start,
        entry_end=args.entry_end,
        max_train_rows=args.max_train_rows,
        position_weight=args.position_weight,
        max_positions=args.max_positions,
        same_category_max_open=args.same_category_max_open,
        etf_daily_trade_limit=args.etf_daily_trade_limit,
        cooldown_minutes=args.cooldown_minutes,
    )

    cost_bps = [float(x) for x in str(args.cost_bps).split(",") if x.strip()]
    prob_thresholds = [float(x) for x in str(args.prob_thresholds).split(",") if x.strip()]

    print("=" * 100)
    print("Train Intraday ML Direction Model")
    print("=" * 100)
    print(f"panel_file : {panel_file.resolve()}")
    print(f"out_dir    : {out_dir.resolve()}")
    print(f"scope      : {args.scope}")
    print(f"horizon    : {cfg.horizon_bars} bars")
    print(f"label      : future_ret > {cfg.label_threshold_bp}bp")
    print("=" * 100)

    print("Reading panel...")
    panel = read_panel(panel_file, args.scope)
    print(f"panel rows after scope: {len(panel):,}, ETFs={panel['ts_code'].nunique()}, days={panel['trade_date'].nunique()}")

    print("Adding daily long-state features...")
    panel = add_daily_long_features(panel)

    print("Adding intraday ML features and labels...")
    data, feature_cols = add_intraday_features(panel, cfg)
    data = data[data["ml_valid_row"] == 1].copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    train = data[(data["trade_date"] >= cfg.train_start) & (data["trade_date"] <= cfg.train_end)].copy()
    test = data[(data["trade_date"] >= cfg.test_start) & (data["trade_date"] <= cfg.test_end)].copy()

    if train.empty or test.empty:
        raise RuntimeError(f"训练集或测试集为空。train={len(train)}, test={len(test)}")

    print(f"valid rows: train={len(train):,}, test={len(test):,}")
    print(f"positive rate: train={train['label_up'].mean():.4f}, test={test['label_up'].mean():.4f}")

    print("Training model...")
    model, used_model, train_sample = train_model(train, feature_cols, cfg, args.model)

    print(f"used model: {used_model}")
    print(f"train sample rows: {len(train_sample):,}")

    print("Predicting train/test...")
    train_eval = train_sample.copy()
    train_eval["pred_up_proba"] = predict_proba(model, train_eval, feature_cols)

    test = test.copy()
    test["pred_up_proba"] = predict_proba(model, test, feature_cols)

    metric_rows = []
    tr_m = model_metrics(train_eval, "pred_up_proba")
    tr_m["period"] = "train_sample"
    te_m = model_metrics(test, "pred_up_proba")
    te_m["period"] = "test"
    metric_rows.extend([tr_m, te_m])
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "ml_prediction_metrics.csv", index=False, encoding="utf-8-sig")

    quantile_table = probability_quantile_table(test, "pred_up_proba")
    quantile_table.to_csv(out_dir / "ml_test_probability_quantiles.csv", index=False, encoding="utf-8-sig")

    # feature importance
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    fi.to_csv(out_dir / "ml_feature_importance.csv", index=False, encoding="utf-8-sig")

    # Save compact test predictions, not all features
    pred_cols = [
        "ts_code", "name", "t0_category", "t0_category_cn",
        "trade_date", "trade_time", "clock_time",
        "entry_open_next", "future_close_h", "future_time_h",
        "future_ret", "future_ret_bp", "label_up", "pred_up_proba",
    ]
    test[pred_cols].to_csv(out_dir / "ml_test_predictions_compact.csv", index=False, encoding="utf-8-sig")

    print("Backtesting probability thresholds...")
    test_dates = sorted(test["trade_date"].unique().tolist())

    bt_rows = []
    all_trades_parts = []
    nav_parts = []

    for th in prob_thresholds:
        trades = make_signal_trades(test, th, cfg)
        if not trades.empty:
            trades["prob_threshold"] = th
            all_trades_parts.append(trades)

        for c in cost_bps:
            s, nav = performance(trades, c, test_dates)
            s["prob_threshold"] = th
            bt_rows.append(s)
            nav["cost_bp"] = c
            nav["prob_threshold"] = th
            nav_parts.append(nav)

    bt_summary = pd.DataFrame(bt_rows)
    bt_summary = bt_summary.sort_values(["cost_bp", "final_nav"], ascending=[True, False])
    bt_summary.to_csv(out_dir / "ml_backtest_summary_by_threshold_cost.csv", index=False, encoding="utf-8-sig")

    daily_nav = pd.concat(nav_parts, ignore_index=True) if nav_parts else pd.DataFrame()
    daily_nav.to_csv(out_dir / "ml_daily_nav_by_threshold_cost.csv", index=False, encoding="utf-8-sig")

    all_trades = pd.concat(all_trades_parts, ignore_index=True) if all_trades_parts else pd.DataFrame()
    all_trades.to_csv(out_dir / "ml_all_threshold_trades.csv", index=False, encoding="utf-8-sig")

    # choose exploratory best at 2bp if available, otherwise first threshold
    if not bt_summary.empty:
        main_cost = 2.0 if 2.0 in cost_bps else cost_bps[0]
        best_row = bt_summary[bt_summary["cost_bp"] == main_cost].sort_values("final_nav", ascending=False).iloc[0]
        best_th = float(best_row["prob_threshold"])
    else:
        best_th = prob_thresholds[0]

    best_trades = make_signal_trades(test, best_th, cfg)
    best_trades.to_csv(out_dir / "ml_best_threshold_trades.csv", index=False, encoding="utf-8-sig")
    etf_sum = group_summary(best_trades, ["ts_code", "name", "t0_category_cn"])
    etf_sum.to_csv(out_dir / "ml_best_threshold_etf_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel_file": str(panel_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "scope": args.scope,
        "model": used_model,
        "horizon_bars": cfg.horizon_bars,
        "label_threshold_bp": cfg.label_threshold_bp,
        "train_start": cfg.train_start,
        "train_end": cfg.train_end,
        "test_start": cfg.test_start,
        "test_end": cfg.test_end,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sample_rows": int(len(train_sample)),
        "train_positive_rate": float(train["label_up"].mean()),
        "test_positive_rate": float(test["label_up"].mean()),
        "prob_thresholds": prob_thresholds,
        "cost_bps": cost_bps,
        "best_threshold_at_2bp_exploratory": best_th,
        "config": asdict(cfg),
    }
    (out_dir / "ml_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(out_dir, config, metrics_df, quantile_table, bt_summary, etf_sum, fi)

    print("\n" + "=" * 100)
    print("Finished")
    print(f"report      : {(out_dir / 'ml_intraday_direction_report.md').resolve()}")
    print(f"summary     : {(out_dir / 'ml_backtest_summary_by_threshold_cost.csv').resolve()}")
    print(f"quantiles   : {(out_dir / 'ml_test_probability_quantiles.csv').resolve()}")
    print(f"predictions : {(out_dir / 'ml_test_predictions_compact.csv').resolve()}")
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
