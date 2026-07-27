#!/usr/bin/env python
"""Day-blocked, purged ML diagnostics for intraday forward returns.

This is a diagnostic model, not a trading strategy. ETF identifiers are never
used as numeric features, and folds are split by complete trading days so that
overlapping minute labels cannot cross from train to validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from t0_research.validation import PurgedDateSplit


DEFAULT_FEATURES = [
    "ret_5m",
    "ret_10m",
    "ret_20m",
    "price_to_vwap",
    "price_to_vwap_z",
    "amount_z_20m",
    "realized_vol_20m",
    "position_in_day_range",
    "distance_to_intraday_high",
    "cat_positive_ratio",
    "cat_ret5_positive_ratio",
    "rv_z",
    "rv_rank_pct",
    "minutes_to_close",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-file", default="./data_t0/processed/t0_intraday_bar_panel.parquet")
    parser.add_argument("--out-dir", default="./results/ml_purged")
    parser.add_argument("--horizon-minutes", type=int, default=15)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--max-rows-per-day", type=int, default=3000)
    return parser.parse_args()


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(spearmanr(y_true[valid], y_pred[valid]).statistic) if valid.sum() >= 10 else np.nan


def main() -> int:
    args = parse_args()
    panel_path = Path(args.panel_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = f"future_ret_{args.horizon_minutes}m"

    panel = pd.read_parquet(panel_path)
    features = [c for c in DEFAULT_FEATURES if c in panel.columns]
    needed = ["trade_date", "trade_time", "ts_code", target] + features
    if target not in panel.columns or not features:
        raise ValueError(f"panel must contain {target} and diagnostic features")
    data = panel[needed].copy()
    data["trade_date"] = data["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    data = data.dropna(subset=[target]).sort_values(["trade_date", "trade_time", "ts_code"])

    # Deterministic within-day thinning limits memory without random minute sampling.
    if args.max_rows_per_day > 0:
        data["_row_in_day"] = data.groupby("trade_date").cumcount()
        day_size = data.groupby("trade_date")["_row_in_day"].transform("max") + 1
        stride = np.ceil(day_size / args.max_rows_per_day).clip(lower=1).astype(int)
        data = data[data["_row_in_day"] % stride == 0].drop(columns="_row_in_day")

    splitter = PurgedDateSplit(
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )
    fold_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data["trade_date"]), start=1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        )
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        daily_frame = pd.DataFrame(
            {"date": test["trade_date"].to_numpy(), "y": test[target].to_numpy(), "pred": pred}
        )
        daily_ic = pd.Series(
            {
                date: rank_ic(group["y"].to_numpy(), group["pred"].to_numpy())
                for date, group in daily_frame.groupby("date")
            },
            dtype=float,
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_start": train["trade_date"].min(),
                "train_end": train["trade_date"].max(),
                "test_start": test["trade_date"].min(),
                "test_end": test["trade_date"].max(),
                "train_days": train["trade_date"].nunique(),
                "test_days": test["trade_date"].nunique(),
                "n_train": len(train),
                "n_test": len(test),
                "pooled_rank_ic": rank_ic(test[target].to_numpy(), pred),
                "mean_daily_rank_ic": daily_ic.mean(),
                "std_daily_rank_ic": daily_ic.std(ddof=1),
                "rmse": mean_squared_error(test[target], pred) ** 0.5,
            }
        )
        part = test[["trade_date", "trade_time", "ts_code", target]].copy()
        part["prediction"] = pred
        part["fold"] = fold
        prediction_parts.append(part)

    if not fold_rows:
        raise RuntimeError("not enough trading days for one purged fold")
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds.to_csv(out_dir / "purged_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(out_dir / "purged_predictions.parquet", index=False)
    summary = {
        "target": target,
        "features": features,
        "explicitly_excluded": ["ts_code_as_numeric_feature"],
        "folds": len(folds),
        "mean_fold_daily_rank_ic": float(folds["mean_daily_rank_ic"].mean()),
        "method": "expanding complete-day folds with purge and embargo",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
