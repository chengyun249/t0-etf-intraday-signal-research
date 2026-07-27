"""Conservative acceptance audit for the current fixed-combination ETF test."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "data_t0/backtest_combined_orb_noise_rv_2025"
TRADES_PATH = RESULT_DIR / "combo_test_trades.csv"
PANEL_PATH = ROOT / "data_t0/processed/t0_intraday_bar_panel.parquet"
OUT_CSV = RESULT_DIR / "combo_acceptance_diagnostics.csv"
OUT_MD = ROOT / "docs/strategy_acceptance_review.md"
POSITION_WEIGHT = 0.20
BASE_COST_BP = 2.0
N_BOOT = 20_000
BLOCK_DAYS = 5
RNG_SEED = 20260727


def circular_block_bootstrap_mean(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    n = len(x)
    blocks = int(np.ceil(n / BLOCK_DAYS))
    starts = rng.integers(0, n, size=(N_BOOT, blocks))
    idx = (starts[..., None] + np.arange(BLOCK_DAYS)) % n
    return x[idx.reshape(N_BOOT, -1)[:, :n]].mean(axis=1)


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)
    best_info = json.loads((RESULT_DIR / "combo_best_params.json").read_text(encoding="utf-8"))
    trades["trade_date"] = trades["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    start, end = trades["trade_date"].min(), trades["trade_date"].max()
    dates = pd.read_parquet(PANEL_PATH, columns=["trade_date"])["trade_date"]
    dates = dates.astype(str).str.replace("-", "", regex=False).str[:8]
    dates = sorted(dates[(dates >= start) & (dates <= end)].unique())

    rows = []
    daily_for_base = None
    for cost_bp in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
        t = trades.copy()
        t["net_ret"] = t["gross_ret"] - 2.0 * cost_bp / 10_000.0
        daily = (POSITION_WEIGHT * t.groupby("trade_date")["net_ret"].sum()).reindex(dates, fill_value=0.0)
        if cost_bp == BASE_COST_BP:
            daily_for_base = daily
        rows.append({
            "oneway_cost_bp": cost_bp,
            "trade_count": len(t),
            "trade_day_count": int(t["trade_date"].nunique()),
            "calendar_trade_days_in_test": len(dates),
            "avg_trade_net_bp": float(t["net_ret"].mean() * 10_000.0),
            "median_trade_net_bp": float(t["net_ret"].median() * 10_000.0),
            "daily_mean_annualized": float(daily.mean() * 252.0),
            "compounded_test_return": float((1.0 + daily).prod() - 1.0),
        })

    boot = circular_block_bootstrap_mean(daily_for_base.to_numpy(float)) * 252.0
    out = pd.DataFrame(rows)
    out["base_2bp_daily_mean_lower_90"] = np.nan
    out["base_2bp_daily_mean_lower_95"] = np.nan
    out["base_2bp_probability_daily_mean_positive"] = np.nan
    base_idx = out.index[np.isclose(out["oneway_cost_bp"], BASE_COST_BP)][0]
    out.loc[base_idx, "base_2bp_daily_mean_lower_90"] = np.quantile(boot, 0.10)
    out.loc[base_idx, "base_2bp_daily_mean_lower_95"] = np.quantile(boot, 0.05)
    out.loc[base_idx, "base_2bp_probability_daily_mean_positive"] = (boot > 0).mean()
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    base = out.loc[base_idx]
    row3 = out.loc[np.isclose(out["oneway_cost_bp"], 3.0)].iloc[0]
    row5 = out.loc[np.isclose(out["oneway_cost_bp"], 5.0)].iloc[0]
    gates = {
        "开发期存在90%净收益下置信界大于0的候选": bool(best_info.get("deployment_eligible", False)),
        "至少100笔锁定样本外交易": bool(base["trade_count"] >= 100),
        "至少60个发生交易的样本外日期": bool(base["trade_day_count"] >= 60),
        "2bp成本下区块bootstrap 95%下界大于0": bool(base["base_2bp_daily_mean_lower_95"] > 0),
        "3bp单边成本下平均单笔净收益为正": bool(row3["avg_trade_net_bp"] > 0),
        "5bp单边成本压力下平均单笔净收益为正": bool(row5["avg_trade_net_bp"] > 0),
    }
    accepted = all(gates.values())
    lines = [
        "# ETF 日内策略验收审查",
        "",
        "当前固定组合测试只有 28 笔交易。开盘区间与噪声构造的是 `movement_budget_bp`（运动空间），不是条件期望收益。",
        "",
        f"- 开发期最高90%净收益下置信界：{best_info.get('dev_objective', np.nan):.2f}bp；未通过正下界门槛，因此测试结果只作诊断；",
        f"- 2bp 单边成本：平均每笔净收益 {base['avg_trade_net_bp']:.2f}bp；",
        f"- 2bp 下按交易日聚类、5日移动区块 bootstrap 的年化日均收益 95% 下界：{base['base_2bp_daily_mean_lower_95']:.2%}；",
        f"- 3bp 单边成本：平均每笔净收益 {row3['avg_trade_net_bp']:.2f}bp；",
        f"- 5bp 单边成本：平均每笔净收益 {row5['avg_trade_net_bp']:.2f}bp。",
        "",
        "## 验收门槛",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in gates.items())
    lines += [
        "",
        f"结论：**{'可进入新的锁定前瞻测试' if accepted else '仅限研究，不具备部署证据'}**。",
        "即使未来通过，也需要逐笔报价、盘口深度、基金申赎/停牌和券商真实费率验证；分钟 OHLC 不能证明可成交性。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
