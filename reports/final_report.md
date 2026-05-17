# Final Report: Minute-Level ETF Intraday Quantitative Research

## 1. Project Positioning

本项目基于 2022-2024 年 A 股 T+0 ETF 1 分钟行情数据，研究短周期收益预测、下行风险识别、日内动量规则和交易成本敏感性。

项目定位为分钟级日内量化研究框架，而非可直接实盘部署的低延迟高频交易系统。

## 2. Data

- Asset universe: China T+0 ETF candidate pool, with commodity-focused subset experiments
- Frequency: 1-minute bar
- Sample period: 2022-2024
- Main panel: approximately 3.97 million rows x 69 feature columns
- Raw data and full parquet panels are not included in this repository

## 3. Strategies and Models

### 3.1 Adaptive Intraday Momentum

- Multi-horizon momentum signal
- Volatility and volume filters
- Adaptive exit logic
- Cost sensitivity tested at 0bp / 5bp / 10bp / 20bp

### 3.2 Combined ORB / Noise / RV Signal

- Opening Range Breakout
- Noise filtering
- Realized volatility regime filter
- Development/test split and walk-forward validation

### 3.3 LightGBM Direction Classification

- Task: direction or downside-risk classification
- Use case: identifying unfavorable trading environments
- Main interpretation: downside-risk detection is more informative than pure positive-return prediction

### 3.4 LightGBM Return Regression

- Task: future return prediction at 15min / 30min / 60min horizons
- Main interpretation: direct short-horizon positive-return prediction is weak and unstable after costs

## 4. Main Findings

1. Short-horizon ETF positive returns are difficult to predict robustly from 1-minute OHLCV data alone.
2. Downside-risk detection is more informative than direct positive-return prediction.
3. Transaction costs are decisive for minute-level long-only strategies.
4. The T+0 ETF universe is relatively small, limiting cross-sectional diversity for machine learning models.
5. Walk-forward validation helps reduce test-period tuning, but it does not fully solve signal decay.

## 5. Project Decision

The legacy return-prediction experiments are retained as negative research evidence. The project is finally positioned as:

> Minute-level ETF intraday signal research and transaction-cost sensitivity analysis.

This positioning is more honest and more useful than presenting the project as a mature high-frequency trading strategy.

## 6. Detailed Reports

- Panel audit: [panel_audit_report.md](panel_audit_report.md)
- Adaptive momentum backtest: [adaptive_momentum_report.md](adaptive_momentum_report.md)
- Commodity adaptive momentum: [adaptive_momentum_commodity_report.md](adaptive_momentum_commodity_report.md)
- Combined strategy dev/test: [combo_dev_test_report.md](combo_dev_test_report.md)
- Walk-forward backtest: [combo_walk_forward_report.md](combo_walk_forward_report.md)
- Signal diagnostics: [signal_diagnostics_report.md](signal_diagnostics_report.md)
- ML direction model h15: [ml_direction_h15_report.md](ml_direction_h15_report.md)
- ML return model h15: [ml_return_h15_report.md](ml_return_h15_report.md)
- ML return model h30: [ml_return_h30_report.md](ml_return_h30_report.md)
- ML return model h60: [ml_return_h60_report.md](ml_return_h60_report.md)

## 7. Key Figures

- Cost sensitivity: [fig_cost_sensitivity.png](figures/fig_cost_sensitivity.png)
- Combined signal dev/test cost stress: [fig_combo_dev_test_cost_sensitivity.png](figures/fig_combo_dev_test_cost_sensitivity.png)
- Signal forward returns: [fig_signal_forward_returns.png](figures/fig_signal_forward_returns.png)
- ML prediction diagnostics: [fig_ml_prediction_diagnostics.png](figures/fig_ml_prediction_diagnostics.png)
- ML feature importance: [fig_ml_feature_importance.png](figures/fig_ml_feature_importance.png)
