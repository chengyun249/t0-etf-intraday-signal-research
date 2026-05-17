# Experiment Log

## V1: Baseline Intraday Momentum

- Content: built a simple momentum signal from 1-minute returns and ran fixed-parameter backtests.
- Result: weak positive performance before costs, but performance deteriorated after transaction costs.
- Lesson: pure momentum is not enough; filters and cost control are necessary.

## V2: Adaptive Momentum and Cost Sensitivity

- Content: added volatility filters, amount z-score filters, and adaptive stop-loss / take-profit logic.
- Result: full-universe no-cost Sharpe was modest; commodity ETF subset performed relatively better.
- Lesson: transaction-cost sensitivity is a core problem for minute-level ETF strategies.

## V3: Combined ORB / Noise / RV Signal

- Content: combined opening range breakout, noise filtering, and realized volatility regime filtering.
- Result: development-period performance was acceptable, but test-period performance decayed.
- Lesson: walk-forward validation helps reduce overfitting but does not eliminate signal decay.

## V4: LightGBM Direction Classification

- Content: used LightGBM to predict short-horizon direction and downside risk.
- Result: downside-risk AUC was around 0.62; direct direction prediction remained weak.
- Lesson: minute-level ETF data is more useful for identifying adverse environments than for predicting positive returns.

## V5: LightGBM Return Regression

- Content: used LightGBM regression to predict 15min / 30min / 60min forward returns.
- Result: out-of-sample prediction correlation was very low, around 0.019 in the legacy experiment.
- Lesson: direct short-horizon positive-return prediction is not robust in this setup.

## Project Decision

After V5, the project stopped positioning itself as a short-horizon return-prediction strategy and was reframed as:

> Minute-level ETF intraday signal research and transaction-cost sensitivity analysis.
