# Data pipeline

1. Discover T+0-eligible ETF candidates from fund metadata.
2. Download all candidates intersecting the research interval; do not require full-sample coverage.
3. Validate duplicate timestamps, OHLC consistency, expected sessions and missing bars.
4. Build point-in-time intraday features and forward-return labels.
5. Apply daily dynamic eligibility using only lagged history.
6. Keep raw data, processed panels and results outside Git.

The current local cache includes a 2025 panel. Reproducing the historical 2022–2024 reports requires the corresponding licensed raw minute data.
