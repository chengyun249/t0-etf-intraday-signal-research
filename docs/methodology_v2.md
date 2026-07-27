# v2 methodology

## Information timing

A signal is evaluated at the close of minute `t` and may enter at the open of minute `t+1`. Signal features may therefore use information completed at `t`. Exit logic at the open or inside minute `t+1` may only use state known by its open; cumulative VWAP is lagged one bar.

## OHLC execution

For a long position, a gap below the stop fills at the bar open. Otherwise an intrabar stop fills at the trigger less the configured adverse slippage. If a bar touches both stop and target, the stop is assumed first. The trailing high-water mark is updated after a bar survives.

This is deliberately conservative but is still not a substitute for quote or order-book replay.

## Dynamic universe

Eligibility on date `d` uses only observations strictly before `d`: prior history count and a shifted rolling median of daily amount. New listings naturally enter after enough history. The downloadable candidate set must still be broad enough; dynamically filtering an ex-post narrow download cannot recover excluded securities.

## Signal semantics and walk-forward selection

`movement_budget_bp` is a deterministic function of the opening range, historical noise boundary and breakout distance. It is a volatility/range budget, not an estimate of conditional expected return. Its only valid interpretation is whether the observed movement scale is large relative to an assumed cost floor.

The daily walk-forward implementation now defaults to weekly parameter updates. A candidate needs at least 30 historical trades across 20 trading days; an ETF needs at least 10 trades, non-negative average net return and profit factor of at least one. Candidate ranking uses a one-sided 90% lower confidence bound for average net return, with uncertainty clustered by trade date. These controls reduce selection noise but do not remove multiple-testing risk.

## Model validation

`PurgedDateSplit` creates expanding folds of complete trading dates with a purge before each validation block and an embargo before the next fold. ETF identifiers are excluded from numeric features. Metrics are reported by fold and by day to avoid presenting overlapping minutes as independent evidence.
