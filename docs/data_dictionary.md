# Data Dictionary

## Raw 1-Minute Bar Fields

| Field | Type | Description |
|---|---|---|
| `ts_code` | string | ETF code, such as `159920.SZ` |
| `trade_time` | datetime | minute-level trading timestamp |
| `trade_date` | string | trading date, `YYYY-MM-DD` |
| `clock_time` | string | intraday clock time, `HH:MM` |
| `open` | float | open price |
| `high` | float | high price |
| `low` | float | low price |
| `close` | float | close price |
| `vol` | float | trading volume |
| `amount` | float | trading amount |

## Processed Panel Key Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | ETF name |
| `t0_category` | string | T+0 category, such as commodity, cross-border, bond |
| `t0_category_cn` | string | Chinese category name |
| `bar_index` | integer | intraday bar index |
| `day_open` | float | daily open price |
| `day_close` | float | daily close price |
| `intraday_move` | float | intraday price move |
| `intraday_vwap` | float | intraday VWAP |
| `ret_1m` / `ret_3m` / `ret_5m` / `ret_10m` / `ret_20m` / `ret_30m` / `ret_60m` | float | multi-horizon returns |
| `ret_1m_vol_5m` / `ret_1m_vol_10m` / `ret_1m_vol_20m` / `ret_1m_vol_60m` | float | realized volatility features |
| `amount_z_20m` | float | 20-minute amount z-score |
| `breakout_high_10m` / `breakout_high_20m` / `breakout_high_30m` / `breakout_high_60m` | boolean | breakout above recent high |
| `breakdown_low_10m` / `breakdown_low_20m` / `breakdown_low_30m` / `breakdown_low_60m` | boolean | breakdown below recent low |
| `valid_entry_bar` | boolean | whether the bar is eligible for entry |
| `force_flat_bar` | boolean | whether the strategy should force flat |
| `daily_ret` | float | daily return |
| `day_amount` | float | daily total amount |
| `zero_amount_bars` | integer | number of zero-amount bars in the day |
| `daily_is_low_liquidity_day` | boolean | low-liquidity day flag |
| `daily_is_extreme_return_day` | boolean | extreme-return day flag |
| `positive_intraday` | boolean | positive intraday return flag |
| `positive_5m` | boolean | positive future 5-minute return flag |
| `rv_z` | float | realized volatility z-score |
| `rv_rank_pct` | float | realized volatility percentile rank |

完整特征清单参见：

```text
reports/tables/t0_feature_manifest.json
```
