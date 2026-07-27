# T+0 ETF 1min Bar 面板构建报告

## 1. 输入

- raw_dir: `.\data_t0\raw_1min\freq=1min`
- universe_file: `.\data_t0\config\t0_etf_selected_30_detail_final.csv`
- daily_regime_file: `.\data_t0\auxiliary\daily_regime_features.csv`

## 2. 原始数据审计

- rows: **1756890**
- etf_count: **30**
- trading_days: **243**
- etf_day_count: **7290**
- duplicate_rows: **0**
- out_of_session_rows: **0**
- bad_ohlc_rows: **0**
- non_positive_price_rows: **0**
- negative_vol_amount_rows: **0**
- incomplete_etf_days: **0**
- zero_amount_etf_days: **10**
- median_rows_per_day: **241.0**
- min_rows_per_day: **241**
- max_rows_per_day: **241**

## 3. 输出面板规模

- bar rows: **1,756,890**
- ETF 数量: **30**
- 交易日数量: **243**
- 特征数量: **88**
- 标签数量: **15**
- valid_entry_bar 数量: **1464486**

## 4. 标签覆盖率

| label          |   non_missing_ratio |         mean |         std |
|:---------------|--------------------:|-------------:|------------:|
| future_ret_1m  |            0.995851 |  1.00888e-07 | 0.000987245 |
| future_ret_3m  |            0.987552 | -6.44842e-07 | 0.00149307  |
| future_ret_5m  |            0.979253 | -3.34942e-06 | 0.00185873  |
| future_ret_10m |            0.958506 | -1.04293e-05 | 0.00250923  |
| future_ret_15m |            0.937759 | -1.74776e-05 | 0.00300531  |
| future_ret_30m |            0.875519 | -4.61527e-05 | 0.00420288  |
| future_ret_60m |            0.751037 | -0.000129841 | 0.00604838  |

## 5. 特征组

- short_momentum: 7
- vwap_deviation: 7
- intraday_position: 4
- breakout_breakdown: 8
- amount_state: 22
- volatility_state: 10
- time_features: 8
- cross_sectional: 20
- daily_regime: 12

## 6. 可疑交易日

| ts_code   |   trade_date |   rows | start_time   | end_time   |   first_close |   last_close |   day_amount |   day_vol |   max_abs_1m_ret |   is_incomplete_day |   is_zero_amount_day |   is_suspect_day |
|:----------|-------------:|-------:|:-------------|:-----------|--------------:|-------------:|-------------:|----------:|-----------------:|--------------------:|---------------------:|-----------------:|
| 159518.SZ |     20250116 |    241 | 09:30        | 15:00      |         1.166 |        1.166 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159518.SZ |     20250124 |    241 | 09:30        | 15:00      |         1.174 |        1.174 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159529.SZ |     20250110 |    241 | 09:30        | 15:00      |         1.899 |        1.899 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159529.SZ |     20250117 |    241 | 09:30        | 15:00      |         1.76  |        1.76  |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159529.SZ |     20250122 |    241 | 09:30        | 15:00      |         1.906 |        1.906 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159529.SZ |     20250124 |    241 | 09:30        | 15:00      |         1.949 |        1.949 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159612.SZ |     20250110 |    241 | 09:30        | 15:00      |         1.87  |        1.87  |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 159612.SZ |     20250124 |    241 | 09:30        | 15:00      |         1.89  |        1.89  |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 513030.SH |     20250123 |    241 | 09:30        | 15:00      |         1.815 |        1.815 |            0 |         0 |                0 |                   0 |                    1 |                1 |
| 513350.SH |     20250124 |    241 | 09:30        | 15:00      |         1.222 |        1.222 |            0 |         0 |                0 |                   0 |                    1 |                1 |

## 7. 后续使用原则

- `valid_entry_bar=1` 才允许策略开仓。
- 复权因子变化日、极端收益日不作为开仓日。
- 均值回归和突破策略都应使用下一根 bar 成交，不能使用信号 bar 的 close 直接成交。
- 第一版主标签建议看 `future_ret_30m`，对应未来 30 分钟收益。