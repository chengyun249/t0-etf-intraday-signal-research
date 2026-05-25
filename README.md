# A 股 T+0 ETF 分钟级日内信号研究

本项目研究 A 股 T+0 ETF 的分钟级日内交易信号，重点不是展示一个"高收益策略"，而是系统检验：在真实交易约束下，基于 1 分钟 OHLCV 数据构造的日内突破、成交额确认、VWAP、噪声边界、类别共振与相对强弱信号，是否能够形成稳定、可交易的样本外收益。

项目最终结论较为审慎：

> 组合信号在部分商品 / 黄金类 ETF 中能够观察到弱毛收益边际，但该边际非常薄，难以覆盖 1—2bp 单边交易成本。在更严格的 daily walk-forward 检验下，滚动参数选择没有显著提升成本后收益。当前版本更适合作为一个"分钟级信号有效性与交易成本敏感性研究项目"，而不是可直接实盘部署的交易策略。

---

## 1. 研究背景

A 股市场中，普通股票通常实行 T+1 交易制度，但部分 ETF 支持 T+0 回转交易。理论上，T+0 ETF 为日内交易策略提供了更高频的交易空间。

但在实际研究中，分钟级 ETF 策略面临几个核心困难：

1. **单笔收益空间很薄** — 1 分钟或几十分钟级别的价格波动通常只有几个 bp，容易被买卖价差、滑点和冲击成本吞掉。
2. **OHLCV 数据粒度有限** — 本项目使用分钟 K 线数据，而非 tick / order book 数据，因此无法精确刻画盘口排队、真实成交价、买卖价差变化和微观结构冲击。
3. **突破信号容易受噪声干扰** — 简单的日内动量、开盘区间突破或 VWAP 信号在无成本回测中可能有效，但在加入交易成本后往往迅速衰减。
4. **参数选择容易过拟合** — 如果只在固定开发期中寻找最优参数，再在后续样本中测试，仍可能高估策略的真实稳定性。因此，本项目进一步加入 daily walk-forward 检验。

---

## 2. 数据与样本

### 2.1 数据来源

项目使用 Tushare Pro 接口获取 A 股 ETF 分钟级行情数据，并构建 1 分钟 bar 面板。

数据主要包括：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日 |
| `trade_time` | 分钟时间 |
| `ts_code` | ETF 代码 |
| `open` | 当前 bar 开盘价 |
| `high` | 当前 bar 最高价 |
| `low` | 当前 bar 最低价 |
| `close` | 当前 bar 收盘价 |
| `vol` | 当前 bar 成交量 |
| `amount` | 当前 bar 成交额 |
| `name` | ETF 名称 |
| `t0_category` / `t0_category_cn` | T+0 ETF 类别 |

### 2.2 样本范围

样本覆盖 2022—2024 年的 T+0 ETF 分钟级行情。

由于原始分钟数据体积较大，且数据来源受接口权限限制，本仓库不上传原始数据和完整中间结果。仓库中仅保留：核心代码、研究报告、汇总结果表、项目说明文档。

完整数据目录 `data_t0_2022_2024/` 已在 `.gitignore` 中忽略。

---

## 3. 项目目标

本项目主要回答四个问题：

1. 分钟级 T+0 ETF 是否存在可识别的日内方向性信号？
2. ORB、VWAP、噪声边界、成交额放大、类别共振和相对强弱过滤能否改善简单动量信号？
3. 固定组合优化结果在更严格的 daily walk-forward 框架下是否仍能延续？
4. 信号毛收益能否覆盖 1—2bp 单边交易成本？

项目重点不是追求最终收益曲线，而是通过逐层检验说明：

> 1 分钟 OHLCV 级别的 T+0 ETF 日内 alpha 边际较薄；即使经过多层信号过滤和 walk-forward 滚动选择，成本后稳定性仍然不足。

---

## 4. 核心信号框架

项目最终组合信号由以下几类特征共同构成。

### 4.1 Opening Range Breakout（开盘区间突破）

首先定义开盘区间（如 09:30—09:45 或 09:30—10:00），在开盘区间内计算：

```text
OR_high = 开盘区间内 high 的最大值
OR_low  = 开盘区间内 low 的最小值
```

当后续价格突破开盘区间上沿并超过一定 buffer 时，认为价格出现日内向上突破：

```text
close_t > OR_high × (1 + breakout_buffer)
```

### 4.2 Relative Volume（开盘成交额放大）

计算当天开盘区间成交额相对于历史均值的放大倍数：

```text
rel_open_amount = 今日开盘区间成交额 / 过去若干日开盘区间成交额均值
```

若 `rel_open_amount` 较高，说明当天市场关注度或资金参与度高于历史正常水平。

### 4.3 Noise Boundary（噪声边界）

使用历史同一 bar_index 的平均绝对日内波动估计正常噪声范围：

```text
noise_upper = day_open × (1 + noise_k × historical_noise)
```

只有当当前价格突破噪声上边界时，才认为价格波动已经超过该时间点的正常波动范围。

### 4.4 VWAP Confirmation（VWAP 确认）

项目要求 `close_t > intraday_vwap_t`，即当前价格位于日内平均成交成本之上，用于确认多头状态。

### 4.5 Category Breadth（类别同步）

在同类 ETF 中计算：

```text
cat_positive_ratio = 同类 ETF 中日内收益为正的比例
cat_ret5_positive_ratio = 同类 ETF 中最近 5 分钟收益为正的比例
```

如果同一类别中多数 ETF 同向上涨，则说明信号不是单只 ETF 的孤立异动。

### 4.6 Relative Value（相对强弱过滤）

计算同一类别内 ETF 的相对强弱：

```text
rv_z = (当前 ETF 日内涨跌幅 - 同类均值) / 同类标准差
rv_rank_pct = 当前 ETF 在同类 ETF 中的日内涨跌幅排名百分位
```

用于避免买入已经明显过热的标的。

### 4.7 Expected Edge（成本覆盖过滤）

估计信号的粗略预期空间 `expected_edge_bp`，要求其至少达到交易成本的一定倍数：

```text
expected_edge_bp >= expected_edge_mult × assumed_roundtrip_cost_bp
```

如果预期空间不足以覆盖成本，则不触发交易。

---

## 5. 交易执行与退出逻辑

### 5.1 入场

信号在当前 bar 收盘后才能确认，因此用下一根 bar 开盘价入场，避免前视偏差：

```text
signal_t 在 t bar 结束后生成
entry_price = open_{t+1}
```

### 5.2 退出

项目设置多类动态退出条件：

| 退出条件 | 含义 |
|---|---|
| `take_profit` | 达到止盈幅度后退出 |
| `hard_stop_loss` | 达到硬止损幅度后退出 |
| `trailing_stop` | 从持仓后最高点回撤一定幅度后退出 |
| `vwap_stop_band` | 跌破 VWAP 附近后退出 |
| `or_high_stop_band` | 跌回 ORB 突破位附近后退出 |
| `max_hold_bars` | 达到最大持有 bar 数后退出 |
| `force_flat_bar` | 收盘前强制平仓，避免隔夜 |

### 5.3 组合层约束

| 约束 | 含义 |
|---|---|
| `max_positions` | 最大同时持仓数 |
| `same_category_max_open` | 同一类别最大同时持仓数 |
| `position_weight` | 单笔仓位权重 |
| `cooldown_minutes` | 同一 ETF 交易后的冷却时间 |
| `etf_daily_trade_limit` | 单只 ETF 单日最大交易次数 |

---

## 6. 固定组合优化与 Walk-forward 检验

### 6.1 固定组合优化

固定组合优化版本采用较传统的开发期 / 测试期框架：

```text
开发期（2022-2023）：选择参数和策略口径
测试期（2024）：检验固定参数是否延续
```

固定组合优化有助于初步验证信号结构，但仍存在一个问题：参数在开发期中被选出后，在测试期固定使用，不能完全模拟真实交易中不断根据近期市场状态校准参数的过程。

### 6.2 Walk-forward 的设计

Walk-forward 版本每天滚动选择参数：

```text
每个交易日 d：
1. 只使用 d 日之前最多 120 个交易日作为历史窗口；
2. 在历史窗口内评估候选 candidate；
3. 选择历史窗口中综合评分较高的 candidate；
4. 用选出的 candidate 交易 d 日当天；
5. 到下一个交易日继续滚动。
```

当前主线 walk-forward 配置：

| 配置项 | 当前设置 |
|---|---|
| `scope` | `commodity_focus` |
| `grid_size` | `mini` |
| `candidate_count` | 48 |
| `walk_forward_trade_dates` | 666 |
| `train_lookback_days` | 120 |
| `min_train_days` | 60 |
| `update_frequency` | daily |
| `main_cost_bp` | 2bp 单边成本 |

### 6.3 为什么删除 ETF 二次筛选

早期版本曾在选出 candidate 后，再基于单 ETF 历史交易表现进行二次筛选（历史交易次数、平均净收益、profit factor）。后续拆解发现，这一层筛选没有稳定的样本外预测力，反而容易在小样本下把历史偶然性误判为标的优劣，最终削弱策略结果。因此，当前主线版本不再使用 ETF 历史收益二次筛选，仅保留 daily rolling candidate selection。

---

## 7. 最新 Walk-forward 主线结果

当前主线版本为：

```text
commodity_focus + daily rolling candidate selection + no ETF secondary filter
```

### 7.1 成本敏感性

| 单边成本 | 交易数 | Final NAV | Sharpe | 最大回撤 | 胜率 | 平均单笔毛收益 | Profit Factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0bp | 266 | 1.00445 | 0.579 | -0.357% | 40.98% | 0.836bp | 1.162 |
| 1bp | 266 | 0.99382 | -0.831 | -0.830% | 39.10% | 0.836bp | 0.816 |
| 2bp | 266 | 0.98330 | -2.251 | -1.770% | 33.46% | 0.836bp | 0.585 |
| 5bp | 266 | 0.95240 | -5.530 | -4.760% | 22.18% | 0.836bp | 0.236 |
| 10bp | 266 | 0.90304 | -7.471 | -9.696% | 9.77% | 0.836bp | 0.061 |

### 7.2 结果解释

0bp 成本下恢复为弱正收益：Final NAV 1.00445，平均单笔毛收益约 0.836bp，Profit Factor 约 1.162。说明原始组合信号并非完全无效，在 `commodity_focus` 池内仍有一定方向性。

但该边际非常薄：1bp 单边成本后 Final NAV 即跌至 0.99382，2bp 后 0.98330。因此当前策略不能被解释为具备实盘可交易性。更准确的结论是：

> 策略在无成本条件下存在弱毛收益边际，但该边际不足以覆盖 1—2bp 单边交易成本。

### 7.3 ETF 归因

| ETF | 交易数 | 平均毛收益 | Profit Factor |
|---|---|---:|---:|---:|
| 有色ETF | 7 | 5.420bp | 1.695 |
| 黄金ETF易方达 | 70 | 1.217bp | 1.253 |
| 黄金ETF博时 | 40 | 0.848bp | 1.192 |
| 黄金ETF华安 | 88 | 0.817bp | 1.179 |
| 黄金ETF国泰 | 42 | 0.812bp | 1.153 |
| 豆粕ETF华夏 | 13 | -0.235bp | 0.972 |
| 能源化工ETF | 6 | -6.269bp | 0.449 |

核心黄金 ETF 信号整体略有正边际，但能源化工 ETF 明显拖累。

### 7.4 退出原因归因

| 退出原因 | 交易数 | 平均毛收益 | 胜率 | Profit Factor |
|---|---:|---:|---:|---:|
| `take_profit` | 5 | 50.000bp | 100.00% | inf |
| `max_hold` | 144 | 5.482bp | 58.33% | 3.778 |
| `force_flat` | 23 | 5.277bp | 65.22% | 3.454 |
| `stop` | 94 | -9.982bp | 5.32% | 0.094 |

`max_hold`、`force_flat` 和少量 `take_profit` 交易为正，但 `stop` 交易数量多且亏损厚，抵消了大部分正向贡献。策略并非完全没有方向性，而是错误信号的止损损失过重。

---

## 8. 项目主要结论

1. **简单分钟级动量信号不足以直接交易。** 单一突破或短期动量信号容易受噪声影响，加入成交额、VWAP、噪声边界和类别同步后可以改善信号质量。
2. **组合信号在 commodity_focus 池内存在弱毛收益边际。** 最新 walk-forward 主线在 0bp 成本下有 266 笔交易，平均单笔毛收益约 0.836bp。
3. **交易成本是决定策略能否成立的核心约束。** 1bp 单边成本后策略已转负，2bp 后 Final NAV 降至 0.98330。
4. **Walk-forward 是更严格的检验，而不是收益增强器。** 滚动选参后表现弱于固定组合优化，说明固定参数版本中的部分优势无法稳定延续到滚动样本外环境中。
5. **ETF 历史收益二次筛选被删除。** 样本外拆解发现该层筛选没有稳定预测力，反而削弱结果。当前主线只保留 daily rolling candidate selection。
6. **当前策略不具备直接实盘交易条件。** 项目价值在于完整展示从信号构造、规则过滤、成本敏感性、组合约束到 walk-forward 样本外检验的量化研究流程。

---

## 9. 仓库结构

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── LICENSE
├── .gitignore
│
├── scripts/
│   ├── download_t0_etf_1min_2022_2024.py
│   ├── build_t0_intraday_bar_panel_2022_2024.py
│   ├── audit_t0_intraday_panel_and_backtest_assumptions.py
│   ├── backtest_adaptive_intraday_momentum.py
│   ├── backtest_combined_orb_noise_rv.py
│   ├── backtest_combo_walk_forward_daily.py
│   ├── diagnose_combo_signal_forward_returns.py
│   ├── train_intraday_lgbm_direction_model.py
│   └── train_intraday_lgbm_return_model.py
│
├── data/
│   ├── README_data.md
│   └── sample_config/
│
├── docs/
│   ├── run_order.md
│   ├── methodology.md
│   ├── limitations.md
│   ├── data_dictionary.md
│   └── experiment_log.md
│
└── reports/
    ├── final_report.md
    ├── combo_dev_test_report.md
    ├── combo_walk_forward_report.md
    ├── ml_direction_h15_report.md
    ├── ml_return_h15_report.md
    ├── ml_return_h30_report.md
    ├── ml_return_h60_report.md
    ├── tables/
    │   ├── table_combo_walk_forward_summary.csv
    │   ├── table_combo_walk_forward_category.csv
    │   ├── table_combo_walk_forward_etf.csv
    │   ├── table_combo_walk_forward_exit.csv
    │   └── wf_config.json
    └── figures/
```

`data_t0_2022_2024/` 为本地数据与中间结果目录，不上传 GitHub。

---

## 10. 安装与运行

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：pandas, numpy, pyarrow, scikit-learn, lightgbm, matplotlib, tqdm, pyyaml, joblib, tushare

### 10.2 配置 Tushare Token

```bash
# Windows
set TUSHARE_TOKEN=your_token_here

# Linux/macOS
export TUSHARE_TOKEN=your_token_here
```

### 10.3 运行顺序

详细说明见 [docs/run_order.md](docs/run_order.md)。

```bash
# 1. 下载原始数据
python scripts/download_t0_etf_1min_2022_2024.py

# 2. 构建分钟级面板
python scripts/build_t0_intraday_bar_panel_2022_2024.py

# 3. 数据质量审计
python scripts/audit_t0_intraday_panel_and_backtest_assumptions.py

# 4. 自适应动量回测
python scripts/backtest_adaptive_intraday_momentum.py

# 5. 组合信号回测（固定 dev/test）
python scripts/backtest_combined_orb_noise_rv.py

# 6. Walk-forward 回测
python scripts/backtest_combo_walk_forward_daily.py

# 7. 信号诊断
python scripts/diagnose_combo_signal_forward_returns.py

# 8. LightGBM 方向分类
python scripts/train_intraday_lgbm_direction_model.py

# 9. LightGBM 收益回归 + 下行风险
python scripts/train_intraday_lgbm_return_model.py
```

---

## 11. 结果文件说明

| 文件 | 含义 |
|---|---|
| `reports/final_report.md` | 项目最终研究报告 |
| `reports/combo_dev_test_report.md` | 固定组合策略 dev/test 报告 |
| `reports/combo_walk_forward_report.md` | walk-forward 主线报告 |
| `reports/tables/table_combo_walk_forward_summary.csv` | walk-forward 成本敏感性汇总 |
| `reports/tables/table_combo_walk_forward_category.csv` | 类别归因 |
| `reports/tables/table_combo_walk_forward_etf.csv` | ETF 归因 |
| `reports/tables/table_combo_walk_forward_exit.csv` | 退出原因归因 |
| `reports/tables/wf_config.json` | walk-forward 配置 |

---

## 12. 局限性

1. **数据粒度限制** — 使用 1 分钟 OHLCV 数据，无法刻画 tick 级成交、盘口深度和真实买卖价差。
2. **交易成本假设仍是近似** — 回测使用 bp 成本敏感性模拟，真实执行中滑点、冲击成本和盘口流动性会随时间变化。
3. **long-only 框架受限** — 项目主要研究做多突破，无法完整表达日内反向或对冲交易机会。
4. **信号边际较薄** — 最新 walk-forward 主线平均单笔毛收益只有约 0.836bp，无法覆盖 1—2bp 单边成本。
5. **walk-forward 选择仍存在 selection noise** — 每日滚动选参更接近真实交易，但过去 120 天表现不一定能稳定预测下一交易日。

---

## 13. 免责声明

本仓库仅用于量化研究和学习交流，不构成任何投资建议。回测结果不代表未来收益。任何真实交易决策都应充分考虑交易成本、流动性、滑点、市场冲击、监管约束和独立风险评估。
