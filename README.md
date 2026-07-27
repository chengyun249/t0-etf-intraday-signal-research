# A股T+0 ETF分钟级信号研究

这是一个1分钟K线级的日内信号、执行假设和交易成本敏感性研究框架。项目的目标是区分“统计上有信号”和“扣除摩擦后可交易”，不是包装出一条好看的收益曲线。

当前定位：研究原型，不是可实盘策略，不构成投资建议。

完整中文介绍见 [项目介绍](docs/project_overview.md)。

## 研究内容

- 开盘区间突破（ORB）、噪声边界、VWAP、类别广度和相对价值；
- 单信号、组合信号、配对价差与walk-forward参数选择；
- 0/1/2/3/5/10bp单边成本敏感性；
- 保守的分钟OHLC成交模型；
- 每日动态标的池；
- 不使用ETF代码数值特征的purged/embargo机器学习诊断；
- 失败实验和负面结论记录。

## v2关键修复

2026-07-27版本针对原项目的未来信息和工程混乱做了重构：

- 信号仍可使用信号分钟收盘时已形成的VWAP，但入场后的止损只使用上一分钟已知的累计VWAP；
- 跳空穿越止损时按更差的当前分钟开盘价成交；
- 盘中止损触发默认再计1bp不利滑点；
- 同一根OHLC同时触发止盈和止损时按止损优先；
- trailing stop只使用上一根已完成bar的高水位，避免假设未知的高低价先后顺序；
- walk-forward入口按历史长度和过去成交额构建每日动态标的池；
- 组合信号中的区间/噪声幅度改名为 `movement_budget_bp`；它只是运动空间过滤器，不再被表述成“预期收益”；
- walk-forward默认改为周度更新，候选参数至少30笔且覆盖20个交易日，单ETF至少10笔；参数排序使用按交易日聚类的90%收益下置信界，而不是奖励样本内profit factor；
- 机器学习按完整交易日分块，并设置purge和embargo，不再随机拆相邻分钟；
- ETF代码不作为数值特征；
- 统一主成本为单边2bp，成本网格为0/1/2/3/5/10bp；
- 原先混在同一文件夹的ETF轮动原型和2025旧脚本已移到本地 `archive/`，不会进入Git；
- 旧报告保留在 `reports_sample/` 仅展示格式，并明确标为v2前结果。

核心实现位于 `src/t0_research/`，单元测试不依赖私有行情。

## 修正后的本地复跑

使用现有2025年、30只ETF、1,756,890条分钟记录，按修正后的执行模型重跑组合策略。开发期为2025-01至2025-08，测试期为2025-09至2025-12。`commodity_focus` 是开发期下置信界最高的候选，但其按交易日聚类的90%净收益下置信界仍为 -6.16bp，因此没有任何 scope 通过开发期门槛。下面28笔测试交易只能作诊断，不能称为“已选中策略”：

| 单边固定成本 | 平均单笔净收益 | 测试期结果 |
|---:|---:|---:|
| 0bp | 7.37bp | 正 |
| 2bp | 3.37bp | 正 |
| 3bp | 1.37bp | 边际 |
| 5bp | -2.63bp | 负 |

止损不利滑点已计入上述毛收益生成过程；但买卖价差、盘口深度、订单延迟和冲击成本仍只有固定bp近似。28笔交易只分布在19个交易日。按交易日聚类的5日移动区块bootstrap下，2bp成本对应的年化日均收益95%下界为 -0.49%；因此正确结论是：尚未证明可交易。验收审查见 [docs/strategy_acceptance_review.md](docs/strategy_acceptance_review.md)，输出位于 `data_t0/backtest_combined_orb_noise_rv_2025/`。

历史2022—2024样本的旧报告不能与v2直接比较，必须用新代码和原始分钟数据重新生成后才能引用。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
```

典型流程：

```bash
# 1. 下载候选ETF分钟数据（需要TUSHARE_TOKEN）
python src/01_download/download_t0_etf_1min_2022_2024_v2.py

# 2. 构建分钟面板
python src/02_build_panel/build_t0_intraday_bar_panel_2022_2024.py

# 3. 固定开发/测试组合策略
python src/03_backtest/backtest_combined_orb_noise_rv_2025.py

# 4. 每日walk-forward + 动态标的池
python src/03_backtest/backtest_combo_walk_forward_daily.py

# 5. purged/embargo机器学习诊断
python src/04_diagnostics/train_purged_signal_models.py

# 6. 固定组合策略验收审查
python src/04_diagnostics/evaluate_strategy_acceptance.py
```

下载凭据通过环境变量提供：

```powershell
$env:TUSHARE_TOKEN = "your-token"
# 如确需兼容端点，再设置TUSHARE_BASE_URL或对应脚本参数。
```

## 目录

```text
src/01_download/       数据发现与下载
src/02_build_panel/    质量检查、特征和标签
src/03_backtest/       ORB、Noise、Pair、Combo、Walk-forward
src/04_diagnostics/    裸信号与purged ML诊断
src/t0_research/       统一执行、动态标的池、时间切分核心
tests/                 无私有数据测试
docs/                  方法、数据和结果解释
reports_sample/        v2前旧报告，仅展示格式
archive/               本地旧原型，Git忽略
```

## 已知限制

- 1分钟OHLC无法识别bar内真实路径；当前采用保守顺序，但仍不等于逐笔回放。
- 没有买一卖一、盘口深度、真实订单与撤单数据。
- 当前本地动态标的池只能在已下载候选集中选择；完整研究应下载每个历史时点的全部合格T+0 ETF。
- T+0资格、跨境额度、申赎状态和临时停牌仍需更细的point-in-time数据。
- 相邻分钟标签重叠会降低有效样本量；按日分块避免跨折泄漏，但不能把每分钟当成独立观测。

更多说明见 [项目介绍](docs/project_overview.md)、[v2方法](docs/methodology_v2.md)、[数据流程](docs/data_pipeline.md)、[成本与成交](docs/cost_and_execution.md)、[策略验收](docs/strategy_acceptance_review.md) 和 [负面结果](docs/negative_results.md)。
