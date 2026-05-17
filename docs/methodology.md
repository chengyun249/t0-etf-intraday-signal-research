# Methodology

本项目的研究流程分为数据构建、特征工程、规则信号、机器学习模型、样本外验证和交易成本分析六个部分。

## 1. Data Pipeline

原始数据为 A 股 T+0 ETF 1 分钟 OHLCV bar。下载后按 ETF 文件保存，并进一步合成为分钟级面板。

核心处理包括：

- ETF 候选池构建；
- 交易时间字段标准化；
- 缺失 bar 与异常交易日检查；
- 价格、成交量和成交额字段清洗；
- 前瞻收益标签构建；
- 面板级特征输出。

## 2. Feature Engineering

特征工程围绕日内交易状态展开，主要包括：

- 短周期收益；
- 日内动量；
- realized volatility；
- high-low range；
- 成交量和成交额变化；
- 流动性代理变量；
- 噪声与波动调整变量；
- 日内时间位置；
- 15min / 30min / 60min forward return labels。

## 3. Rule-Based Signals

规则信号用于提供可解释 baseline：

- Adaptive Intraday Momentum；
- Opening Range Breakout；
- Noise Filter；
- Realized Volatility Filter；
- ORB / noise / RV combined signal。

规则信号不以“堆参数”为目标，而是用于观察日内结构是否存在稳定、可解释、可样本外验证的收益或风险关系。

## 4. Machine Learning Models

机器学习部分使用 LightGBM，包含两类任务：

- Return Regression: 预测未来 15min / 30min / 60min 收益；
- Direction / Downside Classification: 识别未来方向或下行风险。

模型重点不是证明分钟级收益可以稳定预测，而是比较收益预测与风险识别两类任务在样本外的差异。

## 5. Out-of-Sample Validation

项目采用时间序列划分：

| Period | Usage |
|---|---|
| 2022-2023 | development / training |
| 2024 | out-of-sample testing |

组合信号进一步使用 walk-forward 方式进行参数选择，减少直接在测试期调参造成的过拟合。

## 6. Cost Sensitivity

分钟级策略单次收益空间较小，因此项目重点测试不同成本假设：

| Cost | Meaning |
|---|---|
| 0bp | no-cost benchmark |
| 5bp | low-cost assumption |
| 10bp | baseline transaction cost |
| 20bp | stress scenario |

最终结论同时参考成本前表现和成本后表现。
