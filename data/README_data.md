# Data Description

本项目使用 A 股 T+0 ETF 1 分钟行情数据，样本区间为 2022-01-01 至 2024-12-31。

由于原始行情数据体积较大，且可能受到数据源授权限制，本仓库不直接提供原始 1 分钟行情文件和完整 parquet 面板文件。仓库仅保留样本配置、审计摘要、结果表格和研究报告，用于展示研究流程与主要结论。

## Raw Data Structure

原始数据本地目录结构为：

```text
data_t0_2022_2024/raw_1min/freq=1min/
```

每个 ETF 对应一个 parquet 文件，核心字段包括：

| Field | Description |
|---|---|
| `ts_code` | ETF code |
| `trade_time` | minute-level timestamp |
| `trade_date` | trading date |
| `clock_time` | intraday clock time |
| `open` | open price |
| `high` | high price |
| `low` | low price |
| `close` | close price |
| `vol` | trading volume |
| `amount` | trading amount |

## Processed Panel

主面板文件为：

```text
data_t0_2022_2024/processed/t0_intraday_bar_panel.parquet
```

该文件包含约 3,971,198 行和 69 列特征，由以下脚本生成：

```bash
python scripts/build_t0_intraday_bar_panel_2022_2024.py
```

特征清单参见：

```text
reports/tables/t0_feature_manifest.json
```

## Public Files

本仓库公开保留：

- `data/sample_config/`: ETF 候选池与筛选结果样例；
- `reports/tables/`: 面板审计、回测、诊断与模型摘要表；
- `reports/*.md`: 研究报告与实验总结；
- `docs/`: 方法、运行顺序、局限与字段说明。

## Data Source

数据通过 Tushare Pro API 获取。运行下载脚本前需要配置有效的 Tushare Pro token：

```bash
python scripts/download_t0_etf_1min_2022_2024.py
```

## Reproducibility Notes

完整结果复现依赖原始数据源权限、本地运行环境和下载时的数据可用性。仓库中的公开文件用于展示研究流程和主要结论，不保证在缺少原始数据的情况下可以直接复现全部输出。
