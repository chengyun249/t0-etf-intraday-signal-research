# A 股 T+0 ETF 分钟级日内信号研究

本项目基于 2022—2024 年 A 股 T+0 ETF 1 分钟 OHLCV 行情数据，构建了一个从数据下载、分钟级面板构建、数据质量审计、规则信号、组合信号、机器学习预测、交易成本敏感性分析到 walk-forward 样本外验证的完整研究框架。

项目的核心问题不是寻找一条最好看的回测曲线，而是检验：

> 在仅使用 1 分钟 bar 数据的条件下，A 股 T+0 ETF 的短周期收益是否具有稳定可交易性？如果结果不理想，约束主要来自信号本身、交易成本、样本池限制，还是数据粒度不足？

最终结果表明：在当前数据条件和交易结构下，A 股 T+0 ETF 的分钟级正收益预测难度较高。组合信号在零成本假设下存在有限毛收益边际，但加入交易成本后收益迅速衰减；机器学习模型对下行风险的识别相对更稳定，但不足以直接转化为稳定开仓收益。因此，本项目更适合定位为 **A 股 T+0 ETF 分钟级信号有效性检验与交易成本敏感性研究**，而不是可直接部署的实盘交易系统。

---

## 1. 项目背景

A 股市场中，只有部分 ETF 支持 T+0 交易。相比普通 T+1 股票或 ETF，T+0 ETF 理论上更适合进行日内交易和短周期信号研究。

但是，分钟级 ETF 策略并不天然容易成立。主要原因包括：

- A 股 T+0 ETF 数量有限，横截面样本远小于股票市场；
- ETF 本身是一篮子资产，波动率通常低于个股，短周期收益空间较薄；
- 1 分钟 OHLCV 数据只包含开高低收、成交量和成交额，缺少盘口、逐笔成交、主动买卖方向等微观结构信息；
- 分钟级策略对交易成本、买卖价差、滑点和冲击成本极其敏感；
- 简单的日内动量或突破信号容易被短周期噪声淹没；
- long-only 结构难以直接利用负向信号获利。

因此，本项目并不预设“分钟级 ETF 策略一定有效”，而是通过完整实验链条检验它到底是否具有可交易性。

---

## 2. 研究问题

项目围绕以下问题展开：

1. **数据层面**：1 分钟 OHLCV 数据是否足以支持 ETF 短周期收益预测？
2. **规则信号层面**：日内动量、开盘区间突破、噪声边界、VWAP、类别同步、相对强弱等信号是否具有稳定边际？
3. **机器学习层面**：LightGBM 是否能够从分钟级特征中学习到未来 15/30/60 分钟收益信息？
4. **风险识别层面**：相比预测正收益，模型是否更擅长识别未来下行风险？
5. **交易成本层面**：交易成本是否会直接决定分钟级策略是否可交易？
6. **样本外稳定性层面**：在 walk-forward 滚动验证下，策略是否仍然稳定？

---

## 3. 数据说明

### 3.1 数据来源

数据来源为 Tushare Pro API，主要使用 A 股 ETF 1 分钟行情数据。

原始行情数据未随仓库公开，主要原因包括：

- 数据接口需要 Tushare Pro 权限；
- 原始分钟级行情体积较大；
- 行情数据存在版权和再分发限制；
- 本仓库重点展示研究框架、代码结构、实验结果和分析报告。

### 3.2 样本范围

```text
样本区间：2022-01-01 至 2024-12-31
数据频率：1 分钟 bar
研究对象：A 股 T+0 ETF
主要字段：open, high, low, close, volume, amount
```

### 3.3 ETF 标的池

研究对象主要包括具备 T+0 交易属性的 ETF 类型：

- 商品 ETF：黄金、豆粕、有色、能源化工等；
- 跨境 ETF；
- 债券 ETF；
- 货币 ETF。

项目不使用普通 T+1 ETF 构建日内交易策略，以避免交易制度与回测假设不一致。

---

## 4. 整体研究流程

项目按照完整量化研究流程展开，而不是只进行单一模型回测。

```text
T+0 ETF 1 分钟原始行情
        ↓
数据清洗与质量审计
        ↓
分钟级研究面板构建
        ↓
日内特征工程
        ↓
规则信号策略测试
        ↓
组合信号策略测试
        ↓
机器学习预测模型
        ↓
交易成本敏感性分析
        ↓
walk-forward 样本外验证
        ↓
最终结论与局限性分析
```

流程重点包括：

- 先验证数据质量，再进行回测；
- 先构建简单规则基准，再使用机器学习模型；
- 先看零成本信号边际，再加入交易成本；
- 先做开发期测试，再做 walk-forward 样本外验证；
- 对无效或较弱结果进行解释，而不是只展示最好看的回测曲线。

---

## 5. 研究尝试与模型设计

### 5.1 规则策略：自适应日内动量

首先构建自适应日内动量策略，用于检验 ETF 分钟级收益是否存在短周期趋势延续。该策略综合考虑：

- 日内动量；
- 波动率状态；
- 流动性状态；
- 交易成本；
- 不同 ETF 类型。

该部分的意义在于建立基础规则策略基准。如果简单动量策略完全无效，后续复杂模型需要证明自己能够显著改善该基准。

### 5.2 组合信号：ORB + Noise Boundary + VWAP + Category Breadth + Relative Value

在单一动量信号基础上，进一步构建组合信号框架，包括：

1. **Opening Range Breakout（ORB）**：用开盘区间上沿判断日内突破。
2. **Relative Volume**：比较当天开盘成交额与历史同区间成交额，判断开盘成交是否放大。
3. **Noise Boundary**：用历史同一 bar_index 的平均绝对日内波动构建噪声边界，过滤普通波动。
4. **VWAP Confirmation**：要求价格站上日内成交量加权平均价。
5. **Category Breadth**：要求同类 ETF 出现日内方向或短期方向共振。
6. **Relative Value Filter**：过滤相对同类明显过热的 ETF。
7. **Expected Edge Filter**：粗略估计潜在收益空间是否足以覆盖交易成本。
8. **Dynamic Exit**：加入止盈、止损、移动止损、VWAP/ORB 结构止损、最大持有时间和收盘前强制平仓。

该部分试图回答的问题是：如果单一信号较弱，多个弱信号组合后是否能够形成更稳定的交易边际。

### 5.3 方向分类模型

使用 LightGBM Classifier 预测未来收益方向。

```text
模型：LightGBM Classifier
目标：预测未来收益是否为正
预测周期：15 分钟
```

该模型用于检验分钟级特征是否具备基本方向识别能力。

### 5.4 收益回归模型

使用 LightGBM Regressor 直接预测未来收益率。

```text
模型：LightGBM Regressor
目标：预测未来收益率
预测周期：15 分钟 / 30 分钟 / 60 分钟
```

实验结果显示，收益回归模型的预测相关性较低，说明直接预测短周期正收益非常困难。

### 5.5 下行风险分类模型

使用 LightGBM Classifier 识别未来是否可能出现较大负收益。

```text
模型：LightGBM Classifier
目标：识别未来下行风险事件
预测周期：15 分钟 / 30 分钟 / 60 分钟
```

相比直接预测正收益，模型对下行风险的识别更稳定。这说明 1 分钟 OHLCV 特征更适合判断“不适合交易的环境”，而不是直接判断“应该买入的机会”。

---

## 6. 核心实验结论

### 6.1 短周期正收益预测能力较弱

收益回归模型在测试集上的预测相关性较低。

```text
收益回归预测相关性 pred_ret_corr：约 0.019
方向分类 AUC：约 0.594
```

这说明，仅依赖 1 分钟 OHLCV 特征，很难稳定预测未来 15/30/60 分钟正收益。数据并非完全没有信息，而是可预测边际较弱，难以覆盖真实交易成本。

### 6.2 下行风险识别相对更稳定

相比收益回归，下行风险分类模型表现更稳定。

```text
下行风险分类 AUC：约 0.616
Average Precision：约 0.400
```

这表明分钟级 OHLCV 数据对“不利交易环境”的识别能力强于对“正收益机会”的识别能力。模型更适合用于过滤高风险交易时段、避免进入不利市场状态、辅助风控，而不是单独构成稳定盈利的开仓策略。

### 6.3 固定 dev/test 组合信号存在有限边际

组合信号在固定 dev/test 框架下，相较简单规则策略有明显改善。固定组合策略使用 2022—2023 年作为开发期、2024 年作为测试期，开发期选择出的主版本为 `commodity_focus`。

代表性结果如下：

| 期间 | 成本 | 交易数 | Final NAV | Sharpe | 胜率 |
|---|---:|---:|---:|---:|---:|
| 开发期 2022—2023 | 0bp | 178 | 1.0075 | 1.15 | 41.0% |
| 开发期 2022—2023 | 2bp | 178 | 0.9933 | -1.06 | 33.7% |
| 测试期 2024 | 0bp | 117 | 1.0073 | 1.67 | 48.7% |
| 测试期 2024 | 2bp | 117 | 0.9979 | -0.51 | 41.0% |

该结果说明：多层过滤确实能比简单动量更接近有效信号，但信号边际仍然较薄，成本后难以保持稳定正收益。

### 6.4 Walk-forward 检验：为什么滚动选参后表现反而变弱

在固定组合优化版本中，策略先在开发期内选择一组表现较好的参数，再将这组参数应用到后续测试期。这种做法可以检验固定参数在测试期的延续性，但参数一旦确定后并不会随市场状态滚动变化。为了进一步接近真实部署，本项目加入了 daily walk-forward 检验：每个交易日前，只使用该日前最多 120 个交易日的历史窗口，从候选参数组中选择近期表现较好的 candidate，然后只用这组参数交易当天。

需要说明的是，walk-forward 的目的不是保证收益更高，而是进行更严格的样本外检验。与固定组合优化相比，walk-forward 版本表现变弱，主要有三点原因。

第一，固定组合优化使用的是较长开发期内筛选出的参数，参数选择相对稳定；而 walk-forward 每天只参考过去 120 个交易日，窗口更短，且日内信号交易次数本身有限。因此，在每个滚动窗口内，不同参数组之间的表现差异容易受到少数交易和市场状态切换影响，候选参数评分中的平均净收益、profit factor 和交易次数并不一定能稳定预测下一交易日表现。

第二，分钟级 ETF long-only 突破信号的单笔收益空间本身很薄。当前无 ETF 二次筛选的 walk-forward 主线版本在 0bp 成本下共有 266 笔交易，Final NAV 为 1.00445，平均单笔毛收益约 0.836bp，Profit Factor 为 1.162。这说明组合信号在 `commodity_focus` 池内仍存在弱毛收益边际，但该边际非常有限。一旦加入 1bp 或 2bp 单边成本，策略净值迅速转负，说明滚动选参并没有创造出足够厚的可交易收益空间。

第三，早期版本曾加入”单 ETF 历史表现二次筛选”，即在选出 candidate 后，再根据过去窗口内单 ETF 的平均净收益和 profit factor 筛出当天可交易 ETF。后续拆解发现，这一层筛选并没有稳定的样本外预测力，反而容易在小样本下把历史偶然性误判为标的优劣，最终削弱策略结果。因此，当前主线版本不再使用 ETF 历史收益二次筛选，仅保留 daily rolling candidate selection。

当前主线 walk-forward 版本采用：

```text
scope：commodity_focus
grid_size：mini
candidate_count：48
walk_forward_trade_dates：666
train_lookback_days：120
min_train_days：60
update_frequency：daily
ETF 二次筛选：禁用
```

无 ETF 二次筛选的 walk-forward 主线结果如下：

| 单边成本 | 交易数 | Final NAV | Sharpe | 最大回撤 | 胜率 | 平均单笔毛收益 | Profit Factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0bp | 266 | 1.00445 | 0.579 | -0.357% | 40.98% | 0.836bp | 1.162 |
| 1bp | 266 | 0.99382 | -0.831 | -0.830% | 39.10% | 0.836bp | 0.816 |
| 2bp | 266 | 0.98330 | -2.251 | -1.770% | 33.46% | 0.836bp | 0.585 |

因此，walk-forward 结果不应被理解为”滚动优化失败”，而应理解为：当用更接近真实交易的方式进行滚动样本外检验时，固定参数版本中的部分收益优势无法稳定延续。删除 ETF 二次筛选后，策略恢复了弱正毛收益边际，说明原始组合信号并非完全无效；但该边际只有约 0.84bp/笔，无法覆盖实际交易成本。最终结论是：当前 ORB + Noise Boundary + VWAP + 类别同步 + 相对强弱过滤框架具有一定信号方向性，但还没有达到可实盘交易所需的收益厚度和稳定性。

---

## 7. 为什么收益表现仍然较弱

### 7.1 1 分钟 OHLCV 数据信息不足

1 分钟 bar 只包含 open, high, low, close, volume, amount，但真正影响分钟级交易质量的很多信息并不存在，例如买一卖一价差、盘口深度、委托队列变化、主动买卖方向、撤单行为、瞬时流动性变化等。

对于日频策略，OHLCV 可能已经足够构建基础信号；但对于分钟级交易，缺少微观结构信息会严重限制预测能力。

### 7.2 ETF 分钟级收益空间本身较薄

ETF 是一篮子资产的组合，波动率通常低于个股。在 15/30/60 分钟尺度下，很多 ETF 的收益变化幅度非常小。即使方向判断略有优势，单笔收益也可能不足以覆盖交易成本。

### 7.3 A 股 T+0 ETF 样本池较小

A 股可 T+0 交易的 ETF 数量有限，商品子集更小。机器学习模型通常需要大量横截面样本和稳定的特征差异，但 ETF 横截面排序空间不足，模型可学习样本有限，标的之间差异不够丰富，单一市场状态容易影响整体样本。

### 7.4 交易成本对分钟级策略影响过大

无 ETF 二次筛选的 walk-forward 版本在 0bp 成本下平均单笔毛收益约 0.84bp；但 1bp 单边成本对应 round-trip 2bp，2bp 单边成本对应 round-trip 4bp。分钟级策略的单笔边际远小于现实交易成本门槛，因此成本会快速吞噬收益。

### 7.5 Long-only 结构限制了负向信号利用

实验中多数策略采用 long-only 逻辑。模型即使能识别下行风险，也主要只能用于“不买”或“过滤”，而不能直接通过做空获利。下行风险模型虽然表现相对更好，但在 long-only 约束下不能完全转化为收益来源。

---

## 8. 项目价值

虽然最终没有得到可直接实盘部署的高收益策略，但项目仍然具有明确价值。

### 8.1 完成了完整的量化研究闭环

项目覆盖了：数据获取 → 数据清洗 → 面板构建 → 质量审计 → 特征工程 → 规则策略 → 组合信号 → 机器学习模型 → 交易成本分析 → walk-forward 验证 → 结果解释。这比单纯展示一条收益曲线更接近真实量化研究流程。

### 8.2 对无效模块进行了拆解和删除

项目并没有机械堆叠过滤条件。早期 walk-forward 版本曾加入单 ETF 历史表现二次筛选，但拆解发现该层筛选在样本外失效，最终主线版本将其禁用。这一过程体现了量化研究中必要的 ablation 思路：不是过滤越多越好，而是只有能提高样本外表现的过滤才应被保留。

### 8.3 明确了信号的边界

当前结果表明：

- 简单动量策略无效；
- 组合信号在零成本下有弱边际；
- walk-forward 去除 ETF 二次筛选后毛收益恢复为正；
- 但真实交易成本会迅速吞噬信号边际；
- 机器学习直接预测正收益较难；
- 下行风险识别比正收益预测更稳定。

这些结论的“排雷”价值，不亚于找到一个正收益策略。

### 8.4 明确了后续改进方向

如果继续提升分钟级 ETF 策略的可交易性，可能需要：

- tick 级成交数据；
- 盘口和订单簿数据；
- 更精细的执行成本模型；
- 更低交易成本的执行环境；
- 更强的流动性过滤；
- 更适合风险过滤而非直接预测收益的策略结构；
- 不同于 simple long-only 的交易设计。

---

## 9. 仓库结构

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── LICENSE
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
    ├── panel_audit_report.md
    ├── adaptive_momentum_report.md
    ├── adaptive_momentum_commodity_report.md
    ├── combo_dev_test_report.md
    ├── combo_walk_forward_report.md
    ├── signal_diagnostics_report.md
    ├── ml_direction_h15_report.md
    ├── ml_return_h15_report.md
    ├── ml_return_h30_report.md
    ├── ml_return_h60_report.md
    ├── figures/
    └── tables/
```

本地大数据目录 `data_t0_2022_2024/` 不随仓库提供，已被 `.gitignore` 忽略。

---

## 10. 运行方式

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

推荐环境：Python >= 3.9，pandas, numpy, pyarrow, scikit-learn, lightgbm, matplotlib, tqdm, pyyaml, joblib, tushare。

### 10.2 配置 Tushare Token

下载原始 ETF 分钟数据需要 Tushare Pro token。Token 不应提交到 GitHub。

```bash
# Windows PowerShell
$env:TUSHARE_TOKEN="your_token_here"

# Linux/macOS
export TUSHARE_TOKEN=your_token_here
```

### 10.3 典型运行顺序

```bash
# 1. 下载原始数据
python scripts/download_t0_etf_1min_2022_2024.py

# 2. 构建分钟级研究面板
python scripts/build_t0_intraday_bar_panel_2022_2024.py

# 3. 数据质量审计
python scripts/audit_t0_intraday_panel_and_backtest_assumptions.py

# 4. 自适应日内动量基准
python scripts/backtest_adaptive_intraday_momentum.py

# 5. 固定 dev/test 组合信号
python scripts/backtest_combined_orb_noise_rv.py

# 6. Walk-forward 样本外验证
python scripts/backtest_combo_walk_forward_daily.py --scopes commodity_focus --grid-size mini --out-dir data_t0_2022_2024/backtest_combo_wf_no_etf_filter

# 7. 机器学习方向分类 / 收益回归
python scripts/train_intraday_lgbm_direction_model.py
python scripts/train_intraday_lgbm_return_model.py
```

详细说明见 `docs/run_order.md`。

---

## 11. 结果文件说明

主要报告位于 `reports/`：

```text
reports/final_report.md
reports/combo_dev_test_report.md
reports/combo_walk_forward_report.md
reports/signal_diagnostics_report.md
reports/ml_direction_h15_report.md
reports/ml_return_h15_report.md
reports/ml_return_h30_report.md
reports/ml_return_h60_report.md
```

主要汇总表位于 `reports/tables/`：

```text
table_combo_walk_forward_summary.csv
table_combo_walk_forward_category.csv
table_combo_walk_forward_etf.csv
table_combo_walk_forward_exit.csv
wf_config.json
```

逐笔交易、候选参数交易明细、分钟级面板和原始行情数据保留在本地 `data_t0_2022_2024/`，不随仓库公开。

---

## 12. 最终结论

本项目最终没有证明 A 股 T+0 ETF 的 1 分钟 OHLCV 数据可以直接支持稳定实盘日内交易。更准确的结论是：

> 多层组合信号可以在零成本下形成弱毛收益边际；walk-forward 去除无效 ETF 二次筛选后，该边际仍存在，但单笔平均毛收益只有约 0.84bp，无法覆盖 1—2bp 单边成本。当前数据粒度下，分钟级正收益预测难度较高，下行风险识别相对更有价值。

因此，本项目的主要价值在于完整展示了分钟级 ETF 量化研究的构建、验证、证否和迭代过程。
