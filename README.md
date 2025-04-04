# 特斯拉反转策略回测与优化 (TSLA Reversal Strategy Backtest & Optimization)

本项目使用 Python 对基于 Pine Script 的特斯拉股票反转交易策略进行回测和参数优化。主要利用 `vectorbt` 进行高效的回测，`optuna` 进行超参数优化，以及 `yfinance` 获取股票数据。

## 项目结构

```
TSLASTR/
├── tsla_strategy_backtest.py   # 核心回测与优化脚本 (Jupyter Notebook 格式)
├── requirements.txt           # 项目依赖库列表
└── README.md                  # 项目说明文档
```

## 功能特性

- **数据获取**: 使用 `yfinance` 下载特斯拉 (TSLA) 的历史股价数据。
- **指标计算**: 实现 Pine Script 策略中使用的技术指标 (RSI, MACD, ADX, SMA, 模拟 IV Rank, 成交量 MA) 的 Python 版本。
- **策略逻辑**: 将 Pine Script 中的多空评分和入场逻辑转换为 Python 函数。
- **回测**: 使用 `vectorbt` 对策略在不同时间段进行回测，评估性能指标（累计收益、夏普比率、最大回撤）。
- **参数优化**: 利用 `optuna` 自动搜索最优的指标参数和权重组合，以最大化特定目标（如最终收益）。
- **动态止盈止损**: 实现基于历史波动率的动态止盈止损机制。
- **可配置性**: 允许用户轻松调整回测时间段、优化目标和参数范围。
- **可视化**: 使用 `plotly` 生成交互式图表，展示回测结果和优化过程。

## 安装与设置

1.  **创建虚拟环境 (推荐)**:
    ```bash
    conda create -n tsla_backtest python=3.10
    conda activate tsla_backtest
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(可选) 将 .py 转换为 .ipynb**:
    如果您想在 Jupyter Notebook 或 JupyterLab 中交互式运行，可以使用 `jupyter nbconvert`:
    ```bash
    jupyter nbconvert --to notebook tsla_strategy_backtest.py
    ```
    然后打开生成的 `tsla_strategy_backtest.ipynb` 文件。

## 使用说明

1.  **打开脚本**: 在您选择的 Python 环境（直接运行 .py 文件或在 Jupyter 中打开 .ipynb 文件）中打开 `tsla_strategy_backtest.py`。
2.  **配置参数**:
    -   修改 `START_DATE` 和 `END_DATE` 变量来定义回测和优化的时间范围。
    -   (优化时) 调整 `optuna` 优化研究中的 `n_trials` 来控制优化的迭代次数。
    -   (优化时) 检查并根据需要调整 `objective` 函数中参数的建议范围 (`suggest_int`, `suggest_float`)。
3.  **运行代码**: 按顺序执行脚本中的代码单元格。
    -   数据加载和预处理。
    -   指标计算。
    -   策略信号生成。
    -   `vectorbt` 回测执行。
    -   (可选) `optuna` 参数优化执行。
    -   结果分析和可视化。
4.  **分析结果**:
    -   查看 `vectorbt` 输出的回测统计数据，包括总回报、夏普比率、最大回撤等。
    -   检查 `plotly` 生成的图表，例如累计收益曲线、交易信号等。
    -   如果进行了优化，查看 `optuna` 找到的最佳参数组合及其对应的性能。

## 注意事项

-   **隐含波动率 (IV)**: 原始 Pine Script 中的 IV 计算是模拟的。此 Python 实现也使用基于历史波动率 (HV) 的类似模拟。如果您有真实的 IV 数据源，可以替换 `calculate_iv_rank` 函数中的计算逻辑。
-   **数据质量**: `yfinance` 提供的数据可能存在缺失或错误。脚本中包含基本的缺失值处理，但对于生产环境，建议进行更严格的数据清洗。
-   **优化时间**: 参数优化可能需要较长时间，具体取决于 `n_trials` 的设置和参数空间的复杂度。
-   **过拟合风险**: 在优化参数时，请注意过拟合的风险。建议使用一部分数据进行优化，并在另一部分未见过的数据（样本外数据）上验证优化后的参数性能。
-   **API 限制**: `yfinance` 可能会受到雅虎财经 API 的速率限制。如果遇到问题，请尝试减少请求频率。 