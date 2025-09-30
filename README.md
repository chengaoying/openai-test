# AI-Enhanced Public Market Data Pipeline

本项目演示如何结合 Python 与公开数据接口（如新浪财经、雪球）来构建一个简易的量化分析流程。项目包括：

- 对接公开市场数据 API。
- 构造常见技术指标特征。
- 训练一个基于最小二乘法的线性模型来预测未来收益。

> ⚠️ 注意：示例代码使用的 API 均为公开的非官方接口，可能随时调整或限制访问。运行网络请求前请确认符合对应网站的使用政策。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 离线示例

仓库内提供 `src/sample_data.json`，可以直接使用离线数据运行完整流程：

```bash
python -m src.main TEST --source offline --output artifacts
```

离线模式不依赖第三方网络库；若需访问新浪财经或雪球数据，请确保安装 `requests` 并根据命令行参数提供对应凭据。

运行完成后，`artifacts/` 目录下将生成：

- `evaluation.csv`：预测值与真实收益的对比。
- `model.json`：训练好的线性回归模型系数。

### 调用新浪财经接口

```bash
python -m src.main sh600000 --source sina --count 200 --output artifacts
```

### 调用雪球接口

雪球接口需要 `xq_a_token` Cookie，可在浏览器登录雪球后通过开发者工具复制：

```bash
python -m src.main SH600000 --source xueqiu --count 200 --token <your-token> --output artifacts
```

## 模块说明

- `finance_ai.data_sources`：封装新浪财经与雪球的 K 线数据获取逻辑，并提供统一的 `OHLCVBar` 数据结构。
- `finance_ai.features`：对原始数据计算收益率、均线、量能等特征，并生成未来收益作为监督学习目标，返回基于原生 Python 列表的 `FeatureDataset`，无需依赖 `pandas` 或 `numpy`。
- `finance_ai.models`：实现了一个纯 Python 的简易线性回归模型，便于在无额外依赖的情况下快速试验。
- `src/main.py`：命令行脚本，将下载、特征工程与建模步骤串联起来，输出模型参数与评估结果。

## 开发建议

- 在联网情况下可以扩展更多数据源，例如东方财富、聚宽等开放接口。
- 结合 `scikit-learn`、`lightgbm` 等机器学习框架替换当前的线性模型。
- 增加更多技术指标、资金面或宏观经济特征，以提高预测能力。
