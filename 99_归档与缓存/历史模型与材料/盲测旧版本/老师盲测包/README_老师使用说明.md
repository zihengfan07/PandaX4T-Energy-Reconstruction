> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX 能量重建 candidate_v8 独立数据盲测包

## 这个包做什么

本包将已经在 run10972 上训练并冻结的候选模型应用于一份新的标量 TXT，
同时计算现行公式与候选模型的：

- 高斯核心 \(\sigma/\mu\)；
- 非参数 \(R_{68}\)；
- 非参数 \(R_{90}\)；
- 峰形拟合 GOF \(p\) 值；
- 每个事件的现行相对能量与模型相对能量。

模型不会在新数据上重新训练或选择参数。这样新 run 才能作为真正的盲测。

## 输入数据要求

输入应当是与
`light_ana_run10972_finalSS_Egt2MeV_scalar.txt`
具有相同字段定义的制表符分隔 TXT。

至少必须包含模型使用的信号、漂移时间、位置和脉冲形状字段。程序若发现
缺少字段会直接停止并列出字段名，不会静默替代。

## 推荐环境

模型在以下环境中训练：

```text
Python 3.6
NumPy 1.19.5
Pandas 1.1.5
SciPy 1.5.4
scikit-learn 0.24.2
joblib 1.1.1
```

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate pandax-eres-validation
```

也可以在兼容的 Python 3.6 环境中执行：

```bash
python -m pip install -r requirements.txt
```

## Linux/macOS 运行

把新数据放到本文件夹或使用绝对路径，然后运行：

```bash
chmod +x run_validation.sh
./run_validation.sh /path/to/new_run_scalar.txt
```

## Windows 运行

```bat
run_validation.bat C:\path\to\new_run_scalar.txt
```

或直接运行：

```bash
python validate_new_data.py \
  --input /path/to/new_run_scalar.txt \
  --model model/candidate_v8.joblib \
  --output-dir validation_output
```

## 输出文件

运行完成后，`validation_output/` 中包含：

- `validation_report.txt`：最容易阅读的结果；
- `validation_summary.csv`：现行公式与候选模型的指标对比；
- `validation_events.csv`：逐事件重建结果；
- `model_manifest_used.json`：本次使用的模型版本。

若 `validation_summary.csv` 中 candidate_v8 的
`sigma_over_mu_percent` 小于 current_formula，说明模型在新 run 上仍然缩窄。
同时必须检查 \(R_{68}\)、\(R_{90}\) 和 GOF，不能只选择最小的核心数字。

## 重要边界

- 模型训练数据是 run10972。
- 新 run 在报告结果前不能参与重新训练或调参。
- 模型当前是候选版本，不是已经确认的最终物理重建。
- 模型输出首先是相对能量；绝对 MeV 标定需要独立的已知能线。
