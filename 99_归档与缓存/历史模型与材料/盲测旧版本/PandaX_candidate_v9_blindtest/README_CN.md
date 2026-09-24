> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX candidate_v9 独立数据盲测说明

## 目的

该程序使用已经冻结的 `candidate_v9` 模型处理一份模型从未见过的新 run，
并在完全相同的12种峰拟合协议下比较当前公式和候选模型。

模型不会在新数据上重新训练、重新选择变量或重新估计标定系数。

## 输入

输入应为制表符分隔的标量 TXT，字段定义和单位必须与
`light_ana_run10972_finalSS_Egt2MeV_scalar.txt` 一致。

程序会严格检查模型所需列。缺少必要列时会直接报错，不会自动猜测或替换变量。

`fileNumber`可以保留在输入中用于输出身份追踪，但不会进入预测模型。

## 推荐环境

```bash
conda env create -f environment.yml
conda activate pandax-eres-validation
```

也可以在已有 Python 3.6 环境中安装：

```bash
pip install -r requirements.txt
```

## Linux运行

```bash
chmod +x run_validation.sh
./run_validation.sh /path/to/new_run_scalar.txt
```

## Windows运行

```bat
run_validation.bat C:\path\to\new_run_scalar.txt
```

## 输出

运行后生成 `validation_output/`：

- `validation_summary.json`：当前公式与候选模型的汇总指标；
- `validation_protocols.csv`：12种固定峰拟合协议的逐协议结果；
- `validation_events.csv`：逐事件当前能量、候选能量和修正幅度。

重点查看：

- `sigma_median`：12种协议的 \(\sigma/\mu\) 中位数；
- `sigma_worst`：12种协议中的最差结果；
- `center_bias_max`：相对 \(2614.5\,\mathrm{keV}\) 的最大峰中心偏差；
- `r68`、`r90`：核心宽度和尾部宽度；
- `near_cap_fraction`：修正接近平滑上限的事件比例；
- `relative_sigma_improvement`：相对当前公式的峰宽改善。

## 盲测纪律

第一次运行后请原样保留整个 `validation_output/` 并返回。

在报告第一次结果之前，不要：

- 用新数据重新训练；
- 根据新结果调整参数、变量或修正上限；
- 使用新数据峰中心重新标定；
- 改变峰拟合窗口或背景模型；
- 删除表现不好的文件或事件。

否则这批数据将不再是独立盲测。

## 模型边界

`candidate_v9`在 Run 10972 内按五个连续完整文件块完成嵌套验证，但尚未经过独立新 run 验证。
它是候选模型，不是已证明普适的最终能量重建。

模型身份：

```text
spline_ridge_a100_c100_soft
alpha = 100
soft correction bound = 0.10
correction transform = 0.10 * tanh(raw_correction / 0.10)
fileNumber is grouping metadata only, never a predictor
```
