> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX candidate_v10 独立数据验证说明

## 这份程序做什么

程序把冻结的 `candidate_v10` 应用于一份模型从未见过的新 run，并用完全相同的评价流程比较：

1. 当前线性公式；
2. 老师提供的三次多项式 `Energy_cor`；
3. 稀疏候选模型 `candidate_v10`。

模型只使用 20 个变量。`fileNumber` 仅保留作事件追踪，不参与预测。

## 输入

输入为制表符分隔的标量 TXT，字段名和单位应与
`light_ana_run10972_finalSS_Egt2MeV_scalar.txt` 一致。程序会检查必需字段，
缺列时直接报错，不会猜测或替换变量。

## Linux

```bash
conda env create -f environment.yml
conda activate pandax-eres-validation
chmod +x run_validation.sh
./run_validation.sh /path/to/new_run_scalar.txt
```

## Windows

```bat
run_validation.bat C:\path\to\new_run_scalar.txt
```

## 输出

`validation_output/` 中会生成：

- `validation_summary.json`：三种重建方法的汇总指标；
- `validation_protocols.csv`：12 种固定峰拟合协议的逐协议结果；
- `validation_events.csv`：逐事件三种能量及模型修正幅度。

重点查看 `sigma_median`（即多种固定协议下的 \(\sigma/\mu\) 中位数）、
`sigma_worst`、`r68`、`r90` 和 `near_cap_fraction`。

## 盲测纪律

首次运行后请原样保存并返回整个 `validation_output/`。在报告首次结果前不要：

- 用新 run 重新训练或重新选择变量；
- 使用新 run 的峰位重新标定；
- 改变拟合窗口、背景模型或删选事件；
- 因结果不好而重复调参。

否则该 run 将不再是独立盲测。

## 当前证据边界

`candidate_v10` 已在 Run 10972 内按五个互不重叠的完整 `fileNumber`
连续块完成嵌套验证，但尚未经过独立新 run 验证。因此它是待盲测候选模型，
不是已证明适用于所有 run 的最终重建公式。
