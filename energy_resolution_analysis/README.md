> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX-4T 独立能量重建工作区

本目录只读使用上一级的 `light_ana_run10972_finalSS_Egt2MeV_scalar.txt`，不会修改原始 TXT。

## 当前推荐候选（2026-07-29）

当前推荐用于独立新 run 盲测的是 `candidate_v9`，实现位于：

- `grouped_nested_physics_v9.py`
- `apply_grouped_physics_v9.py`
- `server_runs/run10972_grouped_nested_v9_soft/`

v9 使用 `fileNumber` 作为完整采集文件分组键，但绝不作为预测特征。它采用五个连续外层
文件块、训练池内嵌套选参、12种峰拟合协议、file-cluster bootstrap、峰中心和尾部门槛，
最终冻结为平滑 spline + Ridge 模型，修正采用 0.10 的 `tanh` 软边界。

严格外层文件块的 \(\sigma/\mu\) 中位数由 4.1179% 降至 2.6287%，五个文件块全部
改善，七项验收门全部通过。该结果仍不是独立新 run 验证。

## 先阅读

- `INDEPENDENT_REPORT.md`：当前事件级结果、风险与下一步；
- `INDEPENDENT_PROTOCOL.md`：不依赖旧工作的冻结规则；
- `DATA_REQUEST.md`：下一批数据与元信息申请清单；
- `../00_报告与说明/`：三册正式 PDF。

## 推荐复现顺序

```powershell
python .\energy_resolution_analysis\channel_physics_matrix.py
python .\energy_resolution_analysis\basic_experiment_matrix.py
python .\energy_resolution_analysis\robust_peak_evaluation.py `
  --input .\energy_resolution_analysis\basic_matrix_outputs\oof_predictions.csv `
  --bootstrap 200 `
  --output .\energy_resolution_analysis\basic_matrix_outputs\robust_peak_screen.json
```

## 当前协议

- 物理观测量仅为 `qS1_max`、`qS2B_max`、`dt`、`xS2T_max`、`yS2T_max`；
- `sourceRow` 是读入后生成的唯一对齐键，不进入模型；
- 4,500 个有效事件全部保留，不按能量残差或质量分数删事件；
- 固定种子 10972 的平衡随机事件级五折 OOF，每折 900 个事件；
- 主指标为全事件非参数 R68/R90；
- 不确定度为冻结 OOF 预测上的 1,000 次配对事件 bootstrap；
- 参数化峰宽只有在 bootstrap GOF 通过时才可作为补充；当前两条优胜谱形均未通过。

## 主要输出

- `channel_physics_outputs/`：A0--A4、逐折结果、OOF 事件、响应诊断和配对事件 bootstrap；
- `basic_matrix_outputs/`：全部基础方法、内层选择、逐折结果、OOF 事件和配对事件 bootstrap；
- `stage_report/`：原始数据报告与处理方法报告的 LaTeX 源文件和基础数据图；
- `report/`：数据结果分析报告的 LaTeX 源文件与报告辅助脚本；
- `archive/2026-07-17_fileNumber_version/`：已废止的旧分析快照，仅供追溯，不作为当前结论。

当前结果只说明 run10972 的预选样本内事件级 OOF 改善；正式物理结论仍需无阈值样本和完全独立 run。
