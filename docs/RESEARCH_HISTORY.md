# 研究路线与版本时间线

这份时间线按现存脚本、结果说明与报告整理；保留历史结果，不把旧候选重新认定为可部署模型。

| 阶段 | 主要工作 | 证据入口 | 结论边界 |
|---|---|---|---|
| 早期分析 | 基础观测量、S1/S2 组合、响应诊断、事件级 OOF、非参数宽度 | [历史归档](../energy_resolution_analysis/archive/)、[早期协议](../energy_resolution_analysis/INDEPENDENT_PROTOCOL.md) | 预选样本内改善不等于跨 run 泛化；部分早期流程已废止 |
| v2–v10 | 特征与公式搜索、重复 OOF、分组嵌套验证、软边界校正 | [实验代码](../energy_resolution_analysis/archive/历史实验脚本_截至v10/)、[历史运行汇总](../energy_resolution_analysis/archive/历史模型运行输出_截至v10/) | 指标依赖各阶段样本与拟合协议 |
| v11，7 月末 | 基础能量路径、环境变量、样条 Ridge、分组验证、交付准备 | [主程序](../energy_resolution_analysis/foundation_environment_v11.py)、[模型卡](../01_老师盲测交付/PandaX_candidate_v11_blindtest/MODEL_CARD.md) | Run 10972 合并 OOF σ/μ 约 0.9973%，没有独立新 run 证明 |
| v12–v13，8 月初 | 本底检查、可移植模型、能量刻度锚定与问题诊断 | [v12 结果](../energy_resolution_analysis/results/v12_safe_20260803/)、[v13 结果](../energy_resolution_analysis/results/v13_anchored_20260807/) | 为处理早期模型的泛化和本底表现继续探索 |
| v14–v15，8 月 14 日 | 小模型比较、多变量搜索 | [v14 说明](../energy_resolution_analysis/results/v14_smallmodels_20260814/README_结果说明.md)、[v15 说明](../energy_resolution_analysis/results/v15_multivariable_20260814/README_结果说明.md) | 数值最优候选仍需新的标定峰及 run 验证 |
| v16–v19，8 月 14–15 日 | 时间留出、非线性、模型融合、响应图及增量变量 | [v16–v17 结论](../energy_resolution_analysis/results/v17_ensemble_20260814/README_继续提升结论.md)、[v18](../energy_resolution_analysis/results/v18_response_map_20260814/README_结果说明.md)、[v19](../energy_resolution_analysis/results/v19_incremental_features_20260815/README_结果说明.md) | 部分开发集改善未通过时间留出；复杂模型未带来稳定增益 |
| v20，8 月 15 日 | 分离 S1/S2 漂移时间响应与物理修正 | [v20 审计](../energy_resolution_analysis/results/v20_physics_audit_bl0_20260815/README_结果说明.md) | 连续谱缺乏可靠单能约束，留出宽度与能标表现不支持部署 |
| v21，8 月 22 日 | 新 Doke 参数、Kr 残余响应、统一模型否定结果、时间匹配方案及 bootstrap | [v21 最新结论](../energy_resolution_analysis/results/v21_kr_residual_correction_20260822/LATEST_V21_FINDINGS_20260822.md) | 时间匹配方向有希望；低 dt 覆盖与修正触顶仍待独立数据验证 |

## 验证方法如何演进

1. 事件级 OOF：避免同一事件直接参与自己的训练预测，但不能证明时间泛化。
2. 完整采集文件分组、嵌套选参：减少采集相关性对评估的影响。
3. 时间留出与本底谱检查：审查新时期失效和人造谱峰。
4. 单独的 Kr 标定：用 Kr 选择残差修正，再到不同能量的 2615 keV 样本检查。
5. 重采样、逐 run 指标和覆盖诊断：共同衡量候选结果的统计稳定性及适用范围。

v21 全局方案按 Kr run 留出；时间匹配方案在指定 Kr 时段内按事件随机分为四折。这两种验证不能混称为相同的跨 run 验证。

## 后续研究需求

- 补充 P1 低漂移时间 Kr 数据，检查 0.97 下限附近的响应。
- 用多个已知能量峰独立约束能量非线性。
- 用未参与任何模型选择的新 run 做最终盲测。
- 同时报告峰位、逐 run σ/μ、谱形、覆盖范围和修正触顶比例。
