# 研究路线：每一步解决什么问题

本项目的价值既在于候选修正，也在于逐步建立对“分辨率改善”更严格的判断方法。下表只比较研究问题与验证设计，不跨样本排名数值。

| 阶段 | 做了什么 | 学到什么、为什么继续 | 依据 |
|---|---|---|---|
| 基础分析与早期流程 | S1/S2 组合、位置和漂移时间依赖、事件级 OOF | 样本内留出不能证明跨 run 泛化 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/energy_resolution_analysis/INDEPENDENT_PROTOCOL.md) |
| v9–v11 | 完整采集文件分组、嵌套选参、样条 Ridge 与修正幅度约束 | v11 得到较低 OOF 核心宽度，但峰尾和新 run 泛化仍未解决 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/01_%E8%80%81%E5%B8%88%E7%9B%B2%E6%B5%8B%E4%BA%A4%E4%BB%98/PandaX_candidate_v11_blindtest/MODEL_CARD.md) |
| v12–v15 | 本底检查、能量刻度锚定、小模型与变量比较 | 需要兼顾谱形、时间稳定性和修正幅度 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/energy_resolution_analysis/results/v15_multivariable_20260814/README_%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E.md) |
| v16–v17 | 非线性模型与融合、时间留出 | 部分开发集改善在留出段恶化；增加模型复杂度没有稳定收益 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/energy_resolution_analysis/results/v17_ensemble_20260814/README_%E7%BB%A7%E7%BB%AD%E6%8F%90%E5%8D%87%E7%BB%93%E8%AE%BA.md) |
| v18–v19 | 平滑响应图、增量特征 | v19 最终候选未通过时间留出，新增变量没有被接受 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/energy_resolution_analysis/results/v19_incremental_features_20260815/README_%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E.md) |
| v20 | 分别拟合 S1/S2 漂移时间响应 | 连续谱的单能约束不足；全谱指标与能标检查不支持部署 | [原记录](https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction/blob/e98ca8445220d948650019bedcded83c6909d856/energy_resolution_analysis/results/v20_physics_audit_bl0_20260815/README_%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E.md) |
| v21 | 使用新 Doke 参数及 Kr 单能数据，比较全局与时间匹配残差图 | 局部方案更有希望，但边界、峰位与新 run 验证仍是限制 | [结果解读](RESULTS.md) |

## 三条贯穿全程的经验

1. **核心峰变窄不等于整个能谱更可信。** 峰位、尾部、人造结构与背景表现需要同时检查。
2. **更复杂的模型不一定更可靠。** 时间留出和独立标定能够揭示开发集分数无法说明的问题。
3. **响应修正必须尊重标定信息。** Kr 单峰有助于识别残余位置/时间响应，但不能独自建立全能区刻度。

原记录保存了当时的表述和局限。现在的审查结论见[验证说明](VALIDATION.md)，完整旧材料见[归档索引](../history/README.md)。
