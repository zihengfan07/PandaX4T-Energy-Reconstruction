> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# 报告说明

本目录集中保存 PandaX-4T 暑假项目的正式 PDF。前三册记录早期独立研究流程，
第 04 册回答老师本轮提出的变量精简、方法对比和空间诊断问题；
第 06 册是可独立阅读的完整技术报告，第 07 册是用于口头汇报的核心精简版。

## 01 原始数据报告

文件：`01_PandaX4T_run10972_原始数据报告.pdf`

说明当前上游标量 TXT 的结构、字段、质量审计、S1--S2B 反相关、漂移时间与空间覆盖。回答“拿到的数据是什么”，不评价校正方法性能。

LaTeX：`../energy_resolution_analysis/stage_report/PandaX4T_run10972_raw_data_report.tex`

## 02 数据处理方法报告

文件：`02_PandaX4T_run10972_数据处理方法报告.pdf`

记录输入对齐键、基础变量白名单、固定种子的事件级嵌套五折、训练侧响应学习、OOF 汇总、非参数宽度和配对事件 bootstrap。回答“怎样处理数据并避免测试事件泄漏”。

LaTeX：`../energy_resolution_analysis/stage_report/PandaX4T_run10972_data_processing_report.tex`

## 03 数据结果分析报告

文件：`03_PandaX4T_独立能量重建与数据结果分析.pdf`

集中比较通道物理 A4 与组合响应 challenger，给出消融、事件外折稳定性、事件 bootstrap、尾部、峰形 GOF、负结果和结论边界。回答“改善了多少、是否稳定、目前能得出什么”。

LaTeX：`../energy_resolution_analysis/report/PandaX4T_run10972_data_results_analysis_report.tex`

## 04 老师反馈后模型完善与结果汇报

文件：`04_PandaX4T_老师反馈后模型完善与结果汇报.pdf`

在完整 `fileNumber` 块隔离的嵌套验证中，把候选模型由 30 个变量精简到
20 个变量；统一比较当前线性公式、已有三次式 `Energy_cor` 与
`candidate_v10`；给出一维直方图、二维 \(x-y\) 偏差、变量修改前后趋势、
关键变量排序、五个未见文件块稳定性和下一步独立 run 盲测方案。

LaTeX：`../energy_resolution_analysis/v10_report/PandaX4T_v10_teacher_feedback_report.tex`

盲测程序已归档至：
`../99_归档与缓存/历史模型与材料/盲测旧版本/PandaX_candidate_v10_blindtest/`

## 05 旧完整报告（已由 06 替代）

文件：`归档/历史完整报告/05_PandaX4T_独立能量重建完整技术报告.pdf`

该文件是更新前版本，现已归档。请勿再用于汇报。

## 06 candidate_v11 完整技术报告

文件：`06_PandaX4T_candidate_v11_完整技术报告.pdf`

综合原始数据、数据处理、早期结果以及 v10/v11 模型，
从双相液氙背景、S1/S2B 反相关、文件块隔离、固定峰拟合协议、稀疏样条 Ridge、
关键变量、一维与二维结果、数值安全、模型风险审计、老师盲测步骤到下一步研究，
形成完整证据链。当前 v11 使用训练侧选择的 image foundation 和 64 个受控变量，
合并 OOF \(\sigma/\mu=0.9973\%\)。建议今后的完整汇报优先使用本册。

LaTeX：`../energy_resolution_analysis/complete_report/PandaX4T_independent_energy_reconstruction_complete_report.tex`

盲测程序：`../01_老师盲测交付/PandaX_candidate_v11_blindtest/`

## 07 candidate_v11 核心汇报（推荐口头汇报使用）

文件：`07_PandaX4T_candidate_v11_核心汇报.pdf`

9 页精简版，只保留模型工作原理、参量选择逻辑、冻结超参数、关键参量排序，
以及模型进展、一维能谱、外层文件块、二维位置残差和修正幅度等结果图。
删除长篇背景、早期处理历史和完整变量附录。口头阶段性汇报优先使用本册；
需要追溯完整证据链时再查阅第 06 册。

LaTeX：`../energy_resolution_analysis/core_report/PandaX4T_candidate_v11_core_report.tex`

源文件压缩包：`07_PandaX4T_candidate_v11_核心汇报_LaTeX源文件.zip`

## 统一结果口径

- 不同版本使用的输入层级不同；v11 使用 image-derived S1/S2 foundation，
  以及位置、漂移、脉冲形状、PMT 图样和事件环境变量；
- `sourceRow` 只用于表间一一对齐，不是物理特征；
- 身份元数据不参与重建；第 04 册仅用 `fileNumber` 构造互不重叠的文件块，
  防止相邻采集文件跨入训练集与测试集；
- v11 当前主数值是 Run 10972 上完整 `fileNumber` 块隔离的 12 协议 OOF 拟合宽度；
- 不同版本的基础信号路径和验证设计不同，性能数值应结合对应报告解释；
- 这些数值不是完成绝对 MeV 标定与独立 run 盲测后的最终物理能量分辨率。

## 归档

- `归档/阶段性材料/`：被三册正式报告取代的阶段性汇报；
- `归档/历史版本/`：原 39 页完整技术报告的历史快照；
- `../energy_resolution_analysis/archive/2026-07-17_fileNumber_version/`：已废止的旧分析输出，只用于追溯，不作为当前结论。
