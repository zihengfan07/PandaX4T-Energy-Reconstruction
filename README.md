# PandaX-4T 实验能量重建与分辨率优化研究

**从 S1/S2 能量组合、多变量响应校正，到时间匹配 Kr 标定的暑期研究记录。**

This repository documents a student research project on PandaX-4T energy reconstruction, detector-response corrections, and validation. It contains analysis code, historical reports, aggregate results, and figures. Raw detector data are not distributed. This is a research archive, not an official PandaX software release or a validated production calibration.

本仓库按研究阶段整理现有工作，既保留取得改善的方案，也保留未通过验证的实验。研究材料截至 **2026-08-22**；公开整理日期为 **2026-09-24**。

## 从哪里开始

| 你想了解 | 阅读入口 |
|---|---|
| 整个项目做了什么 | [研究路线与版本时间线](docs/RESEARCH_HISTORY.md) |
| 最新结果及其局限 | [v21 最新研究结论](energy_resolution_analysis/results/v21_kr_residual_correction_20260822/LATEST_V21_FINDINGS_20260822.md) |
| 报告与各阶段材料 | [报告索引](docs/REPORTS.md) |
| 如何运行代码 | [环境、数据格式与复现步骤](docs/REPRODUCIBILITY.md) |
| 公开版包含与省略了什么 | [数据与发布范围](docs/DATA_AND_RELEASE_SCOPE.md) |

## 研究问题与方法

液氙探测器的 S1、S2 信号可能残留位置、漂移时间、脉冲形状及采集时期相关的响应差异。本项目研究如何建立能量重建和小幅响应校正，并通过留出样本、跨 run 检查、峰形诊断及重采样，判断分辨率改善是否可靠。

- **能量重建与基线分析：**读取实验标量数据，探索 S1/S2 组合、能谱拟合与非参数宽度指标。
- **模型与验证：**比较线性、样条、Ridge、非线性及融合方案；逐步引入分组验证、时间留出、本底谱检查和修正幅度约束。
- **物理响应修正：**分别分析 S1/S2 随漂移时间等变量的变化，检查单能标定信息不足造成的限制。
- **Kr 标定：**用时间相邻的 Kr 数据学习小幅残差修正，再检查独立的 2615 keV 样本。

技术栈：Python、NumPy、Pandas、SciPy、Matplotlib、scikit-learn、Uproot；报告使用 LaTeX。

## 最新阶段：v21 时间匹配 Kr 修正

基础能量公式为

$$E_{\mathrm{rec}}=0.0137\left(\frac{qS1_{\mathrm{PCs}}}{0.128755}+\frac{qS2Bdes_{\mathrm{PCs}}}{8.6125}\right)\ \mathrm{keV}.$$

输入 PCs 信号已有一轮修正，因此这里只研究小幅残余响应。统一覆盖全部时期的 Kr 模型未达到预设改善门槛；按时间匹配 Kr run 的候选方案得到以下检查结果：

| 2615 keV 样本 | 修正前 σ/μ | 修正后 σ/μ | 相对改善 |
|---|---:|---:|---:|
| 合并样本 | 2.278% | 1.947% | 14.5% |
| run 9563 | 2.516% | 2.218% | 11.8% |
| run 9566 | 2.887% | 2.323% | 19.5% |
| run 9698 | 1.712% | 1.702% | 0.56% |

**这些数值属于诊断性候选结果。** P1 时段约 22.8% 的事件达到 S2 修正下限 0.97；run 9698 的改善区间包含零。run 9566 的 150 次 bootstrap 中有 110 次成功拟合，区间基于这 110 次计算。当前结果尚不能证明新 run 泛化，也不能替代多峰绝对能标标定。合并样本还受到 run 间峰位差异影响，应同时查看逐 run 结果。

![v21 各 run 峰位与分辨率对比](energy_resolution_analysis/results/v21_kr_residual_correction_20260822/08_run_center_and_resolution.png)

详细依据：[拟合及覆盖统计](energy_resolution_analysis/results/v21_kr_residual_correction_20260822/time_matched_summary.json)、[bootstrap 统计](energy_resolution_analysis/results/v21_kr_residual_correction_20260822/time_matched_bootstrap.json)、[覆盖与触顶图](energy_resolution_analysis/results/v21_kr_residual_correction_20260822/09_dt_coverage_and_factor_limits.png)。

## 目录

```text
docs/                         项目导览、时间线、复现与发布范围
energy_resolution_analysis/   分析脚本及历史版本
  results/                    各阶段汇总结果与图表
  archive/                    早期实验、历史输出和集群脚本
00_报告与说明/                 可公开的 PDF 和 LaTeX 报告
01_老师盲测交付/               历史运行程序、模型说明及可读参数
99_归档与缓存/历史模型与材料/    更早的交付材料（已排除缓存）
scripts/                      无需原始数据的结果摘要与完整性校验
```

旧材料中的“当前推荐”“最终”等表述只代表当时的阶段状态。v11、v15 和 v21 的样本、能量定义和验证方法不同，不能按分辨率数值直接排名。

## 快速浏览与运行

无需实验数据或第三方 Python 包即可核对已保存结果：

```bash
git clone https://github.com/zihengfan07/PandaX4T-Energy-Reconstruction.git
cd PandaX4T-Energy-Reconstruction
python scripts/show_results.py
python scripts/verify_snapshot.py
```

完整分析依赖原始数据及相应访问授权。请按[复现指南](docs/REPRODUCIBILITY.md)配置，不要将公开版理解为包含数据的一键复现实验包。

## 数据与使用

原始 ROOT/TXT、逐事件输出、含逐事件信息的 HTML、二进制 joblib 模型、服务器管理脚本及本地缓存不包含在公开仓库中。报告中的汇总图表、可读模型参数和研究结论用于解释研究过程。原始本地材料保持不变。

本仓库不代表 PandaX 合作组的官方结论。尚未授予额外的开源许可证或实验数据再分发权；使用和再分发请先与仓库维护者确认相应权利。
