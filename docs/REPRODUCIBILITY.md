# 复现指南

## 公开版可以复现什么

无需原始数据即可阅读报告、绘图结果、候选比较和汇总 JSON，并运行结果摘要及文件哈希校验。完整训练、能谱拟合及统计复现需要另行获得原始数据；本次公开整理没有重新运行所有历史实验。

保留原目录结构，以尽量维护模块导入和报告图片引用。历史脚本可能依赖旧目录布局、Linux 集群或已省略模型；它们用于研究追溯，不能保证全部开箱即用。

## 环境

推荐 Python 3.10 或以上。在仓库根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.venv/Scripts/Activate.ps1`；Linux/macOS 激活：`source .venv/bin/activate`。

```bash
python -m pip install -r requirements.txt
python scripts/show_results.py
python scripts/verify_snapshot.py
```

依赖列表覆盖分析所用主要库。它不是所有历史实验的锁定环境；历史 joblib 模型的二进制兼容性也不作保证。

## v21 输入格式

准备 `kr.root` 和 `2615.root`，两者都包含名为 `out_tree` 的 TTree。放在本地 `data/` 下，该目录由 `.gitignore` 排除。

| 分支 | 用途 |
|---|---|
| `qS1_PCs` | 已完成已有修正的 S1 |
| `qS2Bdes_PCs` | 已完成已有修正的底部 S2 |
| `dt` | 漂移时间；保持原始数据单位及定义 |
| `wS2CDF_max` | S2 宽度变量 |
| `xS2max_desImageMCPAF_firstS2` | 重建 x 位置 |
| `yS2max_desImageMCPAF_firstS2` | 重建 y 位置 |
| `runNumber` | run 标识，用于分组与时段匹配 |

第一步调查脚本 `new_doke_v21_survey.py` 还会读取 Kr 树中的 `qS1_nn_factor` 和 `qS2_nn_factor`（已有修正因子）；其余四个命令不要求这两个额外分支。

原始 v21 数据规模：Kr 787,039 事件、28 个 run；2615 keV 19,036 事件、3 个 run。文件需与原研究的预选、分支定义和单位一致；仅分支同名不足以保证结果相同。

## v21 命令顺序

以下命令从仓库根目录运行；`local_results/` 自动排除出版本控制。每条命令单独运行。

```bash
python energy_resolution_analysis/new_doke_v21_survey.py --kr data/kr.root --th2615 data/2615.root --output local_results/v21_survey
python energy_resolution_analysis/kr_residual_correction_v21.py --kr data/kr.root --th2615 data/2615.root --output local_results/v21_global
python energy_resolution_analysis/audit_v21_candidates_on_2615.py --kr data/kr.root --th2615 data/2615.root --output local_results/v21_candidate_audit
python energy_resolution_analysis/time_matched_kr_v21.py --kr data/kr.root --th2615 data/2615.root --output local_results/v21_time_matched
python energy_resolution_analysis/bootstrap_time_matched_v21.py --input local_results/v21_time_matched/2615_time_matched_output.txt --output local_results/v21_time_matched --iterations 150
```

`audit_v21_candidates_on_2615.py` 用于诊断候选对高能数据的影响，不能据此反过来选择模型。

时间匹配脚本固定了研究中的配对：Kr 9568 → 2615 run 9563/9566；Kr 9685/9695 → 2615 run 9698。用于新数据时必须根据真实采集时间重新定义配对并验证覆盖，不能直接照搬 run 编号。

脚本在 Kr 峰窗选取、按 run 归一化后进行模型比较；时间匹配四折的随机种子是 20260822，修正因子限制在 [0.97, 1.03]。Bootstrap 也使用固定种子，但依赖库、浮点环境与拟合收敛状态会影响细节。复现应比较完整汇总结果，而不仅是四舍五入后的 σ/μ。

## 历史模型

- v11 等二进制 `.joblib` 未公开；历史应用程序需要在获得数据后按对应版本训练或取得匹配的模型。
- v12/v13 的可读 JSON 参数与运行代码保留，用于理解历史模型结构；它们不代表当前推荐模型。
- 旧文档中的本地用户名和路径已替换为示例值。服务器管理脚本、权限设置和账号配置没有发布。
- 报告源文件可能引用未公开的逐事件输出；需要对应原始数据生成后才能重建这些材料。
