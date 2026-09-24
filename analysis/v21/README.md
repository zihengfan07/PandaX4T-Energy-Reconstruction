# 原始 v21 研究实现

这五个脚本保留原研究代码，SHA-256 由[来源记录](../../results/provenance.json)核对。没有借整理仓库之机改变分折、拟合、模型选择或既有结果。

| 脚本 | 作用 |
|---|---|
| `new_doke_v21_survey.py` | 检查新数据与已有响应修正 |
| `kr_residual_correction_v21.py` | 公共计算函数及全局 Kr run 留出方案 |
| `audit_v21_candidates_on_2615.py` | 在高能样本上诊断全局候选，不用于反向选模 |
| `time_matched_kr_v21.py` | 固定 run 配对的局部 Kr 事件四折方案 |
| `bootstrap_time_matched_v21.py` | 冻结前后能量的配对事件重采样 |

推荐通过 `python scripts/run_analysis.py --help` 使用新增的输入检查和输出保护。直接运行原始脚本不会自动获得这些检查；新 run 需要研究设计上的重新验证。
