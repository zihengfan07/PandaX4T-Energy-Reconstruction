# v21 证据目录

所有 JSON 和 PNG 均从原公开快照逐字节保留；不是合成演示输出。先看[结果解读](../../docs/RESULTS.md)，再定位原始证据。

| 文件 | 回答的问题 |
|---|---|
| [summary.json](summary.json) | 全局方案比较了哪些候选、为什么没有选择修正？ |
| [time_matched_summary.json](time_matched_summary.json) | P1/P2 选择了什么、响应图是什么、逐 run 拟合与覆盖怎样？ |
| [time_matched_bootstrap.json](time_matched_bootstrap.json) | 有效重采样次数、区间及改善为正的比例是多少？ |
| [2615_candidate_audit.json](2615_candidate_audit.json) | 各全局候选作用于高能样本后发生了什么？仅用于诊断。 |
| [residual_maps.json](residual_maps.json) | 全局方案最终选中的额外响应图；本次为空。 |
| [01：全局 Kr 比较](01_kr_cross_validation.png) | 全局候选的留出表现 |
| [06：局部 Kr 比较](06_time_matched_kr_selection.png) | 时间匹配候选的局部验证 |
| [07：高能谱](07_time_matched_2615_comparison.png) | 修正前后完整峰形变化 |
| [08：逐 run 指标](08_run_center_and_resolution.png) | 峰位与 σ/μ 如何同时变化 |
| [09：覆盖与边界](09_dt_coverage_and_factor_limits.png) | 为什么边界命中和 dt 覆盖限制结论 |
| [10：重采样](10_time_matched_bootstrap.png) | 冻结修正后统计稳定性 |

历史图的标题和标注保留原样；更准确的统计解释以[验证说明](../../docs/VALIDATION.md)为准。
