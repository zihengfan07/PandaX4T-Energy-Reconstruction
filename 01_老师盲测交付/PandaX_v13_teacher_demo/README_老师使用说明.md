> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX v13 教师演示包

运行：

```bash
cd ~
/home/researcher/PandaX_v13_teacher_demo_20260807/run.sh /完整路径/待处理的_scalar.txt
```

输出：

```text
~/v13_output/energy_before_after.txt
```

逐事件文件只有两列，以空格分隔：

- `energy_before_kev`：给定三次公式 `Energy_cor`；
- `energy_after_kev`：v13 小幅漂移时间修正后的能量。

本版本不需要 `joblib` 或 `scikit-learn`，只需要 NumPy 和 Pandas。

注意：这是保守研究候选，并非已经证明全能区最优的正式物理模型。当前交叉验证改善非常小，主要用于让老师检查处理流程和逐事件输出。
