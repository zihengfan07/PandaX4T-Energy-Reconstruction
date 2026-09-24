> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX v12_safe 使用说明

运行：

```bash
chmod +x run_validation.sh
./run_validation.sh /完整路径/待处理的_scalar.txt
```

逐事件结果位于：

```text
validation_output/validation_events.txt
```

该文件以空格分隔。主要列：

- `foundation_energy_kev`：仅由稳健的 `qS1ub_C`、`qS2Bdesub_C` 构成的基础能量。
- `candidate_v12_energy_kev`：v12_safe 最终能量。
- `correction_applied`：该事件是否应用几何修正。
- `out_of_distribution`：参量超出训练范围；此时自动退回基础能量。
- `strong_raw_fallback`：模型建议修正过强；此时自动退回基础能量。
- `correction_near_cap`：是否接近修正上限。冻结验证中该比例为 0。

本版本不需要 `joblib` 或 `scikit-learn`。需要 Python 3、NumPy、Pandas。

注意：模型只用 2614.5 keV 单条标定线训练。本底样本只用于否决假峰方案，没有被赋予能量标签。若要证明全能区通用性，仍需多条已知能量线进行独立验证。
