> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX candidate_v11 独立数据验证

该程序把冻结的 `candidate_v11` 应用于一份从未参与开发的新 run，并统一比较：

1. 当前线性公式；
2. 现有三次多项式 `Energy_cor`；
3. 训练侧选择的 image foundation；
4. `candidate_v11`。

Linux：

```bash
conda env create -f environment.yml
conda activate pandax-eres-validation
chmod +x run_validation.sh
./run_validation.sh /path/to/new_run_scalar.txt
```

Windows：

```bat
run_validation.bat C:\path\to\new_run_scalar.txt
```

首次运行后请原样保留 `validation_output/`。在第一次报告结果前，不得在新 run 上：

- 重新训练、调参或选择变量；
- 根据峰中心重新标定；
- 改变 12 种峰拟合协议；
- 删除表现不好的事件或文件；
- 因结果不理想而重复优化后仍称为盲测。

Run 10972 内部结果不能替代独立新 run。v11 使用更丰富的图样和环境变量，
跨 run 稳定性必须通过这一步确认。

## bl-0 上已经部署的版本

服务器上供老师复制的自带依赖版本：

```text
/home/researcher/PandaX_candidate_v11_teacher_ready
```

老师使用自己的 bl-0 账号运行：

```bash
cd ~
cp -a /home/researcher/PandaX_candidate_v11_teacher_ready .
cd PandaX_candidate_v11_teacher_ready
./run_validation.sh /老师的新数据/xxx_scalar.txt
```

结果保存在：

```text
~/PandaX_candidate_v11_teacher_ready/validation_output
```

重点文件为 `validation_summary.json`、`validation_protocols.txt` 和
`validation_events.txt`。两个 TXT 文件使用空格分隔，便于 ROOT 或普通文本程序读取；
同时保留同内容的 CSV 文件。请在运行前确认输入确实是未参与模型开发的新 run。
