> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# 老师盲测交付目录

当前只推荐使用 `candidate_v11`。

## 本地交付

- `PandaX_candidate_v11_blindtest/`：展开后的冻结模型程序；
- `PandaX_candidate_v11_blindtest_FINAL_20260731.zip`：可发送压缩包。

## bl-0 老师专用版本

服务器位置：

```text
/home/researcher/PandaX_candidate_v11_teacher_ready
```

老师使用自己的账号运行：

```bash
cd ~
cp -a /home/researcher/PandaX_candidate_v11_teacher_ready .
cd PandaX_candidate_v11_teacher_ready
./run_validation.sh /新数据的完整路径/xxx_scalar.txt
```

逐事件空格分隔结果为：

```text
validation_output/validation_events.txt
```

其中 `candidate_v11_energy_kev` 是模型最终重建能量。

`90_服务器部署工具/` 仅用于维护服务器版本，不需要发给老师。
