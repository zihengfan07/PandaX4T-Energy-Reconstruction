> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](../../../../README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# candidate_v9 模型卡

## 冻结模型

- 模型：平滑样条基函数 + Ridge 回归
- 正则强度：\(\alpha=100\)
- 修正方式：乘性对数修正
- 平滑上限：

  \[
  \delta_{\mathrm{bounded}}
  =
  0.10\tanh\left(\frac{\delta_{\mathrm{raw}}}{0.10}\right)
  \]

- 训练事件：Run 10972 的4499个有效事件
- 分组：985个 `fileNumber`，按五个连续200-file区间进行外层验证
- `fileNumber`、run/event编号、绝对时间和绝对S1/S2幅度均不进入修正模型

## Run 10972内部验证

以12种预先固定的峰拟合协议汇总，五个外层完整文件块的中位数：

```text
current formula sigma/mu: 4.1179%
candidate_v9 sigma/mu:    2.6287%
relative improvement:     36.16%
```

五个外层块全部改善。峰中心偏差、\(R_{90}\)、有限正值、平滑修正上限、
file-cluster bootstrap 和结构单调性门槛全部通过。

合并 OOF 结果：

```text
current formula R68: 6.7861%
candidate_v9 R68:    3.1123%
current formula R90: 13.0694%
candidate_v9 R90:    6.6932%
```

## 限制

- 仍然只在 Run 10972 内部验证；
- 尚未证明可迁移到其他 run；
- 合并 OOF 峰的拟合优度仍不理想；
- 不能把训练数据应用结果当成独立验证；
- 新 run 首次输出必须冻结保存，之后才能决定是否进入下一开发版本。
