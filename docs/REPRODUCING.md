# 从阅读结果到复现分析

有三种不同的使用方式。请根据是否持有实验数据选择；它们能验证的内容不同。

## A. 查看并核对已保存结果

仅需 Python 3.10+，不安装第三方库：

```bash
python scripts/summarize.py
python scripts/summarize.py --json
python scripts/check_project.py
```

摘要从汇总 JSON 计算点估计、峰位、绝对峰宽及 bootstrap 区间。检查器核对 16 个原始代码/证据文件的 SHA-256、结果内部一致性、Python 语法和本地文档链接。这些检查不会重新拟合峰。

## B. 运行人工数据演示

演示和实验复现已在 Python 3.10.11 验证，建议使用 Python 3.10；其他 Python 版本未验证。在仓库根目录建立隔离环境：

```bash
python -m venv .venv
```

Windows PowerShell：`.venv/Scripts/Activate.ps1`。Linux/macOS：`source .venv/bin/activate`。

```bash
python -m pip install --no-compile -r requirements.txt
python scripts/demo.py --output local_results/demo
```

演示使用固定种子 20260924，生成 18,000 个人工 Kr 事件和 7,500 个人工高能事件，写成真实 TTree 格式，再运行原始时间匹配及配对 bootstrap 脚本。默认只做 12 次重采样，目的是检查完整程序流程，不能用来估计可信物理区间。

```text
local_results/demo/
  SYNTHETIC_DATA.json          明确的人工数据标记
  synthetic_kr.root            人工输入
  synthetic_2615.root
  analysis/
    run_record.json           数据指纹、环境、脚本指纹、状态及耗时
    time_matched_summary.json 修正候选、响应图、峰拟合与覆盖
    time_matched_bootstrap.json
    06_...png 至 10_...png    分析图表
    *.log                    两步运行的输出与报错
```

这些人工数组不是探测器模拟，也不是研究证据。复用了历史 run 标签，仅为了测试固定配对的程序。真实证据一直位于 `results/v21/`，不会被演示覆盖。

输出目录已存在时，演示会停止。重跑请指定新目录，例如 `local_results/demo-2`。

## C. 使用获得授权的实验数据

先核对[输入数据合同](DATA.md)。把数据放在不进入 Git 的 `data/`，运行：

```bash
python scripts/check_inputs.py --kr data/kr.root --th2615 data/2615.root
python scripts/run_analysis.py --kr data/kr.root --th2615 data/2615.root --output local_results/real-v21 --iterations 150
```

默认入口针对原研究的固定 P1/P2 配对。缺少必要 run、有未映射高能 run、字段缺失、非有限值或统计量明显不足时会报出可读错误；它不会静默丢事件、重命名 run 或覆盖已有结果。

输出的 `run_record.json` 记录：输入文件名与 SHA-256、各 run 事件数、依赖版本、源代码指纹、bootstrap 次数、每步返回码及耗时。`status: complete` 只表示两个程序步骤成功执行，不表示通过科学验收。

## 其他原研究命令

需要复现全局方案时单独运行：

```bash
python analysis/v21/kr_residual_correction_v21.py --kr data/kr.root --th2615 data/2615.root --output local_results/global
```

可选调查脚本 `new_doke_v21_survey.py` 需要额外的 Kr 修正因子分支；`audit_v21_candidates_on_2615.py` 只用于诊断，不能用高能检查结果反向挑选候选。它们与全局脚本保留历史行为，没有新增入口的输出保护；务必使用新输出目录。

## 环境与检查

`requirements.txt` 固定此次实际验证的四个直接依赖版本。它是当前 v21 运行环境，不是所有历史版本的锁定环境，也没有锁定所有传递依赖。历史 scikit-learn 等环境请查旧归档。

```bash
python -m unittest discover -s tests -v
python scripts/check_project.py
```

七项针对性检查覆盖误导性结果摘要、非有限输入、未映射 run、缺少 ROOT 树、合并时段统计量和输出覆盖保护。数值复现与软件运行检查的区别见[验证边界](VALIDATION.md)。
