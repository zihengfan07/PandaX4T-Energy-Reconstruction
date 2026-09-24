> 历史研究材料：文中的“当前/推荐”只代表该材料形成时的状态。最新结论与公开版范围请看[项目首页](README.md)。原始数据、逐事件输出和二进制模型不随公开版提供。

# PandaX-4T 暑假项目导航

当前工作区按“报告、老师交付、分析工程、原始数据、历史归档”整理。

## 现在最常用的内容

### 1. 汇报 PDF

目录：`00_报告与说明/`

- `07_PandaX4T_candidate_v11_核心汇报.pdf`：9 页精简版，口头汇报优先使用；
- `06_PandaX4T_candidate_v11_完整技术报告.pdf`：完整方法与证据链；
- `01`--`04`：原始数据、处理方法和早期结果专题报告；
- `归档/`：被替代的历史报告。

### 2. 老师运行模型

目录：`01_老师盲测交付/`

- `PandaX_candidate_v11_blindtest/`：当前冻结模型和运行程序；
- `PandaX_candidate_v11_blindtest_FINAL_20260731.zip`：可发送的 v11 压缩包；
- `90_服务器部署工具/`：bl-0 上传、验证和老师专用包维护脚本。

bl-0 上推荐老师复制的自带依赖版本：

```text
/home/researcher/PandaX_candidate_v11_teacher_ready
```

### 3. 当前分析工程

目录：`energy_resolution_analysis/`

- `foundation_environment_v11.py`：v11 基础能量与环境修正主程序；
- `grouped_nested_physics_v9.py`：分组验证、峰拟合和模型公共函数；
- `server_runs/run10972_foundation_environment_v11/`：当前 v11 结果；
- `complete_report/`、`core_report/`：完整报告和核心汇报的 LaTeX 源文件；
- `cluster_jobs/`：当前 v11 集群任务；
- `archive/`：v2--v10 历史脚本、运行输出和任务文件。

### 4. 原始输入

```text
light_ana_run10972_finalSS_Egt2MeV_scalar.txt
```

该文件仍保留在项目根目录，避免破坏现有分析脚本的相对路径。不要直接修改。

### 5. 历史与缓存

目录：`99_归档与缓存/`

- `历史模型与材料/`：v8--v10 盲测包等旧材料；
- `可删除缓存_20260731/`：PDF 渲染页、临时 Python 环境和冒烟测试副本。

`可删除缓存_20260731/` 不参与当前模型、报告或老师盲测，确认不再追溯排版过程后
可整体删除。

## 当前顶层结构

```text
暑假/
├─ 00_报告与说明/                  正式 PDF 与报告说明
├─ 01_老师盲测交付/                当前 v11 交付与部署工具
├─ energy_resolution_analysis/     当前分析代码、图表和 LaTeX
├─ 99_归档与缓存/                  历史材料与可删除缓存
├─ light_ana_run10972_...txt        Run 10972 标量输入
└─ README_项目导航.md              本导航
```

## 当前结论边界

candidate_v11 在 Run 10972 的分组 OOF 验证中得到
\(\sigma/\mu=0.9973\%\)。它是当前候选模型，不是已经由独立新 run 证明的最终模型。
