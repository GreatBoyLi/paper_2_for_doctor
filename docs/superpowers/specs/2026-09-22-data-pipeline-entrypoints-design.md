# 源域与目标域一键数据处理入口设计

## 目标

增加两个 Python 总入口，分别按正确顺序调用现有源域和目标域数据处理脚本。用户只需运行一条命令，即可完成对应数据域从原始数据清洗到构图的全流程。

一键处理不包含 Node2Vec、Stage 1、Stage 2 或任何模型训练。

## 方案

总入口使用 `subprocess` 逐个启动现有脚本，不将旧脚本直接导入。原因是部分旧脚本在导入时就会执行，并且使用基于脚本目录的相对路径。

每个子脚本使用启动总入口的同一个 Python 解释器，并以子脚本所在目录作为工作目录。因此在 macOS 和 Ubuntu 的 `powersystem` 环境中都可使用，且不会破坏现有相对路径。

## 文件结构

```text
pipeline/
├── runner.py
├── run_source_data_pipeline.py
└── run_target_data_pipeline.py
```

- `runner.py`：通用的步骤执行器，负责路径检查、进度显示、用时统计和失败传递。
- `run_source_data_pipeline.py`：定义源域的处理步骤和显示名称。
- `run_target_data_pipeline.py`：定义目标域的处理步骤和显示名称。

## 源域处理顺序

1. `source_data_preprocess/source_data.py`：读取源域原始数据，生成15分钟归一化数据。
2. `source_data_preprocess/prepare_source_2014.py`：提取2014全年并完成站点质量处理。
3. `dataset/prepare_source_dataset.py`：按月生成80%训练集和20%验证集及滑动窗口。
4. `graph/build_source_graph.py`：只使用源域训练数据构图。

## 目标域处理顺序

1. `target_data_preprocess/prepare_target_data.py`：读取目标域原始数据并生成严格清洗数据。
2. `target_data_preprocess/extract_target_period.py`：提取2021-03-07至2021-03-13的连续7天。
3. `dataset/prepare_target_dataset.py`：按5天训练、2天验证划分，并仅使用训练集计算Q99。
4. `graph/build_target_graph.py`：只使用目标域训练数据构图。

`analyze_target_data.py` 和 `find_complete_target_weeks.py` 是分析和辅助选时段脚本，其输出不被当前正式处理链读取，因此不纳入一键流程。

## 执行行为

运行命令：

```bash
python pipeline/run_source_data_pipeline.py
python pipeline/run_target_data_pipeline.py
```

入口脚本可以从任意工作目录启动。每个步骤开始和完成时都打印序号、名称和用时。

如果某个脚本不存在，总入口在启动任何步骤前立即报错。如果某个子脚本返回非零退出码，总入口保留原子脚本输出、标明失败步骤并立即停止，不运行后续步骤。

总入口不删除旧文件。现有子脚本继续负责创建目录和覆盖自己的输出。

## 验证要求

- 通用执行器使用 `sys.executable` 启动子脚本。
- 子脚本工作目录等于子脚本所在目录。
- 步骤按定义顺序执行。
- 子脚本失败后不再执行后续步骤。
- 源域和目标域入口分别包含四个已确定步骤。
- 单元测试不使用真实大数据，使用临时小脚本检查顺序、工作目录和失败行为。
