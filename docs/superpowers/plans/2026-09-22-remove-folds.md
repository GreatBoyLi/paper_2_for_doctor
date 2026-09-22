# 取消源域与目标域 Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将源域和目标域从四个 Fold 改为各自唯一的训练集和验证集，并消除数据、图、Node2Vec、训练和评估代码中的 Fold 路径依赖。

**Architecture:** 新增一个小型数据划分工具模块，负责源域按月 80/20 分块和滑动窗口合并；目标域固定为前 5 天训练、后 2 天验证。数据和图直接写入现有根目录，下游脚本改为单路径读取；旧 `fold_*` 输出保留但不再使用。

**Tech Stack:** Python 3.12、pandas、NumPy、PyTorch、NetworkX、gensim、unittest。

---

### Task 1: 建立可测试的分块与滑动窗口工具

**Files:**
- Create: `dataset/split_utils.py`
- Create: `tests/test_dataset_split_utils.py`

- [ ] **Step 1: 先写源域按月划分和边界测试**

```python
import unittest

import numpy as np
import pandas as pd

from dataset.split_utils import build_window_batch, split_source_months


class DatasetSplitUtilsTests(unittest.TestCase):
    def test_source_months_use_first_floor_eighty_percent_days_for_training(self):
        index = pd.date_range("2014-01-01", "2014-02-28 23:45", freq="15min")
        data = pd.DataFrame({"A": np.arange(len(index))}, index=index)

        train_blocks, val_blocks, summary = split_source_months(data, 0.8)

        self.assertEqual(len(train_blocks), 2)
        self.assertEqual(len(val_blocks), 2)
        self.assertEqual(train_blocks[0].index.max(), pd.Timestamp("2014-01-24 23:45"))
        self.assertEqual(val_blocks[0].index.min(), pd.Timestamp("2014-01-25 00:00"))
        self.assertEqual(train_blocks[1].index.max(), pd.Timestamp("2014-02-22 23:45"))
        self.assertEqual(val_blocks[1].index.min(), pd.Timestamp("2014-02-23 00:00"))
        self.assertEqual(summary["train_days"].tolist(), [24, 22])
        self.assertEqual(summary["val_days"].tolist(), [7, 6])

    def test_windows_never_cross_a_block_boundary(self):
        first = pd.DataFrame(
            {"A": np.arange(40)},
            index=pd.date_range("2014-01-01", periods=40, freq="15min"),
        )
        second = pd.DataFrame(
            {"A": np.arange(40)},
            index=pd.date_range("2014-02-01", periods=40, freq="15min"),
        )

        batch = build_window_batch([first, second], input_steps=16, output_steps=16)

        self.assertEqual(batch["X"].shape[0], 18)
        self.assertFalse(
            any(
                pd.Timestamp(start).month != pd.Timestamp(end).month
                for start, end in zip(batch["x_start_times"], batch["y_end_times"])
            )
        )
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m unittest tests.test_dataset_split_utils -v`

Expected: `ModuleNotFoundError: No module named 'dataset.split_utils'`

- [ ] **Step 3: 实现按月分块和按块生成窗口**

`dataset/split_utils.py` 实现以下公共接口：

```python
import math

import numpy as np
import pandas as pd


WINDOW_TIME_KEYS = (
    "x_start_times", "x_end_times", "y_start_times", "y_end_times",
)


def split_source_months(data, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio必须介于0和1之间。")

    train_blocks = []
    val_blocks = []
    summary_records = []
    for month, month_data in data.groupby(data.index.to_period("M")):
        dates = pd.Index(month_data.index.normalize().unique()).sort_values()
        train_days = math.floor(len(dates) * train_ratio)
        if train_days == 0 or train_days == len(dates):
            raise ValueError(f"{month}无法同时划分训练集和验证集。")

        train_dates = dates[:train_days]
        val_dates = dates[train_days:]
        normalized_index = month_data.index.normalize()
        train_block = month_data[normalized_index.isin(train_dates)]
        val_block = month_data[normalized_index.isin(val_dates)]
        train_blocks.append(train_block)
        val_blocks.append(val_block)
        summary_records.append({
            "month": str(month),
            "train_start": train_block.index.min(),
            "train_end": train_block.index.max(),
            "train_days": len(train_dates),
            "val_start": val_block.index.min(),
            "val_end": val_block.index.max(),
            "val_days": len(val_dates),
        })

    return train_blocks, val_blocks, pd.DataFrame(summary_records)


def create_sliding_windows(data, input_steps, output_steps):
    values = data.to_numpy(dtype=np.float32)
    total_steps = input_steps + output_steps
    arrays = {"X": [], "Y": []}
    times = {key: [] for key in WINDOW_TIME_KEYS}
    skipped_samples = 0

    for start in range(len(data) - total_steps + 1):
        input_end = start + input_steps
        output_end = input_end + output_steps
        x = values[start:input_end]
        y = values[input_end:output_end]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            skipped_samples += 1
            continue
        arrays["X"].append(x)
        arrays["Y"].append(y)
        times["x_start_times"].append(data.index[start])
        times["x_end_times"].append(data.index[input_end - 1])
        times["y_start_times"].append(data.index[input_end])
        times["y_end_times"].append(data.index[output_end - 1])

    if not arrays["X"]:
        raise RuntimeError("当前数据块没有生成有效滑动窗口。")

    return {
        "X": np.stack(arrays["X"]).astype(np.float32),
        "Y": np.stack(arrays["Y"]).astype(np.float32),
        **{key: np.asarray(value, dtype="datetime64[ns]") for key, value in times.items()},
        "theoretical_sample_count": len(data) - total_steps + 1,
        "skipped_samples": skipped_samples,
    }


def build_window_batch(blocks, input_steps, output_steps):
    batches = [create_sliding_windows(block, input_steps, output_steps) for block in blocks]
    return {
        "X": np.concatenate([batch["X"] for batch in batches]),
        "Y": np.concatenate([batch["Y"] for batch in batches]),
        **{
            key: np.concatenate([batch[key] for batch in batches])
            for key in WINDOW_TIME_KEYS
        },
        "theoretical_sample_count": sum(
            batch["theoretical_sample_count"] for batch in batches
        ),
        "skipped_samples": sum(batch["skipped_samples"] for batch in batches),
    }
```

- [ ] **Step 4: 运行分块工具测试**

Run: `python -m unittest tests.test_dataset_split_utils -v`

Expected: `Ran 2 tests ... OK`

- [ ] **Step 5: 提交分块工具**

```bash
git add dataset/split_utils.py tests/test_dataset_split_utils.py
git commit -m "增加无Fold数据划分工具"
```

### Task 2: 将源域预处理改为全年按月 80/20 划分

**Files:**
- Modify: `dataset/prepare_source_dataset.py`
- Create: `tests/test_prepare_source_split.py`

- [ ] **Step 1: 写源域单一划分输出测试**

```python
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_FILE = PROJECT_DIR / "dataset/prepare_source_dataset.py"


class PrepareSourceSplitTests(unittest.TestCase):
    def test_source_preparation_uses_single_root_level_split(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_CONFIGS", text)
        self.assertNotIn("fold_dir", text)
        self.assertIn("split_source_months", text)
        self.assertIn('"SOURCE_SPLIT_SUMMARY.csv"', text)
        self.assertIn('OUTPUT_DIR / "train.npz"', text)
        self.assertIn('OUTPUT_DIR / "val.npz"', text)
```

- [ ] **Step 2: 运行测试并确认旧 Fold 代码导致失败**

Run: `python -m unittest tests.test_prepare_source_split -v`

Expected: FAIL because `FOLD_CONFIGS` and `fold_dir` still exist.

- [ ] **Step 3: 改写源域数据生成主流程**

保留现有数据读取、时区、节点顺序和数值检查，删除 `FOLD_CONFIGS` 及 Fold 循环。在读取并检查 `df` 后使用：

```python
from dataset.split_utils import build_window_batch, split_source_months

TRAIN_RATIO = 0.8

train_blocks, val_blocks, split_summary = split_source_months(df, TRAIN_RATIO)
train_df = pd.concat(train_blocks).sort_index()
val_df = pd.concat(val_blocks).sort_index()
train_batch = build_window_batch(train_blocks, INPUT_STEPS, OUTPUT_STEPS)
val_batch = build_window_batch(val_blocks, INPUT_STEPS, OUTPUT_STEPS)

np.savez_compressed(OUTPUT_DIR / "train.npz", **train_batch, station_names=station_names)
np.savez_compressed(OUTPUT_DIR / "val.npz", **val_batch, station_names=station_names)
train_df.to_csv(OUTPUT_DIR / "train_timeseries.csv")
val_df.to_csv(OUTPUT_DIR / "val_timeseries.csv")
split_summary.to_csv(OUTPUT_DIR / "SOURCE_SPLIT_SUMMARY.csv", index=False)
```

输出时打印 12 个月的训练、验证日期，总训练样本数和总验证样本数。

- [ ] **Step 4: 运行源域结构测试**

Run: `python -m unittest tests.test_prepare_source_split -v`

Expected: `Ran 1 test ... OK`

- [ ] **Step 5: 运行源域预处理并核对输出**

Run from `dataset/`: `python prepare_source_dataset.py`

Expected:
- 月份汇总为 12 行。
- 31 天月份为 24 天训练、7 天验证。
- 30 天月份为 24 天训练、6 天验证。
- 2014 年 2 月为 22 天训练、6 天验证。
- 输出根目录存在 `train.npz`、`val.npz` 和 `SOURCE_SPLIT_SUMMARY.csv`。

- [ ] **Step 6: 提交源域预处理改动**

```bash
git add dataset/prepare_source_dataset.py tests/test_prepare_source_split.py
git commit -m "取消源域Fold数据划分"
```

### Task 3: 将目标域预处理改为 5 天训练和 2 天验证

**Files:**
- Modify: `dataset/prepare_target_dataset.py`
- Create: `tests/test_prepare_target_split.py`

- [ ] **Step 1: 写目标域日期、Q99 和单一输出测试**

```python
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_FILE = PROJECT_DIR / "dataset/prepare_target_dataset.py"


class PrepareTargetSplitTests(unittest.TestCase):
    def test_target_preparation_uses_five_train_days_and_two_validation_days(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_CONFIGS", text)
        self.assertNotIn("fold_dir", text)
        self.assertIn("TRAIN_DAYS = 5", text)
        self.assertIn('"TARGET_SPLIT_SUMMARY.csv"', text)
        self.assertIn('OUTPUT_DIR / "q99_per_station.csv"', text)
        self.assertIn('OUTPUT_DIR / "train.npz"', text)
        self.assertIn('OUTPUT_DIR / "val.npz"', text)
```

- [ ] **Step 2: 运行测试并确认旧 Fold 代码导致失败**

Run: `python -m unittest tests.test_prepare_target_split -v`

Expected: FAIL because the script still contains `FOLD_CONFIGS` and `fold_dir`.

- [ ] **Step 3: 改写目标域数据生成主流程**

保留现有时区、节点顺序、完整性和 Q99 检查，删除 Fold 配置和循环。使用输入第一天自动确定日期：

```python
from dataset.split_utils import build_window_batch

TRAIN_DAYS = 5
VALIDATION_DAYS = 2

first_day = df.index.min().normalize()
train_end = first_day + pd.Timedelta(days=TRAIN_DAYS) - pd.Timedelta(minutes=15)
val_start = first_day + pd.Timedelta(days=TRAIN_DAYS)
val_end = val_start + pd.Timedelta(days=VALIDATION_DAYS) - pd.Timedelta(minutes=15)
train_raw = df.loc[first_day:train_end]
val_raw = df.loc[val_start:val_end]

q99 = train_raw.quantile(Q99_QUANTILE).clip(lower=EPS)
train_norm = train_raw.divide(q99, axis="columns")
val_norm = val_raw.divide(q99, axis="columns")
train_batch = build_window_batch([train_norm], INPUT_STEPS, OUTPUT_STEPS)
val_batch = build_window_batch([val_norm], INPUT_STEPS, OUTPUT_STEPS)

pd.DataFrame({"NodeID": station_names, "Q99": q99.to_numpy()}).to_csv(
    OUTPUT_DIR / "q99_per_station.csv", index=False
)
np.savez_compressed(OUTPUT_DIR / "train.npz", **train_batch, station_names=station_names)
np.savez_compressed(OUTPUT_DIR / "val.npz", **val_batch, station_names=station_names)
train_raw.to_csv(OUTPUT_DIR / "train_timeseries_raw.csv")
val_raw.to_csv(OUTPUT_DIR / "val_timeseries_raw.csv")
train_norm.to_csv(OUTPUT_DIR / "train_timeseries_normalized.csv")
val_norm.to_csv(OUTPUT_DIR / "val_timeseries_normalized.csv")
```

`TARGET_SPLIT_SUMMARY.csv` 保存两行：`train` 和 `validation`，包含开始时间、结束时间、天数、时间点数和样本数。

- [ ] **Step 4: 运行目标域结构测试**

Run: `python -m unittest tests.test_prepare_target_split -v`

Expected: `Ran 1 test ... OK`

- [ ] **Step 5: 运行目标域预处理并核对输出**

Run from `dataset/`: `python prepare_target_dataset.py`

Expected:
- 训练区间为 `2021-03-07 00:00` ～ `2021-03-11 23:45`，480 个时间点。
- 验证区间为 `2021-03-12 00:00` ～ `2021-03-13 23:45`，192 个时间点。
- 训练样本数为 `480 - 16 - 16 + 1 = 449`。
- 验证样本数为 `192 - 16 - 16 + 1 = 161`。
- Q99 仅使用前 5 天计算。

- [ ] **Step 6: 提交目标域预处理改动**

```bash
git add dataset/prepare_target_dataset.py tests/test_prepare_target_split.py
git commit -m "取消目标域Fold数据划分"
```

### Task 4: 将两域构图改为单一训练图

**Files:**
- Modify: `graph/build_source_graph.py`
- Modify: `graph/build_target_graph.py`
- Create: `tests/test_graph_paths_without_folds.py`

- [ ] **Step 1: 写构图路径测试**

```python
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class GraphPathsWithoutFoldsTests(unittest.TestCase):
    def test_graph_builders_use_root_training_timeseries(self):
        source = (PROJECT_DIR / "graph/build_source_graph.py").read_text(encoding="utf-8")
        target = (PROJECT_DIR / "graph/build_target_graph.py").read_text(encoding="utf-8")
        self.assertNotIn("FOLDS", source)
        self.assertNotIn("fold_output_dir", source)
        self.assertIn('DATASET_DIR / "train_timeseries.csv"', source)
        self.assertNotIn("FOLDS", target)
        self.assertNotIn("fold_output_dir", target)
        self.assertIn('DATASET_DIR / "train_timeseries_normalized.csv"', target)
```

- [ ] **Step 2: 运行测试并确认旧循环导致失败**

Run: `python -m unittest tests.test_graph_paths_without_folds -v`

Expected: FAIL because both scripts still contain `FOLDS` and per-fold output directories.

- [ ] **Step 3: 删除两个构图脚本的 Fold 循环**

源域固定使用：

```python
INPUT_FILE = DATASET_DIR / "train_timeseries.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

目标域固定使用：

```python
INPUT_FILE = DATASET_DIR / "train_timeseries_normalized.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

保留现有 Pearson、Top-K、对称化、自环和数值检查，所有 `.npy` 和 `.csv` 直接保存到 `OUTPUT_DIR`。汇总文件中删除 `fold` 列。

- [ ] **Step 4: 运行构图路径测试**

Run: `python -m unittest tests.test_graph_paths_without_folds -v`

Expected: `Ran 1 test ... OK`

- [ ] **Step 5: 运行两个构图脚本并检查输出**

Run from `graph/`:

```bash
python build_source_graph.py
python build_target_graph.py
```

Expected: 两个 `graph` 根目录都生成 `adjacency_binary.npy`，矩阵分别为 `(104, 104)` 和 `(31, 31)`，且对称。

- [ ] **Step 6: 提交构图改动**

```bash
git add graph/build_source_graph.py graph/build_target_graph.py tests/test_graph_paths_without_folds.py
git commit -m "取消两域图的Fold路径"
```

### Task 5: 删除全局 Fold 配置并更新 Node2Vec 与学习演示

**Files:**
- Modify: `config/config.yaml`
- Modify: `config/config.py`
- Modify: `learning/node2vec/inspect_graph.py`
- Modify: `learning/node2vec/build_networkx_graph.py`
- Modify: `learning/node2vec/demo_single_walk.py`
- Modify: `learning/node2vec/demo_multiple_walks.py`
- Modify: `learning/node2vec/demo_all_nodes_one_walk.py`
- Modify: `learning/node2vec/train_node_embeddings.py`
- Modify: `learning/node2vec/train_target_node_embeddings.py`
- Modify: `learning/gin/demo_shared_gin.py`
- Modify: `learning/domain_adversarial/demo_domain_classifier.py`
- Modify: `learning/forecasting/demo_source_forward.py`
- Modify: `tests/test_project_config.py`
- Modify: `tests/test_build_networkx_graph.py`
- Modify: `tests/test_train_target_node_embeddings.py`

- [ ] **Step 1: 先将配置和学习测试改成无 Fold 期望**

`tests/test_project_config.py` 删除 `config.FOLD_ID` 断言，增加：

```python
self.assertNotIn("experiment", settings)
self.assertFalse(hasattr(config, "FOLD_ID"))
```

`tests/test_build_networkx_graph.py` 改为：

```python
adj = np.load(config.GRAPH_DIR / "adjacency_binary.npy")
```

`tests/test_train_target_node_embeddings.py` 将测试名改为 `test_target_graph_trains_31_node_vectors`，删除 `"Fold： 1"` 输出断言。

- [ ] **Step 2: 运行相关测试并确认失败**

Run:

```bash
python -m unittest tests.test_project_config tests.test_build_networkx_graph tests.test_train_target_node_embeddings -v
```

Expected: FAIL because `FOLD_ID` and `fold_1` paths still exist.

- [ ] **Step 3: 删除配置和学习脚本中的 Fold 依赖**

删除 `config.yaml` 中：

```yaml
experiment:
  fold_id: 1
```

删除 `config.py` 中：

```python
FOLD_ID = settings["experiment"]["fold_id"]
```

所有源域图读取统一改为：

```python
adjacency_file = project_config.GRAPH_DIR / "adjacency_binary.npy"
```

所有目标域图读取统一改为：

```python
adjacency_file = project_config.TARGET_GRAPH_DIR / "adjacency_binary.npy"
```

删除演示输出中的 `Fold：` 以及注释中的 `Fold 1`。

- [ ] **Step 4: 运行配置和学习测试**

Run:

```bash
python -m unittest tests.test_project_config tests.test_build_networkx_graph tests.test_train_node_embeddings tests.test_train_target_node_embeddings -v
```

Expected: all tests pass.

- [ ] **Step 5: 确认学习目录不再使用 Fold**

Run: `rg -n "FOLD_ID|fold_|Fold" learning config`

Expected: no matches.

- [ ] **Step 6: 提交配置和学习脚本改动**

```bash
git add config learning tests/test_project_config.py tests/test_build_networkx_graph.py tests/test_train_target_node_embeddings.py
git commit -m "删除Node2Vec和学习脚本的Fold配置"
```

### Task 6: 更新源域基线、Stage 1 和 Stage 2 训练路径

**Files:**
- Modify: `training/train_source_forecaster.py`
- Modify: `training/train_stage1_domain_adversarial.py`
- Modify: `training/train_stage2_target_finetuning.py`
- Modify: `tests/test_train_source_forecaster.py`
- Modify: `tests/test_stage1_domain_adversarial.py`
- Modify: `tests/test_stage2_target_finetuning.py`

- [ ] **Step 1: 写无 Fold 训练路径测试**

在三个现有测试文件中增加脚本文本断言：

```python
text = SCRIPT_FILE.read_text(encoding="utf-8")
self.assertNotIn("FOLD_ID", text)
self.assertNotIn("fold_", text)
```

并断言对应新检查点名存在：

```python
self.assertIn('"source_best.pt"', source_text)
self.assertIn('"stage1_best.pt"', stage1_text)
self.assertIn('"stage2_best.pt"', stage2_text)
```

- [ ] **Step 2: 运行测试并确认旧路径导致失败**

Run:

```bash
python -m unittest tests.test_train_source_forecaster tests.test_stage1_domain_adversarial tests.test_stage2_target_finetuning -v
```

Expected: FAIL because training scripts still use `FOLD_ID` and fold-specific filenames.

- [ ] **Step 3: 更新三个训练入口**

源域基线使用：

```python
dataset_dir = project_config.DATASET_DIR
graph_dir = project_config.GRAPH_DIR
checkpoint_file = project_config.CHECKPOINT_DIR / "source_best.pt"
history_file = project_config.TRAINING_HISTORY_DIR / "source_history.csv"
```

Stage 1 使用：

```python
source_dataset_dir = project_config.DATASET_DIR
source_graph_dir = project_config.GRAPH_DIR
target_graph_dir = project_config.TARGET_GRAPH_DIR
checkpoint_file = project_config.CHECKPOINT_DIR / "stage1_best.pt"
history_file = project_config.TRAINING_HISTORY_DIR / "stage1_history.csv"
```

Stage 2 使用：

```python
target_dataset_dir = project_config.TARGET_DATASET_DIR
stage1_file = project_config.CHECKPOINT_DIR / "stage1_best.pt"
stage2_file = project_config.CHECKPOINT_DIR / "stage2_best.pt"
history_file = project_config.TRAINING_HISTORY_DIR / "stage2_history.csv"
```

删除所有 `Fold：` 打印，其他训练、Early Stopping、Node2Vec 和历史记录逻辑保持不变。

- [ ] **Step 4: 运行三组训练测试**

Run:

```bash
python -m unittest tests.test_train_source_forecaster tests.test_stage1_domain_adversarial tests.test_stage2_target_finetuning -v
```

Expected: all tests pass.

- [ ] **Step 5: 提交训练路径改动**

```bash
git add training tests/test_train_source_forecaster.py tests/test_stage1_domain_adversarial.py tests/test_stage2_target_finetuning.py
git commit -m "更新无Fold的两阶段训练路径"
```

### Task 7: 将当前目标域评估明确为验证集评估

**Files:**
- Modify: `evaluation/evaluate_target_forecaster.py`
- Modify: `tests/test_evaluate_target_forecaster.py`

- [ ] **Step 1: 写无 Fold 验证评估路径测试**

在 `tests/test_evaluate_target_forecaster.py` 增加：

```python
def test_main_uses_validation_data_and_non_fold_outputs(self):
    text = SCRIPT_FILE.read_text(encoding="utf-8")
    self.assertNotIn("FOLD_ID", text)
    self.assertNotIn("fold_", text)
    self.assertIn('TARGET_DATASET_DIR / "val.npz"', text)
    self.assertIn('CHECKPOINT_DIR / "stage2_best.pt"', text)
    self.assertIn('EVALUATION_DIR / "target_validation"', text)
```

- [ ] **Step 2: 运行新测试并确认失败**

Run: `python -m unittest tests.test_evaluate_target_forecaster -v`

Expected: FAIL because evaluation still reads `fold_1` paths.

- [ ] **Step 3: 更新评估入口路径和输出文字**

```python
validation_file = project_config.TARGET_DATASET_DIR / "val.npz"
checkpoint_file = project_config.CHECKPOINT_DIR / "stage2_best.pt"
output_dir = project_config.EVALUATION_DIR / "target_validation"
q99_file = project_config.TARGET_DATASET_DIR / "q99_per_station.csv"
```

输出使用“目标域验证集”，不使用“Fold”或“最终测试集”描述。

- [ ] **Step 4: 运行评估单元测试**

Run: `python -m unittest tests.test_evaluate_target_forecaster -v`

Expected: all tests pass.

- [ ] **Step 5: 提交评估路径改动**

```bash
git add evaluation/evaluate_target_forecaster.py tests/test_evaluate_target_forecaster.py
git commit -m "明确无Fold的目标域验证评估"
```

### Task 8: 全局清查与端到端验证

**Files:**
- Modify only if verification exposes a missed Fold dependency.

- [ ] **Step 1: 扫描活动代码中的 Fold 依赖**

Run:

```bash
rg -n "FOLD_ID|fold_id|fold_|FOLD_CONFIGS|FOLDS|Fold" config dataset graph learning model training evaluation tests --glob '*.py' --glob '*.yaml'
```

Expected: no active-code matches. Historical design documents and existing generated data are outside this scan.

- [ ] **Step 2: 运行完整测试集**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 3: 运行不含长时训练的数据链路检查**

依次运行数据预处理、两域构图、图检查和目标域 Node2Vec 演示：

```bash
(cd dataset && python prepare_source_dataset.py)
(cd dataset && python prepare_target_dataset.py)
(cd graph && python build_source_graph.py)
(cd graph && python build_target_graph.py)
python learning/node2vec/inspect_graph.py
python learning/node2vec/train_target_node_embeddings.py
```

Expected:
- 源域数据为 104 个节点，目标域数据为 31 个节点。
- 目标域训练、验证样本数分别为 449 和 161。
- 两个邻接矩阵无 NaN/Inf 且对称。
- Node2Vec 为目标域生成 `(31, 32)` 向量。

- [ ] **Step 4: 检查最终工作树**

Run: `git status --short`

Expected: 只有用户原有的 `.DS_Store` 未跟踪文件，没有未提交的代码改动。

- [ ] **Step 5: 确认最后一次提交和验证结果对应**

Run: `git log -1 --oneline`

Expected: 最后一次提交是 Task 7 的“明确无Fold的目标域验证评估”，且 Step 1 ～ Step 4 都已通过。如果验证暴露问题，回到问题所属任务的测试、实现和提交步骤，修正后重新执行 Task 8。
