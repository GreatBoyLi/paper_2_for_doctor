# 源域与目标域一键数据处理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加两个一键入口，分别按顺序运行源域和目标域现有数据处理及构图脚本。

**Architecture:** `pipeline/runner.py` 提供通用步骤执行器，负责预检查、同解释器启动、工作目录、进度和失败传递。两个入口脚本只定义各自四个步骤，不重复数据处理逻辑。

**Tech Stack:** Python 3.12、`subprocess`、`pathlib`、`unittest`。

---

### Task 1: 实现通用流程执行器

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/runner.py`
- Create: `tests/test_pipeline_runner.py`

- [ ] **Step 1: 写顺序、解释器、工作目录和失败停止测试**

```python
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pipeline.runner import PipelineStep, run_pipeline


class PipelineRunnerTests(unittest.TestCase):
    def test_steps_use_current_interpreter_and_script_directory_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one/first.py"
            second = root / "two/second.py"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("print('first')", encoding="utf-8")
            second.write_text("print('second')", encoding="utf-8")
            steps = [PipelineStep("第一步", first), PipelineStep("第二步", second)]

            with patch("pipeline.runner.subprocess.run") as run:
                run_pipeline("测试流程", steps, python_executable="test-python")

            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["test-python", str(first)])
            self.assertEqual(run.call_args_list[0].kwargs, {"cwd": first.parent, "check": True})
            self.assertEqual(run.call_args_list[1].args[0], ["test-python", str(second)])

    def test_failure_stops_later_steps_and_keeps_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = [root / f"{name}.py" for name in ("one", "two", "three")]
            for script in scripts:
                script.write_text("", encoding="utf-8")
            steps = [PipelineStep(str(index), script) for index, script in enumerate(scripts)]

            failure = subprocess.CalledProcessError(7, ["python", str(scripts[1])])
            with patch("pipeline.runner.subprocess.run", side_effect=[None, failure]) as run:
                with self.assertRaises(SystemExit) as result:
                    run_pipeline("测试流程", steps)

            self.assertEqual(result.exception.code, 7)
            self.assertEqual(run.call_count, 2)
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `python -m unittest tests.test_pipeline_runner -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: 实现最小通用执行器**

```python
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path


def validate_steps(steps):
    missing = [step.script for step in steps if not step.script.is_file()]
    if missing:
        files = "\n".join(str(file) for file in missing)
        raise FileNotFoundError(f"以下数据处理脚本不存在：\n{files}")


def run_pipeline(title, steps, python_executable=None):
    steps = list(steps)
    validate_steps(steps)
    python_executable = python_executable or sys.executable
    pipeline_start = time.perf_counter()
    print("\n" + "=" * 75)
    print(title)
    print("解释器：", python_executable)
    print("=" * 75)

    for index, step in enumerate(steps, start=1):
        step_start = time.perf_counter()
        print(f"\n[{index}/{len(steps)}] 开始：{step.name}", flush=True)
        try:
            subprocess.run(
                [python_executable, str(step.script)], cwd=step.script.parent, check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"\n[{index}/{len(steps)}] 失败：{step.name}", file=sys.stderr)
            raise SystemExit(error.returncode) from error
        elapsed = time.perf_counter() - step_start
        print(f"[{index}/{len(steps)}] 完成：{step.name}，用时 {elapsed:.1f} 秒", flush=True)

    total = time.perf_counter() - pipeline_start
    print(f"\n{title}完成，总用时 {total:.1f} 秒。")
```

- [ ] **Step 4: 运行通用执行器测试**

Run: `python -m unittest tests.test_pipeline_runner -v`

Expected: all tests pass.

- [ ] **Step 5: 提交通用执行器**

```bash
git add pipeline/__init__.py pipeline/runner.py
git add -f tests/test_pipeline_runner.py
git commit -m "增加数据处理流程执行器"
```

### Task 2: 增加源域和目标域总入口

**Files:**
- Create: `pipeline/run_source_data_pipeline.py`
- Create: `pipeline/run_target_data_pipeline.py`
- Create: `tests/test_data_pipeline_entrypoints.py`

- [ ] **Step 1: 写两域步骤顺序测试**

```python
import unittest

from pipeline.run_source_data_pipeline import STEPS as SOURCE_STEPS
from pipeline.run_target_data_pipeline import STEPS as TARGET_STEPS


class DataPipelineEntrypointsTests(unittest.TestCase):
    def test_source_pipeline_has_the_four_required_steps_in_order(self):
        self.assertEqual(
            [step.script.name for step in SOURCE_STEPS],
            ["source_data.py", "prepare_source_2014.py",
             "prepare_source_dataset.py", "build_source_graph.py"],
        )

    def test_target_pipeline_has_the_four_required_steps_in_order(self):
        self.assertEqual(
            [step.script.name for step in TARGET_STEPS],
            ["prepare_target_data.py", "extract_target_period.py",
             "prepare_target_dataset.py", "build_target_graph.py"],
        )

    def test_every_configured_script_exists(self):
        for step in SOURCE_STEPS + TARGET_STEPS:
            self.assertTrue(step.script.is_file(), str(step.script))
```

- [ ] **Step 2: 运行测试并确认入口模块不存在**

Run: `python -m unittest tests.test_data_pipeline_entrypoints -v`

Expected: FAIL with missing entrypoint module.

- [ ] **Step 3: 实现源域入口**

```python
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from pipeline.runner import PipelineStep, run_pipeline


STEPS = [
    PipelineStep("源域原始数据清洗与15分钟归一化",
                 PROJECT_DIR / "source_data_preprocess/source_data.py"),
    PipelineStep("提取并整理源域2014全年数据",
                 PROJECT_DIR / "source_data_preprocess/prepare_source_2014.py"),
    PipelineStep("生成源域训练集、验证集和滑动窗口",
                 PROJECT_DIR / "dataset/prepare_source_dataset.py"),
    PipelineStep("使用源域训练数据构图",
                 PROJECT_DIR / "graph/build_source_graph.py"),
]


def main():
    run_pipeline("源域一键数据处理", STEPS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 实现目标域入口**

```python
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from pipeline.runner import PipelineStep, run_pipeline


STEPS = [
    PipelineStep("目标域原始数据清洗",
                 PROJECT_DIR / "target_data_preprocess/prepare_target_data.py"),
    PipelineStep("提取目标域连续7天数据",
                 PROJECT_DIR / "target_data_preprocess/extract_target_period.py"),
    PipelineStep("生成目标域训练集、验证集和滑动窗口",
                 PROJECT_DIR / "dataset/prepare_target_dataset.py"),
    PipelineStep("使用目标域训练数据构图",
                 PROJECT_DIR / "graph/build_target_graph.py"),
]


def main():
    run_pipeline("目标域一键数据处理", STEPS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行入口配置测试**

Run: `python -m unittest tests.test_data_pipeline_entrypoints -v`

Expected: all tests pass without processing real data.

- [ ] **Step 6: 提交两域入口**

```bash
git add pipeline/run_source_data_pipeline.py pipeline/run_target_data_pipeline.py
git add -f tests/test_data_pipeline_entrypoints.py
git commit -m "增加两域一键数据处理入口"
```

### Task 3: 全量验证

**Files:**
- Modify only if verification exposes a directly related defect.

- [ ] **Step 1: 运行新流程测试**

Run:

```bash
/Users/liwenpeng/miniconda3/envs/powersystem/bin/python -m unittest \
  tests.test_pipeline_runner tests.test_data_pipeline_entrypoints -v
```

Expected: 5 tests pass.

- [ ] **Step 2: 运行全部测试**

Run: `/Users/liwenpeng/miniconda3/envs/powersystem/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 3: 编译新入口并检查版本状态**

Run:

```bash
/Users/liwenpeng/miniconda3/envs/powersystem/bin/python -m compileall -q pipeline
git status --short
```

Expected: compile command exits with code 0; Git status contains no uncommitted pipeline or test changes and may retain the user's existing `.DS_Store`.

- [ ] **Step 4: 进行入口帮助性检查**

Run:

```bash
python -c "from pipeline.run_source_data_pipeline import STEPS; print([step.name for step in STEPS])"
python -c "from pipeline.run_target_data_pipeline import STEPS; print([step.name for step in STEPS])"
```

Expected: each command prints four steps in the documented order without starting real data processing.
