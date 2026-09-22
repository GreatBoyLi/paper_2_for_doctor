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
            subprocess.run([python_executable, str(step.script)], cwd=step.script.parent, check=True)
        except subprocess.CalledProcessError as error:
            print(f"\n[{index}/{len(steps)}] 失败：{step.name}", file=sys.stderr)
            raise SystemExit(error.returncode) from error

        elapsed = time.perf_counter() - step_start
        print(f"[{index}/{len(steps)}] 完成：{step.name}，用时 {elapsed:.1f} 秒", flush=True)

    total = time.perf_counter() - pipeline_start
    print(f"\n{title}完成，总用时 {total:.1f} 秒。")
