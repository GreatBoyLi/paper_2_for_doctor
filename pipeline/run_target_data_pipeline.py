from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from pipeline.runner import PipelineStep, run_pipeline


STEPS = [
    PipelineStep("目标域原始数据清洗", PROJECT_DIR / "target_data_preprocess/prepare_target_data.py"),
    PipelineStep("提取目标域连续7天数据", PROJECT_DIR / "target_data_preprocess/extract_target_period.py"),
    PipelineStep("生成目标域训练集、验证集和滑动窗口", PROJECT_DIR / "dataset/prepare_target_dataset.py"),
    PipelineStep("生成覆盖四季度的目标域测试集", PROJECT_DIR / "dataset/prepare_target_test_dataset.py"),
    PipelineStep("使用目标域训练数据构图", PROJECT_DIR / "graph/build_target_graph.py"),
]


def main():
    run_pipeline("目标域一键数据处理", STEPS)


if __name__ == "__main__":
    main()
