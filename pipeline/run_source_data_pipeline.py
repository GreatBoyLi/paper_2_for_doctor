from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from pipeline.runner import PipelineStep, run_pipeline


STEPS = [
    PipelineStep("源域原始数据清洗与15分钟归一化", PROJECT_DIR / "source_data_preprocess/source_data.py"),
    PipelineStep("提取并整理源域2014全年数据", PROJECT_DIR / "source_data_preprocess/prepare_source_2014.py"),
    PipelineStep("生成源域训练集、验证集和滑动窗口", PROJECT_DIR / "dataset/prepare_source_dataset.py"),
    PipelineStep("使用源域训练数据构图", PROJECT_DIR / "graph/build_source_graph.py"),
]


def main():
    run_pipeline("源域一键数据处理", STEPS)


if __name__ == "__main__":
    main()
