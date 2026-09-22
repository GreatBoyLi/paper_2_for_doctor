from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from graph.graph_utils import build_and_save_graph


DATASET_DIR = PROJECT_DIR / "data/target/processed_target_final/model_dataset"
OUTPUT_DIR = PROJECT_DIR / "data/target/processed_target_final/graph"
INPUT_FILE = DATASET_DIR / "train_timeseries_normalized.csv"
STATION_ORDER_FILE = DATASET_DIR / "TARGET_STATION_ORDER.csv"
TOP_K = 5
REMOVE_ALL_ZERO_TIMESTAMPS = True
POSITIVE_CORRELATION_ONLY = True
ADD_SELF_LOOPS = True


def main():
    station_names = pd.read_csv(STATION_ORDER_FILE)["NodeID"].astype(str).tolist()
    train_data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True).sort_index()
    result = build_and_save_graph(
        train_data, station_names, OUTPUT_DIR, "TARGET_GRAPH_SUMMARY.csv",
        TOP_K, REMOVE_ALL_ZERO_TIMESTAMPS,
        POSITIVE_CORRELATION_ONLY, ADD_SELF_LOOPS,
    )

    print("\n" + "=" * 75)
    print("目标域训练图构建完成")
    print("=" * 75)
    print("节点数量：", len(station_names))
    print("构图时间点：", result["graph_steps"])
    print("无向边数量（不含自环）：", result["edge_count"])
    print("最小Degree：", int(result["degree"].min()))
    print("最大Degree：", int(result["degree"].max()))
    print("平均Degree：", float(result["degree"].mean()))
    print("输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
