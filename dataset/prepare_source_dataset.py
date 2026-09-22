from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from dataset.split_utils import build_window_batch, split_source_months


# ============================================================
# 1. 路径与参数
# ============================================================

INPUT_CSV = PROJECT_DIR / "data/source/final/source_2014/04_SOURCE_2014_FINAL.csv"
OUTPUT_DIR = PROJECT_DIR / "data/source/processed_source/source_2014/model_dataset"
INPUT_STEPS = 16
OUTPUT_STEPS = 16
TRAIN_RATIO = 0.8
EXPECTED_2014_STEPS = 365 * 24 * 4


def save_window_batch(file, batch, station_names):
    np.savez_compressed(
        file,
        X=batch["X"], Y=batch["Y"],
        x_start_time=batch["x_start_times"], x_end_time=batch["x_end_times"],
        y_start_time=batch["y_start_times"], y_end_time=batch["y_end_times"],
        station_names=np.asarray(station_names),
    )


def validate_source_data(data):
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = data.sort_index()
    if data.index.duplicated().any():
        raise RuntimeError("源域数据存在重复时间戳。")
    if data.columns.duplicated().any():
        raise RuntimeError("源域数据存在重复节点名。")
    if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("源域数据存在NaN或Inf。")

    expected_index = pd.date_range(data.index.min(), data.index.max(), freq="15min")
    if not data.index.equals(expected_index):
        raise RuntimeError("源域时间轴不是严格15分钟连续时间轴。")
    if len(data) != EXPECTED_2014_STEPS:
        raise RuntimeError("源域2014年时间点数量不等于35040。")
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("读取源域2014年数据")
    print("=" * 75)

    data = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"
    data = validate_source_data(data)
    station_names = data.columns.astype(str).tolist()

    print("数据形状：", data.shape)
    print("时间范围：", data.index.min(), "->", data.index.max())
    print("节点数量：", len(station_names))

    station_order = pd.DataFrame({
        "node_index": np.arange(len(station_names)),
        "NodeID": station_names,
    })
    station_order.to_csv(OUTPUT_DIR / "SOURCE_STATION_ORDER.csv", index=False)

    train_blocks, val_blocks, split_summary = split_source_months(data, TRAIN_RATIO)
    train_data = pd.concat(train_blocks).sort_index()
    val_data = pd.concat(val_blocks).sort_index()
    train_batch = build_window_batch(train_blocks, INPUT_STEPS, OUTPUT_STEPS)
    val_batch = build_window_batch(val_blocks, INPUT_STEPS, OUTPUT_STEPS)

    split_summary["train_time_steps"] = [len(block) for block in train_blocks]
    split_summary["val_time_steps"] = [len(block) for block in val_blocks]
    split_summary.to_csv(OUTPUT_DIR / "SOURCE_SPLIT_SUMMARY.csv", index=False)

    save_window_batch(OUTPUT_DIR / "train.npz", train_batch, station_names)
    save_window_batch(OUTPUT_DIR / "val.npz", val_batch, station_names)
    train_data.to_csv(OUTPUT_DIR / "train_timeseries.csv")
    val_data.to_csv(OUTPUT_DIR / "val_timeseries.csv")

    print("\n每月划分：")
    print(split_summary.to_string(index=False))
    print("\nSource训练时间点：", len(train_data))
    print("Source验证时间点：", len(val_data))
    print("Source训练样本：", len(train_batch["X"]))
    print("Source验证样本：", len(val_batch["X"]))
    print("训练跳过样本：", train_batch["skipped_samples"])
    print("验证跳过样本：", val_batch["skipped_samples"])
    print("\n数据输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
