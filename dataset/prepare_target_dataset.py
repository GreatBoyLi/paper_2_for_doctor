from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from dataset.split_utils import build_window_batch


# ============================================================
# 1. 路径与参数
# ============================================================

INPUT_CSV = PROJECT_DIR / "data/target/processed_target_final/selected_period/TARGET_MODEL_DATA.csv"
OUTPUT_DIR = PROJECT_DIR / "data/target/processed_target_final/model_dataset"
INPUT_STEPS = 16
OUTPUT_STEPS = 16
TRAIN_DAYS = 5
VALIDATION_DAYS = 2
Q99_QUANTILE = 0.99
EPS = 1e-8
EXPECTED_STEPS = 7 * 24 * 4


def save_window_batch(file, batch, station_names, q99):
    np.savez_compressed(
        file,
        X=batch["X"], Y=batch["Y"],
        x_start_time=batch["x_start_times"], x_end_time=batch["x_end_times"],
        y_start_time=batch["y_start_times"], y_end_time=batch["y_end_times"],
        station_names=np.asarray(station_names),
        q99=q99.to_numpy(dtype=np.float32),
    )


def validate_target_data(data):
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = data.sort_index()
    if data.index.duplicated().any():
        raise RuntimeError("目标域数据存在重复时间戳。")
    if data.columns.duplicated().any():
        raise RuntimeError("目标域数据存在重复节点名。")
    if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("目标域七天数据存在NaN或Inf。")

    expected_index = pd.date_range(data.index.min(), data.index.max(), freq="15min")
    if not data.index.equals(expected_index):
        raise RuntimeError("目标域时间轴不是严格15分钟连续时间轴。")
    if len(data) != EXPECTED_STEPS:
        raise RuntimeError("目标域当前数据不是完整的672个时间点。")
    if (data.index[0].hour, data.index[0].minute) != (0, 0):
        raise RuntimeError("目标域第一天应该从00:00开始。")
    if (data.index[-1].hour, data.index[-1].minute) != (23, 45):
        raise RuntimeError("目标域最后一天应该到23:45结束。")
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("读取目标域连续七天数据")
    print("=" * 75)

    data = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"
    data = validate_target_data(data)
    station_names = data.columns.astype(str).tolist()

    first_day = data.index.min().normalize()
    train_end = first_day + pd.Timedelta(days=TRAIN_DAYS) - pd.Timedelta(minutes=15)
    val_start = first_day + pd.Timedelta(days=TRAIN_DAYS)
    val_end = val_start + pd.Timedelta(days=VALIDATION_DAYS) - pd.Timedelta(minutes=15)
    train_raw = data.loc[first_day:train_end].copy()
    val_raw = data.loc[val_start:val_end].copy()

    if len(train_raw) != TRAIN_DAYS * 24 * 4:
        raise RuntimeError("目标域训练集不是完整的5天。")
    if len(val_raw) != VALIDATION_DAYS * 24 * 4:
        raise RuntimeError("目标域验证集不是完整的2天。")

    q99 = train_raw.quantile(Q99_QUANTILE).clip(lower=EPS)
    train_norm = train_raw.divide(q99, axis="columns")
    val_norm = val_raw.divide(q99, axis="columns")
    if not np.isfinite(train_norm.to_numpy()).all() or not np.isfinite(val_norm.to_numpy()).all():
        raise RuntimeError("目标域Q99归一化后存在NaN或Inf。")

    train_batch = build_window_batch([train_norm], INPUT_STEPS, OUTPUT_STEPS)
    val_batch = build_window_batch([val_norm], INPUT_STEPS, OUTPUT_STEPS)

    station_order = pd.DataFrame({
        "node_index": np.arange(len(station_names)),
        "NodeID": station_names,
    })
    station_order.to_csv(OUTPUT_DIR / "TARGET_STATION_ORDER.csv", index=False)
    pd.DataFrame({"NodeID": station_names, "Q99": q99.to_numpy()}).to_csv(
        OUTPUT_DIR / "q99_per_station.csv", index=False
    )

    save_window_batch(OUTPUT_DIR / "train.npz", train_batch, station_names, q99)
    save_window_batch(OUTPUT_DIR / "val.npz", val_batch, station_names, q99)
    train_raw.to_csv(OUTPUT_DIR / "train_timeseries_raw.csv")
    val_raw.to_csv(OUTPUT_DIR / "val_timeseries_raw.csv")
    train_norm.to_csv(OUTPUT_DIR / "train_timeseries_normalized.csv")
    val_norm.to_csv(OUTPUT_DIR / "val_timeseries_normalized.csv")

    split_summary = pd.DataFrame([
        {
            "dataset": "train", "start": train_raw.index.min(), "end": train_raw.index.max(),
            "days": TRAIN_DAYS, "time_steps": len(train_raw), "samples": len(train_batch["X"]),
        },
        {
            "dataset": "validation", "start": val_raw.index.min(), "end": val_raw.index.max(),
            "days": VALIDATION_DAYS, "time_steps": len(val_raw), "samples": len(val_batch["X"]),
        },
    ])
    split_summary.to_csv(OUTPUT_DIR / "TARGET_SPLIT_SUMMARY.csv", index=False)

    print("数据形状：", data.shape)
    print("节点数量：", len(station_names))
    print("训练区间：", train_raw.index.min(), "->", train_raw.index.max())
    print("验证区间：", val_raw.index.min(), "->", val_raw.index.max())
    print("训练样本：", len(train_batch["X"]))
    print("验证样本：", len(val_batch["X"]))
    print("Q99只由目标域前5天训练数据计算。")
    print("\n数据输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
