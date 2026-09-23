from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import config as project_config
from dataset.prepare_target_dataset import save_window_batch
from dataset.split_utils import build_window_batch


# ============================================================
# 1. 路径与参数
# ============================================================

TARGET_DIR = PROJECT_DIR / "data/target/processed_target_final"
INPUT_CSV = TARGET_DIR / "08_TARGET_CLEAN_STRICT_UNNORMALIZED.csv"
OUTPUT_DIR = project_config.TARGET_DATASET_DIR
STATION_ORDER_FILE = OUTPUT_DIR / "TARGET_STATION_ORDER.csv"
Q99_FILE = OUTPUT_DIR / "q99_per_station.csv"
SPLIT_SUMMARY_FILE = OUTPUT_DIR / "TARGET_SPLIT_SUMMARY.csv"
INPUT_STEPS = 16
OUTPUT_STEPS = 16
INTERVAL_MINUTES = 15


def validate_periods(periods, forbidden_start=None, forbidden_end=None):
    validated = []
    previous_end = None

    for period in periods:
        quarter = str(period["quarter"])
        start = pd.Timestamp(period["start"]).normalize()
        end = pd.Timestamp(period["end"]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=INTERVAL_MINUTES)

        if start > end:
            raise ValueError(f"{quarter}测试开始日期晚于结束日期。")
        if start.quarter != end.quarter or quarter != f"Q{start.quarter}":
            raise ValueError(f"{quarter}测试日期与自然季度不一致。")
        if previous_end is not None and start <= previous_end:
            raise ValueError("目标域测试时段重叠或顺序错误。")
        if forbidden_start is not None and forbidden_end is not None:
            forbidden_start = pd.Timestamp(forbidden_start)
            forbidden_end = pd.Timestamp(forbidden_end)
            if start <= forbidden_end and end >= forbidden_start:
                raise ValueError(f"{quarter}测试时段与目标域训练或验证时间重叠。")

        validated.append({"quarter": quarter, "start": start, "end": end})
        previous_end = end

    return validated


def build_test_blocks(data, periods, station_names, q99):
    data = data.copy()
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data.sort_index()

    if data.index.duplicated().any():
        raise RuntimeError("目标域完整数据存在重复时间戳。")
    if len(station_names) != len(set(station_names)):
        raise RuntimeError("目标域训练站点顺序存在重复名称。")

    missing_stations = [station for station in station_names if station not in data.columns]
    if missing_stations:
        raise RuntimeError(f"目标域完整数据缺少站点：{missing_stations}")

    q99 = pd.Series(q99, index=q99.index, dtype=np.float64).reindex(station_names)
    if q99.isna().any() or not np.isfinite(q99.to_numpy()).all() or (q99 <= 0).any():
        raise RuntimeError("目标域训练集Q99不完整或不是正数。")

    raw_blocks = []
    normalized_blocks = []
    summary_records = []

    for period in validate_periods(periods):
        expected_index = pd.date_range(period["start"], period["end"], freq=f"{INTERVAL_MINUTES}min")
        raw_block = data.loc[period["start"]:period["end"], station_names].copy()

        if not raw_block.index.equals(expected_index):
            raise RuntimeError(f"{period['quarter']}测试数据不是严格15分钟连续时间轴。")
        if not np.isfinite(raw_block.to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{period['quarter']}测试数据存在NaN或Inf。")

        normalized_block = raw_block.divide(q99, axis="columns")
        total_steps = INPUT_STEPS + OUTPUT_STEPS
        sample_count = len(raw_block) - total_steps + 1
        if sample_count <= 0:
            raise RuntimeError(f"{period['quarter']}测试时段不足以生成滑动窗口。")

        raw_blocks.append(raw_block)
        normalized_blocks.append(normalized_block)
        summary_records.append({
            "quarter": period["quarter"],
            "start": period["start"],
            "end": period["end"],
            "days": len(raw_block) // (24 * 4),
            "time_steps": len(raw_block),
            "samples": sample_count,
        })

    return raw_blocks, normalized_blocks, pd.DataFrame(summary_records)


def combine_blocks(blocks, periods):
    labeled_blocks = []
    for block, period in zip(blocks, periods):
        labeled = block.copy()
        labeled.insert(0, "quarter", period["quarter"])
        labeled_blocks.append(labeled)
    return pd.concat(labeled_blocks)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"
    station_names = pd.read_csv(STATION_ORDER_FILE)["NodeID"].astype(str).tolist()
    q99_frame = pd.read_csv(Q99_FILE, dtype={"NodeID": str})
    q99 = q99_frame.set_index("NodeID")["Q99"].reindex(station_names)
    split_summary = pd.read_csv(SPLIT_SUMMARY_FILE, parse_dates=["start", "end"])

    forbidden_start = split_summary["start"].min()
    forbidden_end = split_summary["end"].max()
    periods = validate_periods(project_config.TARGET_TEST_PERIODS, forbidden_start, forbidden_end)
    raw_blocks, normalized_blocks, summary = build_test_blocks(data, periods, station_names, q99)
    test_batch = build_window_batch(normalized_blocks, INPUT_STEPS, OUTPUT_STEPS)

    save_window_batch(OUTPUT_DIR / "test.npz", test_batch, station_names, q99)
    combine_blocks(raw_blocks, periods).to_csv(OUTPUT_DIR / "test_timeseries_raw.csv")
    combine_blocks(normalized_blocks, periods).to_csv(OUTPUT_DIR / "test_timeseries_normalized.csv")
    summary.to_csv(OUTPUT_DIR / "TARGET_TEST_PERIODS.csv", index=False)

    print("\n" + "=" * 75)
    print("目标域四季度测试集生成完成")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("站点数量：", len(station_names))
    print("测试天数：", int(summary["days"].sum()))
    print("测试样本：", len(test_batch["X"]))
    print("测试集只使用目标域训练集Q99，不重新计算归一化参数。")
    print("数据输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
