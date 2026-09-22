import math

import numpy as np
import pandas as pd


WINDOW_TIME_KEYS = ("x_start_times", "x_end_times", "y_start_times", "y_end_times")


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
