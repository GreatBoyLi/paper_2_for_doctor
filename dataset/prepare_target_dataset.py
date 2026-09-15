from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径设置
#
# Target Domain:
# UNISOLAR
# Australia
#
# 当前输入：
# 已经人工选择的一周数据
#
# 特点：
# 1. 31个目标域节点
# 2. 无NaN
# 3. 尚未归一化
# ============================================================

INPUT_CSV = Path(
    "../data/target/processed_target_final/"
    "selected_period/"
    "TARGET_MODEL_DATA.csv"
)

OUTPUT_DIR = Path(
    "../data/target/processed_target_final/"
    "model_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 时间窗口参数
#
# 15 min数据
#
# 过去16步：
# 16 × 15 min = 4 h
#
# 预测未来16步：
# 16 × 15 min = 4 h
# ============================================================

INPUT_STEPS = 16

OUTPUT_STEPS = 16

# ============================================================
# 3. Q99参数
#
# 每一个Fold：
#
# 只使用该Fold Train计算Q99
#
# Validation绝对不能参与Q99计算。
# ============================================================

Q99_QUANTILE = 0.99

EPS = 1e-8

# ============================================================
# 4. 目标域分折方案
#
# 一周共7天。
#
# Fold 1:
# Train = Day1-Day3
# Val   = Day4
#
# Fold 2:
# Train = Day1-Day4
# Val   = Day5
#
# Fold 3:
# Train = Day1-Day5
# Val   = Day6
#
# Fold 4:
# Train = Day1-Day6
# Val   = Day7
#
# 注意：
#
# 这里不是固定具体日期，
# 而是根据INPUT_CSV的第一天自动生成。
# ============================================================

FOLD_CONFIGS = [

    {
        "fold": 1,
        "train_days": 3,
        "val_day": 4
    },

    {
        "fold": 2,
        "train_days": 4,
        "val_day": 5
    },

    {
        "fold": 3,
        "train_days": 5,
        "val_day": 6
    },

    {
        "fold": 4,
        "train_days": 6,
        "val_day": 7
    },
]


# ============================================================
# 5. 滑动窗口函数
#
# 输入：
#
# data:
# [T, N]
#
# 输出：
#
# X:
# [samples, 16, N]
#
# Y:
# [samples, 16, N]
# ============================================================

def create_sliding_windows(
        data,
        input_steps,
        output_steps
):
    values = (
        data
        .to_numpy(
            dtype=np.float32
        )
    )

    timestamps = (
        data.index
    )

    total_steps = (
            input_steps
            +
            output_steps
    )

    theoretical_sample_count = (
            len(data)
            -
            total_steps
            +
            1
    )

    if theoretical_sample_count <= 0:
        raise ValueError(
            "当前时间段长度不足以生成滑动窗口。"
        )

    X = []

    Y = []

    x_start_times = []

    x_end_times = []

    y_start_times = []

    y_end_times = []

    skipped_samples = 0

    for start_idx in range(
            theoretical_sample_count
    ):

        input_start = (
            start_idx
        )

        input_end = (
                start_idx
                +
                input_steps
        )

        output_start = (
            input_end
        )

        output_end = (
                output_start
                +
                output_steps
        )

        x = values[
            input_start:
            input_end
        ]

        y = values[
            output_start:
            output_end
        ]

        # ====================================================
        # 如果存在NaN / Inf，
        # 当前样本跳过
        #
        # 理论上目标域选择的一周应该没有NaN。
        # ====================================================

        if not np.isfinite(
                x
        ).all():
            skipped_samples += 1

            continue

        if not np.isfinite(
                y
        ).all():
            skipped_samples += 1

            continue

        X.append(
            x
        )

        Y.append(
            y
        )

        x_start_times.append(
            timestamps[
                input_start
            ]
        )

        x_end_times.append(
            timestamps[
                input_end
                -
                1
                ]
        )

        y_start_times.append(
            timestamps[
                output_start
            ]
        )

        y_end_times.append(
            timestamps[
                output_end
                -
                1
                ]
        )

    if len(X) == 0:
        raise RuntimeError(
            "没有生成任何有效滑动窗口。"
        )

    X = (
        np.stack(
            X
        )
        .astype(
            np.float32
        )
    )

    Y = (
        np.stack(
            Y
        )
        .astype(
            np.float32
        )
    )

    return (

        X,

        Y,

        np.array(
            x_start_times,
            dtype="datetime64[ns]"
        ),

        np.array(
            x_end_times,
            dtype="datetime64[ns]"
        ),

        np.array(
            y_start_times,
            dtype="datetime64[ns]"
        ),

        np.array(
            y_end_times,
            dtype="datetime64[ns]"
        ),

        theoretical_sample_count,

        skipped_samples
    )


# ============================================================
# 6. 读取目标域一周数据
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "读取目标域最终一周数据"
)

print(
    "=" * 75
)

df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True
)

df.index.name = (
    "Timestamp"
)

df = (
    df
    .sort_index()
)

# ============================================================
# 7. 如果存在时区，统一去掉时区
# ============================================================

if isinstance(
        df.index,
        pd.DatetimeIndex
):

    if df.index.tz is not None:
        print(
            "\n检测到目标域时间索引带时区：",
            df.index.tz
        )

        df.index = (
            df.index
            .tz_localize(
                None
            )
        )

        print(
            "已去除时区信息。"
        )

# ============================================================
# 8. 基础检查
# ============================================================

print(
    "\n数据形状：",
    df.shape
)

print(
    "时间范围：",
    df.index.min(),
    "->",
    df.index.max()
)

print(
    "节点数量：",
    df.shape[1]
)

total_nan = int(
    df
    .isna()
    .sum()
    .sum()
)

print(
    "NaN总数：",
    total_nan
)

values = (
    df
    .to_numpy(
        dtype=np.float64
    )
)

inf_count = int(
    np.isinf(
        values
    )
    .sum()
)

print(
    "Inf数量：",
    inf_count
)

print(
    "最小值：",
    df.min().min()
)

print(
    "最大值：",
    df.max().max()
)

# ============================================================
# 9. NaN / Inf严格检查
# ============================================================

if total_nan > 0:
    nodes_with_nan = (
        df.columns[
            df.isna()
            .any(axis=0)
        ]
            .tolist()
    )

    print(
        "\n存在NaN的节点："
    )

    print(
        nodes_with_nan
    )

    raise RuntimeError(
        "目标域最终一周仍然存在NaN。"
    )

if inf_count > 0:
    raise RuntimeError(
        "目标域数据中存在Inf。"
    )

# ============================================================
# 10. 节点信息
# ============================================================

station_names = (
    df.columns
    .astype(str)
    .tolist()
)

print(
    "\n节点数量：",
    len(
        station_names
    )
)

print(
    "节点：",
    station_names
)

# ============================================================
# 11. 检查节点名是否重复
# ============================================================

duplicated_station_count = int(
    pd.Index(
        station_names
    )
    .duplicated()
    .sum()
)

print(
    "重复节点名数量：",
    duplicated_station_count
)

if duplicated_station_count > 0:
    raise RuntimeError(
        "目标域存在重复节点名称。"
    )

# ============================================================
# 12. 检查重复时间戳
# ============================================================

duplicate_time_count = int(
    df.index
    .duplicated()
    .sum()
)

print(
    "重复时间戳数量：",
    duplicate_time_count
)

if duplicate_time_count > 0:
    raise RuntimeError(
        "目标域存在重复时间戳。"
    )

# ============================================================
# 13. 检查15分钟连续时间轴
# ============================================================

expected_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="15min"
)

if df.index.equals(
        expected_index
):

    print(
        "\n时间轴严格连续15分钟：是"
    )

else:

    print(
        "\n时间轴严格连续15分钟：否"
    )

    missing_times = (
        expected_index
        .difference(
            df.index
        )
    )

    print(
        "缺失时间戳数量：",
        len(
            missing_times
        )
    )

    if len(
            missing_times
    ) > 0:
        print(
            missing_times[
                :20
            ]
        )

    raise RuntimeError(
        "目标域时间轴不是严格15分钟连续时间轴。"
    )

# ============================================================
# 14. 检查是否完整7天
#
# 7 × 24 × 4
# = 672
# ============================================================

EXPECTED_STEPS = (
        7
        *
        24
        *
        4
)

print(
    "\n一周理论时间点数：",
    EXPECTED_STEPS
)

print(
    "实际时间点数：",
    len(
        df
    )
)

if len(
        df
) != EXPECTED_STEPS:
    raise RuntimeError(
        "\n目标域当前文件不是完整672个15分钟时间点。\n"
        "请检查extract_target_period.py。\n"
        "完整7天应该是：\n"
        "第1天 00:00 -> 第7天 23:45\n"
        "共672个时间点。"
    )

# ============================================================
# 15. 检查起止时间
# ============================================================

if not (
        df.index[0].hour == 0
        and
        df.index[0].minute == 0
):
    raise RuntimeError(
        "目标域第一天应该从00:00开始。"
    )

if not (
        df.index[-1].hour == 23
        and
        df.index[-1].minute == 45
):
    raise RuntimeError(
        "目标域最后一天应该到23:45结束。"
    )

# ============================================================
# 16. 保存节点顺序
#
# 后续：
#
# Target Graph
# GIN
# GRU
#
# 都必须保持这个顺序。
# ============================================================

STATION_FILE = (
        OUTPUT_DIR
        /
        "TARGET_STATION_ORDER.csv"
)

pd.DataFrame(
    {

        "node_index":
            np.arange(
                len(
                    station_names
                )
            ),

        "NodeID":
            station_names
    }
).to_csv(
    STATION_FILE,
    index=False
)

print(
    "\n节点顺序已保存："
)

print(
    STATION_FILE
)

# ============================================================
# 17. 确定Day 1
# ============================================================

first_day = (
    df.index.min()
    .normalize()
)

print(
    "\n目标域Day 1：",
    first_day.date()
)

# ============================================================
# 18. 创建各Fold
# ============================================================

fold_summary_records = []

for config in FOLD_CONFIGS:

    fold_id = (
        config[
            "fold"
        ]
    )

    train_days = (
        config[
            "train_days"
        ]
    )

    val_day = (
        config[
            "val_day"
        ]
    )

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        f"处理 Fold {fold_id}"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # Train时间
    #
    # 永远从Day1开始
    # ========================================================

    train_start = (
        first_day
    )

    train_end = (
            first_day
            +
            pd.Timedelta(
                days=train_days
            )
            -
            pd.Timedelta(
                minutes=15
            )
    )

    # ========================================================
    # Validation时间
    # ========================================================

    val_start = (
            first_day
            +
            pd.Timedelta(
                days=val_day - 1
            )
    )

    val_end = (
            val_start
            +
            pd.Timedelta(
                days=1
            )
            -
            pd.Timedelta(
                minutes=15
            )
    )

    print(
        "Train：",
        train_start,
        "->",
        train_end
    )

    print(
        "Val：",
        val_start,
        "->",
        val_end
    )

    # ========================================================
    # 19. 截取原始数据
    # ========================================================

    train_raw = (
        df.loc[
            train_start:
            train_end
        ]
        .copy()
    )

    val_raw = (
        df.loc[
            val_start:
            val_end
        ]
        .copy()
    )

    print(
        "\nTrain Raw形状：",
        train_raw.shape
    )

    print(
        "Val Raw形状：",
        val_raw.shape
    )

    # ========================================================
    # 20. 检查理论时间长度
    # ========================================================

    expected_train_steps = (
            train_days
            *
            96
    )

    expected_val_steps = (
        96
    )

    if len(
            train_raw
    ) != expected_train_steps:
        raise RuntimeError(
            f"Fold {fold_id} Train时间长度异常。"
        )

    if len(
            val_raw
    ) != expected_val_steps:
        raise RuntimeError(
            f"Fold {fold_id} Val时间长度异常。"
        )

    # ========================================================
    # 21. 只使用Train计算每个节点Q99
    # ========================================================

    q99 = (
        train_raw
        .quantile(
            Q99_QUANTILE,
            axis=0
        )
    )

    # ========================================================
    # 检查Q99
    #
    # 如果某个站Train全是0，
    # 那么Q99=0，无法归一化。
    # ========================================================

    zero_q99_nodes = (
        q99[
            q99
            <=
            EPS
            ]
        .index
        .tolist()
    )

    if len(
            zero_q99_nodes
    ) > 0:
        print(
            "\nQ99接近0的节点："
        )

        print(
            zero_q99_nodes
        )

        raise RuntimeError(
            f"Fold {fold_id} 存在Q99接近0的节点，"
            "无法正常进行Q99归一化。"
        )

    print(
        "\nQ99统计："
    )

    print(
        "最小Q99：",
        q99.min()
    )

    print(
        "最大Q99：",
        q99.max()
    )

    print(
        "平均Q99：",
        q99.mean()
    )

    # ========================================================
    # 22. Q99归一化
    #
    # Train:
    #
    # Train / Q99(Train)
    #
    #
    # Val:
    #
    # Val / Q99(Train)
    #
    #
    # 注意：
    # Validation不能单独计算自己的Q99。
    # ========================================================

    train_norm = (
        train_raw
        .divide(
            q99,
            axis="columns"
        )
    )

    val_norm = (
        val_raw
        .divide(
            q99,
            axis="columns"
        )
    )

    # ========================================================
    # 23. 不进行clip
    #
    # Q99归一化后，
    #
    # > 1
    #
    # 是正常的。
    # ========================================================

    print(
        "\n归一化后Train："
    )

    print(
        "最小值：",
        train_norm.min().min()
    )

    print(
        "最大值：",
        train_norm.max().max()
    )

    print(
        "归一化后Val："
    )

    print(
        "最小值：",
        val_norm.min().min()
    )

    print(
        "最大值：",
        val_norm.max().max()
    )

    # ========================================================
    # 24. 检查NaN / Inf
    # ========================================================

    train_norm_nan = int(
        train_norm
        .isna()
        .sum()
        .sum()
    )

    val_norm_nan = int(
        val_norm
        .isna()
        .sum()
        .sum()
    )

    train_norm_inf = int(
        np.isinf(
            train_norm.to_numpy()
        )
        .sum()
    )

    val_norm_inf = int(
        np.isinf(
            val_norm.to_numpy()
        )
        .sum()
    )

    print(
        "\nTrain Norm NaN：",
        train_norm_nan
    )

    print(
        "Val Norm NaN：",
        val_norm_nan
    )

    print(
        "Train Norm Inf：",
        train_norm_inf
    )

    print(
        "Val Norm Inf：",
        val_norm_inf
    )

    if (
            train_norm_nan > 0
            or
            val_norm_nan > 0
            or
            train_norm_inf > 0
            or
            val_norm_inf > 0
    ):
        raise RuntimeError(
            f"Fold {fold_id} 归一化后出现NaN或Inf。"
        )

    # ========================================================
    # 25. 创建Train滑动窗口
    # ========================================================

    (
        X_train,
        Y_train,

        train_x_start,
        train_x_end,

        train_y_start,
        train_y_end,

        train_theoretical_samples,
        train_skipped_samples

    ) = create_sliding_windows(

        train_norm,

        INPUT_STEPS,

        OUTPUT_STEPS
    )

    # ========================================================
    # 26. 创建Validation滑动窗口
    # ========================================================

    (
        X_val,
        Y_val,

        val_x_start,
        val_x_end,

        val_y_start,
        val_y_end,

        val_theoretical_samples,
        val_skipped_samples

    ) = create_sliding_windows(

        val_norm,

        INPUT_STEPS,

        OUTPUT_STEPS
    )

    print(
        "\nTrain X：",
        X_train.shape
    )

    print(
        "Train Y：",
        Y_train.shape
    )

    print(
        "Val X：",
        X_val.shape
    )

    print(
        "Val Y：",
        Y_val.shape
    )

    print(
        "\nTrain理论样本数：",
        train_theoretical_samples
    )

    print(
        "Train跳过样本数：",
        train_skipped_samples
    )

    print(
        "Train实际样本数：",
        len(
            X_train
        )
    )

    print(
        "\nVal理论样本数：",
        val_theoretical_samples
    )

    print(
        "Val跳过样本数：",
        val_skipped_samples
    )

    print(
        "Val实际样本数：",
        len(
            X_val
        )
    )

    # ========================================================
    # 27. 维度检查
    # ========================================================

    assert (
            X_train.shape[1]
            ==
            INPUT_STEPS
    )

    assert (
            Y_train.shape[1]
            ==
            OUTPUT_STEPS
    )

    assert (
            X_train.shape[2]
            ==
            len(
                station_names
            )
    )

    assert (
            Y_train.shape[2]
            ==
            len(
                station_names
            )
    )

    assert (
            X_val.shape[1]
            ==
            INPUT_STEPS
    )

    assert (
            Y_val.shape[1]
            ==
            OUTPUT_STEPS
    )

    assert (
            X_val.shape[2]
            ==
            len(
                station_names
            )
    )

    assert (
            Y_val.shape[2]
            ==
            len(
                station_names
            )
    )

    # ========================================================
    # 28. 创建Fold目录
    # ========================================================

    fold_dir = (
            OUTPUT_DIR
            /
            f"fold_{fold_id}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 29. 保存Q99参数
    #
    # 后面反归一化也需要它。
    # ========================================================

    q99_df = pd.DataFrame(
        {

            "NodeID":
                station_names,

            "Q99":
                q99.values
        }
    )

    Q99_FILE = (
            fold_dir
            /
            "q99_scaler.csv"
    )

    q99_df.to_csv(
        Q99_FILE,
        index=False
    )

    # ========================================================
    # 30. 保存Train NPZ
    # ========================================================

    TRAIN_FILE = (
            fold_dir
            /
            "train.npz"
    )

    np.savez_compressed(

        TRAIN_FILE,

        X=X_train,

        Y=Y_train,

        x_start_time=train_x_start,

        x_end_time=train_x_end,

        y_start_time=train_y_start,

        y_end_time=train_y_end,

        station_names=np.array(
            station_names
        ),

        q99=q99.to_numpy(
            dtype=np.float32
        )
    )

    # ========================================================
    # 31. 保存Validation NPZ
    # ========================================================

    VAL_FILE = (
            fold_dir
            /
            "val.npz"
    )

    np.savez_compressed(

        VAL_FILE,

        X=X_val,

        Y=Y_val,

        x_start_time=val_x_start,

        x_end_time=val_x_end,

        y_start_time=val_y_start,

        y_end_time=val_y_end,

        station_names=np.array(
            station_names
        ),

        q99=q99.to_numpy(
            dtype=np.float32
        )
    )

    # ========================================================
    # 32. 保存原始时间序列
    #
    # 用于以后：
    #
    # 检查原始功率
    # 反归一化
    # 数据分析
    # ========================================================

    train_raw.to_csv(
        fold_dir
        /
        "train_timeseries_raw.csv"
    )

    val_raw.to_csv(
        fold_dir
        /
        "val_timeseries_raw.csv"
    )

    # ========================================================
    # 33. 保存归一化时间序列
    #
    # 后面Target Graph可以使用：
    #
    # train_timeseries_normalized.csv
    #
    # 只使用Train，
    # 不使用Validation。
    # ========================================================

    train_norm.to_csv(
        fold_dir
        /
        "train_timeseries_normalized.csv"
    )

    val_norm.to_csv(
        fold_dir
        /
        "val_timeseries_normalized.csv"
    )

    # ========================================================
    # 34. 保存Fold信息
    # ========================================================

    fold_info = pd.DataFrame(
        [
            {

                "fold":
                    fold_id,

                "train_start":
                    train_start,

                "train_end":
                    train_end,

                "val_start":
                    val_start,

                "val_end":
                    val_end,

                "train_days":
                    train_days,

                "train_time_steps":
                    len(
                        train_raw
                    ),

                "val_time_steps":
                    len(
                        val_raw
                    ),

                "train_samples":
                    len(
                        X_train
                    ),

                "val_samples":
                    len(
                        X_val
                    ),

                "node_count":
                    len(
                        station_names
                    ),

                "input_steps":
                    INPUT_STEPS,

                "output_steps":
                    OUTPUT_STEPS,

                "q99_quantile":
                    Q99_QUANTILE,

                "train_skipped_samples":
                    train_skipped_samples,

                "val_skipped_samples":
                    val_skipped_samples
            }
        ]
    )

    fold_info.to_csv(
        fold_dir
        /
        "fold_info.csv",
        index=False
    )

    # ========================================================
    # 35. 总体汇总
    # ========================================================

    fold_summary_records.append(
        {

            "fold":
                fold_id,

            "train_start":
                train_start.date(),

            "train_end":
                train_end.date(),

            "val_start":
                val_start.date(),

            "val_end":
                val_end.date(),

            "train_days":
                train_days,

            "train_steps":
                len(
                    train_raw
                ),

            "val_steps":
                len(
                    val_raw
                ),

            "train_samples":
                len(
                    X_train
                ),

            "val_samples":
                len(
                    X_val
                ),

            "node_count":
                len(
                    station_names
                ),

            "train_skipped_samples":
                train_skipped_samples,

            "val_skipped_samples":
                val_skipped_samples
        }
    )

    print(
        "\nFold文件已保存："
    )

    print(
        TRAIN_FILE
    )

    print(
        VAL_FILE
    )

    print(
        Q99_FILE
    )

# ============================================================
# 36. 保存总体Fold汇总
# ============================================================

fold_summary_df = pd.DataFrame(
    fold_summary_records
)

SUMMARY_FILE = (
        OUTPUT_DIR
        /
        "TARGET_FOLD_SUMMARY.csv"
)

fold_summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# 37. 完成
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "目标域数据集构建完成"
)

print(
    "=" * 75
)

print(
    "\nFold汇总："
)

print(
    fold_summary_df
    .to_string(
        index=False
    )
)

print(
    "\n数据输出目录："
)

print(
    OUTPUT_DIR
)

print(
    "\n主要文件结构："
)

print(
    """
model_dataset/
│
├── TARGET_STATION_ORDER.csv
├── TARGET_FOLD_SUMMARY.csv
│
├── fold_1/
│   ├── train.npz
│   ├── val.npz
│   ├── q99_scaler.csv
│   ├── train_timeseries_raw.csv
│   ├── val_timeseries_raw.csv
│   ├── train_timeseries_normalized.csv
│   ├── val_timeseries_normalized.csv
│   └── fold_info.csv
│
├── fold_2/
│   └── ...
│
├── fold_3/
│   └── ...
│
└── fold_4/
    └── ...
"""
)
