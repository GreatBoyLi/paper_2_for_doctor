from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径设置
#
# Source Domain:
# Utrecht, Netherlands
#
# 2014年
# 104个最终保留光伏站
# 已经完成容量归一化：
#
# P_norm = P / estimated_ac_capacity
# ============================================================

INPUT_CSV = Path(
    "../data/source/final/"
    "source_2014/"
    "04_SOURCE_2014_FINAL.csv"
)

OUTPUT_DIR = Path(
    "../data/source/processed_source/"
    "source_2014/"
    "model_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 模型时间窗口参数
#
# 原始时间分辨率：
# 15 min
#
# INPUT_STEPS = 16
# 过去4小时作为输入
#
# OUTPUT_STEPS = 16
# 预测未来4小时
#
# 一个样本：
#
# X:
# [16, N]
#
# Y:
# [16, N]
#
# N = 源域节点数
# 当前应该为104
# ============================================================

INPUT_STEPS = 16

OUTPUT_STEPS = 16

# ============================================================
# 3. 源域时间序列分折
#
# 使用 Expanding Window
#
# 不进行随机KFold，
# 防止未来信息泄漏。
#
# Fold 1:
# Train: 2014-01 ~ 2014-08
# Val:   2014-09
#
# Fold 2:
# Train: 2014-01 ~ 2014-09
# Val:   2014-10
#
# Fold 3:
# Train: 2014-01 ~ 2014-10
# Val:   2014-11
#
# Fold 4:
# Train: 2014-01 ~ 2014-11
# Val:   2014-12
# ============================================================

FOLD_CONFIGS = [

    {
        "fold": 1,

        "train_start": "2014-01-01",
        "train_end": "2014-08-31",

        "val_start": "2014-09-01",
        "val_end": "2014-09-30",
    },

    {
        "fold": 2,

        "train_start": "2014-01-01",
        "train_end": "2014-09-30",

        "val_start": "2014-10-01",
        "val_end": "2014-10-31",
    },

    {
        "fold": 3,

        "train_start": "2014-01-01",
        "train_end": "2014-10-31",

        "val_start": "2014-11-01",
        "val_end": "2014-11-30",
    },

    {
        "fold": 4,

        "train_start": "2014-01-01",
        "train_end": "2014-11-30",

        "val_start": "2014-12-01",
        "val_end": "2014-12-31",
    },
]


# ============================================================
# 4. 日期辅助函数
# ============================================================

def get_start_time(
        date_string
):
    return pd.Timestamp(
        date_string
    )


def get_end_time(
        date_string
):
    return (
            pd.Timestamp(
                date_string
            )
            +
            pd.Timedelta(
                hours=23,
                minutes=45
            )
    )


# ============================================================
# 5. 滑动窗口函数
#
# 输入：
#
# data:
# [T, N]
#
#
# 输出：
#
# X:
# [samples, input_steps, N]
#
# Y:
# [samples, output_steps, N]
#
#
# 例如：
#
# input_steps = 16
# output_steps = 16
#
# 使用：
#
# t ~ t+15
#
# 预测：
#
# t+16 ~ t+31
# ============================================================

def create_sliding_windows(
        data,
        input_steps,
        output_steps
):
    values = (
        data
        .values
        .astype(
            np.float32
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
            "当前时间段长度不足以构建滑动窗口。"
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

        # ====================================================
        # 输入窗口
        # ====================================================

        input_start = (
            start_idx
        )

        input_end = (
                start_idx
                +
                input_steps
        )

        # ====================================================
        # 输出窗口
        # ====================================================

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
        # 安全检查
        #
        # 如果未来源域数据有NaN或者Inf，
        # 直接跳过对应样本。
        #
        # 当前源域理论上NaN=0。
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

        # ====================================================
        # 保存样本
        # ====================================================

        X.append(
            x
        )

        Y.append(
            y
        )

        # ====================================================
        # 保存时间信息
        # ====================================================

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

    if len(
            X
    ) == 0:
        raise RuntimeError(
            "没有生成任何有效滑动窗口样本。"
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
# 6. 读取源域数据
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "读取源域2014最终数据"
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

# ============================================================
# 7. 处理时区
#
# Utrecht源数据当前索引类似：
#
# 2014-01-01 00:00:00+00:00
#
# 即：
#
# tz-aware UTC
#
# 而Fold配置：
#
# pd.Timestamp("2014-01-01")
#
# 是tz-naive。
#
# Pandas不允许二者直接比较。
#
# 因此这里去掉时区信息，
# 但不改变实际时钟值。
#
# 例如：
#
# 2014-01-01 00:00:00+00:00
#
# ->
#
# 2014-01-01 00:00:00
# ============================================================

if isinstance(
        df.index,
        pd.DatetimeIndex
):

    if df.index.tz is not None:
        print(
            "\n检测到源域时间索引带时区：",
            df.index.tz
        )

        df.index = (
            df.index
            .tz_localize(
                None
            )
        )

        print(
            "已去除时间索引时区信息。"
        )

df = (
    df
    .sort_index()
)

# ============================================================
# 8. 基础信息
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

print(
    "最小值：",
    df.min().min()
)

print(
    "最大值：",
    df.max().max()
)

# ============================================================
# 9. 检查Inf
# ============================================================

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

if inf_count > 0:
    raise RuntimeError(
        "源域数据中存在Inf，请先检查数据。"
    )

# ============================================================
# 10. 检查节点名称
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
    "前10个节点：",
    station_names[
        :10
    ]
)

# ============================================================
# 11. 检查节点名是否重复
# ============================================================

duplicated_station_names = (
    pd.Index(
        station_names
    )
    .duplicated()
    .sum()
)

print(
    "重复节点名数量：",
    duplicated_station_names
)

if duplicated_station_names > 0:
    raise RuntimeError(
        "存在重复节点名称。"
    )

# ============================================================
# 12. 检查时间索引是否重复
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
        "源域数据存在重复时间戳。"
    )

# ============================================================
# 13. 检查时间轴是否严格15分钟
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

    extra_times = (
        df.index
        .difference(
            expected_index
        )
    )

    print(
        "缺失时间戳数量：",
        len(
            missing_times
        )
    )

    print(
        "异常时间戳数量：",
        len(
            extra_times
        )
    )

    if len(
            missing_times
    ) > 0:
        print(
            "\n前20个缺失时间戳："
        )

        print(
            missing_times[
                :20
            ]
        )

    raise RuntimeError(
        "源域时间轴不是严格15分钟连续时间轴。"
    )

# ============================================================
# 14. 验证2014年完整长度
#
# 2014不是闰年：
#
# 365 × 24 × 4
#
# = 35040
# ============================================================

EXPECTED_2014_STEPS = (
        365
        *
        24
        *
        4
)

print(
    "\n2014理论时间点数：",
    EXPECTED_2014_STEPS
)

print(
    "实际时间点数：",
    len(
        df
    )
)

if len(
        df
) != EXPECTED_2014_STEPS:
    raise RuntimeError(
        "2014源域时间点数量不等于35040。"
    )

# ============================================================
# 15. 保存节点顺序
#
# 后面：
#
# train.npz
# Graph
# adjacency matrix
# GIN
# GRU
#
# 都必须保持完全相同的节点顺序。
# ============================================================

STATION_FILE = (
        OUTPUT_DIR
        /
        "SOURCE_STATION_ORDER.csv"
)

station_order_df = pd.DataFrame(
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
)

station_order_df.to_csv(
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
# 16. 开始创建各Fold
# ============================================================

fold_summary_records = []

for config in FOLD_CONFIGS:

    fold_id = (
        config[
            "fold"
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
    # Fold时间范围
    # ========================================================

    train_start = get_start_time(
        config[
            "train_start"
        ]
    )

    train_end = get_end_time(
        config[
            "train_end"
        ]
    )

    val_start = get_start_time(
        config[
            "val_start"
        ]
    )

    val_end = get_end_time(
        config[
            "val_end"
        ]
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
    # 防止Train和Val重叠
    # ========================================================

    if train_end >= val_start:
        raise RuntimeError(
            f"Fold {fold_id} 的Train和Val时间发生重叠。"
        )

    # ========================================================
    # 截取Train
    # ========================================================

    train_df = (
        df.loc[
            train_start:
            train_end
        ]
        .copy()
    )

    # ========================================================
    # 截取Validation
    # ========================================================

    val_df = (
        df.loc[
            val_start:
            val_end
        ]
        .copy()
    )

    print(
        "\nTrain原始形状：",
        train_df.shape
    )

    print(
        "Val原始形状：",
        val_df.shape
    )

    # ========================================================
    # 检查数据是否为空
    # ========================================================

    if train_df.empty:
        raise RuntimeError(
            f"Fold {fold_id} Train为空。"
        )

    if val_df.empty:
        raise RuntimeError(
            f"Fold {fold_id} Validation为空。"
        )

    # ========================================================
    # 检查节点数
    # ========================================================

    if train_df.shape[1] != len(
            station_names
    ):
        raise RuntimeError(
            f"Fold {fold_id} Train节点数量异常。"
        )

    if val_df.shape[1] != len(
            station_names
    ):
        raise RuntimeError(
            f"Fold {fold_id} Val节点数量异常。"
        )

    # ========================================================
    # 检查NaN
    # ========================================================

    train_nan = int(
        train_df
        .isna()
        .sum()
        .sum()
    )

    val_nan = int(
        val_df
        .isna()
        .sum()
        .sum()
    )

    print(
        "Train NaN：",
        train_nan
    )

    print(
        "Val NaN：",
        val_nan
    )

    # ========================================================
    # 注意：
    #
    # 这里不做Q99。
    #
    # 因为源域已经进行了：
    #
    # P / estimated_ac_capacity
    #
    # 的容量归一化。
    #
    # 因此这里直接生成滑动窗口。
    # ========================================================

    # ========================================================
    # 17. Train滑动窗口
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

        train_df,

        INPUT_STEPS,

        OUTPUT_STEPS
    )

    # ========================================================
    # 18. Validation滑动窗口
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

        val_df,

        INPUT_STEPS,

        OUTPUT_STEPS
    )

    # ========================================================
    # 输出维度
    # ========================================================

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
    # 19. 维度检查
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
    # 20. Fold输出目录
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
    # 21. 保存Train NPZ
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
        )
    )

    # ========================================================
    # 22. 保存Validation NPZ
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
        )
    )

    # ========================================================
    # 23. 保存Train时间序列
    #
    # 这个文件以后用于：
    #
    # Source Graph构建
    #
    # 如果采用功率相关性构图，
    # 只能使用当前Fold的Train。
    # ========================================================

    TRAIN_TIME_SERIES_FILE = (
            fold_dir
            /
            "train_timeseries.csv"
    )

    train_df.to_csv(
        TRAIN_TIME_SERIES_FILE
    )

    # ========================================================
    # 24. 保存Validation时间序列
    # ========================================================

    VAL_TIME_SERIES_FILE = (
            fold_dir
            /
            "val_timeseries.csv"
    )

    val_df.to_csv(
        VAL_TIME_SERIES_FILE
    )

    # ========================================================
    # 25. 保存Fold信息
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

                "train_time_steps":
                    len(
                        train_df
                    ),

                "val_time_steps":
                    len(
                        val_df
                    ),

                "train_theoretical_samples":
                    train_theoretical_samples,

                "train_skipped_samples":
                    train_skipped_samples,

                "train_samples":
                    len(
                        X_train
                    ),

                "val_theoretical_samples":
                    val_theoretical_samples,

                "val_skipped_samples":
                    val_skipped_samples,

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

                "train_nan":
                    train_nan,

                "val_nan":
                    val_nan
            }
        ]
    )

    FOLD_INFO_FILE = (
            fold_dir
            /
            "fold_info.csv"
    )

    fold_info.to_csv(
        FOLD_INFO_FILE,
        index=False
    )

    # ========================================================
    # 26. Fold汇总
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

            "train_steps":
                len(
                    train_df
                ),

            "val_steps":
                len(
                    val_df
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
        TRAIN_TIME_SERIES_FILE
    )

# ============================================================
# 27. 保存总Fold汇总
# ============================================================

fold_summary_df = pd.DataFrame(
    fold_summary_records
)

SUMMARY_FILE = (
        OUTPUT_DIR
        /
        "SOURCE_FOLD_SUMMARY.csv"
)

fold_summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# 28. 最终输出
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "源域数据集构建完成"
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
├── SOURCE_STATION_ORDER.csv
├── SOURCE_FOLD_SUMMARY.csv
│
├── fold_1/
│   ├── train.npz
│   ├── val.npz
│   ├── train_timeseries.csv
│   ├── val_timeseries.csv
│   └── fold_info.csv
│
├── fold_2/
│   ├── train.npz
│   ├── val.npz
│   ├── train_timeseries.csv
│   ├── val_timeseries.csv
│   └── fold_info.csv
│
├── fold_3/
│   └── ...
│
└── fold_4/
    └── ...
"""
)
