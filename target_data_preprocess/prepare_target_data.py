from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径设置
# ============================================================

SOLAR_CSV = Path(
    "../data/target/Solar_Energy_Generation.csv"
)

SITE_DETAILS_CSV = Path(
    "../data/target/Solar_Site_Details.csv"
)

OUTPUT_DIR = Path(
    "../data/target/processed_target_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 参数设置
#
# 注意：
# 以下参数是我们当前复现实验的工程设置，
# 不是原论文明确规定的参数。
# ============================================================

# 42个站共同运行区间
COMMON_START = pd.Timestamp(
    "2021-02-03 00:15:00"
)

COMMON_END = pd.Timestamp(
    "2022-04-23 23:45:00"
)

# 白天有效率阈值
DAY_VALID_THRESHOLD = 0.72

# 允许一个站存在的“整日白天完全缺失”最大天数
MAX_FULLY_MISSING_DAYS = 10

# ------------------------------------------------------------
# 日照边界扩展
#
# 修正版设为 0。
#
# 即：
# 每天第一个出现正功率的时间 -> 白天开始
# 每天最后一个出现正功率的时间 -> 白天结束
#
# 如果某一天全站都没有正功率，
# 则根据前后日期的白天边界插值。
# ------------------------------------------------------------

DAYLIGHT_PADDING_MINUTES = 0

# ------------------------------------------------------------
# 最大允许自动插值的连续缺失长度
#
# 8 × 15min = 2小时
# ------------------------------------------------------------

MAX_INTERP_GAP_STEPS = 8

MAX_INTERP_GAP_HOURS = (
        MAX_INTERP_GAP_STEPS
        * 0.25
)


# ============================================================
# 3. 工具函数
# ============================================================

def minutes_from_midnight(timestamp):
    """
    将Timestamp转换为从当天00:00开始经过的分钟数。
    """

    return (
            timestamp.hour * 60
            +
            timestamp.minute
            +
            timestamp.second / 60
    )


def max_consecutive_true(mask):
    """
    返回布尔序列中最长连续True的长度。
    """

    mask = pd.Series(
        mask
    ).astype(bool)

    if len(mask) == 0:
        return 0

    if not mask.any():
        return 0

    group_id = (
            mask
            !=
            mask.shift()
    ).cumsum()

    lengths = (
        mask[
            mask
        ]
        .groupby(
            group_id[
                mask
            ]
        )
        .size()
    )

    if len(lengths) == 0:
        return 0

    return int(
        lengths.max()
    )


def interpolate_only_short_gaps(
        series,
        max_gap_steps
):
    """
    仅填充长度 <= max_gap_steps 的内部NaN缺口。

    长缺口保持NaN。

    参数：
        series:
            DatetimeIndex的Series

        max_gap_steps:
            允许插值的最大连续缺失点数

    返回：
        filled_series
        short_gap_count
        long_gap_count
    """

    s = series.copy()

    missing = (
        s.isna()
    )

    if not missing.any():
        return (
            s,
            0,
            0
        )

    # 先生成完整时间插值候选值
    interpolated_candidate = (
        s.interpolate(
            method="time",
            limit_area="inside"
        )
    )

    # 区分不同连续缺失段
    group_id = (
            missing
            !=
            missing.shift()
    ).cumsum()

    short_gap_count = 0

    long_gap_count = 0

    # 找所有NaN段
    missing_groups = (
        s[
            missing
        ]
        .groupby(
            group_id[
                missing
            ]
        )
    )

    for _, gap_series in missing_groups:

        gap_index = (
            gap_series.index
        )

        gap_length = len(
            gap_index
        )

        # 当前gap在完整序列中的位置
        first_pos = (
            s.index.get_loc(
                gap_index[0]
            )
        )

        last_pos = (
            s.index.get_loc(
                gap_index[-1]
            )
        )

        # 必须左右两侧都有真实值
        has_left = (
                first_pos > 0
                and
                pd.notna(
                    s.iloc[
                        first_pos - 1
                        ]
                )
        )

        has_right = (
                last_pos
                <
                len(s) - 1
                and
                pd.notna(
                    s.iloc[
                        last_pos + 1
                        ]
                )
        )

        # 小缺口且是内部缺口
        if (
                gap_length
                <=
                max_gap_steps
                and
                has_left
                and
                has_right
        ):

            s.loc[
                gap_index
            ] = (
                interpolated_candidate
                .loc[
                    gap_index
                ]
            )

            short_gap_count += 1


        else:

            long_gap_count += 1

    return (
        s,
        short_gap_count,
        long_gap_count
    )


# ============================================================
# 4. 读取光伏功率数据
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "读取 Solar_Energy_Generation.csv"
)

print(
    "=" * 75
)

solar = pd.read_csv(
    SOLAR_CSV
)

print(
    "原始数据形状：",
    solar.shape
)

# ============================================================
# 5. 字段类型处理
# ============================================================

solar[
    "Timestamp"
] = pd.to_datetime(
    solar[
        "Timestamp"
    ],
    errors="coerce"
)

solar[
    "SolarGeneration"
] = pd.to_numeric(
    solar[
        "SolarGeneration"
    ],
    errors="coerce"
)

solar[
    "CampusKey"
] = pd.to_numeric(
    solar[
        "CampusKey"
    ],
    errors="coerce"
)

solar[
    "SiteKey"
] = pd.to_numeric(
    solar[
        "SiteKey"
    ],
    errors="coerce"
)

# 删除时间异常记录
solar = solar.dropna(
    subset=[
        "Timestamp"
    ]
)

# ============================================================
# 6. 构造唯一NodeID
# ============================================================

solar[
    "NodeID"
] = (
        "C"
        +
        solar[
            "CampusKey"
        ]
        .astype(
            "Int64"
        )
        .astype(
            str
        )
        +
        "_S"
        +
        solar[
            "SiteKey"
        ]
        .astype(
            "Int64"
        )
        .astype(
            str
        )
)

print(
    "原始行数：",
    len(
        solar
    )
)

print(
    "Campus数量：",
    solar[
        "CampusKey"
    ]
    .nunique()
)

print(
    "节点数量：",
    solar[
        "NodeID"
    ]
    .nunique()
)

# ============================================================
# 7. 检查重复记录
# ============================================================

duplicate_count = (
    solar
    .duplicated(
        subset=[
            "Timestamp",
            "NodeID"
        ]
    )
    .sum()
)

print(
    "Timestamp + NodeID重复数量：",
    duplicate_count
)

# ============================================================
# 8. 长表 -> 宽表
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "转换为宽表"
)

print(
    "=" * 75
)

if duplicate_count == 0:

    wide = (
        solar
        .pivot(
            index="Timestamp",
            columns="NodeID",
            values="SolarGeneration"
        )
        .sort_index()
    )

else:

    wide = (
        solar
        .groupby(
            [
                "Timestamp",
                "NodeID"
            ],
            dropna=False
        )[
            "SolarGeneration"
        ]
        .mean()
        .unstack(
            "NodeID"
        )
        .sort_index()
    )

print(
    "原始宽表形状：",
    wide.shape
)

# ============================================================
# 9. 建立严格连续15分钟时间轴
# ============================================================

full_index = pd.date_range(
    start=solar[
        "Timestamp"
    ].min(),
    end=solar[
        "Timestamp"
    ].max(),
    freq="15min"
)

wide = (
    wide
    .reindex(
        full_index
    )
)

wide.index.name = (
    "Timestamp"
)

print(
    "完整15min宽表形状：",
    wide.shape
)

# ============================================================
# 10. 截取42个站共同运行区间
# ============================================================

target = (
    wide
    .loc[
        COMMON_START:
        COMMON_END
    ]
    .copy()
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "42站共同运行区间"
)

print(
    "=" * 75
)

print(
    "时间范围：",
    target.index.min(),
    "->",
    target.index.max()
)

print(
    "数据形状：",
    target.shape
)

print(
    "节点数量：",
    target.shape[1]
)

target.to_csv(
    OUTPUT_DIR
    /
    "01_TARGET_COMMON_PERIOD_RAW.csv"
)

# ============================================================
# 11. 根据每天全站功率估计白天起止时间
#
# 每一天：
#
# 最早出现任意站正功率 -> 白天开始
# 最晚出现任意站正功率 -> 白天结束
#
# 当前padding = 0分钟
#
# 如果某一天完全没有任何正功率，
# 则该天的白天起止时间先记为NaN，
# 后面根据前后日期插值。
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "估计每日有效白天区间"
)

print(
    "=" * 75
)

date_list = (
    pd.Index(
        target.index.normalize()
    )
    .unique()
)

daylight_records = []

for date in date_list:

    day_start = (
        pd.Timestamp(
            date
        )
    )

    day_end = (
            day_start
            +
            pd.Timedelta(
                days=1
            )
            -
            pd.Timedelta(
                minutes=15
            )
    )

    day_values = (
        target
        .loc[
            day_start:
            day_end
        ]
    )

    positive_by_time = (
        day_values
        .gt(0)
        .any(
            axis=1
        )
    )

    positive_times = (
        day_values.index[
            positive_by_time
        ]
    )

    if len(
            positive_times
    ) > 0:

        first_positive = (
            positive_times[0]
        )

        last_positive = (
            positive_times[-1]
        )

        sunrise_minute = (
                minutes_from_midnight(
                    first_positive
                )
                -
                DAYLIGHT_PADDING_MINUTES
        )

        sunset_minute = (
                minutes_from_midnight(
                    last_positive
                )
                +
                DAYLIGHT_PADDING_MINUTES
        )

        sunrise_minute = max(
            0,
            sunrise_minute
        )

        sunset_minute = min(
            24 * 60 - 15,
            sunset_minute
        )

        has_positive = True


    else:

        sunrise_minute = (
            np.nan
        )

        sunset_minute = (
            np.nan
        )

        has_positive = False

    daylight_records.append(
        {
            "date":
                day_start,

            "sunrise_minute_raw":
                sunrise_minute,

            "sunset_minute_raw":
                sunset_minute,

            "has_positive_generation":
                has_positive
        }
    )

daylight_df = pd.DataFrame(
    daylight_records
)

daylight_df[
    "date"
] = pd.to_datetime(
    daylight_df[
        "date"
    ]
)

daylight_df = (
    daylight_df
    .set_index(
        "date"
    )
    .sort_index()
)

# ============================================================
# 12. 对没有任何正功率的日期估计白天边界
#
# 注意：
# 这里插值的是“每天白天起止时间”，
# 不是光伏功率。
# ============================================================

daylight_df[
    "sunrise_minute"
] = (
    daylight_df[
        "sunrise_minute_raw"
    ]
    .interpolate(
        method="time",
        limit_direction="both"
    )
)

daylight_df[
    "sunset_minute"
] = (
    daylight_df[
        "sunset_minute_raw"
    ]
    .interpolate(
        method="time",
        limit_direction="both"
    )
)

no_positive_days = int(
    (
        ~daylight_df[
            "has_positive_generation"
        ]
    )
    .sum()
)

print(
    "完全没有正功率的日期数量：",
    no_positive_days
)

print(
    "\n日照区间示例："
)

print(
    daylight_df
    .head(10)
    .to_string()
)

daylight_df.to_csv(
    OUTPUT_DIR
    /
    "02_TARGET_ESTIMATED_DAYLIGHT.csv"
)

# ============================================================
# 13. 创建白天/夜间mask
# ============================================================

day_mask = pd.Series(
    False,
    index=target.index,
    dtype=bool
)

for date, row in daylight_df.iterrows():
    sunrise_time = (
            date
            +
            pd.Timedelta(
                minutes=float(
                    row[
                        "sunrise_minute"
                    ]
                )
            )
    )

    sunset_time = (
            date
            +
            pd.Timedelta(
                minutes=float(
                    row[
                        "sunset_minute"
                    ]
                )
            )
    )

    current_mask = (
            (
                    target.index
                    >=
                    sunrise_time
            )
            &
            (
                    target.index
                    <=
                    sunset_time
            )
    )

    day_mask.loc[
        current_mask
    ] = True

night_mask = (
    ~day_mask
)

print(
    "\n白天15min时间点数：",
    int(
        day_mask.sum()
    )
)

print(
    "夜间15min时间点数：",
    int(
        night_mask.sum()
    )
)

# ============================================================
# 14. 夜间NaN -> 0
#
# 仅修改夜间。
# 白天NaN继续保留。
# ============================================================

target_night_zero = (
    target.copy()
)

target_night_zero.loc[
    night_mask
] = (
    target_night_zero
    .loc[
        night_mask
    ]
    .fillna(
        0.0
    )
)

# ============================================================
# 15. 统计白天总体覆盖率
# ============================================================

day_data = (
    target
    .loc[
        day_mask
    ]
)

day_valid_ratio = (
    day_data
    .notna()
    .mean()
)

day_missing_ratio = (
        1
        -
        day_valid_ratio
)

# ============================================================
# 16. 每个节点每天白天覆盖率
# ============================================================

daily_records = []

for node in target.columns:

    for date, daylight_row in daylight_df.iterrows():

        sunrise_time = (
                date
                +
                pd.Timedelta(
                    minutes=float(
                        daylight_row[
                            "sunrise_minute"
                        ]
                    )
                )
        )

        sunset_time = (
                date
                +
                pd.Timedelta(
                    minutes=float(
                        daylight_row[
                            "sunset_minute"
                        ]
                    )
                )
        )

        daily_series = (
            target[
                node
            ]
            .loc[
                sunrise_time:
                sunset_time
            ]
        )

        if len(
                daily_series
        ) == 0:
            continue

        valid_count = int(
            daily_series
            .notna()
            .sum()
        )

        day_steps = len(
            daily_series
        )

        valid_ratio = (
                valid_count
                /
                day_steps
        )

        daily_records.append(
            {
                "NodeID":
                    node,

                "date":
                    date,

                "day_steps":
                    day_steps,

                "valid_steps":
                    valid_count,

                "day_valid_ratio":
                    valid_ratio
            }
        )

daily_df = pd.DataFrame(
    daily_records
)

daily_df.to_csv(
    OUTPUT_DIR
    /
    "03_TARGET_DAILY_DAY_COVERAGE.csv",
    index=False
)

# ============================================================
# 17. 每站“整日白天完全缺失”天数
# ============================================================

fully_missing_day_count = (
    daily_df
    .groupby(
        "NodeID"
    )[
        "day_valid_ratio"
    ]
    .apply(
        lambda x:
        int(
            (
                    x == 0
            )
            .sum()
        )
    )
)

# ============================================================
# 18. 最长连续“整日白天完全缺失”
# ============================================================

max_consecutive_missing_days = {}

for node in target.columns:
    node_daily = (
        daily_df[
            daily_df[
                "NodeID"
            ]
            ==
            node
            ]
        .sort_values(
            "date"
        )
    )

    missing_day_mask = (
            node_daily[
                "day_valid_ratio"
            ]
            ==
            0
    )

    max_days = (
        max_consecutive_true(
            missing_day_mask
            .to_numpy()
        )
    )

    max_consecutive_missing_days[
        node
    ] = (
        max_days
    )

# ============================================================
# 19. 节点质量汇总
# ============================================================

quality_records = []

for node in target.columns:
    quality_records.append(
        {
            "NodeID":
                node,

            "day_valid_ratio":
                float(
                    day_valid_ratio[
                        node
                    ]
                ),

            "day_missing_ratio":
                float(
                    day_missing_ratio[
                        node
                    ]
                ),

            "fully_missing_day_count":
                int(
                    fully_missing_day_count
                    .get(
                        node,
                        0
                    )
                ),

            "max_consecutive_fully_missing_days":
                int(
                    max_consecutive_missing_days[
                        node
                    ]
                )
        }
    )

quality_df = pd.DataFrame(
    quality_records
)

quality_df = (
    quality_df
    .sort_values(
        "day_valid_ratio",
        ascending=False
    )
)

quality_df.to_csv(
    OUTPUT_DIR
    /
    "04_TARGET_NODE_QUALITY.csv",
    index=False
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "节点质量统计"
)

print(
    "=" * 75
)

print(
    quality_df.to_string(
        index=False
    )
)

# ============================================================
# 20. 双重筛选
#
# 条件1：
# 白天有效率 >= 0.80
#
# 条件2：
# 整日白天完全缺失天数 <= 5
# ============================================================

selected_quality = (
    quality_df[
        (
                quality_df[
                    "day_valid_ratio"
                ]
                >=
                DAY_VALID_THRESHOLD
        )
        &
        (
                quality_df[
                    "fully_missing_day_count"
                ]
                <=
                MAX_FULLY_MISSING_DAYS
        )
        ]
    .copy()
)

removed_quality = (
    quality_df[
        ~quality_df[
            "NodeID"
        ]
        .isin(
            selected_quality[
                "NodeID"
            ]
        )
    ]
        .copy()
)

selected_nodes = (
    selected_quality[
        "NodeID"
    ]
    .tolist()
)

removed_nodes = (
    removed_quality[
        "NodeID"
    ]
    .tolist()
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "双重筛选结果"
)

print(
    "=" * 75
)

print(
    "DAY_VALID_THRESHOLD =",
    DAY_VALID_THRESHOLD
)

print(
    "MAX_FULLY_MISSING_DAYS =",
    MAX_FULLY_MISSING_DAYS
)

print(
    "原始节点数：",
    target.shape[1]
)

print(
    "保留节点数：",
    len(
        selected_nodes
    )
)

print(
    "删除节点数：",
    len(
        removed_nodes
    )
)

print(
    "\n删除节点："
)

if len(
        removed_quality
) > 0:

    print(
        removed_quality.to_string(
            index=False
        )
    )

else:

    print(
        "无"
    )

selected_quality.to_csv(
    OUTPUT_DIR
    /
    "05_TARGET_SELECTED_NODES.csv",
    index=False
)

removed_quality.to_csv(
    OUTPUT_DIR
    /
    "06_TARGET_REMOVED_NODES.csv",
    index=False
)

# ============================================================
# 21. 防止“0个节点”继续往后运行
# ============================================================

if len(
        selected_nodes
) == 0:
    raise RuntimeError(
        "\n当前筛选条件下没有任何目标节点被保留。\n"
        "请检查：\n"
        "1. DAY_VALID_THRESHOLD\n"
        "2. MAX_FULLY_MISSING_DAYS\n"
        "3. 白天区间定义\n"
        "\n"
        f"当前 DAY_VALID_THRESHOLD = "
        f"{DAY_VALID_THRESHOLD}\n"
        f"当前 MAX_FULLY_MISSING_DAYS = "
        f"{MAX_FULLY_MISSING_DAYS}\n"
    )

# ============================================================
# 22. 提取保留节点
# ============================================================

selected_data = (
    target_night_zero[
        selected_nodes
    ]
    .copy()
)

# ============================================================
# 23. 只对短缺口进行时间线性插值
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "短缺口线性插值"
)

print(
    "=" * 75
)

print(
    "最大允许插值缺口：",
    MAX_INTERP_GAP_STEPS,
    "个15min时间步"
)

print(
    "即约：",
    MAX_INTERP_GAP_HOURS,
    "小时"
)

before_nan = int(
    selected_data
    .isna()
    .sum()
    .sum()
)

interpolated_data = (
    selected_data.copy()
)

interp_records = []

for node in selected_nodes:
    original_series = (
        selected_data[
            node
        ]
        .copy()
    )

    (
        filled_series,
        short_gap_count,
        long_gap_count
    ) = interpolate_only_short_gaps(
        original_series,
        MAX_INTERP_GAP_STEPS
    )

    interpolated_data[
        node
    ] = (
        filled_series
    )

    interp_records.append(
        {
            "NodeID":
                node,

            "nan_before":
                int(
                    original_series
                    .isna()
                    .sum()
                ),

            "nan_after":
                int(
                    filled_series
                    .isna()
                    .sum()
                ),

            "short_gap_count_filled":
                int(
                    short_gap_count
                ),

            "long_gap_count_kept":
                int(
                    long_gap_count
                )
        }
    )

# ============================================================
# 24. 构造插值报告
#
# 明确指定列名，
# 即便理论上记录为空，也不会因为nan_after不存在而报KeyError。
# ============================================================

interp_df = pd.DataFrame(
    interp_records,
    columns=[
        "NodeID",
        "nan_before",
        "nan_after",
        "short_gap_count_filled",
        "long_gap_count_kept"
    ]
)

if not interp_df.empty:
    interp_df = (
        interp_df
        .sort_values(
            "nan_after",
            ascending=False
        )
    )

interp_df.to_csv(
    OUTPUT_DIR
    /
    "07_TARGET_INTERPOLATION_REPORT.csv",
    index=False
)

after_nan = int(
    interpolated_data
    .isna()
    .sum()
    .sum()
)

print(
    "插值前NaN总数：",
    before_nan
)

print(
    "短缺口插值后剩余NaN总数：",
    after_nan
)

print(
    "\n各节点插值情况："
)

if not interp_df.empty:

    print(
        interp_df.to_string(
            index=False
        )
    )

else:

    print(
        "无插值记录"
    )

# ============================================================
# 25. 负值检查
# ============================================================

negative_count = int(
    interpolated_data
    .lt(0)
    .sum()
    .sum()
)

print(
    "\n负值数量：",
    negative_count
)

if negative_count > 0:
    print(
        "发现负值，裁剪为0。"
    )

    interpolated_data = (
        interpolated_data
        .clip(
            lower=0
        )
    )

# ============================================================
# 26. 保存严格版未归一化目标域
#
# 注意：
# 长缺口NaN会被保留。
# 这是故意的。
# ============================================================

STRICT_OUTPUT = (
        OUTPUT_DIR
        /
        "08_TARGET_CLEAN_STRICT_UNNORMALIZED.csv"
)

interpolated_data.to_csv(
    STRICT_OUTPUT
)

# ============================================================
# 27. 生成有效性mask
#
# 1：
# 当前站当前时间点有可使用数据
#
# 0：
# 长缺口仍然缺失
# ============================================================

valid_mask = (
    interpolated_data
    .notna()
    .astype(
        np.int8
    )
)

valid_mask.to_csv(
    OUTPUT_DIR
    /
    "09_TARGET_VALID_MASK.csv"
)

# ============================================================
# 28. 统计所有节点同时有效的时间点
# ============================================================

all_nodes_valid_mask = (
    interpolated_data
    .notna()
    .all(
        axis=1
    )
)

valid_all_nodes_count = int(
    all_nodes_valid_mask
    .sum()
)

invalid_any_nodes_count = int(
    (
        ~all_nodes_valid_mask
    )
    .sum()
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "最终严格数据质量"
)

print(
    "=" * 75
)

print(
    "最终数据形状：",
    interpolated_data.shape
)

print(
    "剩余NaN总数：",
    after_nan
)

print(
    "所有节点都有效的时间点数：",
    valid_all_nodes_count
)

print(
    "至少一个节点存在长缺失的时间点数：",
    invalid_any_nodes_count
)

# ============================================================
# 29. 处理metadata
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "处理目标域metadata"
)

print(
    "=" * 75
)

metadata = pd.read_csv(
    SITE_DETAILS_CSV
)

metadata[
    "CampusKey"
] = pd.to_numeric(
    metadata[
        "CampusKey"
    ],
    errors="coerce"
)

metadata[
    "SiteKey"
] = pd.to_numeric(
    metadata[
        "SiteKey"
    ],
    errors="coerce"
)

metadata[
    "NodeID"
] = (
        "C"
        +
        metadata[
            "CampusKey"
        ]
        .astype(
            "Int64"
        )
        .astype(
            str
        )
        +
        "_S"
        +
        metadata[
            "SiteKey"
        ]
        .astype(
            "Int64"
        )
        .astype(
            str
        )
)

selected_metadata = (
    metadata[
        metadata[
            "NodeID"
        ]
        .isin(
            selected_nodes
        )
    ]
        .copy()
)

# ============================================================
# 30. metadata顺序与功率矩阵列顺序保持一致
# ============================================================

node_order = {
    node:
        idx

    for idx, node
    in enumerate(
        selected_nodes
    )
}

selected_metadata[
    "_order"
] = (
    selected_metadata[
        "NodeID"
    ]
    .map(
        node_order
    )
)

selected_metadata = (
    selected_metadata
    .sort_values(
        "_order"
    )
    .drop(
        columns=[
            "_order"
        ]
    )
)

selected_metadata.to_csv(
    OUTPUT_DIR
    /
    "10_TARGET_SELECTED_METADATA.csv",
    index=False
)

# ============================================================
# 31. 保存最终摘要
# ============================================================

summary_path = (
        OUTPUT_DIR
        /
        "11_TARGET_FINAL_PREPROCESS_SUMMARY.txt"
)

with open(
        summary_path,
        "w",
        encoding="utf-8"
) as f:
    f.write(
        "UNISOLAR Target Final Preprocessing Summary\n"
    )

    f.write(
        "=" * 75
        +
        "\n\n"
    )

    f.write(
        f"Time range: "
        f"{COMMON_START} -> {COMMON_END}\n"
    )

    f.write(
        f"Original nodes: "
        f"{target.shape[1]}\n"
    )

    f.write(
        f"Day valid threshold: "
        f"{DAY_VALID_THRESHOLD}\n"
    )

    f.write(
        f"Max fully missing days: "
        f"{MAX_FULLY_MISSING_DAYS}\n"
    )

    f.write(
        f"Daylight padding minutes: "
        f"{DAYLIGHT_PADDING_MINUTES}\n"
    )

    f.write(
        f"Days without any positive generation: "
        f"{no_positive_days}\n"
    )

    f.write(
        f"Selected nodes: "
        f"{len(selected_nodes)}\n"
    )

    f.write(
        f"Removed nodes: "
        f"{len(removed_nodes)}\n"
    )

    f.write(
        f"Max interpolation gap steps: "
        f"{MAX_INTERP_GAP_STEPS}\n"
    )

    f.write(
        f"Max interpolation gap hours: "
        f"{MAX_INTERP_GAP_HOURS}\n"
    )

    f.write(
        f"NaN before interpolation: "
        f"{before_nan}\n"
    )

    f.write(
        f"NaN after short-gap interpolation: "
        f"{after_nan}\n"
    )

    f.write(
        f"All-node-valid timestamps: "
        f"{valid_all_nodes_count}\n"
    )

    f.write(
        f"Timestamps with at least one long gap: "
        f"{invalid_any_nodes_count}\n"
    )

    f.write(
        "\nRemoved nodes:\n"
    )

    for _, row in removed_quality.iterrows():
        f.write(
            f"{row['NodeID']}: "
            f"day_valid="
            f"{row['day_valid_ratio']:.6f}, "
            f"fully_missing_days="
            f"{int(row['fully_missing_day_count'])}, "
            f"max_consecutive_missing_days="
            f"{int(row['max_consecutive_fully_missing_days'])}\n"
        )

# ============================================================
# 32. 完成
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "目标域严格预处理完成"
)

print(
    "=" * 75
)

print(
    "\n主要输出文件："
)

print(
    "01_TARGET_COMMON_PERIOD_RAW.csv"
)

print(
    "02_TARGET_ESTIMATED_DAYLIGHT.csv"
)

print(
    "03_TARGET_DAILY_DAY_COVERAGE.csv"
)

print(
    "04_TARGET_NODE_QUALITY.csv"
)

print(
    "05_TARGET_SELECTED_NODES.csv"
)

print(
    "06_TARGET_REMOVED_NODES.csv"
)

print(
    "07_TARGET_INTERPOLATION_REPORT.csv"
)

print(
    "08_TARGET_CLEAN_STRICT_UNNORMALIZED.csv"
)

print(
    "09_TARGET_VALID_MASK.csv"
)

print(
    "10_TARGET_SELECTED_METADATA.csv"
)

print(
    "11_TARGET_FINAL_PREPROCESS_SUMMARY.txt"
)
