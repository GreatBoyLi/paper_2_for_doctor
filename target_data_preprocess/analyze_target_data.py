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

MONTHLY_SUMMARY_CSV = Path(
    "../data/target/Monthly_Summary_Solar.csv"
)

OUTPUT_DIR = Path(
    "../data/target/processed_target_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 读取 Solar_Energy_Generation.csv
# ============================================================

print("\n" + "=" * 70)
print("读取 Solar_Energy_Generation.csv")
print("=" * 70)

solar = pd.read_csv(
    SOLAR_CSV
)

print(
    "原始数据形状：",
    solar.shape
)

print(
    "\n字段："
)

print(
    solar.columns.tolist()
)

print(
    "\n前5行："
)

print(
    solar.head()
)

# ============================================================
# 3. 字段类型处理
# ============================================================

solar["Timestamp"] = pd.to_datetime(
    solar["Timestamp"],
    errors="coerce"
)

solar["SolarGeneration"] = pd.to_numeric(
    solar["SolarGeneration"],
    errors="coerce"
)

solar["CampusKey"] = pd.to_numeric(
    solar["CampusKey"],
    errors="coerce"
)

solar["SiteKey"] = pd.to_numeric(
    solar["SiteKey"],
    errors="coerce"
)

# 删除时间异常记录
solar = solar.dropna(
    subset=["Timestamp"]
)

# 按时间排序
solar = solar.sort_values(
    [
        "Timestamp",
        "CampusKey",
        "SiteKey"
    ]
)

# ============================================================
# 4. 构造唯一节点ID
# ============================================================

solar["NodeID"] = (
        "C"
        +
        solar["CampusKey"]
        .astype("Int64")
        .astype(str)
        +
        "_S"
        +
        solar["SiteKey"]
        .astype("Int64")
        .astype(str)
)

# ============================================================
# 5. 基本统计
# ============================================================

print("\n" + "=" * 70)
print("目标域基本统计")
print("=" * 70)

print(
    "Campus数量：",
    solar["CampusKey"].nunique()
)

print(
    "NodeID数量：",
    solar["NodeID"].nunique()
)

print(
    "原始时间范围：",
    solar["Timestamp"].min(),
    "->",
    solar["Timestamp"].max()
)

print(
    "SolarGeneration整体NaN比例：",
    solar["SolarGeneration"]
    .isna()
    .mean()
)

print(
    "\n各Campus节点数量："
)

campus_node_count = (
    solar[
        [
            "CampusKey",
            "NodeID"
        ]
    ]
    .drop_duplicates()
    .groupby(
        "CampusKey"
    )
    .size()
)

print(
    campus_node_count
)

# ============================================================
# 6. 检查重复 Timestamp + NodeID
# ============================================================

print("\n" + "=" * 70)
print("检查 Timestamp + NodeID 重复记录")
print("=" * 70)

duplicate_mask = solar.duplicated(
    subset=[
        "Timestamp",
        "NodeID"
    ],
    keep=False
)

duplicate_rows = solar[
    duplicate_mask
].copy()

duplicate_count = solar.duplicated(
    subset=[
        "Timestamp",
        "NodeID"
    ]
).sum()

print(
    "重复记录数量：",
    duplicate_count
)

if duplicate_count > 0:
    print(
        "\n重复记录示例："
    )

    print(
        duplicate_rows
        .head(20)
        .to_string(
            index=False
        )
    )

    duplicate_rows.to_csv(
        OUTPUT_DIR
        /
        "00_duplicate_timestamp_node_records.csv",
        index=False
    )

# ============================================================
# 7. 检查时间分辨率
# ============================================================

print("\n" + "=" * 70)
print("检查时间间隔")
print("=" * 70)

first_node = (
    solar["NodeID"]
    .drop_duplicates()
    .iloc[0]
)

first_node_data = (
    solar[
        solar["NodeID"]
        ==
        first_node
        ]
    .sort_values(
        "Timestamp"
    )
)

time_diff = (
    first_node_data[
        "Timestamp"
    ]
    .diff()
    .dropna()
)

print(
    "示例节点：",
    first_node
)

print(
    "\n最常见时间间隔："
)

print(
    time_diff
    .value_counts()
    .head(10)
)

# ============================================================
# 8. 每个站点原始时间范围
# ============================================================

print("\n" + "=" * 70)
print("统计各站原始时间范围")
print("=" * 70)

station_records = []

for node_id, group in solar.groupby(
        "NodeID"
):
    group = group.sort_values(
        "Timestamp"
    )

    start_time = (
        group["Timestamp"].min()
    )

    end_time = (
        group["Timestamp"].max()
    )

    total_rows = len(
        group
    )

    valid_rows = (
        group[
            "SolarGeneration"
        ]
        .notna()
        .sum()
    )

    nan_rows = (
        group[
            "SolarGeneration"
        ]
        .isna()
        .sum()
    )

    raw_valid_ratio = (
        valid_rows
        /
        total_rows
        if total_rows > 0
        else np.nan
    )

    station_records.append(
        {
            "NodeID":
                node_id,

            "CampusKey":
                int(
                    group[
                        "CampusKey"
                    ].iloc[0]
                ),

            "SiteKey":
                int(
                    group[
                        "SiteKey"
                    ].iloc[0]
                ),

            "start_time":
                start_time,

            "end_time":
                end_time,

            "total_rows":
                total_rows,

            "valid_rows":
                valid_rows,

            "nan_rows":
                nan_rows,

            "raw_valid_ratio":
                raw_valid_ratio
        }
    )

station_summary = pd.DataFrame(
    station_records
)

station_summary = station_summary.sort_values(
    [
        "CampusKey",
        "SiteKey"
    ]
)

station_summary.to_csv(
    OUTPUT_DIR
    /
    "01_target_station_time_and_raw_coverage.csv",
    index=False
)

print(
    station_summary.to_string(
        index=False
    )
)

# ============================================================
# 9. 长表 -> 宽表
#
# 关键修正：
# 不再直接使用 pivot_table()
# 避免全NaN时间戳被删除
# ============================================================

print("\n" + "=" * 70)
print("转换为宽表")
print("=" * 70)

if duplicate_count == 0:

    solar_wide = (
        solar
        .pivot(
            index="Timestamp",
            columns="NodeID",
            values="SolarGeneration"
        )
        .sort_index()
    )

else:

    # 如果存在重复：
    # 对同一 Timestamp + NodeID 的值取平均
    solar_temp = (
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

    solar_wide = (
        solar_temp
    )

print(
    "宽表形状：",
    solar_wide.shape
)

print(
    "宽表时间范围：",
    solar_wide.index.min(),
    "->",
    solar_wide.index.max()
)

solar_wide.to_csv(
    OUTPUT_DIR
    /
    "02_target_solar_wide_raw.csv"
)

# ============================================================
# 10. 建立严格完整15分钟时间轴
#
# 注意：
# 使用原始solar的时间范围
# ============================================================

print("\n" + "=" * 70)
print("建立连续15分钟时间轴")
print("=" * 70)

full_index = pd.date_range(
    start=solar[
        "Timestamp"
    ].min(),
    end=solar[
        "Timestamp"
    ].max(),
    freq="15min"
)

solar_wide_full = (
    solar_wide
    .reindex(
        full_index
    )
)

solar_wide_full.index.name = (
    "Timestamp"
)

print(
    "完整15min数据形状：",
    solar_wide_full.shape
)

print(
    "完整时间范围：",
    solar_wide_full.index.min(),
    "->",
    solar_wide_full.index.max()
)

print(
    "理论时间步数：",
    len(
        full_index
    )
)

solar_wide_full.to_csv(
    OUTPUT_DIR
    /
    "03_target_solar_wide_full_15min.csv"
)

# ============================================================
# 11. 检查时间轴是否存在缺失时间戳
# ============================================================

original_unique_times = pd.DatetimeIndex(
    solar[
        "Timestamp"
    ]
    .drop_duplicates()
    .sort_values()
)

missing_original_timestamps = (
    full_index
    .difference(
        original_unique_times
    )
)

print(
    "\n原始数据完全没有记录的15min时间戳数量：",
    len(
        missing_original_timestamps
    )
)

if len(
        missing_original_timestamps
) > 0:
    print(
        "\n缺失时间戳前20个："
    )

    print(
        missing_original_timestamps[
            :20
        ]
    )

# ============================================================
# 12. 每个节点生命周期内的数据分析
#
# 防止把“站点尚未上线”误认为缺失
# ============================================================

print("\n" + "=" * 70)
print("统计各节点生命周期内覆盖情况")
print("=" * 70)

lifecycle_records = []

for _, row in station_summary.iterrows():
    node_id = (
        row[
            "NodeID"
        ]
    )

    start_time = (
        row[
            "start_time"
        ]
    )

    end_time = (
        row[
            "end_time"
        ]
    )

    series = (
        solar_wide_full[
            node_id
        ]
        .loc[
            start_time:
            end_time
        ]
    )

    total_steps = len(
        series
    )

    valid_steps = (
        series
        .notna()
        .sum()
    )

    nan_steps = (
        series
        .isna()
        .sum()
    )

    lifecycle_valid_ratio = (
        valid_steps
        /
        total_steps
        if total_steps > 0
        else np.nan
    )

    lifecycle_records.append(
        {
            "NodeID":
                node_id,

            "CampusKey":
                row[
                    "CampusKey"
                ],

            "SiteKey":
                row[
                    "SiteKey"
                ],

            "start_time":
                start_time,

            "end_time":
                end_time,

            "lifecycle_total_steps":
                total_steps,

            "lifecycle_valid_steps":
                valid_steps,

            "lifecycle_nan_steps":
                nan_steps,

            "lifecycle_valid_ratio":
                lifecycle_valid_ratio
        }
    )

lifecycle_df = pd.DataFrame(
    lifecycle_records
)

lifecycle_df = lifecycle_df.sort_values(
    "lifecycle_valid_ratio",
    ascending=False
)

lifecycle_df.to_csv(
    OUTPUT_DIR
    /
    "04_target_lifecycle_coverage.csv",
    index=False
)

print(
    lifecycle_df.to_string(
        index=False
    )
)

# ============================================================
# 13. 初步识别白天/夜间
#
# 规则：
# 同一时间只要任意一个站 SolarGeneration > 0，
# 暂时认为这一时刻属于“可能白天”。
#
# 注意：
# 这只是数据诊断规则，
# 不是最终夜间清洗规则。
# ============================================================

print("\n" + "=" * 70)
print("初步分析白天/夜间")
print("=" * 70)

positive_any = (
    solar_wide_full
    .gt(0)
    .any(
        axis=1
    )
)

day_mask = (
    positive_any
)

night_mask = (
    ~positive_any
)

print(
    "判定为白天的时间点数：",
    day_mask.sum()
)

print(
    "判定为夜间的时间点数：",
    night_mask.sum()
)

# ============================================================
# 14. 每个站生命周期内白天/夜间缺失
# ============================================================

day_night_records = []

for _, row in station_summary.iterrows():
    node_id = (
        row[
            "NodeID"
        ]
    )

    start_time = (
        row[
            "start_time"
        ]
    )

    end_time = (
        row[
            "end_time"
        ]
    )

    series = (
        solar_wide_full[
            node_id
        ]
        .loc[
            start_time:
            end_time
        ]
    )

    station_day_mask = (
        day_mask.loc[
            start_time:
            end_time
        ]
    )

    station_night_mask = (
        night_mask.loc[
            start_time:
            end_time
        ]
    )

    day_data = (
        series[
            station_day_mask
        ]
    )

    night_data = (
        series[
            station_night_mask
        ]
    )

    day_nan_ratio = (
        day_data
        .isna()
        .mean()
        if len(day_data) > 0
        else np.nan
    )

    day_valid_ratio = (
        day_data
        .notna()
        .mean()
        if len(day_data) > 0
        else np.nan
    )

    night_nan_ratio = (
        night_data
        .isna()
        .mean()
        if len(night_data) > 0
        else np.nan
    )

    night_valid_ratio = (
        night_data
        .notna()
        .mean()
        if len(night_data) > 0
        else np.nan
    )

    day_night_records.append(
        {
            "NodeID":
                node_id,

            "day_total_steps":
                len(
                    day_data
                ),

            "day_nan_ratio":
                day_nan_ratio,

            "day_valid_ratio":
                day_valid_ratio,

            "night_total_steps":
                len(
                    night_data
                ),

            "night_nan_ratio":
                night_nan_ratio,

            "night_valid_ratio":
                night_valid_ratio
        }
    )

day_night_df = pd.DataFrame(
    day_night_records
)

day_night_df = day_night_df.sort_values(
    "day_valid_ratio",
    ascending=False
)

day_night_df.to_csv(
    OUTPUT_DIR
    /
    "05_target_day_night_missing_analysis.csv",
    index=False
)

print(
    "\n白天/夜间缺失情况："
)

print(
    day_night_df.to_string(
        index=False
    )
)

# ============================================================
# 15. 找42个站全部上线后的共同时间区间
# ============================================================

print("\n" + "=" * 70)
print("计算42站共同生命周期")
print("=" * 70)

common_start = (
    station_summary[
        "start_time"
    ]
    .max()
)

common_end = (
    station_summary[
        "end_time"
    ]
    .min()
)

print(
    "42站共同开始时间：",
    common_start
)

print(
    "42站共同结束时间：",
    common_end
)

common_data = (
    solar_wide_full.loc[
        common_start:
        common_end
    ]
    .copy()
)

print(
    "共同时间段形状：",
    common_data.shape
)

print(
    "共同时间段NaN比例：",
    common_data
    .isna()
    .mean()
    .mean()
)

common_data.to_csv(
    OUTPUT_DIR
    /
    "06_target_all42_common_period_raw.csv"
)

# ============================================================
# 16. 共同时间段白天缺失率
# ============================================================

common_day_mask = (
    day_mask.loc[
        common_start:
        common_end
    ]
)

common_day_data = (
    common_data.loc[
        common_day_mask
    ]
)

common_day_coverage = (
    common_day_data
    .notna()
    .mean()
)

common_day_coverage_df = pd.DataFrame(
    {
        "NodeID":
            common_day_coverage.index,

        "common_period_day_valid_ratio":
            common_day_coverage.values
    }
)

common_day_coverage_df = (
    common_day_coverage_df
    .sort_values(
        "common_period_day_valid_ratio",
        ascending=False
    )
)

common_day_coverage_df.to_csv(
    OUTPUT_DIR
    /
    "07_target_common_period_day_coverage.csv",
    index=False
)

print(
    "\n42站共同时间段内白天覆盖率："
)

print(
    common_day_coverage_df.to_string(
        index=False
    )
)

# ============================================================
# 17. 按月统计
#
# 分两种：
# 1. 原始全天覆盖率
# 2. 初步白天覆盖率
# ============================================================

print("\n" + "=" * 70)
print("按月统计目标域覆盖率")
print("=" * 70)

monthly_records = []

month_index = (
    solar_wide_full
    .index
    .to_period(
        "M"
    )
)

for month in sorted(
        month_index.unique()
):

    month_mask = (
            month_index
            ==
            month
    )

    month_data = (
        solar_wide_full.loc[
            month_mask
        ]
    )

    month_day_mask = (
        day_mask.loc[
            month_data.index
        ]
    )

    month_day_data = (
        month_data.loc[
            month_day_mask
        ]
    )

    raw_coverage = (
        month_data
        .notna()
        .mean()
    )

    if len(
            month_day_data
    ) > 0:

        day_coverage = (
            month_day_data
            .notna()
            .mean()
        )

    else:

        day_coverage = pd.Series(
            np.nan,
            index=month_data.columns
        )

    monthly_records.append(
        {
            "month":
                str(
                    month
                ),

            "active_station_count":
                (
                        raw_coverage > 0
                ).sum(),

            "raw_coverage_ge_50":
                (
                        raw_coverage
                        >= 0.50
                ).sum(),

            "raw_coverage_ge_70":
                (
                        raw_coverage
                        >= 0.70
                ).sum(),

            "day_coverage_ge_70":
                (
                        day_coverage
                        >= 0.70
                ).sum(),

            "day_coverage_ge_80":
                (
                        day_coverage
                        >= 0.80
                ).sum(),

            "day_coverage_ge_90":
                (
                        day_coverage
                        >= 0.90
                ).sum(),

            "day_coverage_ge_95":
                (
                        day_coverage
                        >= 0.95
                ).sum(),

            "mean_raw_coverage":
                raw_coverage.mean(),

            "mean_day_coverage":
                day_coverage.mean()
        }
    )

monthly_summary = pd.DataFrame(
    monthly_records
)

monthly_summary.to_csv(
    OUTPUT_DIR
    /
    "08_target_monthly_coverage_analysis.csv",
    index=False
)

print(
    monthly_summary.to_string(
        index=False
    )
)

# ============================================================
# 18. 最长连续NaN
#
# 注意：
# 这里只在每个站自己的生命周期内部统计
# ============================================================

print("\n" + "=" * 70)
print("统计生命周期内最长连续缺失")
print("=" * 70)


def max_consecutive_nan(
        series
):
    missing = (
        series.isna()
    )

    if not missing.any():
        return 0

    groups = (
            missing
            != missing.shift()
    ).cumsum()

    missing_lengths = (
        missing[
            missing
        ]
        .groupby(
            groups[
                missing
            ]
        )
        .size()
    )

    if len(
            missing_lengths
    ) == 0:
        return 0

    return int(
        missing_lengths.max()
    )


gap_records = []

for _, row in station_summary.iterrows():
    node_id = (
        row[
            "NodeID"
        ]
    )

    start_time = (
        row[
            "start_time"
        ]
    )

    end_time = (
        row[
            "end_time"
        ]
    )

    series = (
        solar_wide_full[
            node_id
        ]
        .loc[
            start_time:
            end_time
        ]
    )

    max_gap = (
        max_consecutive_nan(
            series
        )
    )

    gap_records.append(
        {
            "NodeID":
                node_id,

            "max_consecutive_nan_steps":
                max_gap,

            "max_consecutive_nan_hours":
                max_gap
                *
                0.25,

            "max_consecutive_nan_days":
                max_gap
                *
                0.25
                /
                24
        }
    )

gap_df = pd.DataFrame(
    gap_records
)

gap_df = gap_df.sort_values(
    "max_consecutive_nan_steps",
    ascending=False
)

gap_df.to_csv(
    OUTPUT_DIR
    /
    "09_target_lifecycle_max_consecutive_missing.csv",
    index=False
)

print(
    gap_df.to_string(
        index=False
    )
)

# ============================================================
# 19. 读取 Solar_Site_Details.csv
# ============================================================

print("\n" + "=" * 70)
print("读取 Solar_Site_Details.csv")
print("=" * 70)

site_details = pd.read_csv(
    SITE_DETAILS_CSV
)

print(
    "Site Details形状：",
    site_details.shape
)

print(
    "\n字段："
)

print(
    site_details.columns.tolist()
)

print(
    "\n前10行："
)

print(
    site_details
    .head(10)
    .to_string(
        index=False
    )
)

# ============================================================
# 20. metadata构造NodeID
# ============================================================

if (
        "CampusKey"
        in
        site_details.columns
        and
        "SiteKey"
        in
        site_details.columns
):
    site_details[
        "CampusKey"
    ] = pd.to_numeric(
        site_details[
            "CampusKey"
        ],
        errors="coerce"
    )

    site_details[
        "SiteKey"
    ] = pd.to_numeric(
        site_details[
            "SiteKey"
        ],
        errors="coerce"
    )

    site_details[
        "NodeID"
    ] = (
            "C"
            +
            site_details[
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
            site_details[
                "SiteKey"
            ]
            .astype(
                "Int64"
            )
            .astype(
                str
            )
    )

# ============================================================
# 21. kWp容量检查
# ============================================================

print("\n" + "=" * 70)
print("容量信息检查")
print("=" * 70)

if "kWp" in site_details.columns:
    site_details[
        "kWp"
    ] = pd.to_numeric(
        site_details[
            "kWp"
        ],
        errors="coerce"
    )

    print(
        "有kWp的节点数量：",
        site_details[
            "kWp"
        ]
        .notna()
        .sum()
    )

    print(
        "缺少kWp的节点数量：",
        site_details[
            "kWp"
        ]
        .isna()
        .sum()
    )

    print(
        "\nkWp统计："
    )

    print(
        site_details[
            "kWp"
        ]
        .describe()
    )

# ============================================================
# 22. 经纬度检查
# ============================================================

for col in [
    "lat",
    "Lon"
]:

    if col in site_details.columns:
        site_details[
            col
        ] = pd.to_numeric(
            site_details[
                col
            ],
            errors="coerce"
        )

if (
        "lat"
        in
        site_details.columns
        and
        "Lon"
        in
        site_details.columns
):
    print(
        "\n经纬度缺失："
    )

    print(
        site_details[
            [
                "lat",
                "Lon"
            ]
        ]
        .isna()
        .sum()
    )

    duplicate_location = (
        site_details
        .groupby(
            [
                "lat",
                "Lon"
            ],
            dropna=False
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\n相同经纬度位置对应的节点数："
    )

    print(
        duplicate_location
    )

# ============================================================
# 23. 功率节点与metadata节点匹配
# ============================================================

print("\n" + "=" * 70)
print("检查功率与metadata节点匹配")
print("=" * 70)

solar_nodes = set(
    solar_wide_full.columns
)

metadata_nodes = set(
    site_details[
        "NodeID"
    ]
    .dropna()
)

only_in_solar = (
        solar_nodes
        -
        metadata_nodes
)

only_in_metadata = (
        metadata_nodes
        -
        solar_nodes
)

print(
    "功率数据节点数：",
    len(
        solar_nodes
    )
)

print(
    "metadata节点数：",
    len(
        metadata_nodes
    )
)

print(
    "只在功率中存在：",
    sorted(
        only_in_solar
    )
)

print(
    "只在metadata中存在：",
    sorted(
        only_in_metadata
    )
)

site_details.to_csv(
    OUTPUT_DIR
    /
    "10_target_site_details_processed.csv",
    index=False
)

# ============================================================
# 24. 读取 Monthly_Summary_Solar.csv
# ============================================================

print("\n" + "=" * 70)
print("读取 Monthly_Summary_Solar.csv")
print("=" * 70)

monthly_file = pd.read_csv(
    MONTHLY_SUMMARY_CSV
)

print(
    "Monthly Summary形状：",
    monthly_file.shape
)

print(
    "\n字段："
)

print(
    monthly_file.columns.tolist()
)

print(
    "\n前10行："
)

print(
    monthly_file
    .head(10)
    .to_string(
        index=False
    )
)

if (
        "DataStatus"
        in
        monthly_file.columns
):
    print(
        "\nDataStatus统计："
    )

    print(
        monthly_file[
            "DataStatus"
        ]
        .value_counts(
            dropna=False
        )
    )

# ============================================================
# 25. 保存分析摘要
# ============================================================

summary_output = (
        OUTPUT_DIR
        /
        "11_TARGET_ANALYSIS_SUMMARY.txt"
)

with open(
        summary_output,
        "w",
        encoding="utf-8"
) as f:
    f.write(
        "UNISOLAR Target Domain Analysis\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )

    f.write(
        f"Number of campuses: "
        f"{solar['CampusKey'].nunique()}\n"
    )

    f.write(
        f"Number of nodes: "
        f"{solar['NodeID'].nunique()}\n"
    )

    f.write(
        f"Original time range: "
        f"{solar['Timestamp'].min()} "
        f"to "
        f"{solar['Timestamp'].max()}\n"
    )

    f.write(
        f"Full 15-min shape: "
        f"{solar_wide_full.shape}\n"
    )

    f.write(
        f"Common start of all 42 nodes: "
        f"{common_start}\n"
    )

    f.write(
        f"Common end of all 42 nodes: "
        f"{common_end}\n"
    )

    f.write(
        f"All-42 common-period shape: "
        f"{common_data.shape}\n"
    )

    f.write(
        f"Raw NaN ratio in all-42 common period: "
        f"{common_data.isna().mean().mean()}\n"
    )

    f.write(
        f"Missing full timestamps in original dataset: "
        f"{len(missing_original_timestamps)}\n"
    )

    if (
            "kWp"
            in
            site_details.columns
    ):
        f.write(
            f"Nodes with kWp: "
            f"{site_details['kWp'].notna().sum()}\n"
        )

        f.write(
            f"Nodes without kWp: "
            f"{site_details['kWp'].isna().sum()}\n"
        )

# ============================================================
# 26. 完成
# ============================================================

print("\n" + "=" * 70)
print("目标域修正版分析完成")
print("=" * 70)

print(
    "\n结果目录："
)

print(
    OUTPUT_DIR
)

print(
    "\n下一步重点查看："
)

print(
    "04_target_lifecycle_coverage.csv"
)

print(
    "05_target_day_night_missing_analysis.csv"
)

print(
    "07_target_common_period_day_coverage.csv"
)

print(
    "08_target_monthly_coverage_analysis.csv"
)

print(
    "09_target_lifecycle_max_consecutive_missing.csv"
)
