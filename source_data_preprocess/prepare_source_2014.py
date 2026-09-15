from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径设置
# ============================================================

INPUT_CSV = Path(
    "../data/source/processed_source/"
    "02_source_15min_normalized_full.csv"
)

OUTPUT_DIR = Path(
    "../data/source/final/source_2014"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 参数设置
# ============================================================

# 固定使用 2014 全年
START_TIME = "2014-01-01 00:00:00+00:00"
END_TIME = "2014-12-31 23:45:00+00:00"

# 站点全年最低数据覆盖率
MIN_COVERAGE = 0.90

# ============================================================
# 3. 读取已经归一化的15min数据
# ============================================================

print("\n" + "=" * 60)
print("读取源域15min归一化数据")
print("=" * 60)

df = pd.read_csv(
    INPUT_CSV,
    index_col="DateTime",
    parse_dates=True
)

# 保证按时间排序
df = df.sort_index()

print(
    "原始数据形状：",
    df.shape
)

print(
    "原始时间范围：",
    df.index.min(),
    "->",
    df.index.max()
)

# ============================================================
# 4. 截取2014全年
# ============================================================

print("\n" + "=" * 60)
print("截取2014全年")
print("=" * 60)

source_2014 = df.loc[
    START_TIME:END_TIME
].copy()

print(
    "2014数据形状：",
    source_2014.shape
)

print(
    "2014时间范围：",
    source_2014.index.min(),
    "->",
    source_2014.index.max()
)

# ============================================================
# 5. 检查时间索引
# ============================================================

# 2014不是闰年
EXPECTED_STEPS = (
        365
        * 24
        * 4
)

print(
    "理论时间步数：",
    EXPECTED_STEPS
)

print(
    "实际时间步数：",
    len(source_2014)
)

if len(source_2014) != EXPECTED_STEPS:
    print(
        "\n警告："
        "2014实际时间步数不是35040。"
    )

# ============================================================
# 6. 计算每个站2014全年覆盖率
# ============================================================

print("\n" + "=" * 60)
print("统计2014全年站点覆盖率")
print("=" * 60)

coverage = (
    source_2014
    .notna()
    .mean()
)

coverage = coverage.sort_values(
    ascending=False
)

coverage_df = pd.DataFrame(
    {
        "station":
            coverage.index,

        "coverage":
            coverage.values
    }
)

coverage_output = (
        OUTPUT_DIR
        / "01_station_coverage_2014.csv"
)

coverage_df.to_csv(
    coverage_output,
    index=False
)

print(
    "覆盖率 >= 90% 的站点数：",
    (
            coverage
            >= MIN_COVERAGE
    ).sum()
)

print(
    "\n覆盖率最高的20个站："
)

print(
    coverage.head(20)
)

print(
    "\n覆盖率最低的20个站："
)

print(
    coverage.tail(20)
)

# ============================================================
# 7. 筛选全年覆盖率 >= 90%的站点
# ============================================================

print("\n" + "=" * 60)
print("筛选源域节点")
print("=" * 60)

selected_stations = (
    coverage[
        coverage >= MIN_COVERAGE
        ]
    .index
    .tolist()
)

print(
    "最终候选站点数量：",
    len(selected_stations)
)

print(
    "\n候选站点："
)

print(
    selected_stations
)

source_selected = (
    source_2014[
        selected_stations
    ]
    .copy()
)

print(
    "\n筛选后数据形状：",
    source_selected.shape
)

# ============================================================
# 8. 统计插值前缺失情况
# ============================================================

print("\n" + "=" * 60)
print("统计插值前缺失情况")
print("=" * 60)

missing_count_before = (
    source_selected
    .isna()
    .sum()
)

missing_ratio_before = (
    source_selected
    .isna()
    .mean()
)

missing_before_df = pd.DataFrame(
    {
        "station":
            source_selected.columns,

        "missing_count":
            missing_count_before.values,

        "missing_ratio":
            missing_ratio_before.values
    }
)

missing_before_df = (
    missing_before_df
    .sort_values(
        "missing_ratio",
        ascending=False
    )
)

missing_before_df.to_csv(
    OUTPUT_DIR
    / "02_missing_before_interpolation.csv",
    index=False
)

print(
    "缺失点总数：",
    source_selected
    .isna()
    .sum()
    .sum()
)

print(
    "总体缺失率：",
    source_selected
    .isna()
    .mean()
    .mean()
)


# ============================================================
# 9. 计算每个站最长连续缺失长度
# ============================================================

def calculate_max_consecutive_nan(series):
    """
    计算一个站点最长连续NaN长度。

    返回：
        max_gap_steps:
            最长连续缺失的15min时间步数量

        max_gap_hours:
            最长连续缺失对应小时数
    """

    is_nan = series.isna()

    if not is_nan.any():
        return 0

    # 连续区间编号
    groups = (
            is_nan
            != is_nan.shift()
    ).cumsum()

    # 每一段连续缺失长度
    lengths = (
        is_nan
        .groupby(groups)
        .sum()
    )

    return int(
        lengths.max()
    )


gap_records = []

for station in source_selected.columns:
    max_gap_steps = (
        calculate_max_consecutive_nan(
            source_selected[
                station
            ]
        )
    )

    gap_records.append(
        {
            "station":
                station,

            "max_consecutive_nan_steps":
                max_gap_steps,

            "max_consecutive_nan_hours":
                max_gap_steps * 0.25
        }
    )

gap_df = pd.DataFrame(
    gap_records
)

gap_df = (
    gap_df
    .sort_values(
        "max_consecutive_nan_steps",
        ascending=False
    )
)

gap_df.to_csv(
    OUTPUT_DIR
    / "03_max_consecutive_missing.csv",
    index=False
)

print(
    "\n最长连续缺失最大的20个站："
)

print(
    gap_df.head(20).to_string(
        index=False
    )
)

# ============================================================
# 10. 对内部缺失进行时间线性插值
# ============================================================

print("\n" + "=" * 60)
print("开始线性插值")
print("=" * 60)

source_interp = (
    source_selected
    .interpolate(
        method="time",
        axis=0,
        limit_area="inside"
    )
)

# 功率不允许为负
source_interp = (
    source_interp
    .clip(
        lower=0
    )
)

print(
    "内部缺失线性插值完成。"
)

# ============================================================
# 11. 检查插值后的缺失
# ============================================================

print("\n" + "=" * 60)
print("检查插值后剩余缺失")
print("=" * 60)

missing_after = (
    source_interp
    .isna()
    .sum()
)

remaining_missing = (
    missing_after[
        missing_after > 0
        ]
)

print(
    "插值后剩余NaN总数：",
    missing_after.sum()
)

print(
    "仍有NaN的站点数量：",
    len(
        remaining_missing
    )
)

if len(remaining_missing) > 0:
    print(
        "\n仍存在NaN的站点："
    )

    print(
        remaining_missing
        .sort_values(
            ascending=False
        )
    )

# ============================================================
# 12. 处理序列边界缺失
# ============================================================

# interpolate(limit_area="inside")
# 不会填充：
#
# NaN NaN 有效值 ...
#
# 或
#
# ... 有效值 NaN NaN
#
# 如果这里还有少量边界缺失，
# 用最近的有效观测值补齐。
#
# 注意：
# 这里不是填补大段内部缺失；
# 内部缺失已经通过时间线性插值处理。


if source_interp.isna().sum().sum() > 0:

    print(
        "\n存在头部/尾部边界缺失，"
        "使用最近有效值补齐。"
    )

    source_final = (
        source_interp
        .ffill()
        .bfill()
    )

else:

    source_final = (
        source_interp.copy()
    )

# ============================================================
# 13. 最终质量检查
# ============================================================

print("\n" + "=" * 60)
print("最终2014源域数据检查")
print("=" * 60)

print(
    "最终数据形状：",
    source_final.shape
)

print(
    "最终时间范围：",
    source_final.index.min(),
    "->",
    source_final.index.max()
)

print(
    "最终时间步数：",
    source_final.shape[0]
)

print(
    "最终节点数量：",
    source_final.shape[1]
)

print(
    "剩余NaN数量：",
    source_final
    .isna()
    .sum()
    .sum()
)

print(
    "最小值：",
    source_final
    .min()
    .min()
)

print(
    "最大值：",
    source_final
    .max()
    .max()
)

# ============================================================
# 14. 检查时间是否连续
# ============================================================

expected_index = pd.date_range(
    start=START_TIME,
    end=END_TIME,
    freq="15min"
)

missing_timestamps = (
    expected_index
    .difference(
        source_final.index
    )
)

print(
    "缺失时间戳数量：",
    len(
        missing_timestamps
    )
)

if len(missing_timestamps) > 0:
    print(
        "\n缺失时间戳前20个："
    )

    print(
        missing_timestamps[:20]
    )

# ============================================================
# 15. 保存最终2014源域数据
# ============================================================

final_output = (
        OUTPUT_DIR
        / "04_SOURCE_2014_FINAL.csv"
)

source_final.to_csv(
    final_output
)

print(
    "\n已保存最终源域数据："
)

print(
    final_output
)

# ============================================================
# 16. 保存最终站点列表
# ============================================================

station_list_df = pd.DataFrame(
    {
        "station":
            source_final.columns,

        "coverage_2014":
            coverage.reindex(
                source_final.columns
            ).values
    }
)

station_list_output = (
        OUTPUT_DIR
        / "05_SOURCE_2014_STATIONS.csv"
)

station_list_df.to_csv(
    station_list_output,
    index=False
)

print(
    "\n已保存最终站点列表："
)

print(
    station_list_output
)

# ============================================================
# 17. 保存数据处理摘要
# ============================================================

summary_output = (
        OUTPUT_DIR
        / "06_SOURCE_2014_SUMMARY.txt"
)

with open(
        summary_output,
        "w",
        encoding="utf-8"
) as f:
    f.write(
        "Utrecht Source Domain "
        "2014 Preprocessing Summary\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )

    f.write(
        f"Time range:\n"
        f"{source_final.index.min()} "
        f"to "
        f"{source_final.index.max()}\n\n"
    )

    f.write(
        f"Sampling interval:\n"
        f"15 minutes\n\n"
    )

    f.write(
        f"Coverage threshold:\n"
        f"{MIN_COVERAGE}\n\n"
    )

    f.write(
        f"Original node number:\n"
        f"{source_2014.shape[1]}\n\n"
    )

    f.write(
        f"Selected node number:\n"
        f"{source_final.shape[1]}\n\n"
    )

    f.write(
        f"Time steps:\n"
        f"{source_final.shape[0]}\n\n"
    )

    f.write(
        f"Final shape:\n"
        f"{source_final.shape}\n\n"
    )

    f.write(
        f"Remaining NaNs:\n"
        f"{source_final.isna().sum().sum()}\n\n"
    )

    f.write(
        f"Minimum value:\n"
        f"{source_final.min().min()}\n\n"
    )

    f.write(
        f"Maximum value:\n"
        f"{source_final.max().max()}\n\n"
    )

    f.write(
        "Normalization:\n"
        "P_norm = P / estimated_ac_capacity\n\n"
    )

    f.write(
        "Missing value processing:\n"
        "Internal missing values: "
        "time-based linear interpolation\n"
    )

    f.write(
        "Boundary missing values: "
        "forward/backward nearest valid value\n\n"
    )

    f.write(
        "Dataset split:\n"
        "Not performed during preprocessing.\n"
    )

    f.write(
        "Training/validation folds will be generated "
        "dynamically during Stage 1 training "
        "using time-series cross-validation.\n"
    )

print(
    "\n已保存处理摘要："
)

print(
    summary_output
)

# ============================================================
# 18. 完成
# ============================================================

print("\n" + "=" * 60)

print(
    "2014全年源域数据预处理完成"
)

print("=" * 60)

print(
    "\n注意：本脚本没有划分"
    "Train / Validation / Test。"
)

print(
    "后续Stage 1训练时，"
    "再进行时间序列多折划分。"
)
