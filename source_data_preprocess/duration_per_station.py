import pandas as pd

METADATA_CSV = "../dataset/source/metadata.csv"

metadata = pd.read_csv(
    METADATA_CSV,
    sep=";"
)

# 删除多余空列
metadata = metadata.loc[
    :,
    ~metadata.columns.str.startswith("Unnamed")
]

metadata = metadata.dropna(
    axis=1,
    how="all"
)

# 转换时间
metadata["begin_ts"] = pd.to_datetime(
    metadata["begin_ts"],
    utc=True
)

metadata["end_ts"] = pd.to_datetime(
    metadata["end_ts"],
    utc=True
)

# 提取年份
metadata["begin_year"] = (
    metadata["begin_ts"].dt.year
)

metadata["end_year"] = (
    metadata["end_ts"].dt.year
)

# 计算持续天数
metadata["duration_days"] = (
                                    metadata["end_ts"]
                                    -
                                    metadata["begin_ts"]
                            ).dt.total_seconds() / 86400

# ==================================================
# 每个站点的时间范围
# ==================================================

station_period = metadata[
    [
        "ID",
        "begin_ts",
        "end_ts",
        "begin_year",
        "end_year",
        "duration_days"
    ]
].copy()

print("\n每个站点的数据时间范围：")
print(
    station_period.to_string(
        index=False
    )
)

# ==================================================
# 按开始年份统计
# ==================================================

print("\n按开始年份统计：")

print(
    metadata[
        "begin_year"
    ]
    .value_counts()
    .sort_index()
)

# ==================================================
# 按结束年份统计
# ==================================================

print("\n按结束年份统计：")

print(
    metadata[
        "end_year"
    ]
    .value_counts()
    .sort_index()
)

# ==================================================
# 保存结果
# ==================================================

station_period.to_csv(
    "station_time_distribution.csv",
    index=False
)

print(
    "\n已保存："
    "station_time_distribution.csv"
)
