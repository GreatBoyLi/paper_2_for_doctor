from pathlib import Path

import pandas as pd

# ============================================================
# 1. 路径设置
# ============================================================

INPUT_CSV = Path(
    "../data/target/processed_target_final/"
    "08_TARGET_CLEAN_STRICT_UNNORMALIZED.csv"
)

OUTPUT_DIR = Path(
    "../data/target/processed_target_final/"
    "selected_period"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 设置开始日期和结束日期
#
# 这里只写日期，不需要写时间
# ============================================================

START_DATE = "2021-03-07"

END_DATE = "2021-03-13"

# ============================================================
# 3. 输出文件名
# ============================================================

OUTPUT_FILENAME = (
    "TARGET_MODEL_DATA.csv"
)

# ============================================================
# 4. 是否要求选择的数据完全没有NaN
#
# True：
# 只要存在NaN就报错，不保存
#
# False：
# 即使有NaN也保存
#
# 后面直接用于模型时，建议保持True
# ============================================================

REQUIRE_NO_NAN = True

# ============================================================
# 5. 自动把日期转换成实际时间范围
#
# 例如：
#
# START_DATE = 2021-08-01
# END_DATE   = 2021-08-07
#
# 实际截取：
#
# 2021-08-01 00:15:00
# ->
# 2021-08-07 23:45:00
# ============================================================

start_time = pd.Timestamp(
    START_DATE
)

end_time = (
        pd.Timestamp(
            END_DATE
        )
        +
        pd.Timedelta(
            hours=23,
            minutes=45
        )
)

# ============================================================
# 6. 读取目标域数据
# ============================================================

print("\n" + "=" * 75)
print("读取目标域最终数据")
print("=" * 75)

df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True
)

df.index.name = (
    "Timestamp"
)

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

print(
    "节点数量：",
    df.shape[1]
)

# ============================================================
# 7. 检查日期是否合法
# ============================================================

if start_time > end_time:
    raise ValueError(
        "START_DATE不能晚于END_DATE"
    )

if start_time < df.index.min():
    raise ValueError(
        f"开始日期 {START_DATE} "
        f"早于数据最早日期 "
        f"{df.index.min().date()}"
    )

if end_time > df.index.max():
    raise ValueError(
        f"结束日期 {END_DATE} "
        f"晚于数据最晚日期 "
        f"{df.index.max().date()}"
    )

# ============================================================
# 8. 截取指定日期范围
# ============================================================

selected = (
    df.loc[
        start_time:
        end_time
    ]
    .copy()
)

print("\n" + "=" * 75)
print("截取结果")
print("=" * 75)

print(
    "设置的开始日期：",
    START_DATE
)

print(
    "设置的结束日期：",
    END_DATE
)

print(
    "实际开始时间：",
    selected.index.min()
)

print(
    "实际结束时间：",
    selected.index.max()
)

print(
    "数据形状：",
    selected.shape
)

# ============================================================
# 9. 计算理论时间点数量
#
# 每天：
# 24 × 4 = 96个15min点
# ============================================================

start_date_only = pd.Timestamp(
    START_DATE
)

end_date_only = pd.Timestamp(
    END_DATE
)

number_of_days = (
                         end_date_only
                         -
                         start_date_only
                 ).days + 1

expected_steps = (
        number_of_days
        *
        24
        *
        4
)

print(
    "\n总天数：",
    number_of_days
)

print(
    "理论15min时间点数：",
    expected_steps
)

print(
    "实际时间点数：",
    len(
        selected
    )
)

# ============================================================
# 10. 检查15分钟时间轴是否完整
# ============================================================

expected_index = pd.date_range(
    start=start_time,
    end=end_time,
    freq="15min"
)

missing_timestamps = (
    expected_index
    .difference(
        selected.index
    )
)

if len(
        missing_timestamps
) == 0:

    print(
        "时间轴完整：是"
    )

else:

    print(
        "时间轴完整：否"
    )

    print(
        "缺失时间戳数量：",
        len(
            missing_timestamps
        )
    )

    print(
        "缺失时间戳前20个："
    )

    print(
        missing_timestamps[
            :20
        ]
    )

# ============================================================
# 11. NaN检查
# ============================================================

total_nan = int(
    selected
    .isna()
    .sum()
    .sum()
)

nodes_with_nan_mask = (
    selected
    .isna()
    .any(
        axis=0
    )
)

nodes_with_nan = (
    nodes_with_nan_mask[
        nodes_with_nan_mask
    ]
    .index
    .tolist()
)

print(
    "\nNaN总数：",
    total_nan
)

print(
    "存在NaN的节点数：",
    len(
        nodes_with_nan
    )
)

if len(
        nodes_with_nan
) > 0:
    print(
        "存在NaN的节点："
    )

    print(
        nodes_with_nan
    )

# ============================================================
# 12. 严格检查
#
# 后面直接作为模型输入的话，
# 推荐只使用完全无NaN的时间段。
# ============================================================

if (
        REQUIRE_NO_NAN
        and
        total_nan > 0
):
    raise RuntimeError(
        "\n当前选择的日期范围中仍然存在NaN。\n"
        "为了直接用于模型，建议重新选择"
        "一个完整的数据时间段。\n"
        f"当前NaN总数：{total_nan}\n"
    )

# ============================================================
# 13. 数据基础统计
# ============================================================

print("\n" + "=" * 75)
print("数据统计")
print("=" * 75)

if not selected.empty:
    print(
        "最小值：",
        selected.min().min()
    )

    print(
        "最大值：",
        selected.max().max()
    )

    print(
        "平均值：",
        selected.stack().mean()
    )

# ============================================================
# 14. 保存截取后的目标域模型数据
# ============================================================

OUTPUT_FILE = (
        OUTPUT_DIR
        /
        OUTPUT_FILENAME
)

selected.to_csv(
    OUTPUT_FILE
)

print("\n" + "=" * 75)
print("保存完成")
print("=" * 75)

print(
    "模型数据文件："
)

print(
    OUTPUT_FILE
)

print(
    "\n最终形状：",
    selected.shape
)

print(
    "最终NaN数量：",
    total_nan
)

# ============================================================
# 15. 保存截取信息
# ============================================================

SUMMARY_FILE = (
        OUTPUT_DIR
        /
        "TARGET_MODEL_DATA_SUMMARY.txt"
)

with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
) as f:
    f.write(
        "Target Model Data Summary\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )

    f.write(
        f"Source file: {INPUT_CSV}\n"
    )

    f.write(
        f"Start date: {START_DATE}\n"
    )

    f.write(
        f"End date: {END_DATE}\n"
    )

    f.write(
        f"Actual start time: "
        f"{selected.index.min()}\n"
    )

    f.write(
        f"Actual end time: "
        f"{selected.index.max()}\n"
    )

    f.write(
        f"Number of days: "
        f"{number_of_days}\n"
    )

    f.write(
        f"Rows: "
        f"{selected.shape[0]}\n"
    )

    f.write(
        f"Nodes: "
        f"{selected.shape[1]}\n"
    )

    f.write(
        f"NaN count: "
        f"{total_nan}\n"
    )

print(
    "\n说明文件："
)

print(
    SUMMARY_FILE
)
