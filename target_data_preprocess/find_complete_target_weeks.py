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
    "week_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 参数
# ============================================================

DAYS_PER_WEEK = 7

STEPS_PER_DAY = 24 * 4

STEPS_PER_WEEK = (
        DAYS_PER_WEEK
        *
        STEPS_PER_DAY
)

# ============================================================
# 3. 读取数据
# ============================================================

print("\n" + "=" * 75)
print("读取目标域数据")
print("=" * 75)

df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True
)

df.index.name = "Timestamp"

df = df.sort_index()

print(
    "数据形状：",
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

print(
    "NaN总数：",
    int(
        df.isna()
        .sum()
        .sum()
    )
)

# ============================================================
# 4. 检查时间轴
# ============================================================

expected_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="15min"
)

if df.index.equals(expected_index):

    print(
        "时间轴严格连续15分钟。"
    )

else:

    print(
        "警告：时间轴存在缺失或错位。"
    )

# ============================================================
# 5. 获取所有自然日
#
# 每个候选周：
# 第1天 00:00
# 到
# 第7天 23:45
# ============================================================

all_dates = pd.date_range(
    start=df.index.min().normalize(),
    end=df.index.max().normalize(),
    freq="D"
)

# ============================================================
# 6. 分别记录
#
# complete_weeks:
# 整周31个站全部无NaN
#
# nan_weeks:
# 整周至少存在一个NaN
# ============================================================

complete_weeks = []

nan_weeks = []

print("\n" + "=" * 75)
print("扫描所有连续7天窗口")
print("=" * 75)

for start_date in all_dates:

    end_date = (
            start_date
            +
            pd.Timedelta(
                days=6
            )
    )

    start_time = start_date

    end_time = (
            end_date
            +
            pd.Timedelta(
                hours=23,
                minutes=45
            )
    )

    # 超出原始数据范围
    if start_time < df.index.min():
        continue

    if end_time > df.index.max():
        continue

    week_data = df.loc[
        start_time:
        end_time
    ]

    # 必须是完整7天
    # 7 × 24 × 4 = 672
    if len(week_data) != STEPS_PER_WEEK:
        continue

    total_nan = int(
        week_data
        .isna()
        .sum()
        .sum()
    )

    # ========================================================
    # 完全没有NaN
    # ========================================================

    if total_nan == 0:

        complete_weeks.append(
            {
                "start_date":
                    start_date.strftime(
                        "%Y-%m-%d"
                    ),

                "end_date":
                    end_date.strftime(
                        "%Y-%m-%d"
                    )
            }
        )


    # ========================================================
    # 存在NaN
    # ========================================================

    else:

        nan_weeks.append(
            {
                "start_date":
                    start_date.strftime(
                        "%Y-%m-%d"
                    ),

                "end_date":
                    end_date.strftime(
                        "%Y-%m-%d"
                    )
            }
        )

# ============================================================
# 7. 转成DataFrame
# ============================================================

complete_weeks_df = pd.DataFrame(
    complete_weeks,
    columns=[
        "start_date",
        "end_date"
    ]
)

nan_weeks_df = pd.DataFrame(
    nan_weeks,
    columns=[
        "start_date",
        "end_date"
    ]
)

# ============================================================
# 8. 保存“没有NaN”的周
# ============================================================

COMPLETE_FILE = (
        OUTPUT_DIR
        /
        "01_WEEKS_WITHOUT_NAN.csv"
)

complete_weeks_df.to_csv(
    COMPLETE_FILE,
    index=False
)

# ============================================================
# 9. 保存“有NaN”的周
# ============================================================

NAN_FILE = (
        OUTPUT_DIR
        /
        "02_WEEKS_WITH_NAN.csv"
)

nan_weeks_df.to_csv(
    NAN_FILE,
    index=False
)

# ============================================================
# 10. 打印结果
# ============================================================

print("\n" + "=" * 75)
print("完全没有NaN的7天窗口")
print("=" * 75)

print(
    "数量：",
    len(
        complete_weeks_df
    )
)

if not complete_weeks_df.empty:

    print(
        complete_weeks_df
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有找到。"
    )

print("\n" + "=" * 75)
print("存在NaN的7天窗口")
print("=" * 75)

print(
    "数量：",
    len(
        nan_weeks_df
    )
)

if not nan_weeks_df.empty:

    print(
        nan_weeks_df
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有找到。"
    )

# ============================================================
# 11. 完成
# ============================================================

print("\n" + "=" * 75)
print("分析完成")
print("=" * 75)

print(
    "\n生成文件："
)

print(
    COMPLETE_FILE
)

print(
    NAN_FILE
)
