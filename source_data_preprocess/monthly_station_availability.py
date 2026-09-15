import pandas as pd

# ============================================================
# 路径
# ============================================================

DATA_PATH = (
    "../dataset/source/processed_source/"
    "02_source_15min_normalized_full.csv"
)

OUTPUT_PATH = (
    "../dataset/source/processed_source/"
    "monthly_station_availability.csv"
)

# ============================================================
# 读取数据
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    index_col="DateTime",
    parse_dates=True
)

print("数据形状：", df.shape)

print(
    "时间范围：",
    df.index.min(),
    "->",
    df.index.max()
)

# ============================================================
# 按月统计
# ============================================================

results = []

# 按月份分组
for month, group in df.groupby(
        df.index.to_period("M")
):
    # 每个站在这个月的数据覆盖率
    coverage = group.notna().mean()

    result = {
        "month": str(month),

        # 至少出现过一个数据的站
        "active_station_count":
            (coverage > 0).sum(),

        # 不同覆盖率阈值
        "coverage_ge_80":
            (coverage >= 0.80).sum(),

        "coverage_ge_90":
            (coverage >= 0.90).sum(),

        "coverage_ge_95":
            (coverage >= 0.95).sum(),

        "coverage_ge_99":
            (coverage >= 0.99).sum(),

        "coverage_eq_100":
            (coverage == 1.0).sum(),

        # 所有站平均覆盖率
        "mean_coverage":
            coverage.mean()
    }

    results.append(result)

monthly = pd.DataFrame(results)

# ============================================================
# 输出
# ============================================================

print("\n每月站点可用情况：")

print(
    monthly.to_string(
        index=False
    )
)

# ============================================================
# 找覆盖率95%以上站点最多的月份
# ============================================================

best_months = monthly.sort_values(
    "coverage_ge_95",
    ascending=False
)

print("\n覆盖率 >= 95% 的站点数最多的月份：")

print(
    best_months.head(12).to_string(
        index=False
    )
)

# ============================================================
# 保存
# ============================================================

monthly.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\n已保存：",
    OUTPUT_PATH
)
