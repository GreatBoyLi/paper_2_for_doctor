import pandas as pd

# ============================================================
# 路径
# ============================================================

DATA_PATH = (
    "../dataset/source/processed_source/"
    "02_source_15min_normalized_full.csv"
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
# 定义候选1年时间段
# ============================================================

windows = [

    (
        "2014全年",
        "2014-01-01 00:00:00+00:00",
        "2014-12-31 23:45:00+00:00"
    ),

    (
        "2015全年",
        "2015-01-01 00:00:00+00:00",
        "2015-12-31 23:45:00+00:00"
    ),

    (
        "2016全年",
        "2016-01-01 00:00:00+00:00",
        "2016-12-31 23:45:00+00:00"
    ),

    (
        "2017全年",
        "2017-01-01 00:00:00+00:00",
        "2017-12-31 23:45:00+00:00"
    ),

    # 连续跨年一年
    (
        "2014-07到2015-06",
        "2014-07-01 00:00:00+00:00",
        "2015-06-30 23:45:00+00:00"
    ),

    (
        "2015-07到2016-06",
        "2015-07-01 00:00:00+00:00",
        "2016-06-30 23:45:00+00:00"
    ),

]

# ============================================================
# 统计
# ============================================================

results = []

coverage_dict = {}

for name, start, end in windows:
    window = df.loc[start:end]

    # 每个站整个时间段的有效数据比例
    coverage = (
        window
        .notna()
        .mean()
    )

    coverage_dict[name] = coverage

    result = {

        "window":
            name,

        "time_steps":
            len(window),

        "active_stations":
            (coverage > 0).sum(),

        "coverage_ge_70":
            (coverage >= 0.70).sum(),

        "coverage_ge_80":
            (coverage >= 0.80).sum(),

        "coverage_ge_90":
            (coverage >= 0.90).sum(),

        "coverage_ge_95":
            (coverage >= 0.95).sum(),

        "coverage_ge_99":
            (coverage >= 0.99).sum(),

        "mean_coverage":
            coverage.mean()
    }

    results.append(result)

results_df = pd.DataFrame(results)

# ============================================================
# 打印结果
# ============================================================

print("\n======================================")
print("不同一年时间段数据质量比较")
print("======================================")

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# 按90%覆盖率排序
# ============================================================

best = results_df.sort_values(
    [
        "coverage_ge_90",
        "mean_coverage"
    ],
    ascending=False
)

print("\n======================================")
print("按全年覆盖率 >= 90% 排序")
print("======================================")

print(
    best.to_string(
        index=False
    )
)

# ============================================================
# 打印最佳时间段各站覆盖率
# ============================================================

best_name = best.iloc[0]["window"]

best_coverage = coverage_dict[
    best_name
].sort_values(
    ascending=False
)

print("\n======================================")
print("最佳一年时间段：", best_name)
print("======================================")

print("\n站点覆盖率：")

print(
    best_coverage.to_string()
)

# ============================================================
# 保存
# ============================================================

results_df.to_csv(
    "../dataset/source/processed_source/"
    "one_year_window_comparison.csv",
    index=False
)

best_coverage.to_csv(
    "../dataset/source/processed_source/"
    "best_one_year_station_coverage.csv",
    header=["coverage"]
)

print("\n结果已保存。")
