# ============================================================
# Utrecht 175-site PV dataset preprocessing
#
# 功能：
# 1. 读取 metadata.csv
# 2. 分块读取约 3GB 的 1 min 功率数据
# 3. 1 min -> 15 min
# 4. 根据 metadata 中 begin_ts / end_ts 屏蔽无效区间
# 5. 按 estimated_ac_capacity 做容量归一化
# 6. 统计各站点数据质量
# 7. 自动寻找数据质量最好的连续 90 天
# 8. 选择该时间段内有效率较高的站点
# 9. 对短时间缺失进行线性插值
# 10. 输出最终无缺失、同步的源域数据
#
# 依赖：
# pip install pandas numpy
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 一、参数设置
# ============================================================

# -----------------------------
# 1. 修改成你自己的文件位置
# -----------------------------

POWER_CSV = Path(
    r"../data/source/filtered_pv_power_measurements_ac.csv"
)

METADATA_CSV = Path(
    r"../data/source/metadata.csv"
)

OUTPUT_DIR = Path(
    r"../data/source/processed_source"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------
# 2. 大文件分块读取大小
# -----------------------------

# 如果内存较小：
# 可以改成 50_000
#
# 如果电脑内存较大：
# 可以改成 200_000

CHUNK_SIZE = 100_000

# -----------------------------
# 3. 15 min 重采样有效性要求
# -----------------------------

# 原始每 15 min 理论上有 15 个 1 min 数据点。
#
# 设置为 12 表示：
# 至少有 12 个有效分钟数据，
# 才认为这个 15 min 功率有效。
#
# 这是我们的工程复现设定，
# 并不是论文明确给出的参数。

MIN_VALID_1MIN_PER_15MIN = 12

# -----------------------------
# 4. 自动选择连续多少天
# -----------------------------

# 原论文主要使用约三个月的数据，
# 所以这里先选择 90 天。
#
# 后面也可以改：
# 60
# 120
# 180
# 365

WINDOW_DAYS = 90

# -----------------------------
# 5. 站点最低数据覆盖率
# -----------------------------

# 0.95 表示：
# 在选定的90天内，
# 至少95%的15min数据必须存在。

MIN_STATION_COVERAGE = 0.95

# -----------------------------
# 6. 最大允许插值的连续缺失长度
# -----------------------------

# 15 min × 4 = 1小时
#
# 意思：
# 连续缺失 <= 1小时，进行线性插值
# 连续缺失 > 1小时，不强行插值
#
# 原论文只说明使用线性插值，
# 没说明最大连续缺失时间。
# 所以这里属于我们的复现设定。

MAX_INTERPOLATE_GAP = 4


# ============================================================
# 二、读取 metadata
# ============================================================

def load_metadata(metadata_path):
    print("\n========================================")
    print("读取 metadata.csv")
    print("========================================")

    # Utrecht metadata 使用 ; 分隔
    metadata = pd.read_csv(
        metadata_path,
        sep=";"
    )

    # 去掉由于文件末尾多余 ; 产生的 Unnamed 列
    metadata = metadata.loc[
        :,
        ~metadata.columns.str.startswith("Unnamed")
    ]

    # 去掉完全为空的列
    metadata = metadata.dropna(
        axis=1,
        how="all"
    )

    # 时间
    metadata["begin_ts"] = pd.to_datetime(
        metadata["begin_ts"],
        utc=True,
        errors="coerce"
    )

    metadata["end_ts"] = pd.to_datetime(
        metadata["end_ts"],
        utc=True,
        errors="coerce"
    )

    # 容量
    metadata["estimated_ac_capacity"] = pd.to_numeric(
        metadata["estimated_ac_capacity"],
        errors="coerce"
    )

    metadata["estimated_dc_capacity"] = pd.to_numeric(
        metadata["estimated_dc_capacity"],
        errors="coerce"
    )

    # ID 转字符串
    metadata["ID"] = metadata["ID"].astype(str)

    print(
        "metadata 站点数：",
        len(metadata)
    )

    print("\nmetadata 前5行：")
    print(
        metadata[
            [
                "ID",
                "begin_ts",
                "end_ts",
                "estimated_ac_capacity",
                "estimated_dc_capacity"
            ]
        ].head()
    )

    # 检查容量
    invalid_capacity = metadata[
        metadata["estimated_ac_capacity"].isna()
        |
        (
                metadata["estimated_ac_capacity"]
                <= 0
        )
        ]

    if len(invalid_capacity) > 0:
        print(
            "\n警告：以下站点没有有效AC容量："
        )

        print(
            invalid_capacity["ID"].tolist()
        )

    return metadata


# ============================================================
# 三、1 min -> 15 min
# ============================================================

def resample_power_to_15min(
        csv_path,
        station_ids
):
    print("\n========================================")
    print("开始处理1 min功率数据")
    print("========================================")

    # ----------------------------
    # 检查CSV表头
    # ----------------------------

    header = pd.read_csv(
        csv_path,
        nrows=0
    )

    available_columns = set(
        header.columns
    )

    missing_stations = [
        station
        for station in station_ids
        if station not in available_columns
    ]

    if missing_stations:
        raise ValueError(
            "以下站点在功率CSV中不存在：\n"
            + str(missing_stations)
        )

    use_columns = (
            ["DateTime"]
            +
            station_ids
    )

    # 用来存储每个chunk的
    # 15min功率和、有效数据个数
    sum_parts = []
    count_parts = []

    reader = pd.read_csv(
        csv_path,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(
            reader,
            start=1
    ):
        print(
            f"正在处理 chunk {chunk_number} ..."
        )

        # ----------------------------
        # 时间格式转换
        # ----------------------------

        chunk["DateTime"] = pd.to_datetime(
            chunk["DateTime"],
            utc=True,
            errors="coerce"
        )

        # 删除时间异常数据
        chunk = chunk.dropna(
            subset=["DateTime"]
        )

        # ----------------------------
        # 功率转换成数值
        # ----------------------------

        power = chunk[
            station_ids
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # 转成 float32 节省内存
        power = power.astype(
            np.float32
        )

        # ----------------------------
        # 负功率认为异常
        # ----------------------------

        power = power.mask(
            power < 0
        )

        # ----------------------------
        # 所属15min时间窗口
        # ----------------------------

        time_15min = chunk[
            "DateTime"
        ].dt.floor(
            "15min"
        )

        # ----------------------------
        # 15min功率总和
        # ----------------------------

        part_sum = power.groupby(
            time_15min
        ).sum(
            min_count=1
        )

        # ----------------------------
        # 15min有效1min数据数量
        # ----------------------------

        part_count = (
            power.notna()
            .groupby(
                time_15min
            )
            .sum()
        )

        sum_parts.append(
            part_sum
        )

        count_parts.append(
            part_count
        )

    print("\n所有chunk读取完成。")
    print("开始合并15min数据 ...")

    # ----------------------------
    # 合并所有chunk
    # ----------------------------

    total_sum = (
        pd.concat(
            sum_parts
        )
        .groupby(level=0)
        .sum(
            min_count=1
        )
        .sort_index()
    )

    total_count = (
        pd.concat(
            count_parts
        )
        .groupby(level=0)
        .sum()
        .sort_index()
    )

    # ----------------------------
    # 求15min平均功率
    # ----------------------------

    power_15min = (
            total_sum
            /
            total_count
    )

    # 有效分钟数不足则认为该15min无效
    power_15min = power_15min.mask(
        total_count
        <
        MIN_VALID_1MIN_PER_15MIN
    )

    # ----------------------------
    # 建立严格连续15min时间索引
    # ----------------------------

    full_index = pd.date_range(
        start=power_15min.index.min(),
        end=power_15min.index.max(),
        freq="15min"
    )

    power_15min = power_15min.reindex(
        full_index
    )

    power_15min.index.name = (
        "DateTime"
    )

    power_15min = power_15min.astype(
        np.float32
    )

    print(
        "\n15min数据形状：",
        power_15min.shape
    )

    print(
        "开始时间：",
        power_15min.index.min()
    )

    print(
        "结束时间：",
        power_15min.index.max()
    )

    return power_15min


# ============================================================
# 四、根据metadata屏蔽站点非运行区间
# ============================================================

def apply_metadata_time_mask(
        power_df,
        metadata
):
    print("\n========================================")
    print("根据metadata限制每个站点有效时间")
    print("========================================")

    result = power_df.copy()

    meta = metadata.set_index(
        "ID"
    )

    for station in result.columns:

        if station not in meta.index:
            print(
                f"警告：{station} "
                f"不存在metadata"
            )

            result[station] = np.nan

            continue

        begin = meta.loc[
            station,
            "begin_ts"
        ]

        end = meta.loc[
            station,
            "end_ts"
        ]

        if (
                pd.isna(begin)
                or
                pd.isna(end)
        ):
            result[station] = np.nan

            continue

        begin = begin.floor(
            "15min"
        )

        end = end.floor(
            "15min"
        )

        # 开始时间以前无效
        result.loc[
            result.index < begin,
            station
        ] = np.nan

        # 结束时间以后无效
        result.loc[
            result.index > end,
            station
        ] = np.nan

    return result


# ============================================================
# 五、按AC容量归一化
# ============================================================

def normalize_by_ac_capacity(
        power_df,
        metadata
):
    print("\n========================================")
    print("开始容量归一化")
    print("========================================")

    meta = metadata.set_index(
        "ID"
    )

    capacities = meta[
        "estimated_ac_capacity"
    ].reindex(
        power_df.columns
    )

    # ----------------------------
    # 检查容量
    # ----------------------------

    invalid = capacities[
        capacities.isna()
        |
        (
                capacities <= 0
        )
        ]

    if len(invalid) > 0:
        raise ValueError(
            "以下站点没有有效的 "
            "estimated_ac_capacity：\n"
            +
            str(
                invalid.index.tolist()
            )
        )

    # ----------------------------
    # 容量归一化
    #
    # P_norm = P / P_rated
    # ----------------------------

    normalized = power_df.div(
        capacities,
        axis="columns"
    )

    # 负数裁成0
    # 上限不裁成1
    normalized = normalized.clip(
        lower=0
    )

    normalized = normalized.astype(
        np.float32
    )

    print(
        "容量归一化完成。"
    )

    print(
        "总体最小值：",
        normalized.min().min()
    )

    print(
        "总体最大值：",
        normalized.max().max()
    )

    return normalized


# ============================================================
# 六、统计各站点数据质量
# ============================================================

def calculate_station_quality(
        df,
        metadata
):
    print("\n========================================")
    print("统计站点数据质量")
    print("========================================")

    meta = metadata.set_index(
        "ID"
    )

    records = []

    for station in df.columns:

        series = df[
            station
        ]

        first_valid = (
            series.first_valid_index()
        )

        last_valid = (
            series.last_valid_index()
        )

        begin = meta.loc[
            station,
            "begin_ts"
        ]

        end = meta.loc[
            station,
            "end_ts"
        ]

        if (
                first_valid is None
                or
                last_valid is None
        ):
            records.append(
                {
                    "station": station,
                    "metadata_begin": begin,
                    "metadata_end": end,
                    "first_valid": None,
                    "last_valid": None,
                    "coverage_active": 0.0,
                    "missing_ratio_active": 1.0
                }
            )

            continue

        active_start = max(
            df.index.min(),
            begin.floor("15min")
        )

        active_end = min(
            df.index.max(),
            end.floor("15min")
        )

        active = series.loc[
            active_start:active_end
        ]

        coverage = (
            active.notna().mean()
            if len(active) > 0
            else 0.0
        )

        records.append(
            {
                "station": station,
                "metadata_begin": begin,
                "metadata_end": end,
                "first_valid": first_valid,
                "last_valid": last_valid,
                "coverage_active": coverage,
                "missing_ratio_active": (
                        1.0 - coverage
                )
            }
        )

    quality = pd.DataFrame(
        records
    )

    return quality


# ============================================================
# 七、寻找最佳连续时间窗口
# ============================================================

def find_best_time_window(
        df,
        window_days,
        min_station_coverage
):
    print("\n========================================")
    print(
        f"自动寻找最佳连续{window_days}天"
    )
    print("========================================")

    # ----------------------------
    # 每一天每个站点的有效率
    # ----------------------------

    daily_coverage = (
        df.notna()
        .astype(np.float32)
        .resample("1D")
        .mean()
    )

    # ----------------------------
    # 滚动计算连续N天覆盖率
    # ----------------------------

    rolling_coverage = (
        daily_coverage
        .rolling(
            window=window_days,
            min_periods=window_days
        )
        .mean()
    )

    # ----------------------------
    # 每个时间窗口有多少站点
    # 覆盖率 >= 指定阈值
    # ----------------------------

    station_counts = (
            rolling_coverage
            >= min_station_coverage
    ).sum(
        axis=1
    )

    valid_counts = station_counts.dropna()

    if len(valid_counts) == 0:
        raise ValueError(
            "找不到有效时间窗口。"
        )

    best_end_day = (
        station_counts.idxmax()
    )

    best_start = (
            best_end_day
            -
            pd.Timedelta(
                days=window_days - 1
            )
    )

    best_end = (
            best_end_day
            +
            pd.Timedelta(
                hours=23,
                minutes=45
            )
    )

    station_coverage = (
        rolling_coverage.loc[
            best_end_day
        ]
    )

    selected_stations = (
        station_coverage[
            station_coverage
            >= min_station_coverage
            ]
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    print(
        "最佳时间窗口："
    )

    print(
        "开始：",
        best_start
    )

    print(
        "结束：",
        best_end
    )

    print(
        "满足覆盖率条件的站点数：",
        len(selected_stations)
    )

    return (
        best_start,
        best_end,
        selected_stations,
        station_coverage
    )


# ============================================================
# 八、只填补较短的连续缺失
# ============================================================

def interpolate_short_gaps_series(
        series,
        max_gap
):
    """
    只对长度 <= max_gap 的连续缺失区间
    做线性插值。

    与 pandas interpolate(limit=...) 不同，
    这里不会对超长缺失区间只填前几个点。
    """

    original = series.copy()

    is_missing = original.isna()

    # 没缺失直接返回
    if not is_missing.any():
        return original

    # ----------------------------
    # 找连续缺失区间
    # ----------------------------

    group = (
            is_missing
            != is_missing.shift()
    ).cumsum()

    run_length = (
        is_missing
        .groupby(group)
        .transform("sum")
    )

    # ----------------------------
    # 先生成完整线性插值结果
    # ----------------------------

    interpolated = (
        original.interpolate(
            method="time",
            limit_area="inside"
        )
    )

    # ----------------------------
    # 只允许短缺失使用插值值
    # ----------------------------

    fillable = (
            is_missing
            &
            (
                    run_length <= max_gap
            )
    )

    result = original.copy()

    result.loc[
        fillable
    ] = interpolated.loc[
        fillable
    ]

    return result


def interpolate_short_gaps(
        df,
        max_gap
):
    print("\n========================================")
    print(
        "开始短缺失线性插值"
    )
    print("========================================")

    result = df.copy()

    for number, station in enumerate(
            result.columns,
            start=1
    ):

        result[station] = (
            interpolate_short_gaps_series(
                result[station],
                max_gap=max_gap
            )
        )

        if (
                number % 20 == 0
                or
                number == len(result.columns)
        ):
            print(
                f"已处理 "
                f"{number}/"
                f"{len(result.columns)} "
                f"个站点"
            )

    result = result.clip(
        lower=0
    )

    return result


# ============================================================
# 九、主程序
# ============================================================

def main():
    print("\n")
    print("=" * 60)
    print("Utrecht PV源域数据预处理")
    print("=" * 60)

    # ========================================================
    # Step 1：metadata
    # ========================================================

    metadata = load_metadata(
        METADATA_CSV
    )

    station_ids = (
        metadata["ID"]
        .tolist()
    )

    print(
        "\n站点数量：",
        len(station_ids)
    )

    # ========================================================
    # Step 2：1min -> 15min
    # ========================================================

    source_15min = (
        resample_power_to_15min(
            POWER_CSV,
            station_ids
        )
    )

    # 保存重采样但尚未归一化的数据
    source_15min.to_csv(
        OUTPUT_DIR
        /
        "01_source_15min_raw.csv"
    )

    print(
        "\n已保存："
        "01_source_15min_raw.csv"
    )

    # ========================================================
    # Step 3：根据metadata屏蔽无效时间
    # ========================================================

    source_15min = (
        apply_metadata_time_mask(
            source_15min,
            metadata
        )
    )

    # ========================================================
    # Step 4：容量归一化
    # ========================================================

    source_norm = (
        normalize_by_ac_capacity(
            source_15min,
            metadata
        )
    )

    source_norm.to_csv(
        OUTPUT_DIR
        /
        "02_source_15min_normalized_full.csv"
    )

    print(
        "\n已保存："
        "02_source_15min_normalized_full.csv"
    )

    # ========================================================
    # Step 5：总体站点质量
    # ========================================================

    quality = (
        calculate_station_quality(
            source_norm,
            metadata
        )
    )

    quality = quality.sort_values(
        "coverage_active",
        ascending=False
    )

    quality.to_csv(
        OUTPUT_DIR
        /
        "03_source_station_quality.csv",
        index=False
    )

    print(
        "\n已保存："
        "03_source_station_quality.csv"
    )

    print(
        "\n数据质量最好的10个站："
    )

    print(
        quality[
            [
                "station",
                "first_valid",
                "last_valid",
                "coverage_active"
            ]
        ].head(10)
    )

    # ========================================================
    # Step 6：寻找最佳90天
    # ========================================================

    (
        best_start,
        best_end,
        candidate_stations,
        station_coverage
    ) = find_best_time_window(
        source_norm,
        window_days=WINDOW_DAYS,
        min_station_coverage=MIN_STATION_COVERAGE
    )

    coverage_table = pd.DataFrame(
        {
            "station":
                station_coverage.index,

            "coverage":
                station_coverage.values
        }
    )

    coverage_table = (
        coverage_table
        .sort_values(
            "coverage",
            ascending=False
        )
    )

    coverage_table.to_csv(
        OUTPUT_DIR
        /
        "04_best_window_station_coverage.csv",
        index=False
    )

    print(
        "\n已保存："
        "04_best_window_station_coverage.csv"
    )

    # ========================================================
    # Step 7：提取候选源域
    # ========================================================

    source_candidate = (
        source_norm.loc[
            best_start:best_end,
            candidate_stations
        ]
        .copy()
    )

    print(
        "\n候选源域形状：",
        source_candidate.shape
    )

    print(
        "候选站点数：",
        len(candidate_stations)
    )

    # ========================================================
    # Step 8：短缺失线性插值
    # ========================================================

    source_interpolated = (
        interpolate_short_gaps(
            source_candidate,
            max_gap=MAX_INTERPOLATE_GAP
        )
    )

    source_interpolated.to_csv(
        OUTPUT_DIR
        /
        "05_source_candidate_after_interpolation.csv"
    )

    # ========================================================
    # Step 9：查看剩余缺失
    # ========================================================

    remaining_missing_ratio = (
        source_interpolated
        .isna()
        .mean()
    )

    missing_report = pd.DataFrame(
        {
            "station":
                remaining_missing_ratio.index,

            "remaining_missing_ratio":
                remaining_missing_ratio.values
        }
    )

    missing_report = (
        missing_report
        .sort_values(
            "remaining_missing_ratio"
        )
    )

    missing_report.to_csv(
        OUTPUT_DIR
        /
        "06_remaining_missing_report.csv",
        index=False
    )

    # ========================================================
    # Step 10：
    # 只保留插值后完全无缺失的站点
    #
    # 因为后面GIN + GRU默认不处理missing mask。
    # ========================================================

    final_stations = (
        remaining_missing_ratio[
            remaining_missing_ratio == 0
            ]
        .index
        .tolist()
    )

    source_final = (
        source_interpolated[
            final_stations
        ]
        .copy()
    )

    print("\n========================================")
    print("最终源域数据")
    print("========================================")

    print(
        "时间范围："
    )

    print(
        source_final.index.min(),
        "->",
        source_final.index.max()
    )

    print(
        "时间步数：",
        len(source_final)
    )

    print(
        "最终站点数：",
        len(final_stations)
    )

    print(
        "最终数据形状：",
        source_final.shape
    )

    print(
        "剩余NaN总数：",
        source_final.isna()
        .sum()
        .sum()
    )

    print(
        "归一化最小值：",
        source_final.min().min()
    )

    print(
        "归一化最大值：",
        source_final.max().max()
    )

    # ========================================================
    # Step 11：保存最终训练数据
    # ========================================================

    source_final.to_csv(
        OUTPUT_DIR
        /
        "07_SOURCE_FINAL_15MIN_NORMALIZED.csv"
    )

    # 保存对应metadata
    selected_metadata = (
        metadata[
            metadata["ID"].isin(
                final_stations
            )
        ]
        .copy()
    )

    selected_metadata.to_csv(
        OUTPUT_DIR
        /
        "08_SOURCE_FINAL_METADATA.csv",
        index=False
    )

    # ========================================================
    # Step 12：生成处理摘要
    # ========================================================

    summary_path = (
            OUTPUT_DIR
            /
            "09_processing_summary.txt"
    )

    with open(
            summary_path,
            "w",
            encoding="utf-8"
    ) as f:
        f.write(
            "Utrecht PV Source Dataset "
            "Preprocessing Summary\n"
        )

        f.write(
            "=" * 50
            +
            "\n"
        )

        f.write(
            f"Original stations: "
            f"{len(station_ids)}\n"
        )

        f.write(
            f"Sampling interval: "
            f"15 min\n"
        )

        f.write(
            f"Selected window: "
            f"{best_start} "
            f"to "
            f"{best_end}\n"
        )

        f.write(
            f"Window days: "
            f"{WINDOW_DAYS}\n"
        )

        f.write(
            f"Minimum station coverage: "
            f"{MIN_STATION_COVERAGE}\n"
        )

        f.write(
            f"Candidate stations: "
            f"{len(candidate_stations)}\n"
        )

        f.write(
            f"Final stations: "
            f"{len(final_stations)}\n"
        )

        f.write(
            f"Final time steps: "
            f"{len(source_final)}\n"
        )

        f.write(
            f"Final shape: "
            f"{source_final.shape}\n"
        )

        f.write(
            f"Remaining NaNs: "
            f"{source_final.isna().sum().sum()}\n"
        )

        f.write(
            f"Maximum interpolation gap: "
            f"{MAX_INTERPOLATE_GAP} "
            f"x15min\n"
        )

        f.write(
            "Normalization: "
            "P / estimated_ac_capacity\n"
        )

    print(
        "\n最终文件已经保存："
    )

    print(
        OUTPUT_DIR
        /
        "07_SOURCE_FINAL_15MIN_NORMALIZED.csv"
    )

    print(
        "\n对应metadata："
    )

    print(
        OUTPUT_DIR
        /
        "08_SOURCE_FINAL_METADATA.csv"
    )

    print(
        "\n处理摘要："
    )

    print(
        OUTPUT_DIR
        /
        "09_processing_summary.txt"
    )

    print("\n")
    print("=" * 60)
    print("源域数据预处理完成")
    print("=" * 60)


# ============================================================
# 十、运行
# ============================================================

if __name__ == "__main__":
    main()
