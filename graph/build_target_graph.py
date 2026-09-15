from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径设置
#
# Target Domain:
# UNISOLAR, Australia
#
# 输入：
# prepare_target_dataset.py
# 生成的每个Fold训练时间序列
#
# 注意：
# 这里只使用Train，
# 不允许使用Validation数据构图。
# ============================================================

DATASET_DIR = Path(
    "../data/target/processed_target_final/"
    "model_dataset"
)

OUTPUT_DIR = Path(
    "../data/target/processed_target_final/"
    "graph"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 构图参数
#
# 每个节点主动选择Pearson相关性最高的5个节点。
#
# TOP_K = 5
#
# 这是我们的复现设置，
# 不是原论文明确给出的参数。
# ============================================================

TOP_K = 5

# ============================================================
# 3. 是否删除所有站同时为0的时间点
#
# True：
# 如果某一个15min时刻31个目标站全部为0，
# 则该时间点不参与Pearson相关性计算。
#
# 主要用于排除纯夜间时段。
#
# 注意：
# 这里只影响Graph计算。
#
# train.npz中的夜间数据不会删除，
# 模型训练依然使用完整昼夜数据。
# ============================================================

REMOVE_ALL_ZERO_TIMESTAMPS = True

# 判断“全0”的数值容差
ZERO_EPS = 1e-8

# ============================================================
# 4. 是否只允许正相关边
#
# True：
# 只从Pearson > 0的节点中选择Top-K。
#
# 对光伏站来说，
# 第一版我们主要考虑正相关空间关系。
# ============================================================

POSITIVE_CORRELATION_ONLY = True

# ============================================================
# 5. 是否添加自环
#
# A_ii = 1
# ============================================================

ADD_SELF_LOOPS = True

# ============================================================
# 6. 需要处理的Fold
# ============================================================

FOLDS = [
    1,
    2,
    3,
    4
]

# ============================================================
# 7. 读取目标域节点顺序
#
# 这个顺序必须和：
#
# train.npz
# val.npz
# Target Graph
# GIN
# GRU
#
# 完全一致。
# ============================================================

STATION_ORDER_FILE = (
        DATASET_DIR
        /
        "TARGET_STATION_ORDER.csv"
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "读取目标域节点顺序"
)

print(
    "=" * 75
)

station_order_df = pd.read_csv(
    STATION_ORDER_FILE
)

station_names = (
    station_order_df[
        "NodeID"
    ]
    .astype(str)
    .tolist()
)

N = len(
    station_names
)

print(
    "节点数量：",
    N
)

print(
    "节点列表："
)

print(
    station_names
)

# ============================================================
# 8. 检查节点名称是否重复
# ============================================================

duplicate_node_count = int(
    pd.Index(
        station_names
    )
    .duplicated()
    .sum()
)

print(
    "重复节点名数量：",
    duplicate_node_count
)

if duplicate_node_count > 0:
    raise RuntimeError(
        "TARGET_STATION_ORDER.csv中存在重复节点名称。"
    )


# ============================================================
# 9. Top-K构图函数
#
# 输入：
#
# correlation_df
# NxN Pearson矩阵
#
# 输出：
#
# weighted_adj
# binary_adj
# directed_neighbor_records
# ============================================================

def build_topk_graph(
        correlation_df,
        top_k,
        positive_only=True,
        add_self_loops=True
):
    node_names = (
        correlation_df
        .columns
        .astype(str)
        .tolist()
    )

    n = len(
        node_names
    )

    if top_k >= n:
        raise ValueError(
            f"TOP_K={top_k} 必须小于节点数 N={n}"
        )

    # ========================================================
    # 有向Top-K邻接矩阵
    #
    # 每个节点先主动选自己的Top-K邻居。
    # ========================================================

    directed_weighted = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float32
    )

    directed_neighbor_records = []

    for i in range(
            n
    ):

        source_name = (
            node_names[
                i
            ]
        )

        correlations = (
            correlation_df
            .iloc[
                i
            ]
            .copy()
        )

        # ----------------------------------------------------
        # 排除节点自己
        # ----------------------------------------------------

        correlations.iloc[
            i
        ] = np.nan

        # ----------------------------------------------------
        # 去掉NaN
        # ----------------------------------------------------

        correlations = (
            correlations
            .dropna()
        )

        # ----------------------------------------------------
        # 只允许正相关
        # ----------------------------------------------------

        if positive_only:
            correlations = (
                correlations[
                    correlations
                    >
                    0
                    ]
            )

        # ----------------------------------------------------
        # 从大到小排序
        # ----------------------------------------------------

        correlations = (
            correlations
            .sort_values(
                ascending=False
            )
        )

        current_k = min(
            top_k,
            len(
                correlations
            )
        )

        if current_k == 0:
            raise RuntimeError(
                f"节点 {source_name} 没有可用正相关邻居。"
            )

        top_neighbors = (
            correlations
            .head(
                current_k
            )
        )

        # ----------------------------------------------------
        # 保存有向Top-K关系
        # ----------------------------------------------------

        for rank, (
                neighbor_name,
                corr_value
        ) in enumerate(
            top_neighbors.items(),
            start=1
        ):
            j = (
                node_names
                .index(
                    str(
                        neighbor_name
                    )
                )
            )

            directed_weighted[
                i,
                j
            ] = float(
                corr_value
            )

            directed_neighbor_records.append(
                {

                    "source_index":
                        i,

                    "source":
                        source_name,

                    "rank":
                        rank,

                    "target_index":
                        j,

                    "target":
                        str(
                            neighbor_name
                        ),

                    "correlation":
                        float(
                            corr_value
                        )
                }
            )

    # ========================================================
    # 10. 无向化
    #
    # 如果：
    #
    # i -> j
    #
    # 或
    #
    # j -> i
    #
    # 任意一个存在，
    # 最终建立：
    #
    # i <-> j
    #
    # 权重取两个方向最大值。
    # ========================================================

    weighted_adj = np.maximum(
        directed_weighted,
        directed_weighted.T
    )

    # ========================================================
    # 11. Binary adjacency
    # ========================================================

    binary_adj = (
            weighted_adj
            >
            0
    ).astype(
        np.float32
    )

    # ========================================================
    # 12. 添加自环
    # ========================================================

    if add_self_loops:
        np.fill_diagonal(
            weighted_adj,
            1.0
        )

        np.fill_diagonal(
            binary_adj,
            1.0
        )

    return (
        weighted_adj,
        binary_adj,
        directed_neighbor_records
    )


# ============================================================
# 13. Graph汇总
# ============================================================

graph_summary_records = []

# ============================================================
# 14. 逐Fold构建Target Graph
# ============================================================

for fold_id in FOLDS:

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        f"构建 Target Graph - Fold {fold_id}"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # 15. 当前Fold输入文件
    #
    # 直接使用Q99归一化后的Train。
    # ========================================================

    INPUT_FILE = (
            DATASET_DIR
            /
            f"fold_{fold_id}"
            /
            "train_timeseries_normalized.csv"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}"
        )

    print(
        "训练时间序列："
    )

    print(
        INPUT_FILE
    )

    # ========================================================
    # 16. 读取当前Fold训练数据
    # ========================================================

    train_df = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True
    )

    train_df.index.name = (
        "Timestamp"
    )

    # 如果存在时区，去掉
    if isinstance(
            train_df.index,
            pd.DatetimeIndex
    ):

        if train_df.index.tz is not None:
            train_df.index = (
                train_df.index
                .tz_localize(
                    None
                )
            )

    train_df = (
        train_df
        .sort_index()
    )

    print(
        "\n原始Train形状：",
        train_df.shape
    )

    print(
        "时间范围：",
        train_df.index.min(),
        "->",
        train_df.index.max()
    )

    # ========================================================
    # 17. 节点数量检查
    # ========================================================

    if train_df.shape[1] != N:
        raise RuntimeError(
            f"Fold {fold_id} 节点数量异常："
            f"{train_df.shape[1]} != {N}"
        )

    # ========================================================
    # 18. 节点顺序严格检查
    # ========================================================

    current_station_names = (
        train_df
        .columns
        .astype(str)
        .tolist()
    )

    if (
            current_station_names
            !=
            station_names
    ):
        print(
            "\n标准节点顺序："
        )

        print(
            station_names
        )

        print(
            "\n当前Fold节点顺序："
        )

        print(
            current_station_names
        )

        raise RuntimeError(
            f"Fold {fold_id} 节点顺序与"
            "TARGET_STATION_ORDER.csv不一致。"
        )

    print(
        "节点顺序检查：通过"
    )

    # ========================================================
    # 19. NaN / Inf检查
    # ========================================================

    nan_count = int(
        train_df
        .isna()
        .sum()
        .sum()
    )

    inf_count = int(
        np.isinf(
            train_df
            .to_numpy(
                dtype=np.float64
            )
        )
        .sum()
    )

    print(
        "NaN数量：",
        nan_count
    )

    print(
        "Inf数量：",
        inf_count
    )

    if (
            nan_count > 0
            or
            inf_count > 0
    ):
        raise RuntimeError(
            f"Fold {fold_id} Train存在NaN或Inf。"
        )

    # ========================================================
    # 20. 删除31个站全部为0的时间点
    #
    # 主要用于删除纯夜间时段。
    #
    # 只用于相关性构图。
    #
    # 模型训练数据本身不会改变。
    # ========================================================

    graph_train_df = (
        train_df
        .copy()
    )

    original_steps = len(
        graph_train_df
    )

    if REMOVE_ALL_ZERO_TIMESTAMPS:
        active_mask = (
                graph_train_df
                .abs()
                .max(
                    axis=1
                )
                >
                ZERO_EPS
        )

        graph_train_df = (
            graph_train_df.loc[
                active_mask
            ]
            .copy()
        )

    retained_steps = len(
        graph_train_df
    )

    removed_steps = (
            original_steps
            -
            retained_steps
    )

    print(
        "\n构图前删除全0时间点："
    )

    print(
        "原始时间点数：",
        original_steps
    )

    print(
        "保留时间点数：",
        retained_steps
    )

    print(
        "删除全0时间点数：",
        removed_steps
    )

    print(
        "保留比例：",
        retained_steps
        /
        original_steps
    )

    # ========================================================
    # 21. 检查剩余时间长度
    #
    # 如果数据太少，
    # Pearson会非常不稳定。
    # ========================================================

    if retained_steps < 30:
        raise RuntimeError(
            f"Fold {fold_id} 删除全0时间后"
            f"只剩{retained_steps}个时间点，"
            "不足以稳定计算Pearson相关性。"
        )

    # ========================================================
    # 22. 检查每个节点方差
    #
    # Pearson要求变量不能完全是常数。
    # ========================================================

    node_std = (
        graph_train_df
        .std(
            axis=0
        )
    )

    constant_nodes = (
        node_std[
            node_std
            <=
            ZERO_EPS
            ]
        .index
        .tolist()
    )

    if len(
            constant_nodes
    ) > 0:
        print(
            "\n训练期几乎不变化的节点："
        )

        print(
            constant_nodes
        )

        raise RuntimeError(
            f"Fold {fold_id} 存在近似常数节点，"
            "无法可靠计算Pearson相关性。"
        )

    # ========================================================
    # 23. Pearson相关矩阵
    #
    # 输出：
    #
    # 31 × 31
    # ========================================================

    correlation_df = (
        graph_train_df
        .corr(
            method="pearson"
        )
    )

    print(
        "\nCorrelation矩阵形状：",
        correlation_df.shape
    )

    # ========================================================
    # 24. 相关矩阵节点顺序检查
    # ========================================================

    if (
            correlation_df.columns.tolist()
            !=
            station_names
    ):
        raise RuntimeError(
            f"Fold {fold_id} Correlation节点顺序异常。"
        )

    # ========================================================
    # 25. Correlation NaN检查
    # ========================================================

    correlation_nan = int(
        correlation_df
        .isna()
        .sum()
        .sum()
    )

    print(
        "Correlation NaN数量：",
        correlation_nan
    )

    if correlation_nan > 0:
        bad_nodes = (
            correlation_df
            .columns[
                correlation_df
            .isna()
            .any(
                    axis=0
                )
            ]
            .tolist()
        )

        print(
            "Correlation出现NaN的节点："
        )

        print(
            bad_nodes
        )

        raise RuntimeError(
            f"Fold {fold_id} Pearson矩阵存在NaN。"
        )

    # ========================================================
    # 26. 检查相关矩阵对称性
    # ========================================================

    corr_array = (
        correlation_df
        .to_numpy(
            dtype=np.float64
        )
    )

    symmetric_error = float(
        np.max(
            np.abs(
                corr_array
                -
                corr_array.T
            )
        )
    )

    print(
        "Correlation最大对称误差：",
        symmetric_error
    )

    # ========================================================
    # 27. 非对角Pearson统计
    # ========================================================

    off_diagonal_mask = (
        ~np.eye(
            N,
            dtype=bool
        )
    )

    off_diagonal_corr = (
        corr_array[
            off_diagonal_mask
        ]
    )

    print(
        "\n站点间Pearson相关性统计："
    )

    print(
        "最小值：",
        float(
            np.min(
                off_diagonal_corr
            )
        )
    )

    print(
        "最大值：",
        float(
            np.max(
                off_diagonal_corr
            )
        )
    )

    print(
        "平均值：",
        float(
            np.mean(
                off_diagonal_corr
            )
        )
    )

    print(
        "中位数：",
        float(
            np.median(
                off_diagonal_corr
            )
        )
    )

    print(
        "标准差：",
        float(
            np.std(
                off_diagonal_corr
            )
        )
    )

    # ========================================================
    # 28. 正相关/负相关统计
    # ========================================================

    positive_count = int(
        np.sum(
            off_diagonal_corr
            >
            0
        )
    )

    negative_count = int(
        np.sum(
            off_diagonal_corr
            <
            0
        )
    )

    zero_count = int(
        np.sum(
            off_diagonal_corr
            ==
            0
        )
    )

    print(
        "\n正相关元素数量：",
        positive_count
    )

    print(
        "负相关元素数量：",
        negative_count
    )

    print(
        "零相关元素数量：",
        zero_count
    )

    # ========================================================
    # 29. Top-K构图
    # ========================================================

    (
        weighted_adj,
        binary_adj,
        directed_neighbor_records

    ) = build_topk_graph(

        correlation_df=

        correlation_df,

        top_k=

        TOP_K,

        positive_only=

        POSITIVE_CORRELATION_ONLY,

        add_self_loops=

        ADD_SELF_LOOPS
    )

    # ========================================================
    # 30. 去掉自环后统计图结构
    # ========================================================

    binary_without_self = (
        binary_adj
        .copy()
    )

    np.fill_diagonal(
        binary_without_self,
        0
    )

    weighted_without_self = (
        weighted_adj
        .copy()
    )

    np.fill_diagonal(
        weighted_without_self,
        0
    )

    # ========================================================
    # 31. Degree
    # ========================================================

    degree = (
        binary_without_self
        .sum(
            axis=1
        )
    )

    # ========================================================
    # 32. 无向边数量
    #
    # 每条边在矩阵中出现两次。
    # ========================================================

    edge_count = int(
        binary_without_self
        .sum()
        /
        2
    )

    print(
        "\n图结构统计："
    )

    print(
        "节点数量：",
        N
    )

    print(
        "无向边数量（不含自环）：",
        edge_count
    )

    print(
        "最小Degree：",
        int(
            degree.min()
        )
    )

    print(
        "最大Degree：",
        int(
            degree.max()
        )
    )

    print(
        "平均Degree：",
        float(
            degree.mean()
        )
    )

    # ========================================================
    # 33. Graph Density
    # ========================================================

    max_possible_edges = (
            N
            *
            (
                    N - 1
            )
            /
            2
    )

    density = (
            edge_count
            /
            max_possible_edges
    )

    print(
        "Graph Density：",
        density
    )

    # ========================================================
    # 34. 最终图边权重统计
    # ========================================================

    edge_weights = (
        weighted_without_self[
            weighted_without_self
            >
            0
            ]
    )

    if len(
            edge_weights
    ) > 0:
        print(
            "\n最终图边权重统计："
        )

        print(
            "最小边权重：",
            float(
                edge_weights.min()
            )
        )

        print(
            "最大边权重：",
            float(
                edge_weights.max()
            )
        )

        print(
            "平均边权重：",
            float(
                edge_weights.mean()
            )
        )

    # ========================================================
    # 35. Fold输出目录
    # ========================================================

    fold_output_dir = (
            OUTPUT_DIR
            /
            f"fold_{fold_id}"
    )

    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 36. 保存完整Pearson矩阵
    # ========================================================

    CORRELATION_FILE = (
            fold_output_dir
            /
            "correlation_matrix.csv"
    )

    correlation_df.to_csv(
        CORRELATION_FILE
    )

    # ========================================================
    # 37. 保存Weighted adjacency
    # ========================================================

    WEIGHTED_NPY_FILE = (
            fold_output_dir
            /
            "adjacency_weighted.npy"
    )

    np.save(
        WEIGHTED_NPY_FILE,
        weighted_adj
    )

    weighted_adj_df = pd.DataFrame(
        weighted_adj,
        index=station_names,
        columns=station_names
    )

    weighted_adj_df.to_csv(
        fold_output_dir
        /
        "adjacency_weighted.csv"
    )

    # ========================================================
    # 38. 保存Binary adjacency
    # ========================================================

    BINARY_NPY_FILE = (
            fold_output_dir
            /
            "adjacency_binary.npy"
    )

    np.save(
        BINARY_NPY_FILE,
        binary_adj
    )

    binary_adj_df = pd.DataFrame(
        binary_adj.astype(
            int
        ),
        index=station_names,
        columns=station_names
    )

    binary_adj_df.to_csv(
        fold_output_dir
        /
        "adjacency_binary.csv"
    )

    # ========================================================
    # 39. 保存每个节点“主动选择的Top-K”
    #
    # 这个文件非常方便后面检查：
    #
    # 某个节点到底选了哪些邻居。
    # ========================================================

    directed_neighbor_df = pd.DataFrame(
        directed_neighbor_records
    )

    directed_neighbor_df.to_csv(
        fold_output_dir
        /
        "topk_directed_neighbors.csv",
        index=False
    )

    # ========================================================
    # 40. 保存最终无向Edge List
    # ========================================================

    edge_records = []

    for i in range(
            N
    ):

        for j in range(
                i + 1,
                N
        ):

            if (
                    binary_without_self[
                        i,
                        j
                    ]
                    >
                    0
            ):
                edge_records.append(
                    {

                        "source_index":
                            i,

                        "source":
                            station_names[
                                i
                            ],

                        "target_index":
                            j,

                        "target":
                            station_names[
                                j
                            ],

                        "correlation":
                            float(
                                weighted_adj[
                                    i,
                                    j
                                ]
                            )
                    }
                )

    edge_df = pd.DataFrame(
        edge_records,
        columns=[
            "source_index",
            "source",
            "target_index",
            "target",
            "correlation"
        ]
    )

    edge_df.to_csv(
        fold_output_dir
        /
        "edge_list.csv",
        index=False
    )

    # ========================================================
    # 41. 保存Node Degree
    # ========================================================

    degree_df = pd.DataFrame(
        {

            "node_index":
                np.arange(
                    N
                ),

            "NodeID":
                station_names,

            "degree":
                degree.astype(
                    int
                )
        }
    )

    degree_df.to_csv(
        fold_output_dir
        /
        "node_degree.csv",
        index=False
    )

    # ========================================================
    # 42. 保存实际用于构图的时间序列
    #
    # 即：
    #
    # Train normalized
    # 去除全0时间点以后
    #
    # 后面如果发现图异常，
    # 可以直接检查这个文件。
    # ========================================================

    graph_train_df.to_csv(
        fold_output_dir
        /
        "graph_input_timeseries.csv"
    )

    # ========================================================
    # 43. 保存Graph信息
    # ========================================================

    graph_info_df = pd.DataFrame(
        [
            {

                "fold":
                    fold_id,

                "node_count":
                    N,

                "top_k":
                    TOP_K,

                "edge_count_no_self_loop":
                    edge_count,

                "min_degree":
                    int(
                        degree.min()
                    ),

                "max_degree":
                    int(
                        degree.max()
                    ),

                "mean_degree":
                    float(
                        degree.mean()
                    ),

                "graph_density":
                    density,

                "train_time_steps_original":
                    original_steps,

                "train_time_steps_for_graph":
                    retained_steps,

                "removed_all_zero_steps":
                    removed_steps,

                "retained_ratio":
                    retained_steps
                    /
                    original_steps,

                "remove_all_zero_timestamps":
                    REMOVE_ALL_ZERO_TIMESTAMPS,

                "positive_correlation_only":
                    POSITIVE_CORRELATION_ONLY,

                "self_loop":
                    ADD_SELF_LOOPS,

                "correlation_method":
                    "pearson",

                "correlation_min":
                    float(
                        np.min(
                            off_diagonal_corr
                        )
                    ),

                "correlation_max":
                    float(
                        np.max(
                            off_diagonal_corr
                        )
                    ),

                "correlation_mean":
                    float(
                        np.mean(
                            off_diagonal_corr
                        )
                    ),

                "correlation_median":
                    float(
                        np.median(
                            off_diagonal_corr
                        )
                    )
            }
        ]
    )

    graph_info_df.to_csv(
        fold_output_dir
        /
        "graph_info.csv",
        index=False
    )

    # ========================================================
    # 44. Fold汇总
    # ========================================================

    graph_summary_records.append(
        {

            "fold":
                fold_id,

            "nodes":
                N,

            "edges":
                edge_count,

            "min_degree":
                int(
                    degree.min()
                ),

            "max_degree":
                int(
                    degree.max()
                ),

            "mean_degree":
                float(
                    degree.mean()
                ),

            "density":
                density,

            "original_train_steps":
                original_steps,

            "graph_time_steps":
                retained_steps,

            "removed_zero_steps":
                removed_steps,

            "corr_min":
                float(
                    np.min(
                        off_diagonal_corr
                    )
                ),

            "corr_max":
                float(
                    np.max(
                        off_diagonal_corr
                    )
                ),

            "corr_mean":
                float(
                    np.mean(
                        off_diagonal_corr
                    )
                ),

            "corr_median":
                float(
                    np.median(
                        off_diagonal_corr
                    )
                )
        }
    )

    print(
        "\nFold图文件已保存："
    )

    print(
        fold_output_dir
    )

# ============================================================
# 45. 保存Target Graph总体汇总
# ============================================================

graph_summary_df = pd.DataFrame(
    graph_summary_records
)

GRAPH_SUMMARY_FILE = (
        OUTPUT_DIR
        /
        "TARGET_GRAPH_SUMMARY.csv"
)

graph_summary_df.to_csv(
    GRAPH_SUMMARY_FILE,
    index=False
)

# ============================================================
# 46. 最终输出
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "目标域Graph构建完成"
)

print(
    "=" * 75
)

print(
    "\nGraph汇总："
)

print(
    graph_summary_df
    .to_string(
        index=False
    )
)

print(
    "\n输出目录："
)

print(
    OUTPUT_DIR
)

print(
    "\n文件结构："
)

print(
    """
graph/
│
├── TARGET_GRAPH_SUMMARY.csv
│
├── fold_1/
│   ├── correlation_matrix.csv
│   ├── adjacency_weighted.npy
│   ├── adjacency_weighted.csv
│   ├── adjacency_binary.npy
│   ├── adjacency_binary.csv
│   ├── topk_directed_neighbors.csv
│   ├── edge_list.csv
│   ├── node_degree.csv
│   ├── graph_input_timeseries.csv
│   └── graph_info.csv
│
├── fold_2/
│   └── ...
│
├── fold_3/
│   └── ...
│
└── fold_4/
    └── ...
"""
)
