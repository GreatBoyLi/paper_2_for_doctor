from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 基础路径
#
# Source Domain:
# Utrecht, Netherlands
#
# 输入：
# prepare_source_dataset.py生成的各Fold训练时间序列
# ============================================================

DATASET_DIR = Path(
    "../data/source/processed_source/"
    "source_2014/"
    "model_dataset"
)

OUTPUT_DIR = Path(
    "../data/source/processed_source/"
    "source_2014/"
    "graph"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. 构图参数
#
# 每个节点选Pearson相关性最高的K个邻居
#
# 注意：
# TOP_K = 5 是我们的复现设置，
# 不是原论文明确给出的参数。
# ============================================================

TOP_K = 5

# ============================================================
# 3. 是否排除全站均为0的时间点
#
# True:
# 如果某个时间点104个站全部为0，
# 不参与相关系数计算。
#
# 主要是排除纯夜间时段，
# 防止大量共同0值人为提高相关性。
# ============================================================

REMOVE_ALL_ZERO_TIMESTAMPS = True

# 判断是否为0的小阈值
ZERO_EPS = 1e-8

# ============================================================
# 4. 是否只允许正相关边
#
# 光伏站之间通常希望建立正相关连接。
#
# True:
# 负相关和0相关不进入候选邻居。
#
# False:
# 直接按相关系数从大到小选择。
# ============================================================

POSITIVE_CORRELATION_ONLY = True

# ============================================================
# 5. 是否添加自环
#
# GNN通常需要保留节点自己的信息。
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
# 7. 读取统一节点顺序
#
# 图矩阵的行列顺序必须与：
#
# train.npz
# val.npz
#
# 中station_names完全一致。
# ============================================================

STATION_ORDER_FILE = (
        DATASET_DIR
        /
        "SOURCE_STATION_ORDER.csv"
)

print(
    "\n"
    +
    "=" * 75
)

print(
    "读取源域节点顺序"
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
    "前10个节点：",
    station_names[:10]
)


# ============================================================
# 8. 工具函数：
# 构建Top-K图
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
        .tolist()
    )

    n = len(
        node_names
    )

    # --------------------------------------------------------
    # directed_weighted
    #
    # 先创建有向Top-K图。
    #
    # 第i个节点选择自己的K个最相关邻居。
    # --------------------------------------------------------

    directed_weighted = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float32
    )

    for i in range(
            n
    ):

        correlations = (
            correlation_df
            .iloc[
                i
            ]
            .copy()
        )

        # 自己不能作为自己的Top-K邻居
        correlations.iloc[
            i
        ] = np.nan

        # 只保留正相关
        if positive_only:
            correlations = (
                correlations[
                    correlations
                    >
                    0
                    ]
            )

        correlations = (
            correlations
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        # 当前节点实际可用邻居数量
        current_k = min(
            top_k,
            len(
                correlations
            )
        )

        top_neighbors = (
            correlations
            .head(
                current_k
            )
        )

        for neighbor_name, corr_value in top_neighbors.items():
            j = (
                node_names
                .index(
                    neighbor_name
                )
            )

            directed_weighted[
                i,
                j
            ] = float(
                corr_value
            )

    # ========================================================
    # 9. 对称化
    #
    # 如果：
    #
    # i选择了j
    #
    # 或者：
    #
    # j选择了i
    #
    # 那么最终都建立无向边。
    #
    # 权重取两者较大值。
    # ========================================================

    weighted_adj = np.maximum(
        directed_weighted,
        directed_weighted.T
    )

    # ========================================================
    # 10. Binary邻接矩阵
    # ========================================================

    binary_adj = (
            weighted_adj
            >
            0
    ).astype(
        np.float32
    )

    # ========================================================
    # 11. 添加自环
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
        binary_adj
    )


# ============================================================
# 12. 逐Fold构图
# ============================================================

graph_summary_records = []

for fold_id in FOLDS:

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        f"构建 Source Graph - Fold {fold_id}"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # 13. 输入文件
    # ========================================================

    INPUT_FILE = (
            DATASET_DIR
            /
            f"fold_{fold_id}"
            /
            "train_timeseries.csv"
    )

    print(
        "训练时间序列："
    )

    print(
        INPUT_FILE
    )

    # ========================================================
    # 14. 读取训练数据
    # ========================================================

    train_df = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True
    )

    train_df.index.name = (
        "Timestamp"
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
    # 15. 节点顺序严格检查
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
        raise RuntimeError(
            f"\nFold {fold_id} 节点顺序与"
            "SOURCE_STATION_ORDER.csv不一致。"
        )

    print(
        "节点顺序检查：通过"
    )

    # ========================================================
    # 16. NaN / Inf检查
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
    # 17. 删除所有节点同时为0的时间点
    #
    # 这些主要是纯夜间。
    #
    # 注意：
    # 这里只用于计算Graph。
    #
    # 不会修改train.npz，
    # 模型训练依然使用完整昼夜数据。
    # ========================================================

    graph_train_df = (
        train_df
        .copy()
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

        original_steps = len(
            graph_train_df
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

    else:

        original_steps = len(
            graph_train_df
        )

        retained_steps = (
            original_steps
        )

        removed_steps = 0

    # ========================================================
    # 18. 计算Pearson相关矩阵
    #
    # shape:
    #
    # [104, 104]
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
    # 19. 检查相关矩阵
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
            "相关矩阵出现NaN的节点："
        )

        print(
            bad_nodes
        )

        raise RuntimeError(
            f"Fold {fold_id} 相关矩阵存在NaN，"
            "可能存在整个训练期功率不变化的节点。"
        )

    # ========================================================
    # 20. 输出非对角线相关系数统计
    # ========================================================

    corr_values = (
        correlation_df
        .to_numpy(
            dtype=np.float64
        )
    )

    off_diagonal_mask = (
        ~np.eye(
            N,
            dtype=bool
        )
    )

    off_diagonal_corr = (
        corr_values[
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

    # ========================================================
    # 21. Top-K构图
    # ========================================================

    (
        weighted_adj,
        binary_adj
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
    # 22. 图统计
    # ========================================================

    # 去掉自环后统计实际无向边
    binary_without_self = (
        binary_adj
        .copy()
    )

    np.fill_diagonal(
        binary_without_self,
        0
    )

    degree = (
        binary_without_self
        .sum(
            axis=1
        )
    )

    # 因为是无向图
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
    # 23. 图密度
    #
    # 无向简单图最大边数：
    #
    # N(N-1)/2
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
    # 24. 创建Fold输出目录
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
    # 25. 保存完整Correlation矩阵
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
    # 26. 保存Weighted邻接矩阵
    #
    # 元素：
    #
    # A_ij = Pearson相关系数
    #
    # 无边：
    #
    # A_ij = 0
    # ========================================================

    WEIGHTED_ADJ_FILE = (
            fold_output_dir
            /
            "adjacency_weighted.npy"
    )

    np.save(
        WEIGHTED_ADJ_FILE,
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
    # 27. 保存Binary邻接矩阵
    #
    # 有边：
    #
    # 1
    #
    # 无边：
    #
    # 0
    # ========================================================

    BINARY_ADJ_FILE = (
            fold_output_dir
            /
            "adjacency_binary.npy"
    )

    np.save(
        BINARY_ADJ_FILE,
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
    # 28. 保存Edge List
    #
    # 每一条无向边只保存一次：
    #
    # source
    # target
    # correlation
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
        edge_records
    )

    EDGE_FILE = (
            fold_output_dir
            /
            "edge_list.csv"
    )

    edge_df.to_csv(
        EDGE_FILE,
        index=False
    )

    # ========================================================
    # 29. 保存Degree
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
    # 30. 保存Graph基本信息
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

                "remove_all_zero_timestamps":
                    REMOVE_ALL_ZERO_TIMESTAMPS,

                "positive_correlation_only":
                    POSITIVE_CORRELATION_ONLY,

                "self_loop":
                    ADD_SELF_LOOPS,

                "correlation_method":
                    "pearson"
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
    # 31. 汇总
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

            "graph_time_steps":
                retained_steps
        }
    )

    print(
        "\nFold图文件已保存："
    )

    print(
        fold_output_dir
    )

# ============================================================
# 32. 保存所有Fold图汇总
# ============================================================

graph_summary_df = pd.DataFrame(
    graph_summary_records
)

GRAPH_SUMMARY_FILE = (
        OUTPUT_DIR
        /
        "SOURCE_GRAPH_SUMMARY.csv"
)

graph_summary_df.to_csv(
    GRAPH_SUMMARY_FILE,
    index=False
)

# ============================================================
# 33. 最终输出
# ============================================================

print(
    "\n"
    +
    "=" * 75
)

print(
    "源域Graph构建完成"
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
├── SOURCE_GRAPH_SUMMARY.csv
│
├── fold_1/
│   ├── correlation_matrix.csv
│   ├── adjacency_weighted.npy
│   ├── adjacency_weighted.csv
│   ├── adjacency_binary.npy
│   ├── adjacency_binary.csv
│   ├── edge_list.csv
│   ├── node_degree.csv
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
