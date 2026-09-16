from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config

# ============================================================
# 1. 将邻接矩阵转换成 NetworkX 无向图
# ============================================================

def build_networkx_graph(station_names, adj):
    """按节点顺序把对称邻接矩阵转换成无自环的无向图。"""
    n = len(station_names)

    if len(set(station_names)) != n:
        raise ValueError("节点名称存在重复。")
    if adj.shape != (n, n):
        raise ValueError(f"邻接矩阵尺寸 {adj.shape} 与节点数量 {n} 不一致。")
    if not np.allclose(adj, adj.T):
        raise ValueError("邻接矩阵不是对称矩阵。")

    graph = nx.Graph()
    graph.add_nodes_from(station_names)

    # 保留原始邻接矩阵；随机游走用图不需要自环。
    adj_no_self = adj.copy()
    np.fill_diagonal(adj_no_self, 0)

    # 无向矩阵的上下三角代表同一批边，所以只读取上三角。
    row_indices, col_indices = np.where(np.triu(adj_no_self, k=1) > 0)
    for i, j in zip(row_indices, col_indices):
        graph.add_edge(station_names[i], station_names[j])

    return graph


# ============================================================
# 2. 读取实际数据并检查转换结果
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    if not station_order_file.is_file():
        raise FileNotFoundError(f"找不到节点顺序文件：{station_order_file}")
    if not adjacency_file.is_file():
        raise FileNotFoundError(f"找不到邻接矩阵文件：{adjacency_file}")

    station_df = pd.read_csv(station_order_file)
    station_names = station_df["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)

    graph = build_networkx_graph(station_names, adj)
    matrix_edge_count = int(np.count_nonzero(np.triu(adj, k=1) > 0))

    if graph.number_of_edges() != matrix_edge_count:
        raise RuntimeError("NetworkX 图的边数与邻接矩阵不一致。")

    print("节点数量：", graph.number_of_nodes())
    print("矩阵无向边数量：", matrix_edge_count)
    print("无向边数量：", graph.number_of_edges())
    print("自环数量：", nx.number_of_selfloops(graph))
    print("\n前5个节点的邻居：")

    for station_name in station_names[:5]:
        print(station_name, "->", list(graph.neighbors(station_name)))


if __name__ == "__main__":
    main()
