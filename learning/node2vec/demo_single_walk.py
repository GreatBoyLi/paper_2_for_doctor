import random

import numpy as np
import pandas as pd

import build_networkx_graph as graph_builder


# ============================================================
# 1. 本次只观察 Fold 1 中的一条游走
#
# walk_length 包括起点，因此 20 个节点对应 19 次移动。
# p = q = 1 时，候选邻居的权重相同，可以等概率选择。
# ============================================================

START_NODE = "ID089"
WALK_LENGTH = 20
SEED = 42


# ============================================================
# 2. 从指定节点生成一条随机游走
# ============================================================

def generate_single_walk(graph, start_node, walk_length, rng):
    if start_node not in graph:
        raise ValueError(f"起点不在图中：{start_node}")
    if walk_length < 1:
        raise ValueError("walk_length 必须至少为 1。")

    walk = [start_node]

    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))

        if not neighbors:
            break

        next_node = rng.choice(neighbors)
        walk.append(next_node)

    return walk


# ============================================================
# 3. 读取 Fold 1 图，展示并检查这条游走
# ============================================================

def main():
    station_order_file = graph_builder.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = graph_builder.GRAPH_DIR / f"fold_{graph_builder.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    walk = generate_single_walk(graph, START_NODE, WALK_LENGTH, random.Random(SEED))
    valid_edges = all(graph.has_edge(a, b) for a, b in zip(walk, walk[1:]))

    if not valid_edges:
        raise RuntimeError("游走中存在图里没有的边。")

    print("起点：", START_NODE)
    print("随机种子：", SEED)
    print("游走节点数量：", len(walk))
    print("游走序列：", " -> ".join(walk))
    print("每一步均沿图中的边：", valid_edges)


if __name__ == "__main__":
    main()
