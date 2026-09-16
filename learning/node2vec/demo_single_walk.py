from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec import build_networkx_graph as graph_builder


# ============================================================
# 1. 本次只观察配置指定 Fold 中的一条游走
#
# walk_length 包括起点；设为20时，对应19次移动。
# p 和 q 从配置读取；两者都为1时，候选邻居等概率。
# ============================================================

# ============================================================
# 2. 从指定节点生成一条随机游走
# ============================================================

def transition_weights(graph, previous_node, neighbors, p, q):
    """按返回、附近、向外三种关系计算候选邻居的权重。"""
    weights = []

    for next_node in neighbors:
        if next_node == previous_node:
            weight = 1 / p
        elif graph.has_edge(previous_node, next_node):
            weight = 1.0
        else:
            weight = 1 / q
        weights.append(weight)

    return weights


def generate_single_walk(graph, start_node, walk_length, rng, p=1.0, q=1.0):
    if start_node not in graph:
        raise ValueError(f"起点不在图中：{start_node}")
    if walk_length < 1:
        raise ValueError("walk_length 必须至少为 1。")
    if p <= 0 or q <= 0:
        raise ValueError("p 和 q 必须大于 0。")

    walk = [start_node]

    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))

        if not neighbors:
            break

        # 第一步没有“上一节点”；p=q=1时也保持原来的等概率选择。
        if len(walk) == 1 or (p == 1 and q == 1):
            next_node = rng.choice(neighbors)
        else:
            weights = transition_weights(graph, walk[-2], neighbors, p, q)
            next_node = rng.choices(neighbors, weights=weights, k=1)[0]
        walk.append(next_node)

    return walk


# ============================================================
# 3. 读取配置指定 Fold 的图，展示并检查这条游走
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    walk = generate_single_walk(graph, project_config.START_NODE, project_config.WALK_LENGTH,
                                random.Random(project_config.SEED), project_config.P, project_config.Q)
    valid_edges = all(graph.has_edge(a, b) for a, b in zip(walk, walk[1:]))

    if not valid_edges:
        raise RuntimeError("游走中存在图里没有的边。")

    print("起点：", project_config.START_NODE)
    print("随机种子：", project_config.SEED)
    print("p：", project_config.P)
    print("q：", project_config.Q)
    print("游走节点数量：", len(walk))
    print("游走序列：", " -> ".join(walk))
    print("每一步均沿图中的边：", valid_edges)


if __name__ == "__main__":
    main()
