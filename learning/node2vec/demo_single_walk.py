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
# p = q = 1 时，候选邻居的权重相同，可以等概率选择。
# ============================================================

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
# 3. 读取配置指定 Fold 的图，展示并检查这条游走
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    walk = generate_single_walk(graph, project_config.START_NODE, project_config.WALK_LENGTH, random.Random(project_config.SEED))
    valid_edges = all(graph.has_edge(a, b) for a, b in zip(walk, walk[1:]))

    if not valid_edges:
        raise RuntimeError("游走中存在图里没有的边。")

    print("起点：", project_config.START_NODE)
    print("随机种子：", project_config.SEED)
    print("游走节点数量：", len(walk))
    print("游走序列：", " -> ".join(walk))
    print("每一步均沿图中的边：", valid_edges)


if __name__ == "__main__":
    main()
