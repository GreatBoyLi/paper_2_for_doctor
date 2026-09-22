from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec import build_networkx_graph as graph_builder
from learning.node2vec.demo_all_nodes_one_walk import generate_one_walk_per_node


# ============================================================
# 1. 重复多轮；每一轮让所有节点各作为起点一次
#
# 共用同一个随机数生成器，所以每轮不必得到相同的路径。
# ============================================================

def generate_walks_per_node(graph, num_walks, walk_length, rng, p=1.0, q=1.0):
    walks = []

    for _ in range(num_walks):
        walks.extend(generate_one_walk_per_node(graph, walk_length, rng, p, q))

    return walks


# ============================================================
# 2. 在源域训练图上展示并检查结果
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    rng = random.Random(project_config.SEED)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    rng, project_config.P, project_config.Q)

    start_counts = Counter(walk[0] for walk in walks)
    starts_match = len(walks) == len(station_names) * project_config.NUM_WALKS and all(
        start_counts[node] == project_config.NUM_WALKS for node in station_names
    )
    lengths_match = all(len(walk) == project_config.WALK_LENGTH for walk in walks)
    valid_edges = all(graph.has_edge(a, b) for walk in walks for a, b in zip(walk, walk[1:]))

    if not starts_match or not lengths_match or not valid_edges:
        raise RuntimeError("游走的起点、长度或相邻节点检查未通过。")

    print("节点数量：", graph.number_of_nodes())
    print("每个节点作为起点的次数：", project_config.NUM_WALKS)
    print("游走总数：", len(walks))
    print("每条游走的节点数量：", project_config.WALK_LENGTH)
    print("每个节点的起点次数正确：", starts_match)
    print("游走长度正确：", lengths_match)
    print("每一步均沿图中的边：", valid_edges)

    first_node = station_names[0]
    print(f"\n从 {first_node} 出发的前3条游走：")
    for walk in [walk for walk in walks if walk[0] == first_node][:3]:
        print(" -> ".join(walk))


if __name__ == "__main__":
    main()
