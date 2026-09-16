from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec import build_networkx_graph as graph_builder
from learning.node2vec.demo_single_walk import generate_single_walk


# ============================================================
# 1. 每个节点作为起点，各生成一条游走
#
# 所有起点共用一个随机数生成器，不要为每个节点重新设置种子。
# ============================================================

def generate_one_walk_per_node(graph, walk_length, rng, p=1.0, q=1.0):
    walks = []

    for start_node in graph.nodes:
        walk = generate_single_walk(graph, start_node, walk_length, rng, p, q)
        walks.append(walk)

    return walks


# ============================================================
# 2. 在配置指定的 Fold 上展示结果
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    rng = random.Random(project_config.SEED)
    walks = generate_one_walk_per_node(graph, project_config.WALK_LENGTH, rng, project_config.P, project_config.Q)

    starts_match = len(walks) == len(station_names) and all(walk[0] == node for walk, node in zip(walks, station_names))
    lengths_match = all(len(walk) == project_config.WALK_LENGTH for walk in walks)
    valid_edges = all(graph.has_edge(a, b) for walk in walks for a, b in zip(walk, walk[1:]))

    if not starts_match or not lengths_match or not valid_edges:
        raise RuntimeError("游走的起点、长度或相邻节点检查未通过。")

    print("Fold：", project_config.FOLD_ID)
    print("节点数量：", graph.number_of_nodes())
    print("游走数量：", len(walks))
    print("每条游走的目标节点数量：", project_config.WALK_LENGTH)
    print("起点顺序正确：", starts_match)
    print("游走长度正确：", lengths_match)
    print("每一步均沿图中的边：", valid_edges)
    print("\n前3条游走：")

    for walk in walks[:3]:
        print(" -> ".join(walk))


if __name__ == "__main__":
    main()
