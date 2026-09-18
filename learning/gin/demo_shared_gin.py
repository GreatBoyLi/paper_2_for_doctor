from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec import build_networkx_graph as graph_builder
from learning.node2vec.demo_multiple_walks import generate_walks_per_node
from learning.node2vec.train_node_embeddings import embeddings_in_station_order, train_word2vec


# ============================================================
# 1. 一层 GIN 的前向计算：自身向量 + 邻居平均向量
#
# 邻接矩阵原有的自环在这里去掉，因为自身向量已单独加入。
# 同一组 weights、bias、epsilon 可以处理不同节点数量的图。
# 这里只演示计算，不进行参数训练。
# ============================================================

def mean_gin_layer(vectors, adjacency, weights, bias, epsilon=0.0):
    neighbors = (adjacency > 0).astype(np.float32)
    np.fill_diagonal(neighbors, 0)
    degrees = neighbors.sum(axis=1, keepdims=True)
    neighbor_means = np.divide(neighbors @ vectors, degrees, out=np.zeros_like(vectors), where=degrees > 0)
    combined = (1 + epsilon) * vectors + neighbor_means
    return np.maximum(combined @ weights + bias, 0)


# ============================================================
# 2. Source 和 Target 各自生成 Node2Vec 向量，再通过同一层 GIN
# ============================================================

def main():
    dim = project_config.EMBEDDING_DIM
    rng = np.random.default_rng(project_config.SEED)
    weights = rng.normal(0, 1 / np.sqrt(dim), size=(dim, dim)).astype(np.float32)
    bias = np.zeros(dim, dtype=np.float32)
    epsilon = 0.0

    domains = [
        ("Source", project_config.DATASET_DIR, project_config.GRAPH_DIR, "SOURCE_STATION_ORDER.csv"),
        ("Target", project_config.TARGET_DATASET_DIR, project_config.TARGET_GRAPH_DIR, "TARGET_STATION_ORDER.csv"),
    ]

    print("Fold：", project_config.FOLD_ID)
    print(f"GIN 参数： {weights.shape}（Source/Target 共用）")

    for domain, dataset_dir, graph_dir, order_file in domains:
        station_names = pd.read_csv(dataset_dir / order_file)["NodeID"].astype(str).tolist()
        adjacency = np.load(graph_dir / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy")
        graph = graph_builder.build_networkx_graph(station_names, adjacency)
        walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                        random.Random(project_config.SEED), project_config.P, project_config.Q)
        model = train_word2vec(walks, dim, project_config.WINDOW_SIZE, project_config.EPOCHS, project_config.SEED)
        vectors = embeddings_in_station_order(model, station_names)
        output = mean_gin_layer(vectors, adjacency, weights, bias, epsilon)

        print(f"{domain} 输入：", vectors.shape, "输出：", output.shape)
        print(f"{domain} 首站 {station_names[0]}：输入前5维", vectors[0, :5])
        print(f"{domain} 首站 {station_names[0]}：输出前5维", output[0, :5])


if __name__ == "__main__":
    main()
