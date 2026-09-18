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
# 1. 读取 Target Fold 1 的节点顺序和图
# ============================================================

def main():
    station_order_file = project_config.TARGET_DATASET_DIR / "TARGET_STATION_ORDER.csv"
    adjacency_file = project_config.TARGET_GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    # 与 Source 使用相同的参数，但只在 Target 自己的图上游走和训练。
    rng = random.Random(project_config.SEED)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    rng, project_config.P, project_config.Q)
    model = train_word2vec(walks, project_config.EMBEDDING_DIM, project_config.WINDOW_SIZE,
                           project_config.EPOCHS, project_config.SEED)
    vectors = embeddings_in_station_order(model, station_names)

    if set(model.wv.key_to_index) != set(station_names):
        raise RuntimeError("Word2Vec 学到的节点与目标域节点顺序文件不一致。")
    if vectors.shape != (len(station_names), project_config.EMBEDDING_DIM):
        raise RuntimeError(f"目标域节点向量尺寸异常：{vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("目标域节点向量中存在 NaN 或 Inf。")

    print("Fold：", project_config.FOLD_ID)
    print("游走数量：", len(walks))
    print("节点向量形状：", vectors.shape)
    print("全部节点都有向量：", set(model.wv.key_to_index) == set(station_names))
    print("向量中无 NaN/Inf：", np.isfinite(vectors).all())
    print(f"{station_names[0]} 的向量前5维：", vectors[0, :5])


if __name__ == "__main__":
    main()
