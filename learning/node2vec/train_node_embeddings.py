from pathlib import Path
import random
import sys
import zlib

from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec import build_networkx_graph as graph_builder
from learning.node2vec.demo_multiple_walks import generate_walks_per_node


# ============================================================
# 1. 把每条游走当作一个“句子”，训练 Skip-gram
# ============================================================

def train_word2vec(walks, vector_size, window, epochs, seed):
    # sg=1 使用 Skip-gram；小图关闭高频下采样，并用稳定哈希保证重复运行一致。
    return Word2Vec(
        sentences=walks, vector_size=vector_size, window=window, epochs=epochs,
        min_count=1, sg=1, sample=0, workers=1, seed=seed,
        hashfxn=lambda node: zlib.crc32(node.encode("utf-8")),
    )


def embeddings_in_station_order(model, station_names):
    """按 SOURCE_STATION_ORDER.csv 的顺序排列节点向量。"""
    return np.stack([model.wv[node] for node in station_names])


# ============================================================
# 2. 只训练 Source Fold 1，并检查得到的向量
# ============================================================

def main():
    station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    adjacency_file = project_config.GRAPH_DIR / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy"

    station_names = pd.read_csv(station_order_file)["NodeID"].astype(str).tolist()
    adj = np.load(adjacency_file)
    graph = graph_builder.build_networkx_graph(station_names, adj)

    rng = random.Random(project_config.SEED)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    rng, project_config.P, project_config.Q)
    model = train_word2vec(walks, project_config.EMBEDDING_DIM, project_config.WINDOW_SIZE,
                           project_config.EPOCHS, project_config.SEED)
    vectors = embeddings_in_station_order(model, station_names)

    if set(model.wv.key_to_index) != set(station_names):
        raise RuntimeError("Word2Vec 学到的节点与节点顺序文件不一致。")
    if vectors.shape != (len(station_names), project_config.EMBEDDING_DIM):
        raise RuntimeError(f"节点向量尺寸异常：{vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("节点向量中存在 NaN 或 Inf。")

    print("Fold：", project_config.FOLD_ID)
    print("游走数量：", len(walks))
    print("节点向量形状：", vectors.shape)
    print("全部节点都有向量：", set(model.wv.key_to_index) == set(station_names))
    print("向量中无 NaN/Inf：", np.isfinite(vectors).all())
    print(f"{station_names[0]} 的向量前5维：", vectors[0, :5])


if __name__ == "__main__":
    main()
