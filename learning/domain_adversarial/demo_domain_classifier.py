from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config
from learning.node2vec.build_networkx_graph import build_networkx_graph
from learning.node2vec.demo_multiple_walks import generate_walks_per_node
from learning.node2vec.train_node_embeddings import embeddings_in_station_order, train_word2vec
from model.domain_adversarial import DomainClassifier
from model.gin import MeanGINLayer


# ============================================================
# 1. 为一个域生成Node2Vec向量，并读取它自己的邻接矩阵
# ============================================================

def load_domain(dataset_dir, graph_dir, station_order_file):
    station_names = pd.read_csv(dataset_dir / station_order_file)["NodeID"].astype(str).tolist()
    adjacency = np.load(graph_dir / f"fold_{project_config.FOLD_ID}" / "adjacency_binary.npy")
    graph = build_networkx_graph(station_names, adjacency)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    random.Random(project_config.SEED), project_config.P, project_config.Q)
    node2vec = train_word2vec(walks, project_config.EMBEDDING_DIM, project_config.WINDOW_SIZE,
                              project_config.EPOCHS, project_config.SEED)
    vectors = torch.from_numpy(embeddings_in_station_order(node2vec, station_names)).float()
    return vectors, torch.from_numpy(adjacency).float()


# ============================================================
# 2. 两个域使用同一个GIN，再通过GRL和域分类器
# ============================================================

def main():
    source_vectors, source_adjacency = load_domain(
        project_config.DATASET_DIR, project_config.GRAPH_DIR, "SOURCE_STATION_ORDER.csv"
    )
    target_vectors, target_adjacency = load_domain(
        project_config.TARGET_DATASET_DIR, project_config.TARGET_GRAPH_DIR, "TARGET_STATION_ORDER.csv"
    )

    torch.manual_seed(project_config.SEED)
    shared_gin = MeanGINLayer(project_config.EMBEDDING_DIM)
    domain_classifier = DomainClassifier(project_config.EMBEDDING_DIM, project_config.DOMAIN_HIDDEN_DIM)

    source_features = shared_gin(source_vectors, source_adjacency)
    target_features = shared_gin(target_vectors, target_adjacency)
    domain_features = torch.cat((source_features, target_features), dim=0)

    # Source标记为0，Target标记为1；域损失同时更新分类器和共享GIN。
    source_labels = torch.zeros(source_features.shape[0], dtype=torch.long)
    target_labels = torch.ones(target_features.shape[0], dtype=torch.long)
    domain_labels = torch.cat((source_labels, target_labels), dim=0)
    domain_logits = domain_classifier(domain_features, project_config.GRL_LAMBDA)
    domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
    domain_loss.backward()

    print("Fold：", project_config.FOLD_ID)
    print("Source 空间特征：", tuple(source_features.shape))
    print("Target 空间特征：", tuple(target_features.shape))
    print("域分类输出：", tuple(domain_logits.shape))
    print("域分类损失：", float(domain_loss.detach()))
    print("共享 GIN 参数有梯度：", all(parameter.grad is not None for parameter in shared_gin.parameters()))
    print("域分类器参数有梯度：", all(parameter.grad is not None for parameter in domain_classifier.parameters()))


if __name__ == "__main__":
    main()
