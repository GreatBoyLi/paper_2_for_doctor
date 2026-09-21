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
from model.power_forecaster import GraphPowerForecaster


# ============================================================
# 1. 用 Source Fold 1 的两个真实样本演示一次前向与反向传播
#
# 这里只验证完整预测主干能够工作，不更新参数，也不进行正式训练。
# ============================================================

def main():
    fold = project_config.FOLD_ID
    dataset_dir = project_config.DATASET_DIR / f"fold_{fold}"
    graph_dir = project_config.GRAPH_DIR / f"fold_{fold}"
    station_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"

    station_names = pd.read_csv(station_file)["NodeID"].astype(str).tolist()
    adjacency = np.load(graph_dir / "adjacency_binary.npy")
    with np.load(dataset_dir / "train.npz") as data:
        history = torch.from_numpy(data["X"][:2]).float()
        targets = torch.from_numpy(data["Y"][:2]).float()
        sample_station_names = data["station_names"].astype(str).tolist()

    if sample_station_names != station_names:
        raise RuntimeError("功率样本的站点顺序与 Source 节点顺序不一致。")

    # Node2Vec 仍在每次完整实验开始时运行一次，然后把固定向量交给预测模型。
    graph = build_networkx_graph(station_names, adjacency)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    random.Random(project_config.SEED), project_config.P, project_config.Q)
    node2vec = train_word2vec(walks, project_config.EMBEDDING_DIM, project_config.WINDOW_SIZE,
                              project_config.EPOCHS, project_config.SEED)
    node_vectors = torch.from_numpy(embeddings_in_station_order(node2vec, station_names)).float()

    torch.manual_seed(project_config.SEED)
    model = GraphPowerForecaster(project_config.EMBEDDING_DIM, project_config.HIDDEN_DIM,
                                 project_config.OUTPUT_STEPS)
    predictions = model(history, node_vectors, torch.from_numpy(adjacency))
    loss = nn.functional.l1_loss(predictions, targets)
    loss.backward()

    print("Fold：", fold)
    print("历史功率：", tuple(history.shape))
    print("Node2Vec 向量：", tuple(node_vectors.shape))
    print("真实未来功率：", tuple(targets.shape))
    print("预测未来功率：", tuple(predictions.shape))
    print("MAE 损失：", float(loss.detach()))
    print("全部模型参数都有梯度：", all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    main()
