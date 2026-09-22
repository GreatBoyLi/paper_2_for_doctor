from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config as project_config
from learning.node2vec.build_networkx_graph import build_networkx_graph
from learning.node2vec.demo_multiple_walks import generate_walks_per_node
from learning.node2vec.train_node_embeddings import embeddings_in_station_order, train_word2vec
from model.power_forecaster import GraphPowerForecaster
from training.device import describe_device, select_device
from training.early_stopping import EarlyStopping
from training.history import add_horizon_metrics, save_training_history


# ============================================================
# 1. 单轮训练与验证
# ============================================================

def train_one_epoch(model, data_loader, node_vectors, adjacency, optimizer):
    model.train()
    total_loss = 0.0
    sample_count = 0
    device = next(model.parameters()).device

    for history, targets in data_loader:
        history = history.to(device)
        targets = targets.to(device)
        predictions = model(history, node_vectors, adjacency)
        loss = nn.functional.l1_loss(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * history.shape[0]
        sample_count += history.shape[0]

    return total_loss / sample_count


def calculate_horizon_mae(predictions, targets, horizon_steps):
    horizon_mae = {}
    for step in horizon_steps:
        if step < 1 or step > predictions.shape[1]:
            raise ValueError(f"预测步{step}超出1到{predictions.shape[1]}的范围。")
        horizon_mae[step] = nn.functional.l1_loss(
            predictions[:, step - 1], targets[:, step - 1]
        ).item()
    return horizon_mae


def evaluate_with_horizons(model, data_loader, node_vectors, adjacency, horizon_steps):
    model.eval()
    total_loss = 0.0
    horizon_totals = {step: 0.0 for step in horizon_steps}
    sample_count = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for history, targets in data_loader:
            history = history.to(device)
            targets = targets.to(device)
            predictions = model(history, node_vectors, adjacency)
            loss = nn.functional.l1_loss(predictions, targets)
            batch_size = history.shape[0]
            batch_horizon_mae = calculate_horizon_mae(predictions, targets, horizon_steps)
            total_loss += loss.item() * batch_size
            for step, value in batch_horizon_mae.items():
                horizon_totals[step] += value * batch_size
            sample_count += batch_size

    horizon_mae = {step: value / sample_count for step, value in horizon_totals.items()}
    return total_loss / sample_count, horizon_mae


def evaluate(model, data_loader, node_vectors, adjacency):
    val_loss, _ = evaluate_with_horizons(model, data_loader, node_vectors, adjacency, [])
    return val_loss


def format_horizon_mae(horizon_mae):
    parts = []
    for step, value in horizon_mae.items():
        minutes = step * 15
        label = f"{minutes}min" if minutes < 60 else f"{minutes // 60}h"
        parts.append(f"{label}: {value:.6f}")
    return " | ".join(parts)


# ============================================================
# 2. 保存最佳模型，同时保存预测时必需的图和节点信息
# ============================================================

def save_checkpoint(file, model, node_vectors, adjacency, station_names, epoch, val_loss):
    file.parent.mkdir(parents=True, exist_ok=True)
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "model_state_dict": model_state,
        "node_vectors": node_vectors.detach().cpu(),
        "adjacency": adjacency.detach().cpu(),
        "station_names": station_names,
        "epoch": epoch,
        "val_loss": val_loss,
    }, file)


# ============================================================
# 3. 读取 Source Fold 1 数据
# ============================================================

def load_power_dataset(file, station_names):
    with np.load(file) as data:
        sample_station_names = data["station_names"].astype(str).tolist()
        if sample_station_names != station_names:
            raise RuntimeError(f"{file.name} 的站点顺序与 Source 节点顺序不一致。")
        history = torch.from_numpy(data["X"]).float()
        targets = torch.from_numpy(data["Y"]).float()

    return TensorDataset(history, targets)


def create_node_vectors(station_names, adjacency):
    graph = build_networkx_graph(station_names, adjacency)
    walks = generate_walks_per_node(graph, project_config.NUM_WALKS, project_config.WALK_LENGTH,
                                    random.Random(project_config.SEED), project_config.P, project_config.Q)
    node2vec = train_word2vec(walks, project_config.EMBEDDING_DIM, project_config.WINDOW_SIZE,
                              project_config.EPOCHS, project_config.SEED)
    return torch.from_numpy(embeddings_in_station_order(node2vec, station_names)).float()


# ============================================================
# 4. 正式训练 Source Fold 1 基线
# ============================================================

def main():
    fold = project_config.FOLD_ID
    fold_dataset_dir = project_config.DATASET_DIR / f"fold_{fold}"
    fold_graph_dir = project_config.GRAPH_DIR / f"fold_{fold}"
    station_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    checkpoint_file = project_config.CHECKPOINT_DIR / f"source_fold_{fold}_best.pt"
    history_file = project_config.TRAINING_HISTORY_DIR / f"source_fold_{fold}_history.csv"

    station_names = pd.read_csv(station_file)["NodeID"].astype(str).tolist()
    adjacency = np.load(fold_graph_dir / "adjacency_binary.npy")
    train_dataset = load_power_dataset(fold_dataset_dir / "train.npz", station_names)
    val_dataset = load_power_dataset(fold_dataset_dir / "val.npz", station_names)

    generator = torch.Generator().manual_seed(project_config.SEED)
    train_loader = DataLoader(train_dataset, batch_size=project_config.BATCH_SIZE, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=project_config.BATCH_SIZE, shuffle=False)

    torch.manual_seed(project_config.SEED)
    device = select_device()
    node_vectors = create_node_vectors(station_names, adjacency).to(device)
    adjacency_tensor = torch.from_numpy(adjacency).float().to(device)
    model = GraphPowerForecaster(project_config.EMBEDDING_DIM, project_config.HIDDEN_DIM,
                                 project_config.OUTPUT_STEPS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=project_config.LEARNING_RATE)

    print("Fold：", fold)
    print("设备：", describe_device(device))
    print("训练样本：", len(train_dataset), "验证样本：", len(val_dataset))
    print("Node2Vec 只在正式训练开始前生成一次。")
    print(
        "最多Epoch：", project_config.TRAINING_EPOCHS,
        "Early Stopping耐心值：", project_config.EARLY_STOPPING_PATIENCE,
    )

    best_val_loss = float("inf")
    history_rows = []
    early_stopping = EarlyStopping(
        project_config.EARLY_STOPPING_PATIENCE,
        project_config.EARLY_STOPPING_MIN_DELTA,
    )
    for epoch in range(1, project_config.TRAINING_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, node_vectors, adjacency_tensor, optimizer)
        val_loss, horizon_mae = evaluate_with_horizons(
            model, val_loader, node_vectors, adjacency_tensor,
            project_config.HORIZON_STEPS,
        )
        print(f"Epoch {epoch:02d} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}")
        print("  Val分尺度 MAE |", format_horizon_mae(horizon_mae))

        improved, should_stop = early_stopping.update(val_loss)
        if improved:
            best_val_loss = early_stopping.best_loss
            save_checkpoint(checkpoint_file, model, node_vectors, adjacency_tensor,
                            station_names, epoch, best_val_loss)

        history_row = {
            "epoch": epoch,
            "train_mae": train_loss,
            "val_mae": val_loss,
            "is_best": improved,
            "early_stopping_wait_count": early_stopping.wait_count,
        }
        add_horizon_metrics(history_row, horizon_mae)
        history_rows.append(history_row)
        save_training_history(history_file, history_rows)

        if should_stop:
            print(f"Early Stopping：验证MAE连续{early_stopping.wait_count}轮没有明显改善。")
            break

    print("最佳验证 MAE：", best_val_loss)
    print("最佳模型：", checkpoint_file)
    print("训练历史：", history_file)


if __name__ == "__main__":
    main()
