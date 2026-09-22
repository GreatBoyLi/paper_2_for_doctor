from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config as project_config
from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
from training.device import describe_device, select_device
from training.early_stopping import EarlyStopping
from training.history import add_horizon_metrics, save_training_history
from training.train_source_forecaster import (
    calculate_horizon_mae, create_node_vectors, format_horizon_mae, load_power_dataset,
)


def create_domain_labels(source_nodes, target_nodes, device):
    source_labels = torch.zeros(source_nodes, dtype=torch.long, device=device)
    target_labels = torch.ones(target_nodes, dtype=torch.long, device=device)
    return torch.cat((source_labels, target_labels), dim=0)


# ============================================================
# 1. 一轮Stage 1联合训练
# ============================================================

def train_one_epoch(model, source_loader, source_vectors, source_adjacency,
                    target_vectors, target_adjacency, optimizer, grl_lambda):
    model.train()
    device = next(model.parameters()).device
    domain_labels = create_domain_labels(source_vectors.shape[0], target_vectors.shape[0], device)
    totals = {"forecast_loss": 0.0, "domain_loss": 0.0, "domain_accuracy": 0.0, "total_loss": 0.0}
    sample_count = 0

    for source_history, source_targets in source_loader:
        source_history = source_history.to(device)
        source_targets = source_targets.to(device)
        predictions, domain_logits = model(
            source_history, source_vectors, source_adjacency,
            target_vectors, target_adjacency, grl_lambda,
        )
        forecast_loss = nn.functional.l1_loss(predictions, source_targets)
        domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
        total_loss = forecast_loss + domain_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        batch_size = source_history.shape[0]
        domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()
        totals["forecast_loss"] += forecast_loss.item() * batch_size
        totals["domain_loss"] += domain_loss.item() * batch_size
        totals["domain_accuracy"] += domain_accuracy * batch_size
        totals["total_loss"] += total_loss.item() * batch_size
        sample_count += batch_size

    return {name: value / sample_count for name, value in totals.items()}


def evaluate(model, source_loader, source_vectors, source_adjacency,
             target_vectors, target_adjacency, grl_lambda, horizon_steps=()):
    model.eval()
    device = next(model.parameters()).device
    domain_labels = create_domain_labels(source_vectors.shape[0], target_vectors.shape[0], device)
    totals = {"forecast_loss": 0.0, "domain_loss": 0.0, "domain_accuracy": 0.0, "total_loss": 0.0}
    horizon_totals = {step: 0.0 for step in horizon_steps}
    sample_count = 0

    with torch.no_grad():
        for source_history, source_targets in source_loader:
            source_history = source_history.to(device)
            source_targets = source_targets.to(device)
            predictions, domain_logits = model(
                source_history, source_vectors, source_adjacency,
                target_vectors, target_adjacency, grl_lambda,
            )
            forecast_loss = nn.functional.l1_loss(predictions, source_targets)
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
            total_loss = forecast_loss + domain_loss

            batch_size = source_history.shape[0]
            domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()
            totals["forecast_loss"] += forecast_loss.item() * batch_size
            totals["domain_loss"] += domain_loss.item() * batch_size
            totals["domain_accuracy"] += domain_accuracy * batch_size
            totals["total_loss"] += total_loss.item() * batch_size
            batch_horizon_mae = calculate_horizon_mae(
                predictions, source_targets, horizon_steps
            )
            for step, value in batch_horizon_mae.items():
                horizon_totals[step] += value * batch_size
            sample_count += batch_size

    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics["horizon_mae"] = {
        step: value / sample_count for step, value in horizon_totals.items()
    }
    return metrics


# ============================================================
# 2. 保存Stage 1最佳模型和两域图信息
# ============================================================

def save_checkpoint(file, model, source_vectors, source_adjacency, source_station_names,
                    target_vectors, target_adjacency, target_station_names, epoch, val_forecast_loss):
    file.parent.mkdir(parents=True, exist_ok=True)
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "model_state_dict": model_state,
        "source_node_vectors": source_vectors.detach().cpu(),
        "source_adjacency": source_adjacency.detach().cpu(),
        "source_station_names": source_station_names,
        "target_node_vectors": target_vectors.detach().cpu(),
        "target_adjacency": target_adjacency.detach().cpu(),
        "target_station_names": target_station_names,
        "epoch": epoch,
        "val_forecast_loss": val_forecast_loss,
    }, file)


# ============================================================
# 3. 正式训练Stage 1
# ============================================================

def main():
    source_dataset_dir = project_config.DATASET_DIR
    source_graph_dir = project_config.GRAPH_DIR
    target_graph_dir = project_config.TARGET_GRAPH_DIR
    checkpoint_file = project_config.CHECKPOINT_DIR / "stage1_best.pt"
    history_file = project_config.TRAINING_HISTORY_DIR / "stage1_history.csv"

    source_station_names = pd.read_csv(
        project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"
    )["NodeID"].astype(str).tolist()
    target_station_names = pd.read_csv(
        project_config.TARGET_DATASET_DIR / "TARGET_STATION_ORDER.csv"
    )["NodeID"].astype(str).tolist()
    source_adjacency = np.load(source_graph_dir / "adjacency_binary.npy")
    target_adjacency = np.load(target_graph_dir / "adjacency_binary.npy")

    train_dataset = load_power_dataset(source_dataset_dir / "train.npz", source_station_names)
    val_dataset = load_power_dataset(source_dataset_dir / "val.npz", source_station_names)
    generator = torch.Generator().manual_seed(project_config.SEED)
    train_loader = DataLoader(train_dataset, batch_size=project_config.BATCH_SIZE, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=project_config.BATCH_SIZE, shuffle=False)

    torch.manual_seed(project_config.SEED)
    device = select_device()
    source_vectors = create_node_vectors(source_station_names, source_adjacency).to(device)
    target_vectors = create_node_vectors(target_station_names, target_adjacency).to(device)
    source_adjacency = torch.from_numpy(source_adjacency).float().to(device)
    target_adjacency = torch.from_numpy(target_adjacency).float().to(device)

    model = Stage1DomainAdversarialModel(
        project_config.EMBEDDING_DIM, project_config.HIDDEN_DIM,
        project_config.OUTPUT_STEPS, project_config.DOMAIN_HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=project_config.LEARNING_RATE)

    print("设备：", describe_device(device))
    print("Source训练样本：", len(train_dataset), "Source验证样本：", len(val_dataset))
    print("Source节点：", len(source_station_names), "Target节点：", len(target_station_names))
    print("两域Node2Vec均只在Stage 1开始前生成一次。")
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
        train_metrics = train_one_epoch(
            model, train_loader, source_vectors, source_adjacency,
            target_vectors, target_adjacency, optimizer, project_config.GRL_LAMBDA,
        )
        val_metrics = evaluate(
            model, val_loader, source_vectors, source_adjacency,
            target_vectors, target_adjacency, project_config.GRL_LAMBDA,
            project_config.HORIZON_STEPS,
        )
        print(
            f"Epoch {epoch:02d} | Train Forecast: {train_metrics['forecast_loss']:.6f} "
            f"| Train Domain: {train_metrics['domain_loss']:.6f} "
            f"| Domain Acc: {train_metrics['domain_accuracy']:.4f} "
            f"| Val Forecast: {val_metrics['forecast_loss']:.6f}"
        )
        print("  Val分尺度 MAE |", format_horizon_mae(val_metrics["horizon_mae"]))

        improved, should_stop = early_stopping.update(val_metrics["forecast_loss"])
        if improved:
            best_val_loss = early_stopping.best_loss
            save_checkpoint(
                checkpoint_file, model, source_vectors, source_adjacency, source_station_names,
                target_vectors, target_adjacency, target_station_names, epoch, best_val_loss,
            )

        history_row = {
            "epoch": epoch,
            "train_forecast_loss": train_metrics["forecast_loss"],
            "train_domain_loss": train_metrics["domain_loss"],
            "train_domain_accuracy": train_metrics["domain_accuracy"],
            "train_total_loss": train_metrics["total_loss"],
            "val_forecast_loss": val_metrics["forecast_loss"],
            "val_domain_loss": val_metrics["domain_loss"],
            "val_domain_accuracy": val_metrics["domain_accuracy"],
            "val_total_loss": val_metrics["total_loss"],
            "is_best": improved,
            "early_stopping_wait_count": early_stopping.wait_count,
        }
        add_horizon_metrics(history_row, val_metrics["horizon_mae"])
        history_rows.append(history_row)
        save_training_history(history_file, history_rows)

        if should_stop:
            print(f"Early Stopping：验证MAE连续{early_stopping.wait_count}轮没有明显改善。")
            break

    print("最佳Source验证MAE：", best_val_loss)
    print("Stage 1最佳模型：", checkpoint_file)
    print("训练历史：", history_file)


if __name__ == "__main__":
    main()
