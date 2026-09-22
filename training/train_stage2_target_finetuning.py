from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config as project_config
from model.stage2_target_finetuning import Stage2TargetFineTuningModel
from training.device import describe_device, select_device
from training.early_stopping import EarlyStopping
from training.history import add_horizon_metrics, save_training_history
from training.train_source_forecaster import (
    evaluate_with_horizons, format_horizon_mae, load_power_dataset, train_one_epoch,
)


# ============================================================
# 1. 用Stage 1参数初始化Stage 2
# ============================================================

def load_prefixed_state(module, stage1_state, prefix):
    module_state = {
        name.removeprefix(prefix): value
        for name, value in stage1_state.items()
        if name.startswith(prefix)
    }
    module.load_state_dict(module_state)


def initialize_from_stage1(model, stage1_state):
    load_prefixed_state(model.shared_spatial_encoder, stage1_state, "spatial_encoder.")
    load_prefixed_state(model.private_spatial_encoder, stage1_state, "spatial_encoder.")
    load_prefixed_state(model.temporal_regression, stage1_state, "temporal_regression.")
    load_prefixed_state(model.forecast_head, stage1_state, "forecast_head.")


# ============================================================
# 2. 保存Target微调后的最佳模型
# ============================================================

def save_checkpoint(file, model, target_vectors, target_adjacency,
                    target_station_names, epoch, val_loss):
    file.parent.mkdir(parents=True, exist_ok=True)
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "model_state_dict": model_state,
        "target_node_vectors": target_vectors.detach().cpu(),
        "target_adjacency": target_adjacency.detach().cpu(),
        "target_station_names": target_station_names,
        "epoch": epoch,
        "val_loss": val_loss,
    }, file)


# ============================================================
# 3. Fold 1的Stage 2目标域微调
# ============================================================

def main():
    fold = project_config.FOLD_ID
    target_fold_dir = project_config.TARGET_DATASET_DIR / f"fold_{fold}"
    stage1_file = project_config.CHECKPOINT_DIR / f"stage1_fold_{fold}_best.pt"
    stage2_file = project_config.CHECKPOINT_DIR / f"stage2_fold_{fold}_best.pt"
    history_file = project_config.TRAINING_HISTORY_DIR / f"stage2_fold_{fold}_history.csv"

    if not stage1_file.is_file():
        raise FileNotFoundError(f"找不到Stage 1检查点：{stage1_file}")

    target_station_names = pd.read_csv(
        project_config.TARGET_DATASET_DIR / "TARGET_STATION_ORDER.csv"
    )["NodeID"].astype(str).tolist()
    train_dataset = load_power_dataset(target_fold_dir / "train.npz", target_station_names)
    val_dataset = load_power_dataset(target_fold_dir / "val.npz", target_station_names)

    generator = torch.Generator().manual_seed(project_config.SEED)
    train_loader = DataLoader(train_dataset, batch_size=project_config.BATCH_SIZE, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=project_config.BATCH_SIZE, shuffle=False)

    device = select_device()
    stage1 = torch.load(stage1_file, map_location="cpu", weights_only=False)
    if stage1["target_station_names"] != target_station_names:
        raise RuntimeError("Stage 1检查点的Target站点顺序与Target数据集不一致。")

    target_vectors = stage1["target_node_vectors"].to(device)
    target_adjacency = stage1["target_adjacency"].to(device)
    torch.manual_seed(project_config.SEED)
    model = Stage2TargetFineTuningModel(
        project_config.EMBEDDING_DIM, project_config.HIDDEN_DIM, project_config.OUTPUT_STEPS
    )
    initialize_from_stage1(model, stage1["model_state_dict"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=project_config.LEARNING_RATE)

    print("Fold：", fold)
    print("设备：", describe_device(device))
    print("Target训练样本：", len(train_dataset), "Target验证样本：", len(val_dataset))
    print("Stage 1检查点：", stage1_file)
    print("Target Node2Vec直接从Stage 1读取，不重新训练。")
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
        train_loss = train_one_epoch(model, train_loader, target_vectors, target_adjacency, optimizer)
        val_loss, horizon_mae = evaluate_with_horizons(
            model, val_loader, target_vectors, target_adjacency,
            project_config.HORIZON_STEPS,
        )
        print(
            f"Epoch {epoch:02d} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f} "
            f"| w_private: {model.private_weight.item():.4f} | w_shared: {model.shared_weight.item():.4f}"
        )
        print("  Val分尺度 MAE |", format_horizon_mae(horizon_mae))

        improved, should_stop = early_stopping.update(val_loss)
        if improved:
            best_val_loss = early_stopping.best_loss
            save_checkpoint(
                stage2_file, model, target_vectors, target_adjacency,
                target_station_names, epoch, best_val_loss,
            )

        history_row = {
            "epoch": epoch,
            "train_mae": train_loss,
            "val_mae": val_loss,
            "private_weight": model.private_weight.item(),
            "shared_weight": model.shared_weight.item(),
            "is_best": improved,
            "early_stopping_wait_count": early_stopping.wait_count,
        }
        add_horizon_metrics(history_row, horizon_mae)
        history_rows.append(history_row)
        save_training_history(history_file, history_rows)

        if should_stop:
            print(f"Early Stopping：验证MAE连续{early_stopping.wait_count}轮没有明显改善。")
            break

    print("最佳Target验证MAE：", best_val_loss)
    print("Stage 2最佳模型：", stage2_file)
    print("训练历史：", history_file)


if __name__ == "__main__":
    main()
