from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config as project_config
from model.stage2_target_finetuning import Stage2TargetFineTuningModel
from training.device import describe_device, select_device


# ============================================================
# 1. 将Q99归一化功率恢复为原始功率
# ============================================================

def denormalize_power(normalized_power, q99):
    normalized_power = np.asarray(normalized_power)
    q99 = np.asarray(q99)
    if normalized_power.ndim != 3:
        raise ValueError("功率数组必须是[样本, 预测步, 站点]三维数组。")
    if normalized_power.shape[2] != len(q99):
        raise ValueError("功率数组的站点数量与Q99数量不一致。")
    return normalized_power * q99[None, None, :]


# ============================================================
# 2. 计算一组真实值与预测值的误差指标
# ============================================================

def calculate_error_metrics(predictions, targets, p_max):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # 光伏夜间真实功率接近0，MAPE只使用不低于峰值1%的样本。
    mape_mask = targets >= 0.01 * p_max
    mape_sample_count = int(mape_mask.sum())
    if mape_sample_count == 0:
        mape_percent = float("nan")
    else:
        mape_percent = float(
            np.mean(np.abs(errors[mape_mask]) / targets[mape_mask]) * 100.0
        )

    if np.std(predictions) == 0 or np.std(targets) == 0:
        correlation_percent = float("nan")
    else:
        correlation_percent = float(np.corrcoef(predictions, targets)[0, 1] * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape_percent,
        "correlation_percent": correlation_percent,
        "mape_sample_count": mape_sample_count,
    }


# ============================================================
# 3. 按论文公式计算目标域总功率指标
# ============================================================

def calculate_aggregate_metrics(predictions, targets, horizon_steps):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("预测值与真实值的形状不一致。")

    aggregate_predictions = predictions.sum(axis=2)
    aggregate_targets = targets.sum(axis=2)
    p_max = float(aggregate_targets.max())
    if p_max <= 0:
        raise ValueError("目标域总功率最大值必须大于0。")

    rows = []
    for step in horizon_steps:
        if step < 1 or step > targets.shape[1]:
            raise ValueError(f"预测步{step}超出1到{targets.shape[1]}的范围。")
        step_metrics = calculate_error_metrics(
            aggregate_predictions[:, step - 1], aggregate_targets[:, step - 1], p_max
        )
        rows.append({
            "horizon_step": step,
            "horizon_minutes": step * 15,
            "p_max": p_max,
            "nmae": step_metrics["mae"] / p_max,
            "nrmse": step_metrics["rmse"] / p_max,
            **step_metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. 计算每个站点的误差指标
# ============================================================

def calculate_station_metrics(predictions, targets, station_names, horizon_steps):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("预测值与真实值的形状不一致。")
    if targets.shape[2] != len(station_names):
        raise ValueError("功率数组的站点数量与站点名称数量不一致。")

    station_p_max = targets.max(axis=(0, 1))
    if np.any(station_p_max <= 0):
        raise ValueError("每个站点的最大真实功率都必须大于0。")

    rows = []
    for step in horizon_steps:
        if step < 1 or step > targets.shape[1]:
            raise ValueError(f"预测步{step}超出1到{targets.shape[1]}的范围。")
        for index, station in enumerate(station_names):
            step_metrics = calculate_error_metrics(
                predictions[:, step - 1, index], targets[:, step - 1, index],
                station_p_max[index],
            )
            rows.append({
                "horizon_step": step,
                "horizon_minutes": step * 15,
                "station": station,
                "p_max": float(station_p_max[index]),
                "nmae": step_metrics["mae"] / station_p_max[index],
                "nrmse": step_metrics["rmse"] / station_p_max[index],
                **step_metrics,
            })

    return pd.DataFrame(rows)


# ============================================================
# 5. 根据预测时间和经纬度计算太阳高度角
# ============================================================

def build_target_times(y_start_time, output_steps, interval_minutes=15):
    y_start_time = np.asarray(y_start_time, dtype="datetime64[ns]")
    offsets = np.arange(output_steps) * np.timedelta64(interval_minutes, "m")
    return y_start_time[:, None] + offsets[None, :]


def calculate_solar_elevation(target_times, latitudes, longitudes, timezone_name):
    target_times = np.asarray(target_times, dtype="datetime64[ns]")
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    if target_times.ndim != 2:
        raise ValueError("预测时间必须是[样本, 预测步]二维数组。")
    if latitudes.shape != longitudes.shape:
        raise ValueError("纬度和经度数量不一致。")

    local_times = pd.DatetimeIndex(target_times.reshape(-1)).tz_localize(timezone_name)
    utc_seconds = local_times.tz_convert("UTC").asi8.astype(np.float64) / 1e9
    julian_day = utc_seconds / 86400.0 + 2440587.5
    days_from_j2000 = julian_day - 2451545.0

    mean_longitude = np.deg2rad((280.460 + 0.9856474 * days_from_j2000) % 360.0)
    mean_anomaly = np.deg2rad((357.528 + 0.9856003 * days_from_j2000) % 360.0)
    ecliptic_longitude = (
        mean_longitude
        + np.deg2rad(1.915) * np.sin(mean_anomaly)
        + np.deg2rad(0.020) * np.sin(2.0 * mean_anomaly)
    )
    obliquity = np.deg2rad(23.439 - 0.0000004 * days_from_j2000)
    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude), np.cos(ecliptic_longitude)
    )
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))
    sidereal_hours = (
        18.697374558 + 24.06570982441908 * days_from_j2000
    ) % 24.0

    shape = target_times.shape + (1,)
    right_ascension = right_ascension.reshape(shape)
    declination = declination.reshape(shape)
    sidereal_degrees = (sidereal_hours * 15.0).reshape(shape)
    latitude_radians = np.deg2rad(latitudes)[None, None, :]
    local_sidereal_degrees = sidereal_degrees + longitudes[None, None, :]
    hour_angle_degrees = (
        local_sidereal_degrees - np.rad2deg(right_ascension) + 180.0
    ) % 360.0 - 180.0
    hour_angle = np.deg2rad(hour_angle_degrees)
    elevation = np.arcsin(
        np.sin(latitude_radians) * np.sin(declination)
        + np.cos(latitude_radians) * np.cos(declination) * np.cos(hour_angle)
    )
    return np.rad2deg(elevation)


def apply_pv_physical_constraints(predictions, target_times, latitudes,
                                  longitudes, timezone_name):
    predictions = np.asarray(predictions)
    target_times = np.asarray(target_times, dtype="datetime64[ns]")
    if predictions.ndim != 3:
        raise ValueError("预测值必须是[样本, 预测步, 站点]三维数组。")
    if predictions.shape[:2] != target_times.shape:
        raise ValueError("预测值的样本和预测步与预测时间不一致。")
    if predictions.shape[2] != len(latitudes):
        raise ValueError("预测值的站点数量与经纬度数量不一致。")

    solar_elevation = calculate_solar_elevation(
        target_times, latitudes, longitudes, timezone_name
    )
    night_mask = solar_elevation <= 0.0
    constrained = np.maximum(predictions, 0).copy()
    constrained[night_mask] = 0
    return constrained, night_mask


# ============================================================
# 6. 使用Stage 2最佳模型生成全部预测
# ============================================================

def predict_all(model, histories, node_vectors, adjacency, batch_size):
    data_loader = DataLoader(TensorDataset(histories), batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    predictions = []
    model.eval()

    with torch.no_grad():
        for (history,) in data_loader:
            prediction = model(history.to(device), node_vectors, adjacency)
            predictions.append(prediction.cpu())

    return torch.cat(predictions).numpy()


# ============================================================
# 7. 保存预测值、真实值和指标
# ============================================================

def save_results(output_dir, predictions_normalized, targets_normalized,
                 predictions_raw, targets_raw, q99, station_names,
                 aggregate_metrics, station_metrics,
                 predictions_normalized_before_constraints,
                 predictions_raw_before_constraints, target_times,
                 night_mask, metrics_before_constraints):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions_normalized=predictions_normalized,
        predictions_normalized_before_constraints=predictions_normalized_before_constraints,
        targets_normalized=targets_normalized,
        predictions_raw=predictions_raw,
        predictions_raw_before_constraints=predictions_raw_before_constraints,
        targets_raw=targets_raw,
        q99=q99,
        station_names=np.asarray(station_names),
        target_times=target_times,
        night_mask=night_mask,
    )
    aggregate_metrics.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    station_metrics.to_csv(output_dir / "metrics_by_station.csv", index=False)
    metrics_before_constraints.to_csv(
        output_dir / "metrics_before_constraints.csv", index=False
    )


# ============================================================
# 8. 评估Target测试集
# ============================================================

def main():
    test_file = project_config.TARGET_DATASET_DIR / "test.npz"
    checkpoint_file = project_config.CHECKPOINT_DIR / "stage2_best.pt"
    output_dir = project_config.EVALUATION_DIR / "target_test"

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"找不到Stage 2检查点：{checkpoint_file}")

    with np.load(test_file) as data:
        histories = torch.from_numpy(data["X"]).float()
        targets_normalized = data["Y"].astype(np.float32)
        q99 = data["q99"].astype(np.float32)
        station_names = data["station_names"].astype(str).tolist()
        y_start_time = data["y_start_time"].astype("datetime64[ns]")
        y_end_time = data["y_end_time"].astype("datetime64[ns]")

    metadata = pd.read_csv(project_config.TARGET_METADATA_FILE).set_index("NodeID")
    if not set(station_names).issubset(metadata.index):
        raise RuntimeError("Target元数据缺少测试集中的站点。")
    metadata = metadata.loc[station_names]
    latitudes = metadata["lat"].to_numpy(dtype=np.float64)
    longitudes = metadata["Lon"].to_numpy(dtype=np.float64)

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint["target_station_names"] != station_names:
        raise RuntimeError("Stage 2检查点的Target站点顺序与测试集不一致。")

    device = select_device()
    node_vectors = checkpoint["target_node_vectors"].to(device)
    adjacency = checkpoint["target_adjacency"].to(device)
    model = Stage2TargetFineTuningModel(
        project_config.EMBEDDING_DIM, project_config.HIDDEN_DIM, project_config.OUTPUT_STEPS
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    predictions_normalized_before_constraints = predict_all(
        model, histories, node_vectors, adjacency, project_config.BATCH_SIZE
    )
    target_times = build_target_times(y_start_time, project_config.OUTPUT_STEPS)
    if not np.array_equal(target_times[:, -1], y_end_time):
        raise RuntimeError("生成的最后一个预测时间与测试集y_end_time不一致。")
    predictions_normalized, night_mask = apply_pv_physical_constraints(
        predictions_normalized_before_constraints, target_times, latitudes,
        longitudes, project_config.TARGET_TIMEZONE,
    )
    predictions_raw_before_constraints = denormalize_power(
        predictions_normalized_before_constraints, q99
    )
    predictions_raw = denormalize_power(predictions_normalized, q99)
    targets_raw = denormalize_power(targets_normalized, q99)
    horizon_steps = project_config.HORIZON_STEPS
    metrics_before_constraints = calculate_aggregate_metrics(
        predictions_raw_before_constraints, targets_raw, horizon_steps
    )
    aggregate_metrics = calculate_aggregate_metrics(
        predictions_raw, targets_raw, horizon_steps
    )
    station_metrics = calculate_station_metrics(
        predictions_raw, targets_raw, station_names, horizon_steps
    )
    save_results(
        output_dir, predictions_normalized, targets_normalized, predictions_raw,
        targets_raw, q99, station_names, aggregate_metrics, station_metrics,
        predictions_normalized_before_constraints,
        predictions_raw_before_constraints, target_times, night_mask,
        metrics_before_constraints,
    )

    normalized_mae = float(np.mean(np.abs(predictions_normalized - targets_normalized)))
    print("设备：", describe_device(device))
    print("Stage 2检查点：", checkpoint_file)
    print("最佳模型Epoch：", checkpoint["epoch"])
    print("夜间置零数量：", int(night_mask.sum()), "/", night_mask.size)
    print("归一化测试MAE：", normalized_mae)
    print("\n目标域测试集总功率指标：")
    print(aggregate_metrics.to_string(index=False))
    print("\n评估结果：", output_dir)


if __name__ == "__main__":
    main()
