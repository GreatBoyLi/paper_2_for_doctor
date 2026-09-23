from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import config as project_config


# ============================================================
# 1. 路径与绘图参数
# ============================================================

INPUT_FILE = project_config.EVALUATION_DIR / "target_test/predictions.npz"
OUTPUT_PNG = project_config.EVALUATION_DIR / "target_test/target_test_four_weeks.png"
OUTPUT_PDF = project_config.EVALUATION_DIR / "target_test/target_test_four_weeks.pdf"
FORECAST_STEP = 0
EXPECTED_OUTPUT_STEPS = 16
EXPECTED_STATIONS = 31
EXPECTED_SAMPLES_PER_PERIOD = 641


def load_weekly_power_series(file, periods, expected_samples_per_period=EXPECTED_SAMPLES_PER_PERIOD):
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"找不到目标域测试预测结果：{file}")

    with np.load(file) as data:
        required = {"predictions_raw", "targets_raw", "target_times"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"预测结果缺少数组：{sorted(missing)}")
        predictions = data["predictions_raw"]
        targets = data["targets_raw"]
        target_times = data["target_times"].astype("datetime64[ns]")

    if predictions.shape != targets.shape:
        raise RuntimeError("目标域预测值与真实值形状不一致。")
    if predictions.ndim != 3 or predictions.shape[1:] != (EXPECTED_OUTPUT_STEPS, EXPECTED_STATIONS):
        raise RuntimeError("目标域预测数组应为[样本, 16预测步, 31站点]。")
    if target_times.shape != predictions.shape[:2]:
        raise RuntimeError("目标域预测时间形状与预测数组不一致。")

    times = target_times[:, FORECAST_STEP]
    predicted_power = predictions[:, FORECAST_STEP, :].sum(axis=1)
    actual_power = targets[:, FORECAST_STEP, :].sum(axis=1)
    weeks = []

    for period in periods:
        start = np.datetime64(pd.Timestamp(period["start"]), "ns")
        end_exclusive = np.datetime64(pd.Timestamp(period["end"]) + pd.Timedelta(days=1), "ns")
        mask = (times >= start) & (times < end_exclusive)
        sample_count = int(mask.sum())
        if sample_count != expected_samples_per_period:
            raise RuntimeError(
                f"{period['quarter']}测试周应有{expected_samples_per_period}个15分钟预测点，"
                f"实际为{sample_count}个。"
            )

        weeks.append({
            "quarter": period["quarter"],
            "start": period["start"],
            "end": period["end"],
            "times": times[mask],
            "actual_power": actual_power[mask],
            "predicted_power": predicted_power[mask],
        })

    return weeks


def format_week_title(week):
    start = pd.Timestamp(week["start"])
    end = pd.Timestamp(week["end"])
    return f"{week['quarter']}: {start:%b %d}-{end:%d, %Y}"


def create_four_week_figure(weeks, output_png, output_pdf):
    if len(weeks) != 4:
        raise ValueError("四季度测试图必须包含4个测试周。")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.ravel()
    maximum_power = max(
        float(np.nanmax(np.concatenate([week["actual_power"], week["predicted_power"]])))
        for week in weeks
    )

    for axis, week in zip(axes, weeks):
        axis.plot(week["times"], week["actual_power"], color="black", linewidth=1.2, label="Actual Power")
        axis.plot(
            week["times"], week["predicted_power"], color="#d62728", linestyle="--",
            linewidth=1.2, label="Predicted Power (15-min Ahead)",
        )
        axis.set_title(format_week_title(week))
        axis.set_ylim(0, maximum_power * 1.05 if maximum_power > 0 else 1)
        axis.xaxis.set_major_locator(mdates.DayLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.grid(True, color="#d0d0d0", linestyle=":", linewidth=0.7)
        axis.tick_params(axis="x", rotation=25)
        axis.margins(x=0)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.supxlabel("Date")
    figure.supylabel("Aggregated PV Power (kW)")
    figure.tight_layout(rect=(0.035, 0.035, 1, 0.94))

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    return figure


def main():
    weeks = load_weekly_power_series(INPUT_FILE, project_config.TARGET_TEST_PERIODS)
    figure = create_four_week_figure(weeks, OUTPUT_PNG, OUTPUT_PDF)
    plt.close(figure)

    print("目标域四周实际功率与预测功率图已生成。")
    print("PNG：", OUTPUT_PNG)
    print("PDF：", OUTPUT_PDF)


if __name__ == "__main__":
    main()
