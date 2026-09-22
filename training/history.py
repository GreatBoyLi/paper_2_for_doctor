import pandas as pd


def horizon_label(step):
    minutes = step * 15
    return f"{minutes}min" if minutes < 60 else f"{minutes // 60}h"


def add_horizon_metrics(row, horizon_mae, prefix="val_mae"):
    for step, value in horizon_mae.items():
        row[f"{prefix}_{horizon_label(step)}"] = value
    return row


def save_training_history(file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(file, index=False)
