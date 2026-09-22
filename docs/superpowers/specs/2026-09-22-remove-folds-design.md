# 源域与目标域取消 Fold 设计

## 目标

取消源域和目标域的 `fold_1` ～ `fold_4` 数据划分及路径。每个数据域只保留一套训练集和验证集，数据集文件和图文件直接保存在各自目录根部。

本次不制作目标域最终测试集。最终测试集将在确定 2021-03-14 之后的连续完整区间后单独生成。

## 源域划分

源域使用 2014 年全年数据。每个月单独按完整日期划分：

- 训练天数为 `floor(当月天数 × 0.8)`。
- 当月前面的日期用于训练。
- 当月剩余日期用于验证。

例如，31 天的月份使用 1 ～ 24 日训练，25 ～ 31 日验证。

每个月的训练段和验证段分别生成滑动窗口，然后再合并全年样本。任何样本都不得跨越月份边界或训练、验证边界。

## 目标域划分

目标域当前使用 2021-03-07 ～ 2021-03-13 的连续 7 天：

- 2021-03-07 ～ 2021-03-11：训练集，共 5 天。
- 2021-03-12 ～ 2021-03-13：验证集，共 2 天。

Q99 只使用目标域训练集计算，然后用同一组 Q99 对训练集和验证集归一化。训练段和验证段分别生成滑动窗口。

## 输出结构

源域 `model_dataset` 根目录直接保存：

- `SOURCE_STATION_ORDER.csv`
- `SOURCE_SPLIT_SUMMARY.csv`
- `train.npz`
- `val.npz`
- `train_timeseries.csv`
- `val_timeseries.csv`

目标域 `model_dataset` 根目录直接保存：

- `TARGET_STATION_ORDER.csv`
- `TARGET_SPLIT_SUMMARY.csv`
- `q99_per_station.csv`
- `train.npz`
- `val.npz`
- `train_timeseries_raw.csv`
- `val_timeseries_raw.csv`
- `train_timeseries_normalized.csv`
- `val_timeseries_normalized.csv`

源域和目标域的图文件也直接保存在各自 `graph` 根目录。构图只能使用对应数据域的训练时间序列，验证数据不参与相关系数统计。

## 下游代码调整

- 从 `config.yaml` 和 `config.py` 删除 `FOLD_ID`。
- Node2Vec 学习和演示脚本改为直接读取图目录根部。
- Stage 1 直接读取源域根目录的 `train.npz` 和 `val.npz`。
- Stage 2 直接读取目标域根目录的 `train.npz` 和 `val.npz`。
- 检查点改为 `source_best.pt`、`stage1_best.pt` 和 `stage2_best.pt`。
- 训练历史改为 `source_history.csv`、`stage1_history.csv` 和 `stage2_history.csv`。
- 当前评估脚本继续评估目标域验证集，输出目录改为 `target_validation`，不将其冒充为最终测试结果。

## 旧文件处理

新代码不再读取 `fold_1` ～ `fold_4` 目录。本次不自动删除旧数据、旧图、旧检查点和旧评估结果，避免不可恢复的数据丢失。新流程验证通过后再由用户决定是否清理。

## 验证要求

- 源域和目标域的训练、验证日期不重叠。
- 滑动窗口不跨越数据划分边界。
- 目标域 Q99 与训练集重新计算的结果一致。
- 构图脚本只读取训练时间序列。
- 配置、学习脚本、训练脚本和评估脚本中不再存在 Fold 路径依赖。
- 现有与路径、数据形状、训练和评估相关的自动检查需同步更新并通过。
