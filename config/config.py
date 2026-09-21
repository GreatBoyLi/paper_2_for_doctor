from pathlib import Path

import yaml


# config/ 位于项目根目录，YAML 中的数据路径相对于项目根目录。
PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(__file__).with_name("config.yaml")

with CONFIG_FILE.open(encoding="utf-8") as file:
    settings = yaml.safe_load(file)

DATASET_DIR = PROJECT_DIR / settings["paths"]["source_model_dataset"]
GRAPH_DIR = PROJECT_DIR / settings["paths"]["source_graph"]
TARGET_DATASET_DIR = PROJECT_DIR / settings["paths"]["target_model_dataset"]
TARGET_GRAPH_DIR = PROJECT_DIR / settings["paths"]["target_graph"]
FOLD_ID = settings["experiment"]["fold_id"]
EMBEDDING_DIM = settings["node2vec"]["embedding_dim"]
WALK_LENGTH = settings["node2vec"]["walk_length"]
NUM_WALKS = settings["node2vec"]["num_walks"]
WINDOW_SIZE = settings["node2vec"]["window_size"]
EPOCHS = settings["node2vec"]["epochs"]
SEED = settings["node2vec"]["seed"]
P = settings["node2vec"]["p"]
Q = settings["node2vec"]["q"]
HIDDEN_DIM = settings["forecast_model"]["hidden_dim"]
OUTPUT_STEPS = settings["forecast_model"]["output_steps"]
START_NODE = settings["demo"]["start_node"]
