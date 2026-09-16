from pathlib import Path

import yaml


# config/ 位于项目根目录，YAML 中的数据路径相对于项目根目录。
PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(__file__).with_name("config.yaml")

with CONFIG_FILE.open(encoding="utf-8") as file:
    settings = yaml.safe_load(file)

DATASET_DIR = PROJECT_DIR / settings["paths"]["source_model_dataset"]
GRAPH_DIR = PROJECT_DIR / settings["paths"]["source_graph"]
FOLD_ID = settings["experiment"]["fold_id"]
WALK_LENGTH = settings["node2vec"]["walk_length"]
SEED = settings["node2vec"]["seed"]
START_NODE = settings["demo"]["start_node"]
