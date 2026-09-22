import importlib.util
from pathlib import Path
import unittest

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
LOADER_FILE = CONFIG_DIR / "config.py"
PACKAGE_FILE = CONFIG_DIR / "__init__.py"


def load_config_module():
    assert LOADER_FILE.is_file(), f"配置读取文件尚不存在：{LOADER_FILE}"
    spec = importlib.util.spec_from_file_location("node2vec_learning_config", LOADER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectConfigTests(unittest.TestCase):
    def test_config_files_are_in_project_level_package(self):
        self.assertTrue(CONFIG_FILE.is_file(), f"项目配置文件尚不存在：{CONFIG_FILE}")
        self.assertTrue(LOADER_FILE.is_file(), f"配置读取文件尚不存在：{LOADER_FILE}")
        self.assertTrue(PACKAGE_FILE.is_file(), f"配置包入口尚不存在：{PACKAGE_FILE}")
        self.assertFalse((PROJECT_DIR / "config.yaml").exists())
        self.assertFalse((PROJECT_DIR / "learning/node2vec/config.py").exists())

    def test_loader_resolves_paths_and_experiment_parameters(self):
        config = load_config_module()
        with CONFIG_FILE.open(encoding="utf-8") as file:
            settings = yaml.safe_load(file)

        self.assertEqual(config.PROJECT_DIR, PROJECT_DIR)
        self.assertEqual(config.DATASET_DIR, PROJECT_DIR / settings["paths"]["source_model_dataset"])
        self.assertEqual(config.GRAPH_DIR, PROJECT_DIR / settings["paths"]["source_graph"])
        self.assertEqual(config.TARGET_DATASET_DIR, PROJECT_DIR / settings["paths"]["target_model_dataset"])
        self.assertEqual(config.TARGET_GRAPH_DIR, PROJECT_DIR / settings["paths"]["target_graph"])
        self.assertNotIn("experiment", settings)
        self.assertFalse(hasattr(config, "FOLD_ID"))
        self.assertEqual(config.START_NODE, settings["demo"]["start_node"])
        self.assertEqual(config.WALK_LENGTH, settings["node2vec"]["walk_length"])
        self.assertEqual(config.NUM_WALKS, settings["node2vec"]["num_walks"])
        self.assertEqual(config.EMBEDDING_DIM, settings["node2vec"]["embedding_dim"])
        self.assertEqual(config.WINDOW_SIZE, settings["node2vec"]["window_size"])
        self.assertEqual(config.EPOCHS, settings["node2vec"]["epochs"])
        self.assertEqual(config.SEED, settings["node2vec"]["seed"])
        self.assertTrue(hasattr(config, "P"))
        self.assertTrue(hasattr(config, "Q"))
        self.assertEqual(config.P, settings["node2vec"]["p"])
        self.assertEqual(config.Q, settings["node2vec"]["q"])
        self.assertEqual(config.HIDDEN_DIM, settings["forecast_model"]["hidden_dim"])
        self.assertEqual(config.OUTPUT_STEPS, settings["forecast_model"]["output_steps"])
        self.assertEqual(config.BATCH_SIZE, settings["training"]["batch_size"])
        self.assertEqual(config.LEARNING_RATE, settings["training"]["learning_rate"])
        self.assertEqual(config.TRAINING_EPOCHS, settings["training"]["epochs"])
        self.assertEqual(
            config.EARLY_STOPPING_PATIENCE, settings["training"]["early_stopping_patience"]
        )
        self.assertEqual(
            config.EARLY_STOPPING_MIN_DELTA, settings["training"]["early_stopping_min_delta"]
        )
        self.assertEqual(config.CHECKPOINT_DIR, PROJECT_DIR / settings["paths"]["checkpoint_dir"])
        self.assertEqual(
            config.TRAINING_HISTORY_DIR, PROJECT_DIR / settings["paths"]["training_history_dir"]
        )
        self.assertEqual(config.DOMAIN_HIDDEN_DIM, settings["domain_adversarial"]["hidden_dim"])
        self.assertEqual(config.GRL_LAMBDA, settings["domain_adversarial"]["grl_lambda"])

    def test_training_uses_long_run_with_early_stopping(self):
        config = load_config_module()

        self.assertEqual(config.TRAINING_EPOCHS, 100)
        self.assertEqual(config.EARLY_STOPPING_PATIENCE, 10)
        self.assertEqual(config.EARLY_STOPPING_MIN_DELTA, 0.0001)


if __name__ == "__main__":
    unittest.main()
