from pathlib import Path
import unittest

import torch
from torch import nn


MODEL_FILE = Path(__file__).resolve().parents[1] / "model" / "power_forecaster.py"
DEMO_FILE = Path(__file__).resolve().parents[1] / "learning" / "forecasting" / "demo_source_forward.py"


class PowerForecasterTests(unittest.TestCase):
    def test_forecast_head_converts_each_node_hidden_state_to_future_steps(self):
        self.assertTrue(MODEL_FILE.is_file(), f"预测模型尚不存在：{MODEL_FILE}")
        from model.power_forecaster import ForecastHead

        head = ForecastHead(hidden_dim=8, output_steps=16)
        predictions = head(torch.rand(2, 104, 8))

        self.assertEqual(predictions.shape, (2, 16, 104))

    def test_complete_model_predicts_and_backpropagates_through_all_modules(self):
        self.assertTrue(MODEL_FILE.is_file(), f"预测模型尚不存在：{MODEL_FILE}")
        from model.power_forecaster import GraphPowerForecaster

        model = GraphPowerForecaster(spatial_dim=4, hidden_dim=8, output_steps=3)
        history = torch.rand(2, 5, 3)
        node_vectors = torch.rand(3, 4, requires_grad=True)
        adjacency = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
        targets = torch.rand(2, 3, 3)

        predictions = model(history, node_vectors, adjacency)
        loss = nn.functional.l1_loss(predictions, targets)
        loss.backward()

        self.assertEqual(predictions.shape, targets.shape)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(node_vectors.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_source_demo_predicts_future_power_and_backpropagates(self):
        import subprocess
        import sys

        self.assertTrue(DEMO_FILE.is_file(), f"Source 前向演示尚不存在：{DEMO_FILE}")
        result = subprocess.run([sys.executable, str(DEMO_FILE)], capture_output=True, text=True, check=True)

        self.assertIn("历史功率： (2, 16, 104)", result.stdout)
        self.assertIn("真实未来功率： (2, 16, 104)", result.stdout)
        self.assertIn("预测未来功率： (2, 16, 104)", result.stdout)
        self.assertIn("全部模型参数都有梯度： True", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
