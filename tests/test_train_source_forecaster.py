from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_FILE = Path(__file__).resolve().parents[1] / "training" / "train_source_forecaster.py"
DEVICE_FILE = Path(__file__).resolve().parents[1] / "training" / "device.py"


class TrainSourceForecasterTests(unittest.TestCase):
    def test_training_entry_uses_non_fold_paths(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_ID", text)
        self.assertNotIn("fold_", text)
        self.assertIn('"source_best.pt"', text)

    def test_device_priority_is_cuda_then_mps_then_cpu(self):
        from unittest.mock import patch

        self.assertTrue(DEVICE_FILE.is_file(), f"设备选择模块尚不存在：{DEVICE_FILE}")
        from training.device import describe_device, select_device

        with patch("training.device.torch.cuda.is_available", return_value=True), \
                patch("training.device.torch.backends.mps.is_available", return_value=True), \
                patch("training.device.torch.cuda.get_device_name", return_value="NVIDIA Test GPU"):
            device = select_device()
            self.assertEqual(device.type, "cuda")
            self.assertEqual(describe_device(device), "cuda（NVIDIA Test GPU）")
        with patch("training.device.torch.cuda.is_available", return_value=False), \
                patch("training.device.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_device().type, "mps")
        with patch("training.device.torch.cuda.is_available", return_value=False), \
                patch("training.device.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device().type, "cpu")

    def test_training_updates_parameters_and_validation_does_not(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Source 训练脚本尚不存在：{SCRIPT_FILE}")
        from model.power_forecaster import GraphPowerForecaster
        from training.train_source_forecaster import evaluate, train_one_epoch

        torch.manual_seed(42)
        model = GraphPowerForecaster(spatial_dim=4, hidden_dim=6, output_steps=3)
        loader = DataLoader(TensorDataset(torch.rand(8, 5, 3), torch.rand(8, 3, 3)), batch_size=4)
        node_vectors = torch.rand(3, 4)
        adjacency = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before_training = [parameter.detach().clone() for parameter in model.parameters()]

        train_loss = train_one_epoch(model, loader, node_vectors, adjacency, optimizer)
        after_training = [parameter.detach().clone() for parameter in model.parameters()]
        val_loss = evaluate(model, loader, node_vectors, adjacency)
        after_validation = [parameter.detach().clone() for parameter in model.parameters()]

        self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_training, after_training)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(after_training, after_validation)))

    def test_validation_reports_mae_for_selected_horizons(self):
        from model.power_forecaster import GraphPowerForecaster
        from training.train_source_forecaster import evaluate_with_horizons

        torch.manual_seed(42)
        model = GraphPowerForecaster(spatial_dim=4, hidden_dim=6, output_steps=4)
        histories = torch.rand(6, 5, 3)
        targets = torch.rand(6, 4, 3)
        loader = DataLoader(TensorDataset(histories, targets), batch_size=2)
        node_vectors = torch.rand(3, 4)
        adjacency = torch.eye(3)

        val_loss, horizon_mae = evaluate_with_horizons(
            model, loader, node_vectors, adjacency, [1, 4]
        )
        with torch.no_grad():
            predictions = model(histories, node_vectors, adjacency)

        self.assertAlmostEqual(
            val_loss, torch.nn.functional.l1_loss(predictions, targets).item(), places=6
        )
        self.assertAlmostEqual(
            horizon_mae[1],
            torch.nn.functional.l1_loss(predictions[:, 0], targets[:, 0]).item(), places=6,
        )
        self.assertAlmostEqual(
            horizon_mae[4],
            torch.nn.functional.l1_loss(predictions[:, 3], targets[:, 3]).item(), places=6,
        )

    def test_checkpoint_keeps_model_inputs_and_station_order(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Source 训练脚本尚不存在：{SCRIPT_FILE}")
        from model.power_forecaster import GraphPowerForecaster
        from training.train_source_forecaster import save_checkpoint

        model = GraphPowerForecaster(spatial_dim=4, hidden_dim=6, output_steps=3)
        node_vectors = torch.rand(3, 4)
        adjacency = torch.eye(3)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_file = Path(directory) / "source.pt"
            save_checkpoint(checkpoint_file, model, node_vectors, adjacency, ["A", "B", "C"], 2, 0.25)
            checkpoint = torch.load(checkpoint_file, weights_only=False)

        self.assertEqual(checkpoint["station_names"], ["A", "B", "C"])
        torch.testing.assert_close(checkpoint["node_vectors"], node_vectors)
        torch.testing.assert_close(checkpoint["adjacency"], adjacency)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["val_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
