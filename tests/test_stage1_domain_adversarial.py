from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


MODEL_FILE = Path(__file__).resolve().parents[1] / "model" / "stage1_domain_adversarial.py"
TRAIN_FILE = Path(__file__).resolve().parents[1] / "training" / "train_stage1_domain_adversarial.py"


def small_domain_inputs():
    source_vectors = torch.rand(3, 4)
    source_adjacency = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
    target_vectors = torch.rand(2, 4)
    target_adjacency = torch.ones(2, 2)
    return source_vectors, source_adjacency, target_vectors, target_adjacency


class Stage1DomainAdversarialTests(unittest.TestCase):
    def test_training_entry_uses_non_fold_paths(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_ID", text)
        self.assertNotIn("fold_", text)
        self.assertIn('"stage1_best.pt"', text)

    def test_model_returns_source_forecast_and_both_domain_logits(self):
        self.assertTrue(MODEL_FILE.is_file(), f"Stage 1模型尚不存在：{MODEL_FILE}")
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel

        model = Stage1DomainAdversarialModel(4, 6, 3, 5)
        inputs = small_domain_inputs()
        predictions, domain_logits = model(torch.rand(2, 5, 3), *inputs, grl_lambda=1.0)
        labels = torch.tensor([0, 0, 0, 1, 1])
        loss = torch.nn.functional.l1_loss(predictions, torch.rand(2, 3, 3))
        loss = loss + torch.nn.functional.cross_entropy(domain_logits, labels)
        loss.backward()

        self.assertEqual(predictions.shape, (2, 3, 3))
        self.assertEqual(domain_logits.shape, (5, 2))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_joint_training_updates_one_model_with_both_losses(self):
        self.assertTrue(TRAIN_FILE.is_file(), f"Stage 1训练脚本尚不存在：{TRAIN_FILE}")
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
        from training.train_stage1_domain_adversarial import train_one_epoch

        torch.manual_seed(42)
        model = Stage1DomainAdversarialModel(4, 6, 3, 5)
        loader = DataLoader(TensorDataset(torch.rand(8, 5, 3), torch.rand(8, 3, 3)), batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        metrics = train_one_epoch(model, loader, *small_domain_inputs(), optimizer, grl_lambda=1.0)
        after = [parameter.detach().clone() for parameter in model.parameters()]

        self.assertEqual(set(metrics), {"forecast_loss", "domain_loss", "domain_accuracy", "total_loss"})
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_validation_reports_selected_forecast_horizons(self):
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
        from training.train_stage1_domain_adversarial import evaluate

        model = Stage1DomainAdversarialModel(4, 6, 3, 5)
        loader = DataLoader(TensorDataset(torch.rand(8, 5, 3), torch.rand(8, 3, 3)), batch_size=4)

        metrics = evaluate(
            model, loader, *small_domain_inputs(), grl_lambda=1.0, horizon_steps=[1, 3]
        )

        self.assertEqual(set(metrics["horizon_mae"]), {1, 3})
        self.assertTrue(all(
            torch.isfinite(torch.tensor(value)) for value in metrics["horizon_mae"].values()
        ))

    def test_checkpoint_contains_both_domains_and_stage1_model(self):
        self.assertTrue(TRAIN_FILE.is_file(), f"Stage 1训练脚本尚不存在：{TRAIN_FILE}")
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
        from training.train_stage1_domain_adversarial import save_checkpoint

        model = Stage1DomainAdversarialModel(4, 6, 3, 5)
        source_vectors, source_adjacency, target_vectors, target_adjacency = small_domain_inputs()

        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "stage1.pt"
            save_checkpoint(file, model, source_vectors, source_adjacency, ["S1", "S2", "S3"],
                            target_vectors, target_adjacency, ["T1", "T2"], 2, 0.25)
            checkpoint = torch.load(file, weights_only=False)

        self.assertEqual(checkpoint["source_station_names"], ["S1", "S2", "S3"])
        self.assertEqual(checkpoint["target_station_names"], ["T1", "T2"])
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["val_forecast_loss"], 0.25)
        self.assertIn("model_state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main()
