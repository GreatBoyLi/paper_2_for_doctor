from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


MODEL_FILE = Path(__file__).resolve().parents[1] / "model" / "stage2_target_finetuning.py"
TRAIN_FILE = Path(__file__).resolve().parents[1] / "training" / "train_stage2_target_finetuning.py"


class Stage2TargetFineTuningTests(unittest.TestCase):
    def test_training_entry_uses_non_fold_paths(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_ID", text)
        self.assertNotIn("fold_", text)
        self.assertIn('"stage2_best.pt"', text)

    def test_stage1_parameters_initialize_both_target_encoders_and_forecaster(self):
        self.assertTrue(MODEL_FILE.is_file(), f"Stage 2模型尚不存在：{MODEL_FILE}")
        self.assertTrue(TRAIN_FILE.is_file(), f"Stage 2训练脚本尚不存在：{TRAIN_FILE}")
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
        from model.stage2_target_finetuning import Stage2TargetFineTuningModel
        from training.train_stage2_target_finetuning import initialize_from_stage1

        stage1 = Stage1DomainAdversarialModel(4, 6, 3, 5)
        stage2 = Stage2TargetFineTuningModel(4, 6, 3)
        initialize_from_stage1(stage2, stage1.state_dict())

        for stage1_parameter, shared_parameter, private_parameter in zip(
                stage1.spatial_encoder.parameters(), stage2.shared_spatial_encoder.parameters(),
                stage2.private_spatial_encoder.parameters()):
            torch.testing.assert_close(shared_parameter, stage1_parameter)
            torch.testing.assert_close(private_parameter, stage1_parameter)
        for stage1_parameter, stage2_parameter in zip(
                stage1.temporal_regression.parameters(), stage2.temporal_regression.parameters()):
            torch.testing.assert_close(stage2_parameter, stage1_parameter)
        for stage1_parameter, stage2_parameter in zip(
                stage1.forecast_head.parameters(), stage2.forecast_head.parameters()):
            torch.testing.assert_close(stage2_parameter, stage1_parameter)

    def test_fused_target_model_predicts_and_all_parameters_receive_gradients(self):
        self.assertTrue(MODEL_FILE.is_file(), f"Stage 2模型尚不存在：{MODEL_FILE}")
        from model.stage2_target_finetuning import Stage2TargetFineTuningModel

        model = Stage2TargetFineTuningModel(4, 6, 3)
        predictions = model(
            torch.rand(2, 5, 3), torch.rand(3, 4),
            torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]]),
        )
        torch.nn.functional.l1_loss(predictions, torch.rand(2, 3, 3)).backward()

        self.assertEqual(predictions.shape, (2, 3, 3))
        self.assertTrue(all(parameter.grad is None for parameter in model.shared_spatial_encoder.parameters()))
        self.assertTrue(all(
            parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
        ))

    def test_target_training_updates_model_and_checkpoint_keeps_target_inputs(self):
        self.assertTrue(TRAIN_FILE.is_file(), f"Stage 2训练脚本尚不存在：{TRAIN_FILE}")
        from model.stage1_domain_adversarial import Stage1DomainAdversarialModel
        from model.stage2_target_finetuning import Stage2TargetFineTuningModel
        from training.train_source_forecaster import train_one_epoch
        from training.train_stage2_target_finetuning import initialize_from_stage1, save_checkpoint

        stage1 = Stage1DomainAdversarialModel(4, 6, 3, 5)
        model = Stage2TargetFineTuningModel(4, 6, 3)
        initialize_from_stage1(model, stage1.state_dict())
        loader = DataLoader(TensorDataset(torch.rand(8, 5, 3), torch.rand(8, 3, 3)), batch_size=4)
        node_vectors = torch.rand(3, 4)
        adjacency = torch.eye(3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        shared_before = [parameter.detach().clone() for parameter in model.shared_spatial_encoder.parameters()]
        private_before = [parameter.detach().clone() for parameter in model.private_spatial_encoder.parameters()]

        loss = train_one_epoch(model, loader, node_vectors, adjacency, optimizer)
        shared_after = [parameter.detach().clone() for parameter in model.shared_spatial_encoder.parameters()]
        private_after = [parameter.detach().clone() for parameter in model.private_spatial_encoder.parameters()]

        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "stage2.pt"
            save_checkpoint(file, model, node_vectors, adjacency, ["T1", "T2", "T3"], 2, 0.25)
            checkpoint = torch.load(file, weights_only=False)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(shared_before, shared_after)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(private_before, private_after)))
        self.assertEqual(checkpoint["target_station_names"], ["T1", "T2", "T3"])
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["val_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
