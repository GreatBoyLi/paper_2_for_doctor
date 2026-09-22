from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch import nn


MODEL_FILE = Path(__file__).resolve().parents[1] / "model" / "domain_adversarial.py"
DEMO_FILE = Path(__file__).resolve().parents[1] / "learning" / "domain_adversarial" / "demo_domain_classifier.py"


class DomainAdversarialTests(unittest.TestCase):
    def test_gradient_reversal_keeps_forward_value_and_reverses_backward_gradient(self):
        self.assertTrue(MODEL_FILE.is_file(), f"域对抗模型尚不存在：{MODEL_FILE}")
        from model.domain_adversarial import gradient_reverse

        features = torch.tensor([1., 2., 3.], requires_grad=True)
        output = gradient_reverse(features, grl_lambda=0.5)
        output.sum().backward()

        torch.testing.assert_close(output, features.detach())
        torch.testing.assert_close(features.grad, torch.full_like(features, -0.5))

    def test_domain_classifier_predicts_two_domains_and_backpropagates(self):
        self.assertTrue(MODEL_FILE.is_file(), f"域对抗模型尚不存在：{MODEL_FILE}")
        from model.domain_adversarial import DomainClassifier

        classifier = DomainClassifier(feature_dim=4, hidden_dim=6)
        features = torch.rand(5, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1])
        logits = classifier(features, grl_lambda=1.0)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        self.assertEqual(logits.shape, (5, 2))
        self.assertIsNotNone(features.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in classifier.parameters()))

    def test_demo_uses_source_and_target_with_one_shared_gin(self):
        self.assertTrue(DEMO_FILE.is_file(), f"域分类演示尚不存在：{DEMO_FILE}")
        result = subprocess.run([sys.executable, str(DEMO_FILE)], capture_output=True, text=True, check=True)

        self.assertIn("Source 空间特征： (104, 32)", result.stdout)
        self.assertIn("Target 空间特征： (31, 32)", result.stdout)
        self.assertIn("域分类输出： (135, 2)", result.stdout)
        self.assertIn("共享 GIN 参数有梯度： True", result.stdout)
        self.assertIn("域分类器参数有梯度： True", result.stdout)
        stderr_lines = set(result.stderr.splitlines())
        self.assertLessEqual(stderr_lines, {
            "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'"
        })


if __name__ == "__main__":
    unittest.main()
