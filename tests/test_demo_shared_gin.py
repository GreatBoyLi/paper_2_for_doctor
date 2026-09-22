from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


SCRIPT_FILE = Path(__file__).resolve().parents[1] / "learning" / "gin" / "demo_shared_gin.py"


class SharedGinDemoTests(unittest.TestCase):
    def test_mean_gin_uses_neighbors_but_not_self_loops(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"GIN 演示脚本尚不存在：{SCRIPT_FILE}")
        from learning.gin.demo_shared_gin import mean_gin_layer

        features = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float32)
        adjacency = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = mean_gin_layer(features, adjacency, np.eye(2), np.zeros(2))

        np.testing.assert_allclose(result, [[1, 2], [1, 2], [3, 4]])
        self.assertEqual(adjacency[0, 0], 1)

    def test_same_layer_processes_both_domains(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"GIN 演示脚本尚不存在：{SCRIPT_FILE}")
        result = subprocess.run([sys.executable, str(SCRIPT_FILE)], capture_output=True, text=True, check=True)

        self.assertIn("GIN 参数： (32, 32)（Source/Target 共用）", result.stdout)
        self.assertIn("Source 输入： (104, 32) 输出： (104, 32)", result.stdout)
        self.assertIn("Target 输入： (31, 32) 输出： (31, 32)", result.stdout)
        stderr_lines = set(result.stderr.splitlines())
        self.assertLessEqual(stderr_lines, {
            "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'"
        })


if __name__ == "__main__":
    unittest.main()
