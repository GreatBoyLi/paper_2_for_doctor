from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT_FILE = Path(__file__).resolve().parents[1] / "learning/node2vec/train_target_node_embeddings.py"


class TrainTargetNodeEmbeddingsTests(unittest.TestCase):
    def test_target_graph_trains_31_node_vectors(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"目标域训练脚本尚不存在：{SCRIPT_FILE}")
        result = subprocess.run([sys.executable, str(SCRIPT_FILE)], capture_output=True, text=True, check=True)

        self.assertIn("游走数量： 620", result.stdout)
        self.assertIn("节点向量形状： (31, 32)", result.stdout)
        self.assertIn("全部节点都有向量： True", result.stdout)
        self.assertIn("向量中无 NaN/Inf： True", result.stdout)
        stderr_lines = set(result.stderr.splitlines())
        self.assertLessEqual(stderr_lines, {
            "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'"
        })


if __name__ == "__main__":
    unittest.main()
