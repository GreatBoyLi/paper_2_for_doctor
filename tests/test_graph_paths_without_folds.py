from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from graph.graph_utils import build_topk_graph


PROJECT_DIR = Path(__file__).resolve().parents[1]


class GraphPathsWithoutFoldsTests(unittest.TestCase):
    def test_topk_graph_is_symmetric_and_keeps_self_loops(self):
        correlation = pd.DataFrame(
            [[1.0, 0.9, 0.2], [0.9, 1.0, 0.8], [0.2, 0.8, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )

        weighted, binary, directed = build_topk_graph(correlation, top_k=1)

        np.testing.assert_array_equal(binary, binary.T)
        np.testing.assert_array_equal(np.diag(binary), np.ones(3))
        self.assertEqual(int((binary.sum() - 3) / 2), 2)
        self.assertEqual(directed.shape, (3, 3))

    def test_graph_builders_use_root_training_timeseries(self):
        source = (PROJECT_DIR / "graph/build_source_graph.py").read_text(encoding="utf-8")
        target = (PROJECT_DIR / "graph/build_target_graph.py").read_text(encoding="utf-8")

        self.assertNotIn("FOLDS", source)
        self.assertNotIn("fold_output_dir", source)
        self.assertIn('DATASET_DIR / "train_timeseries.csv"', source)
        self.assertNotIn("FOLDS", target)
        self.assertNotIn("fold_output_dir", target)
        self.assertIn('DATASET_DIR / "train_timeseries_normalized.csv"', target)


if __name__ == "__main__":
    unittest.main()
