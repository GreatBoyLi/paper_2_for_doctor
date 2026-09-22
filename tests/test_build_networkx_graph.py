import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_FILE = Path(__file__).resolve().parents[1] / "learning" / "node2vec" / "build_networkx_graph.py"
INSPECT_FILE = SCRIPT_FILE.with_name("inspect_graph.py")


def load_graph_module():
    assert SCRIPT_FILE.exists(), f"建图脚本尚不存在：{SCRIPT_FILE}"
    sys.path.insert(0, str(SCRIPT_FILE.parent))
    try:
        spec = importlib.util.spec_from_file_location("build_networkx_graph", SCRIPT_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def configured_graph_sizes():
    config = load_graph_module().project_config
    station_count = len(pd.read_csv(config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"))
    adj = np.load(config.GRAPH_DIR / "adjacency_binary.npy")
    edge_count = (np.count_nonzero(adj) - np.count_nonzero(np.diag(adj))) // 2
    return station_count, edge_count


class BuildNetworkxGraphTests(unittest.TestCase):
    def test_preserves_nodes_and_adds_each_undirected_edge_once(self):
        names = ["A", "B", "C", "D"]
        adj = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        graph = load_graph_module().build_networkx_graph(names, adj)

        self.assertEqual(list(graph.nodes), names)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertEqual(list(graph.neighbors("D")), [])
        self.assertEqual(list(nx.selfloop_edges(graph)), [])

    def test_rejects_non_symmetric_adjacency(self):
        adj = np.array([[0, 1], [0, 0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "对称"):
            load_graph_module().build_networkx_graph(["A", "B"], adj)

    def test_rejects_duplicate_node_names(self):
        adj = np.eye(2, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "重复"):
            load_graph_module().build_networkx_graph(["A", "A"], adj)

    def test_script_prints_configured_graph_summary(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"建图脚本尚不存在：{SCRIPT_FILE}")
        result = subprocess.run([sys.executable, str(SCRIPT_FILE)], capture_output=True, text=True, check=True)
        station_count, edge_count = configured_graph_sizes()

        self.assertIn(f"节点数量： {station_count}", result.stdout)
        self.assertIn(f"矩阵无向边数量： {edge_count}", result.stdout)
        self.assertIn(f"无向边数量： {edge_count}", result.stdout)
        self.assertIn("自环数量： 0", result.stdout)

    def test_inspect_script_summarizes_configured_graph(self):
        self.assertTrue(INSPECT_FILE.is_file(), f"检查脚本尚不存在：{INSPECT_FILE}")
        result = subprocess.run([sys.executable, str(INSPECT_FILE)], capture_output=True, text=True, check=True)
        station_count, edge_count = configured_graph_sizes()

        self.assertIn(f"节点数量： {station_count}", result.stdout)
        self.assertIn(f"无向边数量（不含自环）： {edge_count}", result.stdout)


if __name__ == "__main__":
    unittest.main()
