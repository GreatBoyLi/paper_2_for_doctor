from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 直接运行本文件时，把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config as project_config

# ============================================================
# 1. 读取配置指定的节点顺序
# ============================================================

station_order_file = project_config.DATASET_DIR / "SOURCE_STATION_ORDER.csv"

station_df = pd.read_csv(station_order_file)
station_names = station_df["NodeID"].astype(str).tolist()

print("节点数量：", len(station_names))
print("前10个节点：", station_names[:10])

# ============================================================
# 2. 读取 Binary 邻接矩阵
# ============================================================

adjacency_file = project_config.GRAPH_DIR / "adjacency_binary.npy"

adj = np.load(adjacency_file)

print("\n邻接矩阵形状：", adj.shape)
print("数据类型：", adj.dtype)

# ============================================================
# 3. 检查邻接矩阵尺寸
#
# 节点数量应该等于：
#
# adjacency 行数
# adjacency 列数
# ============================================================

n = len(station_names)

if adj.shape != (n, n):
    raise RuntimeError(
        f"邻接矩阵尺寸异常：{adj.shape}，但节点数量是 {n}"
    )

print("邻接矩阵尺寸检查：通过")

# ============================================================
# 4. 检查邻接矩阵是否对称
#
# 之前构建的是无向图，因此应该满足：
#
# A_ij = A_ji
# ============================================================

symmetric = np.allclose(adj, adj.T)

print("是否对称：", symmetric)

if not symmetric:
    raise RuntimeError("邻接矩阵不是对称矩阵。")

# ============================================================
# 5. 检查自环
#
# 当前 adjacency_binary.npy 中：
#
# A_ii = 1
#
# 所以理论上应该有104个自环。
# ============================================================

diagonal = np.diag(adj)
self_loop_count = int(np.sum(diagonal > 0))

print("自环数量：", self_loop_count)

# ============================================================
# 6. 去掉自环
#
# Node2Vec进行随机游走时，
# 我们不打算把节点自己的自环作为普通边。
# ============================================================

adj_no_self = adj.copy()
np.fill_diagonal(adj_no_self, 0)

# ============================================================
# 7. 统计无向边数量
#
# 无向图中一条边：
#
# A -- B
#
# 在邻接矩阵里会出现两次：
#
# A_ab = 1
# A_ba = 1
#
# 因此总和需要除以2。
# ============================================================

edge_count = int(adj_no_self.sum() / 2)

print("无向边数量（不含自环）：", edge_count)

# ============================================================
# 8. 查看前5个节点的邻居
#
# np.where(adj_no_self[i] > 0)[0]
#
# 找到第i个节点这一行中所有值为1的位置。
#
# 这些位置就是邻居节点的索引。
# ============================================================

print("\n前5个节点的邻居：")

for i in range(5):
    neighbor_indices = np.where(adj_no_self[i] > 0)[0]
    neighbors = [station_names[j] for j in neighbor_indices]

    print(station_names[i], "->", neighbors)
