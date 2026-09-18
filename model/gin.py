import torch
from torch import nn


class MeanGINLayer(nn.Module):
    """用邻居平均聚合的单层 GIN；输入和输出均为 [节点数, 特征维度]。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, node_features, adjacency):
        # 自身向量单独参与计算，所以邻居矩阵不保留原有自环。
        neighbors = (adjacency > 0).to(node_features.dtype)
        neighbors.fill_diagonal_(0)

        # 孤立节点没有邻居；分母至少为 1，此时邻居平均向量自然为 0。
        degrees = neighbors.sum(dim=1, keepdim=True).clamp_min(1)
        neighbor_means = neighbors @ node_features / degrees
        combined = (1 + self.epsilon) * node_features + neighbor_means
        return self.mlp(combined)
