import torch
from torch import nn


class SpatialFeatureGRU(nn.Module):
    """按论文式 (2)–(5) 融合历史功率和 GIN 特征，返回每个站点的最终隐藏状态。"""

    def __init__(self, spatial_dim, hidden_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.reset_gate = nn.Linear(1 + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(1 + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(1 + hidden_dim, hidden_dim)
        self.spatial_fusion = nn.Linear(spatial_dim + hidden_dim, hidden_dim)

    def forward(self, power_history, spatial_features):
        # power_history: [样本数, 历史步数, 节点数]；spatial_features: [节点数, 空间维度]。
        batch_size, steps, nodes = power_history.shape
        if spatial_features.shape != (nodes, self.spatial_dim):
            raise ValueError("GIN 特征形状与历史功率的节点数或空间维度不一致。")

        hidden = power_history.new_zeros(batch_size, nodes, self.hidden_dim)
        spatial = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)

        for step in range(steps):
            power = power_history[:, step, :].unsqueeze(-1)
            gate_input = torch.cat((power, hidden), dim=-1)
            reset = torch.sigmoid(self.reset_gate(gate_input))
            update = torch.sigmoid(self.update_gate(gate_input))
            candidate_input = torch.cat((power, reset * hidden), dim=-1)
            candidate = torch.tanh(self.candidate(candidate_input))

            # 先完成 GRU 更新，再拼入当前站点的 GIN 空间特征。
            gru_hidden = (1 - update) * candidate + update * hidden
            hidden = self.spatial_fusion(torch.cat((spatial, gru_hidden), dim=-1))

        return hidden
