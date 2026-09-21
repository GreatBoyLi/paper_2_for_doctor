from torch import nn

from model.gin import MeanGINLayer
from model.temporal_gru import SpatialFeatureGRU


class ForecastHead(nn.Module):
    """把每个站点的最终隐藏状态转换为未来多步功率。"""

    def __init__(self, hidden_dim, output_steps):
        super().__init__()
        self.output_layer = nn.Linear(hidden_dim, output_steps)

    def forward(self, hidden):
        # [样本, 节点, 隐藏维度] -> [样本, 未来步数, 节点]
        return self.output_layer(hidden).transpose(1, 2)


class GraphPowerForecaster(nn.Module):
    """依次使用 GIN、GRU 和预测头生成每个站点的未来功率。"""

    def __init__(self, spatial_dim, hidden_dim, output_steps):
        super().__init__()
        self.spatial_encoder = MeanGINLayer(spatial_dim)
        self.temporal_regression = SpatialFeatureGRU(spatial_dim, hidden_dim)
        self.forecast_head = ForecastHead(hidden_dim, output_steps)

    def forward(self, power_history, node_vectors, adjacency):
        spatial_features = self.spatial_encoder(node_vectors, adjacency)
        hidden = self.temporal_regression(power_history, spatial_features)
        return self.forecast_head(hidden)
