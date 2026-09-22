import torch
from torch import nn

from model.gin import MeanGINLayer
from model.power_forecaster import ForecastHead
from model.temporal_gru import SpatialFeatureGRU


class Stage2TargetFineTuningModel(nn.Module):
    """融合共享与Target私有空间特征，再预测Target未来功率。"""

    def __init__(self, spatial_dim, hidden_dim, output_steps):
        super().__init__()
        self.shared_spatial_encoder = MeanGINLayer(spatial_dim)
        self.private_spatial_encoder = MeanGINLayer(spatial_dim)
        for parameter in self.shared_spatial_encoder.parameters():
            parameter.requires_grad = False
        self.private_weight = nn.Parameter(torch.tensor(0.5))
        self.shared_weight = nn.Parameter(torch.tensor(0.5))
        self.fusion_bias = nn.Parameter(torch.zeros(spatial_dim))
        self.temporal_regression = SpatialFeatureGRU(spatial_dim, hidden_dim)
        self.forecast_head = ForecastHead(hidden_dim, output_steps)

    def forward(self, target_history, target_vectors, target_adjacency):
        shared_features = self.shared_spatial_encoder(target_vectors, target_adjacency)
        private_features = self.private_spatial_encoder(target_vectors, target_adjacency)
        fused_features = (
            self.private_weight * private_features
            + self.shared_weight * shared_features
            + self.fusion_bias
        )
        hidden = self.temporal_regression(target_history, fused_features)
        return self.forecast_head(hidden)
