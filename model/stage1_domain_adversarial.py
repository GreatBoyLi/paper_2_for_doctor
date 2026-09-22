import torch
from torch import nn

from model.domain_adversarial import DomainClassifier
from model.gin import MeanGINLayer
from model.power_forecaster import ForecastHead
from model.temporal_gru import SpatialFeatureGRU


class Stage1DomainAdversarialModel(nn.Module):
    """论文Stage 1：共享GIN、Source时间回归和Source/Target域分类。"""

    def __init__(self, spatial_dim, hidden_dim, output_steps, domain_hidden_dim):
        super().__init__()
        self.spatial_encoder = MeanGINLayer(spatial_dim)
        self.temporal_regression = SpatialFeatureGRU(spatial_dim, hidden_dim)
        self.forecast_head = ForecastHead(hidden_dim, output_steps)
        self.domain_classifier = DomainClassifier(spatial_dim, domain_hidden_dim)

    def forward(self, source_history, source_vectors, source_adjacency,
                target_vectors, target_adjacency, grl_lambda=1.0):
        source_features = self.spatial_encoder(source_vectors, source_adjacency)
        target_features = self.spatial_encoder(target_vectors, target_adjacency)

        source_hidden = self.temporal_regression(source_history, source_features)
        source_predictions = self.forecast_head(source_hidden)

        domain_features = torch.cat((source_features, target_features), dim=0)
        domain_logits = self.domain_classifier(domain_features, grl_lambda)
        return source_predictions, domain_logits
