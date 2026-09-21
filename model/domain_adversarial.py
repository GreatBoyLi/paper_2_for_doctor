import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grl_lambda):
        ctx.grl_lambda = grl_lambda
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.grl_lambda * gradient, None


def gradient_reverse(features, grl_lambda=1.0):
    """前向保持特征不变，反向把梯度乘以 -grl_lambda。"""
    return _GradientReversal.apply(features, grl_lambda)


class DomainClassifier(nn.Module):
    """通过GRL判断空间特征来自Source还是Target。"""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, spatial_features, grl_lambda=1.0):
        reversed_features = gradient_reverse(spatial_features, grl_lambda)
        return self.network(reversed_features)
