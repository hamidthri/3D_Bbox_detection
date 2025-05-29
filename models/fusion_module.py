import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFusion(nn.Module):
    def __init__(self, feature_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        """
        Transformer-based feature fusion module.
        Args:
            feature_dim (int): Dimension of the input features.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in the transformer.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, rgb_features, pc_features):
        features = torch.stack([rgb_features, pc_features], dim=1)  # (B, 2, F)
        fused = self.transformer(features)  # (B, 2, F)
        return fused.mean(dim=1)  # (B, F)

