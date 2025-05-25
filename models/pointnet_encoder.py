import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedPointNetEncoder(nn.Module):
    def __init__(self, point_dim=3, feat_dim=32, dropout_rate=0.5):
        super().__init__()
        
        # Smaller network with more regularization
        self.conv1 = nn.Conv1d(point_dim, 16, 1)  # Reduced from 32
        self.conv2 = nn.Conv1d(16, 32, 1)         # Reduced from 64
        self.conv3 = nn.Conv1d(32, feat_dim, 1)   # Keep final dim
        
        # Use LayerNorm instead of BatchNorm for better small batch performance
        self.ln1 = nn.LayerNorm(16)
        self.ln2 = nn.LayerNorm(32)
        self.ln3 = nn.LayerNorm(feat_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Add spatial dropout for point clouds
        self.spatial_dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # x shape: [B, N, 3]
        batch_size = x.size(0)
        x = x.transpose(2, 1)  # [B, 3, N]
        
        # Apply spatial dropout to input points
        if self.training:
            x = self.spatial_dropout(x.unsqueeze(-1)).squeeze(-1)
        
        x = self.conv1(x)  # [B, 16, N]
        x = x.transpose(2, 1)  # [B, N, 16] for LayerNorm
        x = self.dropout(self.relu(self.ln1(x)))
        x = x.transpose(2, 1)  # Back to [B, 16, N]
        
        x = self.conv2(x)  # [B, 32, N]
        x = x.transpose(2, 1)  # [B, N, 32]
        x = self.dropout(self.relu(self.ln2(x)))
        x = x.transpose(2, 1)  # Back to [B, 32, N]
        
        x = self.conv3(x)  # [B, feat_dim, N]
        x = x.transpose(2, 1)  # [B, N, feat_dim]
        x = self.relu(self.ln3(x))
        x = x.transpose(2, 1)  # Back to [B, feat_dim, N]
        
        # Global max pooling with some regularization
        x = torch.max(x, 2)[0]  # [B, feat_dim]
        
        return x