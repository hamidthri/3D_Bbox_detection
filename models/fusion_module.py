import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    def __init__(self, point_feat_dim=32, img_feat_dim=64, fusion_dim=128):
        super().__init__()
        self.point_proj = nn.Linear(point_feat_dim, fusion_dim)
        self.img_proj = nn.Linear(img_feat_dim, fusion_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=0.5,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Dropout(0.5)
        )
        
    def forward(self, point_feat, img_feat):
        # Project features to fusion_dim
        point_feat = self.point_proj(point_feat).unsqueeze(1)
        img_feat = self.img_proj(img_feat).unsqueeze(1)
        
        combined = torch.cat([point_feat, img_feat], dim=1)
        
        # Self-attention (query=key=value=combined)
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = self.norm(combined + attn_output)
        
        # Lightweight FFN
        output = self.ffn(attn_output)
        return torch.mean(output, dim=1)