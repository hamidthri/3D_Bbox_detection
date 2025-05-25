import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    def __init__(self, point_feat_dim=64, img_feat_dim=128, fusion_dim=256):
        super().__init__()
        self.point_proj = nn.Linear(point_feat_dim, fusion_dim)
        self.img_proj = nn.Linear(img_feat_dim, fusion_dim)
        self.fusion = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, point_feat, img_feat):
        point_feat = self.point_proj(point_feat).unsqueeze(1)
        img_feat = self.img_proj(img_feat).unsqueeze(1)
        
        combined = torch.cat([point_feat, img_feat], dim=1)
        fused, _ = self.fusion(combined, combined, combined)
        fused = self.norm(fused)
        
        return torch.mean(fused, dim=1)
