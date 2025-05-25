import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedMultiModalFusion(nn.Module):
    def __init__(self, point_feat_dim=32, img_feat_dim=64, fusion_dim=128, dropout_rate=0.3):
        super().__init__()
        
        # Smaller projection layers
        self.point_proj = nn.Sequential(
            nn.Linear(point_feat_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )
        
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )
        
        # Simplified attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=2,  # Reduced from 4
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Smaller FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),  # Much smaller
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 4, fusion_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, point_feat, img_feat):
        # Project features to fusion_dim
        point_feat = self.point_proj(point_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        img_feat = self.img_proj(img_feat).unsqueeze(1)        # [B, 1, fusion_dim]
        
        # Concatenate modalities
        combined = torch.cat([point_feat, img_feat], dim=1)  # [B, 2, fusion_dim]
        
        # Self-attention
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = self.norm1(combined + attn_output)
        
        # FFN
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)
        
        # Average pooling over modalities
        return torch.mean(output, dim=1)  # [B, fusion_dim]