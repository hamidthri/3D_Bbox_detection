import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, feat_dim=64, freeze_backbone=True, backbone_type='mobilenet'):
        super().__init__()
        
        if backbone_type == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=True).features
            backbone_out_dim = 1280
        elif backbone_type == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            backbone_out_dim = 512
        elif backbone_type == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Frozen {backbone_type} backbone - only training the head")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_out_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, feat_dim)
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.backbone.parameters())):
            x = self.backbone(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x