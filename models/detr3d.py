import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_encoder import PointNetPPFeatureExtractor
from models.image_encoder import ImageEncoder
from models.fusion_module import TransformerFusion
from torchvision.models import efficientnet_b3

class BBox3DPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_objects = config['max_objects']

        # Image encoder
        self.rgb_backbone = efficientnet_b3(pretrained=config['model_params']['backbone_pretrained'])
        rgb_feature_dim = self.rgb_backbone.classifier[1].in_features
        self.rgb_backbone.classifier = nn.Identity()
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False

        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_feature_dim, config['model_params']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout'])
        )

        # Point cloud feature extractor
        self.pc_extractor = PointNetPPFeatureExtractor(
            input_dim=3,
            output_dim=config['model_params']['fusion_dim']
        )

        # Fusion
        self.fusion = TransformerFusion(
            feature_dim=config['model_params']['fusion_dim'],
            num_layers=config['model_params']['num_transformer_layers']
        )

        # 3D BBox prediction
        self.bbox_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 512),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_objects * 10)  # (center + size + quat) * max_objects
        )

        # Confidence prediction per object
        self.confidence_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 256),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(256, self.max_objects)  # One confidence score per object
        )

    def forward(self, rgb, pointcloud):
        rgb_feat = self.rgb_backbone(rgb)
        rgb_feat = self.rgb_proj(rgb_feat)

        pc_feat = self.pc_extractor(pointcloud)

        fused_feat = self.fusion(rgb_feat, pc_feat)

        bbox_pred = self.bbox_head(fused_feat).view(-1, self.max_objects, 10)
        conf_pred = torch.sigmoid(self.confidence_head(fused_feat))

        return bbox_pred, conf_pred
