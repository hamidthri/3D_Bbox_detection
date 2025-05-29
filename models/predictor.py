import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_encoder import DGCNN
from models.image_encoder import ImageEncoder
from models.fusion_module import TransformerFusion
from torchvision.models import efficientnet_b3

class BBox3DPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_objects = config['max_objects']
        fusion_dim = config['model_params']['fusion_dim']
        dropout = config['model_params']['dropout']

        """
            This model predicts 3D bounding boxes from RGB images and point cloud data.
            It uses a combination of an EfficientNet backbone for RGB feature extraction,
            a DGCNN for point cloud feature extraction, and a transformer-based fusion module.
            Args:
                config (dict): Configuration dictionary containing model parameters.
        """

        # RGB backbone
        self.rgb_backbone = efficientnet_b3(
            weights="IMAGENET1K_V1" if config['model_params']['backbone_pretrained'] else None
        )
        rgb_feature_dim = self.rgb_backbone.classifier[1].in_features
        self.rgb_backbone.classifier = nn.Identity()

        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Point cloud feature extractor using DGCNN
        self.pc_extractor = DGCNN(
            input_dim=3,
            k=config['model_params'].get('dgcnn_k', 20),
            output_dim=fusion_dim
        )

        # Fusion module
        self.fusion = TransformerFusion(
            feature_dim=fusion_dim,
            num_layers=config['model_params']['num_transformer_layers']
        )

        # Separate heads for center, size, and rotation
        self.center_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.max_objects * 3)
        )

        self.size_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.max_objects * 3)
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.max_objects * 4)
        )

        # Confidence score head
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.max_objects)
        )

    def forward(self, rgb, pointcloud):
        rgb_feat = self.rgb_backbone(rgb)
        rgb_feat = self.rgb_proj(rgb_feat)

        pc_feat = self.pc_extractor(pointcloud)
        fused_feat = self.fusion(rgb_feat, pc_feat)

        # Separate predictions
        center_pred = self.center_head(fused_feat).view(-1, self.max_objects, 3)
        size_pred = self.size_head(fused_feat).view(-1, self.max_objects, 3)
        rot_pred = self.rotation_head(fused_feat).view(-1, self.max_objects, 4)

        # Combine for loss
        bbox_pred = torch.cat([center_pred, size_pred, rot_pred], dim=-1)

        # Confidence
        conf_pred = torch.sigmoid(self.confidence_head(fused_feat))

        return bbox_pred, conf_pred
