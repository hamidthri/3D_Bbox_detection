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

        self.rgb_backbone = efficientnet_b3(weights="IMAGENET1K_V1" if config['model_params']['backbone_pretrained'] else None)
        rgb_feature_dim = self.rgb_backbone.classifier[1].in_features
        self.rgb_backbone.classifier = nn.Identity()

        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_feature_dim, config['model_params']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout'])
        )

        self.pc_extractor = DGCNN(
            input_dim=3,
            k=config['model_params'].get('dgcnn_k', 20),
            output_dim=config['model_params']['fusion_dim']
        )

        self.fusion = TransformerFusion(
            feature_dim=config['model_params']['fusion_dim'],
            num_layers=config['model_params']['num_transformer_layers']
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 512),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_objects * (6 + 4))
        )

        # Confidence score head
        self.confidence_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 256),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(256, self.max_objects)
        )

    def forward(self, rgb, pointcloud):
        rgb_feat = self.rgb_backbone(rgb)
        rgb_feat = self.rgb_proj(rgb_feat)

        pc_feat = self.pc_extractor(pointcloud)

        fused_feat = self.fusion(rgb_feat, pc_feat)

        bbox_pred = self.bbox_head(fused_feat).view(-1, self.max_objects, 6 + 4)
        conf_pred = torch.sigmoid(self.confidence_head(fused_feat))

        return bbox_pred, conf_pred
