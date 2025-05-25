from torch import nn
from models.pointnet_encoder import RegularizedPointNetEncoder
from models.image_encoder import ImageEncoder
from models.fusion_module import EnhancedMultiModalFusion

class Custom3DETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.point_encoder = RegularizedPointNetEncoder(
            point_dim=3,
            feat_dim=config['model_params']['point_feat_dim']
        )
        
        self.img_encoder = ImageEncoder(
            feat_dim=config['model_params']['img_feat_dim']
        )
        
        self.fusion = EnhancedMultiModalFusion(
            point_feat_dim=config['model_params']['point_feat_dim'],
            img_feat_dim=config['model_params']['img_feat_dim'],
            fusion_dim=config['model_params']['fusion_dim']
        )
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config['model_params']['fusion_dim'],
            nhead=8,
            dim_feedforward=1024,
            dropout=config['model_params']['dropout'],
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config['model_params']['num_transformer_layers']
        )
        
        self.query_embed = nn.Embedding(config['max_objects'], config['model_params']['fusion_dim'])
        
        self.bbox_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 256),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(256, 9)
        )
        
        self.conf_head = nn.Sequential(
            nn.Linear(config['model_params']['fusion_dim'], 128),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, pointcloud):
        batch_size = rgb.size(0)
        
        point_feat = self.point_encoder(pointcloud)
        img_feat = self.img_encoder(rgb)
        
        fused_feat = self.fusion(point_feat, img_feat)
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        scene_feat = fused_feat.unsqueeze(1).repeat(1, self.config['max_objects'], 1)
        transformer_input = queries + scene_feat
        
        transformer_output = self.transformer(transformer_input)
        
        bbox_pred = self.bbox_head(transformer_output)
        conf_pred = self.conf_head(transformer_output).squeeze(-1)
        
        return bbox_pred, conf_pred

