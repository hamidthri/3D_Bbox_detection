import torch
import torch.nn as nn
import torch.nn.functional as F

class BBox3DLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # bbox loss weight
        self.beta = beta    # confidence loss weight

    def forward(self, pred_bbox, pred_conf, gt_bbox, gt_conf):
        # Bbox regression loss (L1 + IoU proxy)
        bbox_l1_loss = F.l1_loss(pred_bbox, gt_bbox, reduction='mean')

        # Confidence loss
        conf_loss = F.binary_cross_entropy(pred_conf, gt_conf, reduction='mean')

        total_loss = self.alpha * bbox_l1_loss + self.beta * conf_loss

        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_l1_loss,
            'conf_loss': conf_loss
        }
