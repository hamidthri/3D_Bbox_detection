import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class Combined3DLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_loss = FocalLoss(
            alpha=config['loss_weights']['focal_alpha'],
            gamma=config['loss_weights']['focal_gamma']
        )
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, bbox_pred, conf_pred, bbox_target, conf_target, num_objects):
        batch_size = bbox_pred.size(0)
        
        conf_loss = self.focal_loss(conf_pred, conf_target)
        
        bbox_loss = 0
        for i in range(batch_size):
            n_obj = num_objects[i].item()
            if n_obj > 0:
                bbox_loss += self.bbox_loss(bbox_pred[i, :n_obj], bbox_target[i, :n_obj])
        
        bbox_loss = bbox_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=bbox_pred.device)
        
        total_loss = (self.config['loss_weights']['conf'] * conf_loss + 
                     self.config['loss_weights']['bbox'] * bbox_loss)
        
        return total_loss, conf_loss, bbox_loss
