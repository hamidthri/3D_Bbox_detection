import torch
import torch.nn as nn
import torch.nn.functional as F

class BBox3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    """
        This loss function computes the total loss for 3D bounding box detection, including:
        - Center loss: Smooth L1 loss for the center of the bounding box.
        - Size loss: Smooth L1 loss for the size of the bounding box.
        - Rotation loss: Angle difference between predicted and ground truth quaternion representations.
        - Confidence loss: Focal loss for the confidence score of the bounding box.
        Args:
            pred_bbox (torch.Tensor): Predicted bounding box parameters of shape (B, N, 10) where N is the number of boxes.
            pred_conf (torch.Tensor): Predicted confidence scores of shape (B, N).
            gt_bbox (torch.Tensor): Ground truth bounding box parameters of shape (B, N, 10).
            gt_conf (torch.Tensor): Ground truth confidence scores of shape (B, N).
        Returns:
            dict: A dictionary containing the total loss and individual component losses:
                - 'total_loss': Total loss value.
                - 'bbox_loss': Combined loss for center, size, and rotation.
                - 'center_loss': Loss for the center of the bounding box.
                - 'size_loss': Loss for the size of the bounding box.
                - 'rotation_loss': Loss for the rotation of the bounding box.
                - 'conf_loss': Confidence loss.
    """
        
    def focal_loss(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * ce_loss).mean()
    
    def rotation_loss(self, pred_quat, gt_quat):
        pred_quat = F.normalize(pred_quat, dim=-1)
        gt_quat = F.normalize(gt_quat, dim=-1)
        
        cos_sim = torch.abs((pred_quat * gt_quat).sum(dim=-1))
        cos_sim = torch.clamp(cos_sim, 0, 1)
        angle_diff = torch.acos(cos_sim)
        return angle_diff.mean()
    
    def forward(self, pred_bbox, pred_conf, gt_bbox, gt_conf):
        valid_mask = gt_conf > 0.5
        
        if valid_mask.sum() == 0:
            conf_loss = self.focal_loss(pred_conf, torch.zeros_like(pred_conf))
            return {
                'total_loss': conf_loss,
                'bbox_loss': torch.tensor(0.0, device=pred_bbox.device),
                'center_loss': torch.tensor(0.0, device=pred_bbox.device),
                'size_loss': torch.tensor(0.0, device=pred_bbox.device),
                'rotation_loss': torch.tensor(0.0, device=pred_bbox.device),
                'conf_loss': conf_loss
            }
        
        pred_centers = pred_bbox[..., :3]
        pred_sizes = pred_bbox[..., 3:6]
        pred_quats = pred_bbox[..., 6:10]
        
        gt_centers = gt_bbox[..., :3]
        gt_sizes = gt_bbox[..., 3:6]
        gt_quats = gt_bbox[..., 6:10]
        
        center_loss = F.smooth_l1_loss(pred_centers[valid_mask], gt_centers[valid_mask])
        
        pred_sizes_pos = torch.clamp(pred_sizes[valid_mask], min=0.01)
        gt_sizes_pos = torch.clamp(gt_sizes[valid_mask], min=0.01)
        size_loss = F.smooth_l1_loss(torch.log(pred_sizes_pos), torch.log(gt_sizes_pos))
        
        rotation_loss = self.rotation_loss(pred_quats[valid_mask], gt_quats[valid_mask])
        
        target_conf = valid_mask.float()
        conf_loss = self.focal_loss(pred_conf, target_conf)
        
        bbox_loss = center_loss + size_loss + rotation_loss
        total_loss = 5.0 * bbox_loss + conf_loss
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'center_loss': center_loss,
            'size_loss': size_loss,
            'rotation_loss': rotation_loss,
            'conf_loss': conf_loss
        }
