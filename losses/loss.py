import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class BBox3DLoss(nn.Module):
    def __init__(self, w_center=1.0, w_size=1.0, w_rot=1.0, w_conf=1.0, conf_alpha=0.75):
        super().__init__()
        self.w_center = w_center
        self.w_size = w_size
        self.w_rot = w_rot
        self.w_conf = w_conf
        self.conf_alpha = conf_alpha

    def hungarian_matching(self, pred_bbox, gt_bbox, gt_conf):
        batch_size = pred_bbox.shape[0]
        assignments = []
        
        for b in range(batch_size):
            valid_gt = gt_conf[b] > 0.5
            num_valid_gt = valid_gt.sum().item()
            
            if num_valid_gt == 0:
                assignments.append(([], []))
                continue
                
            pred_centers = pred_bbox[b, :, :3]
            pred_sizes = pred_bbox[b, :, 3:6]
            pred_quats = pred_bbox[b, :, 6:10]
            
            gt_centers = gt_bbox[b, valid_gt, :3]
            gt_sizes = gt_bbox[b, valid_gt, 3:6]
            gt_quats = gt_bbox[b, valid_gt, 6:10]
            
            center_dist = torch.cdist(pred_centers, gt_centers, p=1)
            size_dist = torch.cdist(pred_sizes, gt_sizes, p=1)
            
            pred_quats_norm = F.normalize(pred_quats, dim=-1)
            gt_quats_norm = F.normalize(gt_quats, dim=-1)
            rot_sim = torch.abs(torch.mm(pred_quats_norm, gt_quats_norm.T))
            rot_dist = 1.0 - rot_sim
            
            cost_matrix = center_dist + size_dist + rot_dist
            cost_np = cost_matrix.detach().cpu().numpy()
            
            pred_indices, gt_indices = linear_sum_assignment(cost_np)
            assignments.append((pred_indices, gt_indices))
            
        return assignments

    def forward(self, pred_bbox, pred_conf, gt_bbox, gt_conf):
        batch_size = pred_bbox.shape[0]
        device = pred_bbox.device
        
        assignments = self.hungarian_matching(pred_bbox, gt_bbox, gt_conf)
        
        center_losses = []
        size_losses = []
        rot_losses = []
        conf_losses = []
        
        for b in range(batch_size):
            pred_indices, gt_indices = assignments[b]
            
            if len(pred_indices) == 0:
                center_losses.append(torch.tensor(0.0, device=device))
                size_losses.append(torch.tensor(0.0, device=device))
                rot_losses.append(torch.tensor(0.0, device=device))
            else:
                valid_gt_mask = gt_conf[b] > 0.5
                valid_gt_indices = torch.where(valid_gt_mask)[0]
                matched_gt_indices = valid_gt_indices[gt_indices]
                
                pred_centers = pred_bbox[b, pred_indices, :3]
                pred_sizes = pred_bbox[b, pred_indices, 3:6]
                pred_quats = F.normalize(pred_bbox[b, pred_indices, 6:10], dim=-1)
                
                gt_centers = gt_bbox[b, matched_gt_indices, :3]
                gt_sizes = gt_bbox[b, matched_gt_indices, 3:6]
                gt_quats = F.normalize(gt_bbox[b, matched_gt_indices, 6:10], dim=-1)
                
                center_loss = F.l1_loss(pred_centers, gt_centers)
                size_loss = F.l1_loss(pred_sizes, gt_sizes)
                
                quat_similarity = torch.abs((pred_quats * gt_quats).sum(dim=-1))
                rot_loss = (1.0 - quat_similarity).mean()
                
                center_losses.append(center_loss)
                size_losses.append(size_loss)
                rot_losses.append(rot_loss)
            
            target_conf = torch.zeros_like(pred_conf[b])
            if len(pred_indices) > 0:
                target_conf[pred_indices] = 1.0
            
            pos_mask = target_conf > 0.5
            neg_mask = target_conf <= 0.5
            
            if pos_mask.sum() > 0:
                pos_loss = F.binary_cross_entropy(pred_conf[b][pos_mask], target_conf[pos_mask])
            else:
                pos_loss = torch.tensor(0.0, device=device)
                
            if neg_mask.sum() > 0:
                neg_loss = F.binary_cross_entropy(pred_conf[b][neg_mask], target_conf[neg_mask])
            else:
                neg_loss = torch.tensor(0.0, device=device)
            
            conf_loss = self.conf_alpha * pos_loss + (1 - self.conf_alpha) * neg_loss
            conf_losses.append(conf_loss)
        
        final_center_loss = torch.stack(center_losses).mean()
        final_size_loss = torch.stack(size_losses).mean()
        final_rot_loss = torch.stack(rot_losses).mean()
        final_conf_loss = torch.stack(conf_losses).mean()
        
        total_loss = (self.w_center * final_center_loss + 
                     self.w_size * final_size_loss + 
                     self.w_rot * final_rot_loss + 
                     self.w_conf * final_conf_loss)
        
        return {
            'total_loss': total_loss,
            'bbox_loss': final_center_loss + final_size_loss + final_rot_loss,
            'center_loss': final_center_loss,
            'size_loss': final_size_loss,
            'rotation_loss': final_rot_loss,
            'conf_loss': final_conf_loss
        }