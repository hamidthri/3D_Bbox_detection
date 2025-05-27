import torch
import torch.nn as nn
import torch.nn.functional as F

class BBox3DLoss(nn.Module):
    """
    Computes total loss for 3-D bounding-box prediction.

    Inputs
    ------
    bbox_pred : (B, N, 10)  → [cx,cy,cz, sx,sy,sz, qw,qx,qy,qz]
    conf_pred : (B, N)      → sigmoid scores in [0,1]
    bbox_gt   : (B, N, 10)
    conf_gt   : (B, N)      → 1 = real object, 0 = padded slot
    """

    def __init__(
        self,
        w_center=1.0,
        w_size=1.0,
        w_rot=1.0,
        w_conf=1.0,
        eps=1e-6
    ):
        super().__init__()
        self.w_center = w_center
        self.w_size   = w_size
        self.w_rot    = w_rot
        self.w_conf   = w_conf
        self.eps      = eps          # to avoid /0

    def forward(self, bbox_pred, conf_pred, bbox_gt, conf_gt):
        # ----- split fields --------------------------------------------------
        p_center, p_size, p_quat = torch.split(bbox_pred, [3, 3, 4], dim=-1)
        g_center, g_size, g_quat = torch.split(bbox_gt,   [3, 3, 4], dim=-1)

        # normalise quaternions to unit length
        p_quat = F.normalize(p_quat, dim=-1)
        g_quat = F.normalize(g_quat, dim=-1)

        # ----- masks ---------------------------------------------------------
        valid_mask = (conf_gt > 0).float()            # (B,N)
        valid_cnt  = valid_mask.sum().clamp(min=1.)   # scalar to avoid /0

        # expand mask for vector losses
        vm3  = valid_mask.unsqueeze(-1)               # (B,N,1)

        # ----- sub-losses ----------------------------------------------------
        center_loss = (F.l1_loss(p_center, g_center, reduction='none') * vm3).sum() / valid_cnt
        size_loss   = (F.l1_loss(p_size,   g_size,   reduction='none') * vm3).sum() / valid_cnt

        # quaternion loss: 1 - |dot|
        quat_dot    = torch.abs((p_quat * g_quat).sum(dim=-1))    # (B,N)
        rot_loss    = ((1.0 - quat_dot) * valid_mask).sum() / valid_cnt

        conf_loss   = (F.binary_cross_entropy(conf_pred, conf_gt, reduction='none') *
                       valid_mask).sum() / valid_cnt

        # ----- weighted total -----------------------------------------------
        total = ( self.w_center * center_loss +
                  self.w_size   * size_loss   +
                  self.w_rot    * rot_loss    +
                  self.w_conf   * conf_loss )

        # return both total and breakdown for easy logging
        return total, {
            "center": center_loss.detach(),
            "size"  : size_loss.detach(),
            "rot"   : rot_loss.detach(),
            "conf"  : conf_loss.detach(),
            "total": total.detach()
        }
