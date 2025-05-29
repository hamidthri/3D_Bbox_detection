import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pc_and_boxes_matplotlib(pc, gt_boxes, pred_boxes, path=None):
    """
    Plots point cloud with GT (green) and predicted (red) boxes in 3D using matplotlib.
    pc: (N, 3)
    gt_boxes, pred_boxes: list of (8, 3) arrays
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='gray', s=0.5, alpha=0.4)

    def draw_box(corners, color):
        for start, end in [(0,1),(1,2),(2,3),(3,0),
                           (4,5),(5,6),(6,7),(7,4),
                           (0,4),(1,5),(2,6),(3,7)]:
            xs = [corners[start, 0], corners[end, 0]]
            ys = [corners[start, 1], corners[end, 1]]
            zs = [corners[start, 2], corners[end, 2]]
            ax.plot(xs, ys, zs, c=color, linewidth=1)

    for box in gt_boxes:
        draw_box(box, 'green')
    for box in pred_boxes:
        draw_box(box, 'red')

    ax.set_xlim(pc[:,0].min(), pc[:,0].max())
    ax.set_ylim(pc[:,1].min(), pc[:,1].max())
    ax.set_zlim(pc[:,2].min(), pc[:,2].max())
    ax.set_axis_off()
    if path:
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

def visualize_predictions(rgb, pred_corners, gt_corners, pred_conf, gt_conf, sample_idx=0, conf_threshold=0.5):
    """
    Visualize RGB input, Ground Truth, and Predicted 3D bounding boxes side-by-side.

    Parameters:
        rgb (Tensor): [B, 3, H, W] image tensor
        pred_corners (Tensor): [B, N, 8, 3] predicted box corners
        gt_corners (Tensor): [B, N, 8, 3] ground truth box corners
        pred_conf (Tensor): [B, N] confidence scores
        gt_conf (Tensor): [B, N] validity mask (or soft scores)
        sample_idx (int): which sample in the batch to visualize
        conf_threshold (float): threshold to filter predicted boxes
    """
    fig = plt.figure(figsize=(18, 6))

    # --- RGB Image ---
    ax1 = fig.add_subplot(131)
    img = rgb[sample_idx].cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = torch.clamp(img, 0, 1).numpy()
    ax1.imshow(img)
    ax1.set_title("RGB Image", fontsize=14)
    ax1.axis('off')

    # --- Ground Truth Boxes ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Ground Truth Boxes", fontsize=14)
    _setup_3d_axes(ax2)

    gt_valid = gt_conf[sample_idx]
    if gt_valid.ndim > 1:
        gt_valid = gt_valid.mean(dim=-1)
    gt_mask = gt_valid > 0.5

    num_gt = gt_mask.sum().item()
    for i in range(gt_mask.shape[0]):
        if gt_mask[i].item():
            corners = gt_corners[sample_idx, i].cpu().numpy()
            draw_3d_box(ax2, corners, color='green', alpha=0.5)
    ax2.view_init(elev=20, azim=135)
    ax2.text2D(0.05, 0.95, f"{num_gt} GT boxes", transform=ax2.transAxes, fontsize=12)

    # --- Predicted Boxes ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(f"Predicted Boxes (conf > {conf_threshold})", fontsize=14)
    _setup_3d_axes(ax3)

    pred_valid = pred_conf[sample_idx]
    if pred_valid.ndim > 1:
        pred_valid = pred_valid.mean(dim=-1)
    pred_mask = pred_valid > conf_threshold

    num_pred = pred_mask.sum().item()
    for i in range(pred_mask.shape[0]):
        if pred_mask[i].item():
            corners = pred_corners[sample_idx, i].detach().cpu().numpy()
            draw_3d_box(ax3, corners, color='red', alpha=0.5)
    ax3.view_init(elev=20, azim=135)
    ax3.text2D(0.05, 0.95, f"{num_pred} Predicted boxes", transform=ax3.transAxes, fontsize=12)

    plt.suptitle("3D Bounding Box Visualization", fontsize=16, y=1.02)
    plt.tight_layout()
    # add grid
    for ax in fig.axes:
        ax.grid(False)
    return fig


def draw_3d_box(ax, corners, color='blue', alpha=0.6):
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        p1, p2 = corners[edge]
        ax.plot3D(*zip(p1, p2), color=color, alpha=alpha, linewidth=2.5)

def _setup_3d_axes(ax):
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])