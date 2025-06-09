import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.custom_dataset import BBox3DDataset

# -------------------------
# Config & Dataset Setup
# -------------------------
config = {
    'image_size': (256, 256),
    'max_points': 2048,
    'max_objects': 21,
    'model_params': {'fusion_dim': 256},
    'rotation_degrees': 45  # Max ±45° for testing
}

folder_paths = [
    '/workspaces/3D_Bbox_detection/data/8b061a88-9915-11ee-9103-bbb8eae05561'
]

# Load original raw data (before dataset logic) to compare
raw_corners = np.load(os.path.join(folder_paths[0], 'bbox3d.npy'))  # (N, 8, 3)
raw_pc = np.load(os.path.join(folder_paths[0], 'pc.npy'))  # (3, H, W)
raw_pc = np.transpose(raw_pc, (1, 2, 0)).reshape(-1, 3)
valid = ~np.isnan(raw_pc).any(axis=1) & (raw_pc[:, 2] > 0)
raw_pc = raw_pc[valid]
raw_pc_tensor = torch.from_numpy(raw_pc).float()
raw_corners_tensor = torch.from_numpy(raw_corners).float()

# Load dataset (with augmentation enabled)
dataset = BBox3DDataset(folder_paths, config, split='train')
sample = dataset[0]

rgb = sample['rgb']
pc = sample['pointcloud']  # (P, 3)
bbox_params = sample['bbox_params']
bbox_corners = sample['bbox_corners']
num_objects = sample['num_objects'].item()

# -------------------------
# Visualization
# -------------------------
def draw_3d_box(ax, corners, color='r', label=None):
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        pts = corners[edge, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=color, linewidth=1.5)
    if label:
        center = corners.mean(axis=0)
        ax.text(center[0], center[1], center[2], label, color=color, fontsize=8, weight='bold')

def plot_before_after(pc_before, pc_after, box_before, box_after, num_boxes):
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pc_before[:, 0], pc_before[:, 1], pc_before[:, 2],
                s=3, c=pc_before[:, 2], cmap='plasma', alpha=1.0)
    for i in range(min(num_boxes, box_before.shape[0])):
        draw_3d_box(ax1, box_before[i], color='green', label=f"Before {i}")
    ax1.set_title("Original Point Cloud + BBoxes")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(20, 130)

    ax2 = fig.add_subplot(122, projection='3d')
    pc_after_np = pc_after.cpu().numpy()
    ax2.scatter(pc_after_np[:, 0], pc_after_np[:, 1], pc_after_np[:, 2],
                s=3, c=pc_after_np[:, 2], cmap='plasma', alpha=1.0)
    for i in range(min(num_boxes, box_after.shape[0])):
        draw_3d_box(ax2, box_after[i], color='red', label=f"After {i}")
    ax2.set_title("Augmented Point Cloud + BBoxes")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(20, 130)

    plt.tight_layout()
    plt.savefig("dataset_aug_debug.png")
    plt.show()

# -------------------------
# Run the Plot
# -------------------------
plot_before_after(raw_pc_tensor, pc, raw_corners_tensor, bbox_corners, num_objects)

# -------------------------
# Debug Printouts
# -------------------------
i = 0
if num_objects > 0:
    print("\n🔍 DEBUG CHECKS FOR OBJECT 0")

    # Center shift
    raw_center = raw_corners_tensor[i].mean(dim=0)
    aug_center = bbox_corners[i].mean(dim=0)
    print(f"Center Before: {raw_center.numpy().round(3)}")
    print(f"Center After : {aug_center.numpy().round(3)}")

    # Edge length check
    def edge_lengths(c):
        return torch.norm(c[1] - c[0]), torch.norm(c[3] - c[0]), torch.norm(c[4] - c[0])

    l0, w0, h0 = edge_lengths(raw_corners_tensor[i])
    l1, w1, h1 = edge_lengths(bbox_corners[i])
    print(f"Edge Lengths Before: {[round(x.item(), 3) for x in (l0, w0, h0)]}")
    print(f"Edge Lengths After : {[round(x.item(), 3) for x in (l1, w1, h1)]}")

    # Point cloud difference check
    if raw_pc_tensor.shape[0] >= pc.shape[0]:
        sampled_raw_pc = raw_pc_tensor[torch.randperm(raw_pc_tensor.shape[0])[:pc.shape[0]]]
    else:
        pad = torch.zeros(pc.shape[0] - raw_pc_tensor.shape[0], 3)
        sampled_raw_pc = torch.cat([raw_pc_tensor, pad], dim=0)

    l2_pc_dist = torch.norm(sampled_raw_pc - pc, dim=1).mean()

    print(f"\nMean L2 distance between raw and rotated point cloud: {l2_pc_dist:.4f}")

    # Sanity check: did rotation actually move the box and pc?
    moved_box = torch.norm(raw_center - aug_center) > 1e-3
    moved_pc = l2_pc_dist > 1e-3

    print(f"\n✅ BBox moved? {'Yes' if moved_box else 'No'}")
    print(f"✅ PC moved?   {'Yes' if moved_pc else 'No'}")
