import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datasets.custom_dataset import BBox3DDataset

# ---- CONFIG ---- #
config = {
    'image_size': (256, 256),
    'max_points': 2048,
    'max_objects': 21,
    'model_params': {'fusion_dim': 256}
}

folder_paths = [
    '/workspaces/3D_Bbox_detection/data/8b061a88-9915-11ee-9103-bbb8eae05561'
]

# ---- Load Sample ---- #
dataset = BBox3DDataset(folder_paths, config, split='val')
sample = dataset[0]

pc = sample['pointcloud']                         # (N, 3) tensor
num_objects = sample['num_objects']
raw_bbox_corners = np.load(os.path.join(folder_paths[0], 'bbox3d.npy'))  # (M, 8, 3)
raw_bbox_corners_tensor = torch.from_numpy(raw_bbox_corners).float()     # (M, 8, 3)

# ---- Rotation Function ---- #
def euler_rotate(pc, bboxes, angles_deg):
    rotation = R.from_euler('xyz', angles_deg, degrees=True)
    rot_mat = torch.from_numpy(rotation.as_matrix()).float()
    pc_rotated = torch.matmul(pc, rot_mat.T)
    bboxes_rotated = torch.matmul(bboxes, rot_mat.T)
    return pc_rotated, bboxes_rotated, rot_mat

# ---- Drawing Utility ---- #
def draw_3d_box(ax, corners, color='r', label=None):
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        pts = corners[edge, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=color, linewidth=2)
    if label:
        center = corners.mean(axis=0)
        ax.text(center[0], center[1], center[2], label, color=color, fontsize=9, weight='bold')

# ---- Plot Scene (One Box) ---- #
def plot_scene(ax, pc, box, title, box_color):
    pc_np = pc.cpu().numpy()
    ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
               c=pc_np[:, 2], cmap='viridis', s=2, alpha=0.9)
    draw_3d_box(ax, box.squeeze(0).cpu().numpy(), color=box_color, label="Box")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)

# ---- Edge Length Comparison ---- #
def box_edge_lengths(corners):
    # Use standard opposing corners for size
    l = torch.norm(corners[1] - corners[0])
    w = torch.norm(corners[3] - corners[0])
    h = torch.norm(corners[4] - corners[0])
    return l.item(), w.item(), h.item()

# ---- Apply Rotation ---- #
rotation_angles = (0, 90, 0)
print(f"\n🔁 Applying Euler rotation: {rotation_angles} degrees")
pc_rot, bbox_rot, rot_mat = euler_rotate(pc, raw_bbox_corners_tensor, rotation_angles)

# ---- Pick Box to Test ---- #
box_index = 0
single_box_before = raw_bbox_corners_tensor[box_index].unsqueeze(0)
single_box_after = bbox_rot[box_index].unsqueeze(0)

# ---- Plot ---- #
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection='3d')
plot_scene(ax1, pc, single_box_before, title=f"Before Rotation - Box {box_index}", box_color='green')

ax2 = fig.add_subplot(122, projection='3d')
plot_scene(ax2, pc_rot, single_box_after, title=f"After 90° Z Rotation - Box {box_index}", box_color='red')

plt.tight_layout()
plt.savefig("rotation_verification_box.png")
plt.show()

# ---- Verification: Box Center Shift ---- #
center_before = single_box_before[0].mean(dim=0)
center_after = single_box_after[0].mean(dim=0)
print("\n📍 Box Center Check")
print("Before:", center_before.numpy().round(2))
print("After :", center_after.numpy().round(2))

# ---- Verification: Edge Lengths ---- #
l1, w1, h1 = box_edge_lengths(single_box_before[0])
l2, w2, h2 = box_edge_lengths(single_box_after[0])
print("\n📏 Edge Lengths (L, W, H)")
print(f"Before: {l1:.3f}, {w1:.3f}, {h1:.3f}")
print(f"After : {l2:.3f}, {w2:.3f}, {h2:.3f}")

# ---- Analytical Rotation Check ---- #
sample_point = torch.tensor([[1.0, 0.0, 0.0]])  # Along X
rotated_point = torch.matmul(sample_point, rot_mat.T)
print("\n🧮 Test Point Rotation:")
print("Original point:", sample_point.numpy())
print("Rotated point :", rotated_point.numpy())  # Expected: [0, 1, 0]
