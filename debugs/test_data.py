import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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

rgb = sample['rgb']
pc = sample['pointcloud']
bbox_params = sample['bbox_params']
bbox_corners = sample['bbox_corners']
num_objects = sample['num_objects']
raw_bbox_corners = np.load(os.path.join(folder_paths[0], 'bbox3d.npy'))  # (N, 8, 3)
raw_bbox_corners_tensor = torch.from_numpy(raw_bbox_corners).float()  # (N, 8, 3)


# ---- Plotting Utilities ---- #
def draw_3d_box(ax, corners, color='r', label=None):
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        pts = corners[edge, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=color, linewidth=1.2)
    if label:
        center = corners.mean(axis=0)
        ax.text(center[0], center[1], center[2], label, color=color, fontsize=8, weight='bold')


def plot_overlay_boxes(pc, raw_corners, processed_corners, num_objects):
    pc_np = pc.cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud
    ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
               c=pc_np[:, 2], cmap='gray', s=0.5, alpha=0.5)

    # Plot original boxes (green)
    for i in range(raw_corners.shape[0]):
        draw_3d_box(ax, raw_corners[i], color='green', label=f"Before {i}")

    # Plot processed boxes (red)
    for i in range(num_objects):
        draw_3d_box(ax, processed_corners[i].cpu().numpy(), color='red', label=f"After {i}")

    ax.set_title("3D Bounding Boxes\n🟩 Before (green) — 🟥 After (red)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=120)
    plt.tight_layout()
    plt.savefig("bbox_overlay_labeled.png")
    plt.show()


def compare_boxes(raw_corners, processed_corners, num_to_check):
    print(f"\n🔍 Comparing first {num_to_check} bounding boxes:")
    for i in range(num_to_check):
        if i >= raw_corners.shape[0] or i >= processed_corners.shape[0]:
            print(f"Index {i} out of bounds. Skipping.")
            continue

        raw = raw_corners[i]
        proc = processed_corners[i].cpu()

        l2 = torch.norm(raw - proc, dim=-1).mean().item()
        print(f"\nBox {i} - Mean L2 Distance: {l2:.4f}")
        if l2 > 0.05:
            print("⚠️  Mismatch likely.")
        else:
            print("✅  Match.")

        print("  Raw Center     :", raw.mean(dim=0).numpy().round(2))
        print("  Processed Center:", proc.mean(dim=0).numpy().round(2))


# ---- Run Comparison and Plot ---- #
print("🔍 Visualizing and comparing bounding boxes...")
compare_boxes(raw_bbox_corners_tensor, bbox_corners, num_objects.item())
plot_overlay_boxes(pc, raw_bbox_corners_tensor, bbox_corners, num_objects.item())
