import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class BBoxCornerToParametric:
    def __init__(self, device='cpu'):
        self.device = device
        self.canonical_template = self._get_canonical_template()

    def _get_canonical_template(self):
        return np.array([
            [-0.5, -0.5, -0.5],
            [+0.5, -0.5, -0.5],
            [+0.5, +0.5, -0.5],
            [-0.5, +0.5, -0.5],
            [-0.5, -0.5, +0.5],
            [+0.5, -0.5, +0.5],
            [+0.5, +0.5, +0.5],
            [-0.5, +0.5, +0.5],
        ], dtype=np.float32)

    def convert_corners_to_params(self, corners, method='pca'):
        if corners.ndim == 2:
            return self._convert_single_box(corners, method)
        elif corners.ndim == 3:
            results = []
            for i in range(corners.shape[0]):
                results.append(self._convert_single_box(corners[i], method))
            return self._batch_results(results)
        else:
            raise ValueError("Corners must be shape (8, 3) or (N, 8, 3)")

    def _convert_single_box(self, corners, method='pca'):
        corners = np.array(corners, dtype=np.float32)

        if corners.shape != (8, 3):
            raise ValueError("Corners must be shape (8, 3)")

        center = np.mean(corners, axis=0)
        centered_corners = corners - center

        if method == 'pca':
            size, rotation_matrix = self._pca_box_fitting(centered_corners)
        else:
            size, rotation_matrix = self._minmax_box_fitting(centered_corners)

        size = np.maximum(size, 0.01)

        rotation_quat = self._rotation_matrix_to_quaternion(rotation_matrix)

        return {
            'center': center,
            'size': size,
            'rotation_quat': rotation_quat,
            'rotation_matrix': rotation_matrix,
            'corners_original': corners,
            'corners_centered': centered_corners
        }

    def _pca_box_fitting(self, centered_corners):
        cov_matrix = np.cov(centered_corners.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]

        if np.linalg.det(eigenvecs) < 0:
            eigenvecs[:, -1] *= -1

        projected = centered_corners @ eigenvecs
        size = np.max(projected, axis=0) - np.min(projected, axis=0)

        return size, eigenvecs

    def _minmax_box_fitting(self, centered_corners):
        best_volume = float('inf')
        best_size = None
        best_rotation = np.eye(3)

        test_rotations = [
            np.eye(3),
            self._rotation_matrix_from_euler(0, 0, np.pi/4),
            self._rotation_matrix_from_euler(0, np.pi/4, 0),
            self._rotation_matrix_from_euler(np.pi/4, 0, 0),
        ]

        for rot in test_rotations:
            projected = centered_corners @ rot
            mins = np.min(projected, axis=0)
            maxs = np.max(projected, axis=0)
            size = maxs - mins
            volume = np.prod(size)

            if volume < best_volume:
                best_volume = volume
                best_size = size
                best_rotation = rot

        return best_size, best_rotation

    def _rotation_matrix_from_euler(self, roll, pitch, yaw):
        return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        return R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]

    def _batch_results(self, results):
        batch_size = len(results)
        return {
            'center': np.stack([r['center'] for r in results]),
            'size': np.stack([r['size'] for r in results]),
            'rotation_quat': np.stack([r['rotation_quat'] for r in results]),
            'rotation_matrix': np.stack([r['rotation_matrix'] for r in results]),
            'corners_original': np.stack([r['corners_original'] for r in results]),
            'corners_centered': np.stack([r['corners_centered'] for r in results])
        }

    def reconstruct_corners(self, center, size, rotation_quat):
        template = self.canonical_template * size.reshape(1, 3)
        rotation_matrix = R.from_quat(rotation_quat[[1, 2, 3, 0]]).as_matrix()
        rotated_corners = template @ rotation_matrix.T
        corners = rotated_corners + center.reshape(1, 3)
        return corners

def reconstruct_corners_tensor(center, size, rotation_quat, device='cpu'):
    canonical_template = torch.tensor([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=device)

    batch_size, max_objects = center.shape[:2]

    template = canonical_template.unsqueeze(0).unsqueeze(0) * size.unsqueeze(2)

    w, x, y, z = rotation_quat[..., 0], rotation_quat[..., 1], rotation_quat[..., 2], rotation_quat[..., 3]

    rotation_matrix = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1)
    ], dim=-2)

    rotated_corners = torch.matmul(template, rotation_matrix.transpose(-2, -1))
    corners = rotated_corners + center.unsqueeze(2)

    return corners

def convert_corners_to_params_tensor(corners, converter):
    batch_size, max_objects = corners.shape[:2]
    centers = []
    sizes = []
    rotation_quats = []

    for b in range(batch_size):
        batch_centers = []
        batch_sizes = []
        batch_quats = []

        for o in range(max_objects):
            corner_np = corners[b, o].detach().cpu().numpy()
            if np.all(corner_np == 0):
                batch_centers.append(np.zeros(3))
                batch_sizes.append(np.ones(3) * 0.1)
                batch_quats.append(np.array([1, 0, 0, 0]))
            else:
                try:
                    params = converter.convert_corners_to_params(corner_np)
                    batch_centers.append(params['center'])
                    batch_sizes.append(params['size'])
                    batch_quats.append(params['rotation_quat'])
                except:
                    batch_centers.append(np.mean(corner_np, axis=0))
                    batch_sizes.append(np.ones(3) * 0.1)
                    batch_quats.append(np.array([1, 0, 0, 0]))

        centers.append(np.stack(batch_centers))
        sizes.append(np.stack(batch_sizes))
        rotation_quats.append(np.stack(batch_quats))

    return {
        'center': torch.tensor(np.stack(centers), dtype=torch.float32, device=corners.device),
        'size': torch.tensor(np.stack(sizes), dtype=torch.float32, device=corners.device),
        'rotation_quat': torch.tensor(np.stack(rotation_quats), dtype=torch.float32, device=corners.device)
    }

def visualize_predictions(rgb, pred_corners, gt_corners, pred_conf, gt_conf, sample_idx=0, conf_threshold=0.5):
    fig = plt.figure(figsize=(18, 6))

    # RGB Image
    ax1 = fig.add_subplot(131)
    img = rgb[sample_idx].cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = torch.clamp(img, 0, 1).numpy()
    ax1.imshow(img)
    ax1.set_title("RGB Image", fontsize=14)
    ax1.axis('off')

    # Ground Truth 3D Boxes
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Ground Truth 3D Boxes", fontsize=14)
    _setup_3d_axes(ax2)

    gt_valid = gt_conf[sample_idx]
    if gt_valid.ndim > 1:
        gt_valid = gt_valid.mean(dim=-1)  # ✅ Reduce to scalar confidence
    gt_valid = gt_valid > 0.5

    for i in range(gt_valid.shape[0]):
        if gt_valid[i].item():
            corners = gt_corners[sample_idx, i].cpu().numpy()
            draw_3d_box(ax2, corners, color='green', alpha=0.5)
    ax2.view_init(elev=20, azim=135)

    # Predicted 3D Boxes
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("Predicted 3D Boxes", fontsize=14)
    _setup_3d_axes(ax3)

    pred_valid = pred_conf[sample_idx]
    if pred_valid.ndim > 1:
        pred_valid = pred_valid.mean(dim=-1)  # ✅ Reduce to scalar
    pred_valid = pred_valid > conf_threshold

    for i in range(pred_valid.shape[0]):
        if pred_valid[i].item():
            corners = pred_corners[sample_idx, i].detach().cpu().numpy()
            draw_3d_box(ax3, corners, color='red', alpha=0.5)
    ax3.view_init(elev=20, azim=135)

    plt.tight_layout()
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