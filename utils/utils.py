import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os

class BBoxCornerToParametric:
    def __init__(self, device='cpu'):
        self.device = device
        self.canonical_template = self._get_canonical_template()
        """
        This class converts 3D bounding box corners to parametric representation
        (center, size, rotation quaternion) and vice versa.
        It supports both PCA-based and min-max fitting methods for box parameters.
        The canonical template is a unit cube centered at the origin.
        The corners are expected to be in the shape (8, 3) for a single box or (N, 8, 3) for a batch of boxes.
        The conversion methods return a dictionary with the following keys:
        - 'center': The center of the bounding box (3D vector).
        - 'size': The size of the bounding box (3D vector).
        - 'rotation_quat': The rotation of the bounding box represented as a quaternion (4D vector).
        """

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

# 



def save_checkpoint(model, optimizer, epoch, best_val_loss, path="checkpoints/checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, path)
    print(f"Saved checkpoint to: {path}")

def load_checkpoint(model, optimizer, path="checkpoints/checkpoint.pth", device='cpu'):
    if not os.path.exists(path):
        print(f"No checkpoint found at: {path}")
        return model, optimizer, 1, float('inf')

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Loaded checkpoint from: {path} (epoch {epoch})")
    return model, optimizer, epoch + 1, best_val_loss