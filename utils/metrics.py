import numpy as np
import torch
from scipy.spatial.distance import cdist


def compute_3d_iou_accurate(corners1, corners2):
    """
    More accurate 3D IoU computation using bounding box intersection
    """
    def get_bbox_bounds(corners):
        """Get min/max bounds from 8 corners"""
        return {
            'x_min': corners[:, 0].min(), 'x_max': corners[:, 0].max(),
            'y_min': corners[:, 1].min(), 'y_max': corners[:, 1].max(),
            'z_min': corners[:, 2].min(), 'z_max': corners[:, 2].max()
        }

    bounds1 = get_bbox_bounds(corners1)
    bounds2 = get_bbox_bounds(corners2)

    # Intersection bounds
    x_overlap = max(0, min(bounds1['x_max'], bounds2['x_max']) - max(bounds1['x_min'], bounds2['x_min']))
    y_overlap = max(0, min(bounds1['y_max'], bounds2['y_max']) - max(bounds1['y_min'], bounds2['y_min']))
    z_overlap = max(0, min(bounds1['z_max'], bounds2['z_max']) - max(bounds1['z_min'], bounds2['z_min']))

    intersection = x_overlap * y_overlap * z_overlap

    # Individual volumes
    vol1 = ((bounds1['x_max'] - bounds1['x_min']) *
            (bounds1['y_max'] - bounds1['y_min']) *
            (bounds1['z_max'] - bounds1['z_min']))
    vol2 = ((bounds2['x_max'] - bounds2['x_min']) *
            (bounds2['y_max'] - bounds2['y_min']) *
            (bounds2['z_max'] - bounds2['z_min']))

    union = vol1 + vol2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def compute_translation_error(pred_params, gt_params):
    """Compute L2 distance between predicted and ground truth centers"""
    pred_center = pred_params[:3]  # x, y, z
    gt_center = gt_params[:3]
    return np.linalg.norm(pred_center - gt_center)


def compute_translation_error(pred_params, gt_params):
    """Compute L2 distance between predicted and ground truth centers"""
    pred_center = pred_params[:3]  # x, y, z
    gt_center = gt_params[:3]
    return np.linalg.norm(pred_center - gt_center)

def compute_rotation_error(pred_params, gt_params):
    """Compute angular difference in yaw angle"""
    pred_yaw = pred_params[6]  # yaw angle
    gt_yaw = gt_params[6]

    # Normalize angles to [-pi, pi]
    diff = pred_yaw - gt_yaw
    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
    return abs(diff)

def compute_size_error(pred_params, gt_params):
    """Compute relative size error"""
    pred_size = pred_params[3:6]  # width, length, height
    gt_size = gt_params[3:6]

    relative_error = np.abs(pred_size - gt_size) / (gt_size + 1e-6)
    return relative_error.mean()

def params_to_corners(params):
    """
    Convert 9D bbox parameters [center_x, center_y, center_z, w, l, h, rot_x, rot_y, rot_z]
    into 8 corner points of the 3D bounding box.
    """
    center = params[:3]
    size = params[3:6]
    rotation = params[6:9]

    # Create a unit cube centered at origin
    w, l, h = size
    x_corners = np.array([+0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5, -0.5]) * w
    y_corners = np.array([-0.5, +0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5]) * l
    z_corners = np.array([-0.5, -0.5, -0.5, -0.5, +0.5, +0.5, +0.5, +0.5]) * h

    corners = np.vstack((x_corners, y_corners, z_corners)).T

    # Apply rotation
    r = R.from_euler('xyz', rotation, degrees=False)
    rotated_corners = r.apply(corners)

    # Translate to center
    corners_3d = rotated_corners + center

    return corners_3d