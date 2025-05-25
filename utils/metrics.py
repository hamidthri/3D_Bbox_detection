import numpy as np
import torch
from scipy.spatial.distance import cdist


def calculate_3d_iou(box1, box2):
    x1, y1, z1, w1, h1, d1, rx1, ry1, rz1 = box1
    x2, y2, z2, w2, h2, d2, rx2, ry2, rz2 = box2
    
    corners1 = get_bbox_corners(box1)
    corners2 = get_bbox_corners(box2)
    
    intersection_vol = calculate_intersection_volume(corners1, corners2)
    
    vol1 = w1 * h1 * d1
    vol2 = w2 * h2 * d2
    union_vol = vol1 + vol2 - intersection_vol
    
    if union_vol <= 0:
        return 0.0
    
    return intersection_vol / union_vol


def get_bbox_corners(bbox):
    x, y, z, w, h, d, rx, ry, rz = bbox
    
    corners = np.array([
        [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2],
        [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
        [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
        [w/2, h/2, d/2], [-w/2, h/2, d/2]
    ])
    
    rotation_matrix = get_rotation_matrix(rx, ry, rz)
    rotated_corners = corners @ rotation_matrix.T
    translated_corners = rotated_corners + np.array([x, y, z])
    
    return translated_corners


def get_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx


def calculate_intersection_volume(corners1, corners2):
    min1 = np.min(corners1, axis=0)
    max1 = np.max(corners1, axis=0)
    min2 = np.min(corners2, axis=0)
    max2 = np.max(corners2, axis=0)
    
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    
    intersection_dims = intersection_max - intersection_min
    intersection_dims = np.maximum(intersection_dims, 0)
    
    return np.prod(intersection_dims)


def calculate_ap_3d(predictions, targets, iou_thresholds=[0.5, 0.7]):
    aps = []
    
    for iou_thresh in iou_thresholds:
        ap = calculate_single_ap_3d(predictions, targets, iou_thresh)
        aps.append(ap)
    
    return np.mean(aps)


def calculate_single_ap_3d(predictions, targets, iou_threshold):
    if len(predictions) == 0:
        return 0.0
    
    sorted_indices = np.argsort([-pred['confidence'] for pred in predictions])
    sorted_predictions = [predictions[i] for i in sorted_indices]
    
    tp = np.zeros(len(sorted_predictions))
    fp = np.zeros(len(sorted_predictions))
    
    matched_targets = set()
    
    for pred_idx, pred in enumerate(sorted_predictions):
        best_iou = 0
        best_target_idx = -1
        
        for target_idx, target in enumerate(targets):
            if target_idx in matched_targets:
                continue
            
            iou = calculate_3d_iou(pred['bbox'], target['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_targets.add(best_target_idx)
        else:
            fp[pred_idx] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / len(targets)
    
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))
    
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return ap


def calculate_translation_error(pred_bbox, target_bbox):
    pred_center = pred_bbox[:3]
    target_center = target_bbox[:3]
    return np.linalg.norm(pred_center - target_center)


def calculate_rotation_error(pred_bbox, target_bbox):
    pred_rot = pred_bbox[6:9]
    target_rot = target_bbox[6:9]
    
    diff = np.abs(pred_rot - target_rot)
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    return np.mean(diff)


def calculate_size_error(pred_bbox, target_bbox):
    pred_size = pred_bbox[3:6]
    target_size = target_bbox[3:6]
    
    relative_error = np.abs(pred_size - target_size) / (target_size + 1e-8)
    return np.mean(relative_error)