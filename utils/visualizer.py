import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch


def visualize_3d_predictions(rgb_image, pointcloud, pred_boxes, target_boxes, 
                           pred_confidences=None, save_path=None):
    fig = plt.figure(figsize=(20, 8))
    
    ax1 = fig.add_subplot(141)
    if isinstance(rgb_image, torch.Tensor):
        rgb_np = rgb_image.cpu().permute(1, 2, 0).numpy()
    else:
        rgb_np = rgb_image
    
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
    ax1.imshow(rgb_np)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(142, projection='3d')
    if isinstance(pointcloud, torch.Tensor):
        pc_np = pointcloud.cpu().numpy()
    else:
        pc_np = pointcloud
    
    ax2.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], 
               c=pc_np[:, 2], s=1, alpha=0.6, cmap='viridis')
    ax2.set_title('Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], 
               c='lightgray', s=0.5, alpha=0.3)
    
    for i, box in enumerate(target_boxes):
        draw_3d_bbox(ax3, box, color='green', alpha=0.7, linewidth=2,
                    label='Ground Truth' if i == 0 else "")
    
    ax3.set_title('Ground Truth Boxes')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    if len(target_boxes) > 0:
        ax3.legend()
    
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], 
               c='lightgray', s=0.5, alpha=0.3)
    
    for i, box in enumerate(target_boxes):
        draw_3d_bbox(ax4, box, color='green', alpha=0.5, linewidth=1,
                    label='Ground Truth' if i == 0 else "")
    
    for i, box in enumerate(pred_boxes):
        conf_text = f" (conf: {pred_confidences[i]:.2f})" if pred_confidences is not None else ""
        draw_3d_bbox(ax4, box, color='red', alpha=0.8, linewidth=2,
                    label=f'Prediction{conf_text}' if i == 0 else "")
    
    ax4.set_title('Predictions vs Ground Truth')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    if len(pred_boxes) > 0 or len(target_boxes) > 0:
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def draw_3d_bbox(ax, bbox, color='red', alpha=0.7, linewidth=1, label=None):
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
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    for i, edge in enumerate(edges):
        points = translated_corners[edge]
        ax.plot3D(*points.T, color=color, alpha=alpha, linewidth=linewidth,
                 label=label if i == 0 and label else None)


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


def visualize_confidence_distribution(confidences, save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_conf = sorted(confidences, reverse=True)
    plt.plot(range(len(sorted_conf)), sorted_conf, 'b-', linewidth=2)
    plt.xlabel('Prediction Rank')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores (Sorted)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_detection_summary_plot(metrics, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    performance_metrics = ['Precision', 'Recall', 'F1-Score']
    performance_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    bars1 = ax1.bar(performance_metrics, performance_values, 
                   color=['lightcoral', 'lightgreen', 'lightblue'])
    ax1.set_title('Detection Performance')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    for bar, value in zip(bars1, performance_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ap_metrics = ['AP@0.5', 'AP@0.75']
    ap_values = [metrics['ap_50'], metrics['ap_75']]
    
    bars2 = ax2.bar(ap_metrics, ap_values, color=['gold', 'orange'])
    ax2.set_title('Average Precision')
    ax2.set_ylabel('AP Score')
    ax2.set_ylim(0, 1)
    
    for bar, value in zip(bars2, ap_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax3.text(0.5, 0.7, f"Mean IoU: {metrics['mean_iou']:.3f}", 
            transform=ax3.transAxes, ha='center', va='center', 
            fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax3.text(0.5, 0.3, f"Total Detections: {metrics['total_predictions']}\nTotal Targets: {metrics['total_targets']}", 
            transform=ax3.transAxes, ha='center', va='center', 
            fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax3.set_title('Summary Statistics')
    ax3.axis('off')
    
    if 'iou_distribution' in metrics and len(metrics['iou_distribution']) > 0:
        ax4.hist(metrics['iou_distribution'], bins=20, alpha=0.7, 
                color='purple', edgecolor='black')
        ax4.set_xlabel('IoU Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('IoU Distribution')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No IoU data available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=14)
        ax4.set_title('IoU Distribution')
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def project_3d_to_2d(points_3d, camera_matrix, image_size):
    if len(points_3d) == 0:
        return np.array([])
    
    points_3d_homogeneous = np.column_stack([points_3d, np.ones(len(points_3d))])
    points_2d_homogeneous = camera_matrix @ points_3d_homogeneous.T
    points_2d = points_2d_homogeneous[:2] / points_2d_homogeneous[2]
    points_2d = points_2d.T
    
    valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_size[1]) & 
                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_size[0]))
    
    return points_2d[valid_mask]


def overlay_2d_boxes_on_image(image, boxes_3d, camera_matrix, save_path=None):
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image.copy()
    
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    
    for box in boxes_3d:
        corners_3d = get_bbox_corners_for_projection(box)
        corners_2d = project_3d_to_2d(corners_3d, camera_matrix, img_np.shape[:2])
        
        if len(corners_2d) >= 4:
            hull = cv2.convexHull(corners_2d.astype(np.int32))
            cv2.drawContours(img_np, [hull], -1, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    return img_np


def get_bbox_corners_for_projection(bbox):
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