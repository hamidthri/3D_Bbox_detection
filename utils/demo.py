import numpy as np
import torch
from utils.visualizer_inf import visualize_3d_predictions
from utils.utils import BBoxCornerToParametric
from datasets.custom_dataset import BBox3DDataset
from torch.utils.data import DataLoader

def run_inference_demo(model, test_loader, device, num_samples=5):
    """Run inference on test samples and create visualizations"""
    model.eval()
    converter = BBoxCornerToParametric()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            rgb = batch['rgb'].to(device)
            pointcloud = batch['pointcloud'].to(device)
            gt_bbox_params = batch['bbox_params'].numpy()

            pred_bbox_params, pred_conf = model(rgb, pointcloud)
            pred_bbox_params = pred_bbox_params.cpu().numpy()
            pred_conf = pred_conf.cpu().numpy()

            # Process first sample in batch
            sample_idx = 0
            rgb_img = rgb[sample_idx].cpu().numpy().transpose(1, 2, 0)

            # Denormalize RGB image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = rgb_img * std + mean
            rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)

            # Convert parameters to corners using quaternion logic
            gt_corners = []
            pred_corners = []
            gt_confs = []

            max_objects = gt_bbox_params.shape[1]

            for obj_idx in range(max_objects):
                gt_params = gt_bbox_params[sample_idx, obj_idx]
                pred_params = pred_bbox_params[sample_idx, obj_idx]

                if not (gt_params == 0).all():
                    gt_corner = converter.reconstruct_corners(
                        gt_params[:3], gt_params[3:6], gt_params[6:10]
                    )
                    pred_corner = converter.reconstruct_corners(
                        pred_params[:3], pred_params[3:6], pred_params[6:10]
                    )
                    gt_corners.append(gt_corner)
                    pred_corners.append(pred_corner)
                    gt_confs.append(1.0)
                else:
                    gt_corners.append(np.zeros((8, 3)))
                    pred_corners.append(np.zeros((8, 3)))
                    gt_confs.append(0.0)

            pred_confs = pred_conf[sample_idx]
            pc = pointcloud[sample_idx].cpu().numpy()

            # Create visualization
            visualize_3d_predictions(
                rgb_img, gt_corners, pred_corners, gt_confs, pred_confs,
                pointcloud=pc, save_path=f'prediction_demo_{i}.html'
            )
