from sklearn.metrics import average_precision_score
from datetime import datetime
import numpy as np
import torch
from utils.utils import BBoxCornerToParametric
from utils.metrics import compute_3d_iou_accurate

class BBox3DEvaluator:
    def __init__(self, model, device, iou_thresholds=[0.25, 0.5, 0.7]):
        self.model = model
        self.device = device
        self.iou_thresholds = iou_thresholds
        self.convertor = BBoxCornerToParametric()
        self.reset_metrics()

    def reset_metrics(self):
        self.translation_errors, self.rotation_errors = [], []
        self.size_errors, self.ious = [], []
        self.confidences, self.is_valid = [], []

    def evaluate_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            rgb, pc = batch['rgb'].to(self.device), batch['pointcloud'].to(self.device)
            gt_params = batch['bbox_params'].numpy()
            pred_params, pred_conf = self.model(rgb, pc)
            pred_params, pred_conf = pred_params.cpu().numpy(), pred_conf.cpu().numpy()

            for b in range(rgb.shape[0]):
                for i in range(gt_params.shape[1]):
                    gt_box = gt_params[b, i]
                    pred_box = pred_params[b, i]
                    conf = pred_conf[b, i]

                    valid = not (gt_box == 0).all()
                    self.is_valid.append(valid)
                    self.confidences.append(conf)

                    if valid:
                        gt_corners = self.convertor.reconstruct_corners(gt_box[:3], gt_box[3:6], gt_box[6:10])
                        pred_corners = self.convertor.reconstruct_corners(pred_box[:3], pred_box[3:6], pred_box[6:10])

                        iou = compute_3d_iou_accurate(pred_corners, gt_corners)
                        self.ious.append(iou)
                        self.translation_errors.append(np.linalg.norm(pred_box[:3] - gt_box[:3]))
                        self.rotation_errors.append(np.linalg.norm(pred_box[6:10] - gt_box[6:10]))
                        self.size_errors.append(np.mean(np.abs(pred_box[3:6] - gt_box[3:6]) / (gt_box[3:6] + 1e-6)))
                    else:
                        self.ious.append(0)
                        self.translation_errors.append(0)
                        self.rotation_errors.append(0)
                        self.size_errors.append(0)


    def compute_average_precision(self):
        """Compute Average Precision at different IoU thresholds"""
        if not self.ious:
            return {}

        results = {}
        valid_indices = [i for i, valid in enumerate(self.is_valid) if valid]

        if not valid_indices:
            return results

        valid_ious = [self.ious[i] for i in valid_indices]
        valid_confs = [self.confidences[i] for i in valid_indices]

        for threshold in self.iou_thresholds:
            # Create binary labels (1 if IoU > threshold, 0 otherwise)
            binary_labels = [1 if iou > threshold else 0 for iou in valid_ious]

            if sum(binary_labels) > 0:  # Only compute if there are positive samples
                ap = average_precision_score(binary_labels, valid_confs)
                results[f'AP@{threshold}'] = ap

        return results

    def get_summary_metrics(self):
        """Get summary of all metrics"""
        if not self.translation_errors:
            return {"error": "No valid predictions to evaluate"}

        metrics = {
            'mean_translation_error': np.mean(self.translation_errors),
            'mean_rotation_error': np.mean(self.rotation_errors),
            'mean_size_error': np.mean(self.size_errors),
            'mean_3d_iou': np.mean(self.ious),
            'std_translation_error': np.std(self.translation_errors),
            'std_rotation_error': np.std(self.rotation_errors),
            'std_size_error': np.std(self.size_errors),
            'std_3d_iou': np.std(self.ious),
        }

        # Add AP metrics
        ap_metrics = self.compute_average_precision()
        metrics.update(ap_metrics)

        return metrics

def create_evaluation_report(evaluator, model_name="3D BBox Predictor"):
    """Create comprehensive evaluation report"""
    metrics = evaluator.get_summary_metrics()

    if "error" in metrics:
        return {"error": metrics["error"]}

    # Create report
    report = {
        "model_name": model_name,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "summary": {
            "total_predictions": len(evaluator.translation_errors),
            "mean_3d_iou": metrics["mean_3d_iou"],
            "mean_translation_error_m": metrics["mean_translation_error"],
            "mean_rotation_error_rad": metrics["mean_rotation_error"],
            "mean_size_error_relative": metrics["mean_size_error"]
        }
    }

    # Add interpretation
    interpretation = {
        "3d_iou": "Higher is better (0-1 range)",
        "translation_error": "Lower is better (meters)",
        "rotation_error": "Lower is better (radians)",
        "size_error": "Lower is better (relative error)"
    }

    report["interpretation"] = interpretation

    return report
