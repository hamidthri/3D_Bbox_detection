import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

from config.config import CONFIG
from datasets.custom_dataset import EnhancedBBox3DDataset
from models.detr3d import Custom3DETR
from utils.metrics import calculate_3d_iou, calculate_ap_3d
from utils.visualizer import visualize_3d_predictions


class ModelEvaluator:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device(config['device'])
        
        self.model = Custom3DETR(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.25
        
    def load_test_data(self):
        data_root = self.config['data_root']
        folder_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) 
                       if os.path.isdir(os.path.join(data_root, f))]
        
        with open('data/splits/test.txt', 'r') as f:
            test_folders = [line.strip() for line in f.readlines()]
        
        test_paths = [os.path.join(data_root, folder) for folder in test_folders 
                     if os.path.exists(os.path.join(data_root, folder))]
        
        test_dataset = EnhancedBBox3DDataset(test_paths, self.config, split='test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        return test_loader
    
    def evaluate_model(self):
        test_loader = self.load_test_data()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_ious = []
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        wandb.init(project="3d-bbox-evaluation", config=self.config)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                rgb = batch['rgb'].to(self.device)
                pointcloud = batch['pointcloud'].to(self.device)
                bbox_target = batch['bbox_params'].cpu().numpy()[0]
                num_objects = batch['num_objects'].item()
                
                bbox_pred, conf_pred = self.model(rgb, pointcloud)
                
                bbox_pred = bbox_pred.cpu().numpy()[0]
                conf_pred = torch.sigmoid(conf_pred).cpu().numpy()[0]
                
                valid_targets = bbox_target[:num_objects]
                
                confident_preds = conf_pred > self.confidence_threshold
                confident_boxes = bbox_pred[confident_preds]
                confident_scores = conf_pred[confident_preds]
                
                if len(confident_boxes) > 0 and len(valid_targets) > 0:
                    ious = self.calculate_batch_iou(confident_boxes, valid_targets)
                    max_ious = np.max(ious, axis=1) if len(ious) > 0 else []
                    
                    for i, iou in enumerate(max_ious):
                        if iou > self.iou_threshold:
                            true_positives += 1
                        else:
                            false_positives += 1
                        all_ious.append(iou)
                    
                    false_negatives += max(0, len(valid_targets) - len(max_ious))
                else:
                    false_negatives += len(valid_targets)
                    false_positives += len(confident_boxes)
                
                all_predictions.extend(confident_boxes)
                all_targets.extend(valid_targets)
                all_confidences.extend(confident_scores)
                
                if batch_idx < 5:
                    self.visualize_sample(
                        rgb[0], pointcloud[0], confident_boxes, 
                        valid_targets, batch_idx
                    )
        
        metrics = self.calculate_metrics(
            true_positives, false_positives, false_negatives, 
            all_confidences, all_ious
        )
        
        self.log_metrics(metrics)
        wandb.finish()
        
        return metrics
    
    def calculate_batch_iou(self, pred_boxes, target_boxes):
        ious = []
        for pred_box in pred_boxes:
            box_ious = []
            for target_box in target_boxes:
                iou = calculate_3d_iou(pred_box, target_box)
                box_ious.append(iou)
            ious.append(box_ious)
        return np.array(ious)
    
    def calculate_metrics(self, tp, fp, fn, confidences, ious):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        mean_iou = np.mean(ious) if len(ious) > 0 else 0
        
        ap_50 = self.calculate_ap_at_iou(0.5, confidences, ious)
        ap_75 = self.calculate_ap_at_iou(0.75, confidences, ious)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'ap_50': ap_50,
            'ap_75': ap_75,
            'total_predictions': tp + fp,
            'total_targets': tp + fn
        }
        
        return metrics
    
    def calculate_ap_at_iou(self, iou_thresh, confidences, ious):
        if len(confidences) == 0 or len(ious) == 0:
            return 0
        
        labels = [1 if iou > iou_thresh else 0 for iou in ious]
        
        if len(set(labels)) < 2:
            return 0
        
        try:
            ap = average_precision_score(labels, confidences)
            return ap
        except:
            return 0
    
    def visualize_sample(self, rgb, pointcloud, pred_boxes, target_boxes, sample_idx):
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131)
        rgb_np = rgb.cpu().permute(1, 2, 0).numpy()
        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
        ax1.imshow(rgb_np)
        ax1.set_title('RGB Image')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(132, projection='3d')
        pc_np = pointcloud.cpu().numpy()
        ax2.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], 
                   c=pc_np[:, 2], s=1, alpha=0.6)
        ax2.set_title('Point Cloud')
        
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], 
                   c='lightgray', s=0.5, alpha=0.3)
        
        for box in target_boxes:
            self.draw_3d_bbox(ax3, box, color='green', label='GT')
        
        for box in pred_boxes:
            self.draw_3d_bbox(ax3, box, color='red', label='Pred')
        
        ax3.set_title('3D Bounding Boxes')
        ax3.legend()
        
        plt.tight_layout()
        wandb.log({f"sample_{sample_idx}": wandb.Image(plt)})
        plt.close()
    
    def draw_3d_bbox(self, ax, bbox, color='red', label=None):
        x, y, z, w, h, d, rx, ry, rz = bbox
        
        corners = np.array([
            [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2],
            [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
            [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
            [w/2, h/2, d/2], [-w/2, h/2, d/2]
        ])
        
        rotation_matrix = self.get_rotation_matrix(rx, ry, rz)
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([x, y, z])
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for edge in edges:
            points = translated_corners[edge]
            ax.plot3D(*points.T, color=color, alpha=0.7)
    
    def get_rotation_matrix(self, rx, ry, rz):
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
    
    def log_metrics(self, metrics):
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"AP@0.5: {metrics['ap_50']:.4f}")
        print(f"AP@0.75: {metrics['ap_75']:.4f}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Targets: {metrics['total_targets']}")
        print("="*50)
        
        wandb.log({
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "mean_iou": metrics['mean_iou'],
            "ap_50": metrics['ap_50'],
            "ap_75": metrics['ap_75']
        })
        
        metrics_df = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Mean IoU', 'AP@0.5', 'AP@0.75'],
            'Value': [metrics['precision'], metrics['recall'], metrics['f1_score'], 
                     metrics['mean_iou'], metrics['ap_50'], metrics['ap_75']]
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_df['Metric'], metrics_df['Value'], 
                      color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'orange'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, metrics_df['Value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"metrics_summary": wandb.Image(plt)})
        plt.savefig('outputs/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    evaluator = ModelEvaluator(CONFIG, 'checkpoints/best_model.pth')
    
    os.makedirs('outputs', exist_ok=True)
    
    metrics = evaluator.evaluate_model()
    
    with open('outputs/evaluation_results.txt', 'w') as f:
        f.write("3D BBox Detection - Evaluation Results\n")
        f.write("="*40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")


if __name__ == "__main__":
    main()