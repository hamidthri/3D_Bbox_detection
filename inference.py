import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from config.config import CONFIG
from models.detr3d import Custom3DETR
from utils.visualizer import visualize_3d_predictions


class BBox3DInference:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        self.model = Custom3DETR(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.confidence_threshold = 0.5
        
    def preprocess_rgb(self, rgb_path):
        image = Image.open(rgb_path).convert('RGB')
        image = image.resize(self.config['image_size'][::-1])
        
        image_array = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
    
    def preprocess_pointcloud(self, pc_path):
        if pc_path.endswith('.npy'):
            pointcloud = np.load(pc_path)
        elif pc_path.endswith('.txt'):
            pointcloud = np.loadtxt(pc_path)
        else:
            raise ValueError("Unsupported point cloud format")

        # Handle case where pointcloud is (H, W, 3) or (C, H, W)
        if pointcloud.ndim == 3:
            pointcloud = pointcloud.reshape(-1, 3)

        if pointcloud.shape[1] > 3:
            pointcloud = pointcloud[:, :3]

        if len(pointcloud) > self.config['max_points']:
            indices = np.random.choice(len(pointcloud), self.config['max_points'], replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < self.config['max_points']:
            padding = np.zeros((self.config['max_points'] - len(pointcloud), 3))
            pointcloud = np.vstack([pointcloud, padding])

        return torch.FloatTensor(pointcloud).unsqueeze(0)

    
    def predict(self, rgb_path, pointcloud_path):
        rgb_tensor = self.preprocess_rgb(rgb_path).to(self.device)
        pc_tensor = self.preprocess_pointcloud(pointcloud_path).to(self.device)
        
        with torch.no_grad():
            bbox_pred, conf_pred = self.model(rgb_tensor, pc_tensor)
        
        bbox_pred = bbox_pred.cpu().numpy()[0]
        conf_pred = torch.sigmoid(conf_pred).cpu().numpy()[0]
        
        confident_indices = conf_pred > self.confidence_threshold
        confident_boxes = bbox_pred[confident_indices]
        confident_scores = conf_pred[confident_indices]
        
        results = []
        for i, (box, score) in enumerate(zip(confident_boxes, confident_scores)):
            results.append({
                'bbox': box,
                'confidence': score,
                'id': i
            })
        
        return results, rgb_tensor[0], pc_tensor[0]
    
    def visualize_predictions(self, results, rgb_tensor, pc_tensor, save_path=None):
        pred_boxes = [result['bbox'] for result in results]
        pred_confidences = [result['confidence'] for result in results]
        
        fig = visualize_3d_predictions(
            rgb_tensor, pc_tensor, pred_boxes, [], 
            pred_confidences, save_path
        )
        
        return fig
    
    def save_results(self, results, output_path):
        with open(output_path, 'w') as f:
            f.write("3D Bounding Box Predictions\n")
            f.write("="*50 + "\n")
            f.write(f"Total detections: {len(results)}\n\n")
            
            for i, result in enumerate(results):
                bbox = result['bbox']
                conf = result['confidence']
                
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Confidence: {conf:.4f}\n")
                f.write(f"  Center: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f})\n")
                f.write(f"  Size: ({bbox[3]:.3f}, {bbox[4]:.3f}, {bbox[5]:.3f})\n")
                f.write(f"  Rotation: ({bbox[6]:.3f}, {bbox[7]:.3f}, {bbox[8]:.3f})\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='3D BBox Detection Inference')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--pointcloud', required=True, help='Path to point cloud file')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--output', default='outputs/', help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    inference = BBox3DInference(args.model, CONFIG)
    inference.confidence_threshold = args.confidence
    
    print("Running inference...")
    results, rgb_tensor, pc_tensor = inference.predict(args.rgb, args.pointcloud)
    
    print(f"Found {len(results)} detections with confidence > {args.confidence}")
    
    for i, result in enumerate(results):
        print(f"  Detection {i+1}: confidence = {result['confidence']:.4f}")
    
    output_file = os.path.join(args.output, 'predictions.txt')
    inference.save_results(results, output_file)
    print(f"Results saved to {output_file}")
    
    if args.visualize:
        vis_path = os.path.join(args.output, 'prediction_visualization.png')
        fig = inference.visualize_predictions(results, rgb_tensor, pc_tensor, vis_path)
        print(f"Visualization saved to {vis_path}")
        plt.show()


if __name__ == "__main__":
    main()