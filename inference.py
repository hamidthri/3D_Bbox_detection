from models.detr3d import BBox3DPredictor
from datasets.custom_dataset import BBox3DDataset
from utils.evaluator import BBox3DEvaluator, create_evaluation_report
from utils.visualizer_inf import visualize_3d_predictions
from utils.demo import run_inference_demo
from config.config import CONFIG
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def create_data_splits(data_root, config):
    """You can keep this here or import it from train.py"""
    import os
    from sklearn.model_selection import train_test_split

    all_folders = [os.path.join(data_root, f) for f in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, f))]

    train_folders, temp_folders = train_test_split(
        all_folders, test_size=(config['val_split'] + config['test_split']),
        random_state=42
    )

    val_folders, test_folders = train_test_split(
        temp_folders, test_size=config['test_split'] / (config['val_split'] + config['test_split']),
        random_state=42
    )

    return train_folders, val_folders, test_folders


def run_inference():
    _, _, test_folders = create_data_splits(CONFIG['data_root'], CONFIG)
    test_dataset = BBox3DDataset(test_folders, CONFIG, split='test')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = BBox3DPredictor(CONFIG).to(CONFIG['device'])
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = BBox3DEvaluator(model, CONFIG['device'])
    for batch in tqdm(test_loader, desc="Evaluating"):
        evaluator.evaluate_batch(batch)

    metrics = evaluator.get_summary_metrics()
    print("\n📋 Evaluation Report")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 🎯 Run inference demo visualizations
    run_inference_demo(model, test_loader, CONFIG['device'], num_samples=5)


if __name__ == "__main__":
    run_inference()
