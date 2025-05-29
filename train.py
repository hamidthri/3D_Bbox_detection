import os
import numpy as np
import torch  # ✅ REQUIRED
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.config import CONFIG
from datasets.custom_dataset import BBox3DDataset
from models.detr3d import BBox3DPredictor
from losses.loss import BBox3DLoss
import random
import glob
from utils.utils import reconstruct_corners_tensor
from utils.utils import save_checkpoint, load_checkpoint
from utils.visualizer import visualize_pc_and_boxes_matplotlib
from utils.visualizer import visualize_predictions


def create_data_splits(data_root, config):
    """Create train/val/test splits"""
    all_folders = [os.path.join(data_root, f) for f in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, f))]

    # First split: train vs (val + test)
    train_folders, temp_folders = train_test_split(
        all_folders,
        test_size=(config['val_split'] + config['test_split']),
        random_state=42
    )

    # Second split: val vs test
    val_folders, test_folders = train_test_split(
        temp_folders,
        test_size=config['test_split'] / (config['val_split'] + config['test_split']),
        random_state=42
    )

    print(f"Data splits - Train: {len(train_folders)}, Val: {len(val_folders)}, Test: {len(test_folders)}")

    return train_folders, val_folders, test_folders


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    accumulation_steps = 4  # Effective batch size = batch_size * 4
    optimizer.zero_grad()

    running_stats = {}

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        rgb = batch['rgb'].to(device)
        pc = batch['pointcloud'].to(device)
        gt_bbox = batch['bbox_params'].to(device)
        gt_conf = (gt_bbox.abs().sum(dim=-1) > 1e-6).float()

        pred_bbox, pred_conf = model(rgb, pc)
        loss_dict = criterion(pred_bbox, pred_conf, gt_bbox, gt_conf)
        loss = loss_dict['total_loss']


        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update stats
        for k, v in loss_dict.items():
            running_stats[k] = running_stats.get(k, 0.0) + v.item()

    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    if scheduler:
        scheduler.step()

    return {k: v / len(dataloader) for k, v in running_stats.items()}


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_stats = {}
    iou_accumulator = []
    n_batches = len(dataloader)

    progress = tqdm(enumerate(dataloader), total=n_batches, desc=f"Epoch {epoch} [Val]")

    for i, batch in progress:
        rgb = batch['rgb'].to(device)
        pc = batch['pointcloud'].to(device)
        gt_bbox = batch['bbox_params'].to(device)
        gt_corners = batch['bbox_corners'].to(device)
        gt_conf = (gt_bbox[:, :, :3].abs().sum(dim=-1) > 0).float()

        pred_bbox, pred_conf = model(rgb, pc)
        loss_dict = criterion(pred_bbox, pred_conf, gt_bbox, gt_conf)
        loss = loss_dict['total_loss']

        pred_corners = reconstruct_corners_tensor(
            center=pred_bbox[:, :, :3],
            size=pred_bbox[:, :, 3:6],
            rotation_quat=pred_bbox[:, :, 6:10],
            device=device
        )

        # Visualize first batch sample
        if i == 0:
            b = 0
            pc_np = pc[b].cpu().numpy()
            gt_boxes = gt_corners[b][gt_conf[b] > 0].cpu().numpy()
            pred_boxes = pred_corners[b][pred_conf[b] > 0.5].cpu().numpy()
            os.makedirs("epoch_predictions", exist_ok=True)
            fig_path = f"epoch_predictions/mpl_epoch_{epoch}.png"
            visualize_pc_and_boxes_matplotlib(pc_np, gt_boxes, pred_boxes, path=fig_path)


        # Update loss stats
        for k, v in loss_dict.items():
            running_stats[k] = running_stats.get(k, 0.0) + v.item()

    final_stats = {k: v / n_batches for k, v in running_stats.items()}
    final_stats = {k: v / n_batches for k, v in running_stats.items()}

    return final_stats



def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config):
    seed_everything()
    device = torch.device(config['device'])

    # 1. Create data splits
    train_folders, val_folders, test_folders = create_data_splits(config['data_root'], config)
    train_dataset = BBox3DDataset(train_folders, config, split='train')
    val_dataset = BBox3DDataset(val_folders, config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    # 3. Model
    model = BBox3DPredictor(config).to(device)

    # 4. Loss & Optimizer
    criterion = BBox3DLoss()
    train_losses = []
    val_losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    # 6. Load checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    if config['resume']:
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, path="checkpoints/best_model.pth", device=device
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler)
        train_losses.append(train_stats['total_loss'])

        val_stats = validate_one_epoch(model, val_loader, criterion, device, epoch)
        val_losses.append(val_stats['total_loss'])

        print(f"Train - Total: {train_stats['total_loss']:.4f}, "
              f"BBox: {train_stats['bbox_loss']:.4f}, "
              f"Conf: {train_stats['conf_loss']:.4f}")
        print(f"Val - Total: {val_stats['total_loss']:.4f}, "
              f"BBox: {val_stats['bbox_loss']:.4f}, "
              f"Conf: {val_stats['conf_loss']:.4f}")

        # Plot training curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 4))

            # Full loss curves
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train', marker='o')
            plt.plot(val_losses, label='Val', marker='s')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)

            # Recent loss curves (last 50 epochs or less)
            plt.subplot(1, 2, 2)
            start = max(0, epoch - 49)
            recent_train = train_losses[start:]
            recent_val = val_losses[start:]
            x_vals = list(range(start_epoch + start, start_epoch + start + len(recent_train)))

            plt.plot(x_vals, recent_train, label='Train', marker='o')
            plt.plot(x_vals, recent_val, label='Val', marker='s')
            plt.title('Recent Training Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'training_progress_epoch_{epoch + 1}.png')
            plt.show()

        # Visualization for the first sample of val
        model.eval()
        batch = next(iter(val_loader))
        rgb = batch['rgb'].to(device)
        pc = batch['pointcloud'].to(device)
        gt_bbox = batch['bbox_corners'].to(device)
        gt_conf = (gt_bbox.abs().sum(dim=(-2, -1)) > 0).float()

        with torch.no_grad():
            pred_params, pred_conf = model(rgb, pc)
            pred_corners = reconstruct_corners_tensor(
                center=pred_params[:, :, :3],
                size=pred_params[:, :, 3:6],
                rotation_quat=pred_params[:, :, 6:10],
                device=device
            )

            fig = visualize_predictions(rgb, pred_corners, gt_bbox, pred_conf, gt_conf, sample_idx=0)
            os.makedirs("epoch_predictions", exist_ok=True)
            vis_path = f'epoch_predictions/epoch_{epoch}_viz.png'
            fig.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_stats['total_loss']:.4f} | Val Loss: {val_stats['total_loss']:.4f}")

        # Save best model checkpoint
        if val_stats['total_loss'] < best_val_loss:
            best_val_loss = val_stats['total_loss']
            save_checkpoint(model, optimizer, epoch, best_val_loss, path="checkpoints/best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved.")

    print("🎉 Training complete.")
