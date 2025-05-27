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
from utils.utils import visualize_predictions, reconstruct_corners_tensor


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=5.0):
    model.train()
    running_stats = {}
    n_batches = len(dataloader)

    progress = tqdm(enumerate(dataloader), total=n_batches, desc=f"Epoch {epoch} [Train]")

    for i, batch in progress:
        rgb = batch['rgb'].to(device)
        pc = batch['pointcloud'].to(device)
        gt_bbox = batch['bbox_params'].to(device)
        gt_conf = (gt_bbox[:, :, :3].abs().sum(dim=-1) > 0).float()

        optimizer.zero_grad()

        pred_bbox, pred_conf = model(rgb, pc)
        loss, stats = criterion(pred_bbox, pred_conf, gt_bbox, gt_conf)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in stats.items():
            running_stats[k] = running_stats.get(k, 0.0) + v.item()

        progress.set_postfix({k: f"{running_stats[k]/(i+1):.4f}" for k in running_stats})

    return {k: v / n_batches for k, v in running_stats.items()}

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_stats = {}
    n_batches = len(dataloader)

    progress = tqdm(enumerate(dataloader), total=n_batches, desc=f"Epoch {epoch} [Val]")

    for i, batch in progress:
        rgb = batch['rgb'].to(device)
        pc = batch['pointcloud'].to(device)
        gt_bbox = batch['bbox_params'].to(device)
        gt_conf = (gt_bbox[:, :, :3].abs().sum(dim=-1) > 0).float()

        pred_bbox, pred_conf = model(rgb, pc)
        loss, stats = criterion(pred_bbox, pred_conf, gt_bbox, gt_conf)

        for k, v in stats.items():
            running_stats[k] = running_stats.get(k, 0.0) + v.item()

        progress.set_postfix({k: f"{running_stats[k]/(i+1):.4f}" for k in running_stats})

    return {k: v / n_batches for k, v in running_stats.items()}


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config):
    seed_everything()

    device = torch.device(config['device'])

    # 1. Collect all sample folders
    all_folders = sorted(glob.glob(os.path.join(config['data_root'], '*')))
    random.shuffle(all_folders)

    # 2. Train/val/test split
    N = len(all_folders)
    n_train = int(N * config['train_split'])
    n_val   = int(N * config['val_split'])

    train_folders = all_folders[:n_train]
    val_folders   = all_folders[n_train:n_train+n_val]
    test_folders  = all_folders[n_train+n_val:]
    

    # 3. Datasets and DataLoaders
    train_set = BBox3DDataset(train_folders, config, split='train')
    val_set   = BBox3DDataset(val_folders, config, split='val')

    train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])
    val_loader   = DataLoader(val_set, batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])

    # 4. Model, Loss, Optimizer
    model = BBox3DPredictor(config).to(device)

    # Freeze EfficientNet for now
    for param in model.rgb_backbone.parameters():
        param.requires_grad = False

    criterion = BBox3DLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config['learning_rate'], weight_decay=1e-4)

    # 5. Training loop
    best_val_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        if epoch == 10:
            print("🔓 Unfreezing last EfficientNet blocks...")
            for name, param in model.rgb_backbone.named_parameters():
                if 'blocks.6' in name or 'blocks.5' in name:
                    param.requires_grad = True

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        val_stats = validate_one_epoch(model, val_loader, criterion, device, epoch)
        if epoch % 1 == 0:
            model.eval()
            batch = next(iter(val_loader))

            rgb = batch['rgb'].to(device)
            pc = batch['pointcloud'].to(device)
            gt_bbox = batch['bbox_corners'].to(device)
            gt_conf = (gt_bbox[:, :, :3].abs().sum(dim=-1) > 0).float()

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

        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"Train Loss: {train_stats['total']:.4f} | Val Loss: {val_stats['total']:.4f}")

        # Save best model
        if val_stats['total'] < best_val_loss:
            best_val_loss = val_stats['total']
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ Saved best model")

    print("🎉 Training complete.")

