import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.config import CONFIG
from datasets.custom_dataset import EnhancedBBox3DDataset
from models.detr3d import Custom3DETR
from losses.loss import Combined3DLoss
from torchvision import transforms
from datasets.custom_dataset import EnhancedBBox3DDataset


def create_data_loaders(config):
    data_root = config['data_root']
    folder_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, f))]
    
    train_paths, temp_paths = train_test_split(
        folder_paths, 
        test_size=1-config['train_split'], 
        random_state=42
    )
    
    val_size = config['val_split'] / (config['val_split'] + config['test_split'])
    val_paths, test_paths = train_test_split(
        temp_paths, 
        test_size=1-val_size, 
        random_state=42
    )
    
    train_dataset = EnhancedBBox3DDataset(train_paths, config, split='train')
    val_dataset = EnhancedBBox3DDataset(val_paths, config, split='val')
    test_dataset = EnhancedBBox3DDataset(test_paths, config, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    total_conf_loss = 0
    total_bbox_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        pointcloud = batch['pointcloud'].to(device)
        bbox_target = batch['bbox_params'].to(device)
        conf_target = batch['confidence_target'].to(device)
        num_objects = batch['num_objects'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            bbox_pred, conf_pred = model(rgb, pointcloud)
            loss, conf_loss, bbox_loss = criterion(
                bbox_pred, conf_pred, bbox_target, conf_target, num_objects
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_conf_loss += conf_loss.item()
        total_bbox_loss += bbox_loss.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Conf': f'{conf_loss.item():.4f}',
            'BBox': f'{bbox_loss.item():.4f}'
        })
    
    return total_loss / len(train_loader), total_conf_loss / len(train_loader), total_bbox_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_conf_loss = 0
    total_bbox_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            rgb = batch['rgb'].to(device)
            pointcloud = batch['pointcloud'].to(device)
            bbox_target = batch['bbox_params'].to(device)
            conf_target = batch['confidence_target'].to(device)
            num_objects = batch['num_objects'].to(device)
            
            with autocast():
                bbox_pred, conf_pred = model(rgb, pointcloud)
                loss, conf_loss, bbox_loss = criterion(
                    bbox_pred, conf_pred, bbox_target, conf_target, num_objects
                )
            
            total_loss += loss.item()
            total_conf_loss += conf_loss.item()
            total_bbox_loss += bbox_loss.item()
    
    return total_loss / len(val_loader), total_conf_loss / len(val_loader), total_bbox_loss / len(val_loader)

def main():
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = create_data_loaders(CONFIG)
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    model = Custom3DETR(CONFIG).to(device)
    criterion = Combined3DLoss(CONFIG)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 50)
        
        train_loss, train_conf_loss, train_bbox_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        
        val_loss, val_conf_loss, val_bbox_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Conf: {train_conf_loss:.4f}, BBox: {train_bbox_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Conf: {val_conf_loss:.4f}, BBox: {val_bbox_loss:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG
            }, 'checkpoints/best_model.pth')
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig('training_progress.png')
    plt.show()
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
