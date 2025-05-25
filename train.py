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


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def mixup_data(x1, x2, y_bbox, y_conf, alpha=1.0):
    """Apply mixup augmentation to the data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    
    # For bbox and conf, we use the original targets (no mixing for structured outputs)
    return mixed_x1, mixed_x2, y_bbox, y_conf, lam


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Cosine schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_factor)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    
    print(f"Data splits - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    train_dataset = EnhancedBBox3DDataset(train_paths, config, split='train')
    val_dataset = EnhancedBBox3DDataset(val_paths, config, split='val')
    test_dataset = EnhancedBBox3DDataset(test_paths, config, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


def add_l1_l2_regularization(model, l1_lambda, l2_lambda):
    """Add L1 and L2 regularization to the loss"""
    l1_reg = torch.tensor(0., device=next(model.parameters()).device)
    l2_reg = torch.tensor(0., device=next(model.parameters()).device)
    
    for param in model.parameters():
        if param.requires_grad:
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
    
    return l1_lambda * l1_reg + l2_lambda * l2_reg


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, config):
    model.train()
    total_loss = 0
    total_conf_loss = 0
    total_bbox_loss = 0
    total_reg_loss = 0
    
    accumulate_steps = config['training']['accumulate_grad_batches']
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        pointcloud = batch['pointcloud'].to(device)
        bbox_target = batch['bbox_params'].to(device)
        conf_target = batch['confidence_target'].to(device)
        num_objects = batch['num_objects'].to(device)
        
        # Apply mixup augmentation
        if config['augmentation']['mixup_alpha'] > 0 and np.random.random() < 0.5:
            rgb, pointcloud, bbox_target, conf_target, lam = mixup_data(
                rgb, pointcloud, bbox_target, conf_target, 
                config['augmentation']['mixup_alpha']
            )
        
        with autocast():
            bbox_pred, conf_pred = model(rgb, pointcloud)
            loss, conf_loss, bbox_loss = criterion(
                bbox_pred, conf_pred, bbox_target, conf_target, num_objects
            )
            
            # Add regularization
            reg_loss = add_l1_l2_regularization(
                model, 
                config['loss_weights']['l1_regularization'],
                config['loss_weights']['l2_regularization']
            )
            
            total_loss_batch = loss + reg_loss
            # Scale loss for gradient accumulation
            total_loss_batch = total_loss_batch / accumulate_steps
        
        scaler.scale(total_loss_batch).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulate_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['regularization']['grad_clip_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_conf_loss += conf_loss.item()
        total_bbox_loss += bbox_loss.item()
        total_reg_loss += reg_loss.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Conf': f'{conf_loss.item():.4f}',
            'BBox': f'{bbox_loss.item():.4f}',
            'Reg': f'{reg_loss.item():.6f}'
        })
    
    return (total_loss / len(train_loader), 
            total_conf_loss / len(train_loader), 
            total_bbox_loss / len(train_loader),
            total_reg_loss / len(train_loader))


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
    
    return (total_loss / len(val_loader), 
            total_conf_loss / len(val_loader), 
            total_bbox_loss / len(val_loader))


def main():
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = create_data_loaders(CONFIG)
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    model = Custom3DETR(CONFIG).to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    criterion = Combined3DLoss(CONFIG)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['regularization']['weight_decay']
    )
    
    # Setup scheduler with warmup
    num_training_steps = len(train_loader) * CONFIG['num_epochs']
    num_warmup_steps = len(train_loader) * CONFIG['training']['warmup_epochs']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=CONFIG['training']['min_lr']
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=CONFIG['regularization']['early_stopping_patience'],
        min_delta=1e-4
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        train_loss, train_conf_loss, train_bbox_loss, train_reg_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, CONFIG
        )
        
        val_loss, val_conf_loss, val_bbox_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Conf: {train_conf_loss:.4f}, BBox: {train_bbox_loss:.4f}, Reg: {train_reg_loss:.6f})")
        print(f"Val Loss: {val_loss:.4f} (Conf: {val_conf_loss:.4f}, BBox: {val_bbox_loss:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Step scheduler after each batch (for warmup + cosine)
        if hasattr(scheduler, 'step'):
            scheduler.step()
        
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
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.array(val_losses) - np.array(train_losses), label='Overfitting Gap', color='red', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title('Overfitting Monitor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150)
    plt.show()
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

