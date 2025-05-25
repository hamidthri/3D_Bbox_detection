import torch

CONFIG = {
    'data_root': 'data/dl_challenge',
    'max_objects': 21, 
    'batch_size': 2,  # Reduced for better gradient estimates
    'num_epochs': 50,  # Reduced from 100
    'learning_rate': 5e-5,  # Reduced learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_size': (480, 608),
    'max_points': 1024,  # Reduced from 2048 
    'train_split': 0.7,   # Reduced to have more validation data
    'val_split': 0.2,     # Increased validation split
    'test_split': 0.1,    # Reduced test split
    
    'model_params': {
        'backbone_pretrained': True,
        'freeze_backbone': True,  # NEW: Freeze the backbone
        'backbone_type': 'mobilenet',  # Options: 'mobilenet', 'resnet18', 'resnet34'
        'fusion_dim': 64,  # Reduced from 128
        'num_transformer_layers': 1,  # Reduced from 2
        'dropout': 0.3,  # Reduced dropout in transformer
        'point_feat_dim': 32,
        'img_feat_dim': 32,  # Reduced from 64
        'point_dropout': 0.5,  # NEW: PointNet dropout
    },
    
    'loss_weights': {
        'bbox': 1.0,
        'conf': 1.5,  # Reduced from 2.0
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'iou': 0.5,  # Reduced IoU weight
        'l1_regularization': 1e-5,  # NEW: L1 regularization
        'l2_regularization': 1e-4,  # NEW: L2 regularization
    },
    
    'augmentation': {
        'point_noise_std': 0.01,  # Reduced noise
        'point_dropout_ratio': 0.05,  # Reduced dropout
        'rgb_aug_prob': 0.7,  # Reduced augmentation probability
        'mixup_alpha': 0.2,  # NEW: Mixup augmentation
    },
    
    'regularization': {
        'early_stopping_patience': 10,  # NEW: Early stopping
        'weight_decay': 1e-3,  # NEW: Weight decay
        'grad_clip_norm': 1.0,  # NEW: Gradient clipping
        'use_label_smoothing': True,  # NEW: Label smoothing
        'label_smoothing_factor': 0.1,
    },
    
    'training': {
        'warmup_epochs': 5,  # NEW: Learning rate warmup
        'scheduler_type': 'cosine_with_warmup',  # NEW: Better scheduler
        'min_lr': 1e-6,
        'accumulate_grad_batches': 2,  # NEW: Gradient accumulation
    }
}