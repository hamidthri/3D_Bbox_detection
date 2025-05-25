import torch
CONFIG = {
    'data_root': 'data/dl_challenge',
    'max_objects': 21, 
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_size': (480, 608),
    'max_points': 2048, 
    'train_split': 0.8,
    'val_split': 0.15,
    'test_split': 0.05,
    'model_params': {
        'backbone_pretrained': True,
        'fusion_dim': 128,
        'num_transformer_layers': 2,
        'dropout': 0.5,
        'point_feat_dim': 32,
        'img_feat_dim': 64 
    },
    'loss_weights': {
        'bbox': 1.0,
        'conf': 2.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'iou': 1.0
    },
    'augmentation': {
        'point_noise_std': 0.02,
        'point_dropout_ratio': 0.1,
        'rgb_aug_prob': 0.9
    }
}