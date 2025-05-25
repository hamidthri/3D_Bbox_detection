import torch

CONFIG = {
    'data_root': 'data/dl_challenge',
    'max_objects': 21,
    'batch_size': 4,
    'num_epochs': 150,
    'learning_rate': 5e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_size': (480, 608),
    'max_points': 4096,
    'train_split': 0.85,
    'val_split': 0.1,
    'test_split': 0.05,
    'model_params': {
        'backbone_pretrained': True,
        'fusion_dim': 256,
        'num_transformer_layers': 4,
        'dropout': 0.15,
        'point_feat_dim': 64,
        'img_feat_dim': 128
    },
    'loss_weights': {
        'bbox': 1.0,
        'conf': 5.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'iou': 1.0
    },
    'augmentation': {
        'point_noise_std': 0.01,
        'point_dropout_ratio': 0.05,
        'rgb_aug_prob': 0.7
    }
}
