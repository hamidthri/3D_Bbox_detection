import torch
CONFIG = {
    'data_root': '../data/dl_challenge',
    'max_objects': 21,
    'batch_size': 2,
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_size': (480, 608),
    'max_points': 4096,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'resume': False,
    'model_params': {
        'backbone_pretrained': True,
        'fusion_dim': 512,
        'num_transformer_layers': 8,
        'dropout': 0.4
    }
}