import torch
CONFIG = {
    'data_root': 'data/dl_challenge',
    'max_objects': 21,
    'batch_size': 1,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_size': (480, 608),
    'max_points': 8192,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'model_params': {
        'backbone_pretrained': True,
        'fusion_dim': 256,
        'num_transformer_layers': 4,
        'dropout': 0.2
    }
}