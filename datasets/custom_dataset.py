import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torchvision.transforms as T
import random
from utils.utils import convert_corners_to_params_tensor, BBoxCornerToParametric



class AddGaussianNoise:
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


class BBox3DDataset(Dataset):
    def __init__(self, folder_paths, config, split='train'):
        self.folder_paths = folder_paths
        self.config = config
        self.split = split
        self.converter = BBoxCornerToParametric()

        self.augment = split == 'train'

        # Augmentations that don't alter object positions
        self.safe_img_aug = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])

        # Final image pipeline
        self.img_transform = T.Compose([
            T.Resize(config['image_size']),
            T.ToTensor(),
            AddGaussianNoise(std=0.02),  # optional noise
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        try:
            rgb = cv2.imread(os.path.join(folder_path, 'rgb.jpg'))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            bbox3d = np.load(os.path.join(folder_path, 'bbox3d.npy'))
            pc = np.load(os.path.join(folder_path, 'pc.npy'))
            mask = np.load(os.path.join(folder_path, 'mask.npy'))

            pc = np.transpose(pc, (1, 2, 0)).reshape(-1, 3)
            valid = ~np.isnan(pc).any(axis=1) & (pc[:, 2] > 0)
            pc = pc[valid]

            if len(pc) > self.config['max_points']:
                indices = np.random.choice(len(pc), self.config['max_points'], replace=False)
                pc = pc[indices]
            elif len(pc) < self.config['max_points']:
                pad = np.zeros((self.config['max_points'] - len(pc), 3))
                pc = np.vstack([pc, pad])

            bbox_corners = torch.from_numpy(bbox3d).float()

            params = convert_corners_to_params_tensor(bbox_corners.unsqueeze(0), self.converter)
            centers = params['center'].squeeze(0)
            sizes = params['size'].squeeze(0)
            rotation_quats = params['rotation_quat'].squeeze(0)

            bbox_params = torch.cat([centers, sizes, rotation_quats], dim=-1)

            max_objects = self.config['max_objects']
            num_objects = bbox_params.shape[0]

            if num_objects < max_objects:
                pad = max_objects - num_objects
                bbox_params = F.pad(bbox_params, (0, 0, 0, pad), value=0)
                bbox_corners = F.pad(bbox_corners, (0, 0, 0, 0, 0, pad), value=0)
                mask = np.concatenate([mask, np.zeros((pad, *mask.shape[1:]), dtype=np.float32)], axis=0)
            elif num_objects > max_objects:
                bbox_params = bbox_params[:max_objects]
                bbox_corners = bbox_corners[:max_objects]
                mask = mask[:max_objects]
                num_objects = max_objects

            # Resize masks
            out_H, out_W = self.config['image_size']
            resized_masks = []
            for k in range(mask.shape[0]):
                m = mask[k].astype(np.float32)
                m = cv2.resize(m, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(m)
            mask = np.stack(resized_masks, axis=0)

            # Augment + transform image
            rgb = Image.fromarray(rgb)
            if self.augment:
                rgb = self.safe_img_aug(rgb)
            rgb = self.img_transform(rgb)

            # Convert pc and mask to tensors
            pc = torch.from_numpy(pc).float()
            mask = torch.from_numpy(mask).float()
            num_objects = torch.tensor(num_objects, dtype=torch.long)

            return {
                'rgb': rgb,
                'pointcloud': pc,
                'bbox_params': bbox_params,
                'bbox_corners': bbox_corners,
                'mask': mask,
                'num_objects': num_objects
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {folder_path}: {e}")
            return {
                'rgb': torch.zeros(3, *self.config['image_size']),
                'pointcloud': torch.zeros(self.config['max_points'], 3),
                'bbox_params': torch.zeros(self.config['max_objects'], 10),
                'bbox_corners': torch.zeros(self.config['max_objects'], 8, 3),
                'mask': torch.zeros(self.config['max_objects'], *self.config['image_size']),
                'num_objects': torch.tensor(0, dtype=torch.long)
            }
