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
from tqdm import tqdm
from torchvision import transforms

import os
import cv2
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import torchvision.transforms as T
from utils.utils import convert_corners_to_params_tensor, BBoxCornerToParametric


class BBox3DDataset(Dataset):
    def __init__(self, folder_paths, config, split='train'):
        self.folder_paths = folder_paths
        self.config = config
        self.split = split
        self.converter = BBoxCornerToParametric()

        self.rotation_degrees = config.get('rotation_degrees', 90)
        self.do_rotation = config.get('do_rotation', True)

        # Image transforms
        if split == 'train':
            self.img_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(config['image_size']),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomResizedCrop(config['image_size'], scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(config['image_size']),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]

        if self.split == 'train' and idx % 1 == 0:
            tqdm.write(f"[{self.split.upper()}] Loading index {idx} → {os.path.basename(folder_path)}")

        try:
            # --- Load Data ---
            rgb = cv2.imread(os.path.join(folder_path, 'rgb.jpg'))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            bbox3d = np.load(os.path.join(folder_path, 'bbox3d.npy'))  # (N, 8, 3)
            pc = np.load(os.path.join(folder_path, 'pc.npy'))          # (3, H, W)
            mask = np.load(os.path.join(folder_path, 'mask.npy'))      # (N, H, W)

            # --- Process Point Cloud ---
            pc = np.transpose(pc, (1, 2, 0)).reshape(-1, 3)
            valid = ~np.isnan(pc).any(axis=1) & (pc[:, 2] > 0)
            pc = pc[valid]

            # --- Apply Z-Rotation Augmentation ---
            if self.split == 'train' and self.do_rotation:
                angles = tuple(
                        random.uniform(-self.rotation_degrees, self.rotation_degrees)
                        for _ in range(3)
)
                print(f"🔁 Augmenting idx {idx} with rotation {angles}")

                pc = torch.from_numpy(pc).float()
                corners = torch.from_numpy(bbox3d).float()

                r = R.from_euler('xyz', angles, degrees=True)
                rot_mat = torch.from_numpy(r.as_matrix()).float()

                pc = torch.matmul(pc, rot_mat.T)
                corners = torch.matmul(corners.view(-1, 3), rot_mat.T).view(corners.shape)

                pc = pc.numpy()
                bbox3d = corners.numpy()

            # --- Sample/Pad Point Cloud ---
            if len(pc) > self.config['max_points']:
                indices = np.random.choice(len(pc), self.config['max_points'], replace=False)
                pc = pc[indices]
            elif len(pc) < self.config['max_points']:
                pad = np.zeros((self.config['max_points'] - len(pc), 3))
                pc = np.vstack([pc, pad])

            # --- Convert Corners → (center, size, quaternion) ---
            bbox3d_tensor = torch.from_numpy(bbox3d).float().unsqueeze(0)
            params = convert_corners_to_params_tensor(bbox3d_tensor, self.converter)
            centers = params['center'].squeeze(0)
            sizes = params['size'].squeeze(0)
            quats = params['rotation_quat'].squeeze(0)

            bbox_params = torch.cat([centers, sizes, quats], dim=-1)

            # --- Pad/Trim BBoxes and Masks ---
            max_objects = self.config['max_objects']
            num_objects = bbox_params.shape[0]

            if num_objects < max_objects:
                pad = max_objects - num_objects
                bbox_params = F.pad(bbox_params, (0, 0, 0, pad), value=0)
                mask = np.concatenate([mask, np.zeros((pad, *mask.shape[1:]), dtype=np.float32)], axis=0)
            elif num_objects > max_objects:
                bbox_params = bbox_params[:max_objects]
                mask = mask[:max_objects]
                num_objects = max_objects

            # --- Resize Masks ---
            out_H, out_W = self.config['image_size']
            resized_masks = [
                cv2.resize(mask[k].astype(np.float32), (out_W, out_H), interpolation=cv2.INTER_NEAREST)
                for k in range(mask.shape[0])
            ]
            mask = np.stack(resized_masks, axis=0)

            # --- Final Conversion ---
            rgb = self.img_transform(rgb)
            pc = torch.from_numpy(pc).float()
            mask = torch.from_numpy(mask).float()
            num_objects = torch.tensor(num_objects, dtype=torch.long)

            bbox_corners = torch.zeros((max_objects, 8, 3), dtype=torch.float32)
            n = min(num_objects, max_objects)
            bbox_corners[:n] = torch.from_numpy(bbox3d[:n]).float()

            return {
                'rgb': rgb,
                'pointcloud': pc,
                'bbox_params': bbox_params,
                'mask': mask,
                'num_objects': num_objects,
                'bbox_corners': bbox_corners
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {folder_path}: {e}")
            with open("bad_samples.txt", "a") as f:
                f.write(folder_path + "\n")
            return {
                'rgb': torch.zeros(3, *self.config['image_size']),
                'pointcloud': torch.zeros(self.config['max_points'], 3),
                'bbox_params': torch.zeros(self.config['max_objects'], 10),
                'mask': torch.zeros(self.config['max_objects'], *self.config['image_size']),
                'num_objects': torch.tensor(0, dtype=torch.long),
                'bbox_corners': torch.zeros(self.config['max_objects'], 8, 3, dtype=torch.float32)
            }
