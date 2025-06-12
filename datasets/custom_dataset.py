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
from tqdm import tqdm
from torchvision import transforms


class PointCloudBBox3DDataset(Dataset):
    def __init__(self, folder_paths, config, split='train'):
        self.folder_paths = folder_paths
        self.config = config
        self.split = split

        self.rotation_degrees = config.get('rotation_degrees', 90)
        self.do_rotation = config.get('do_rotation', True)

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]

        if self.split == 'train' and idx % 10 == 0:
            tqdm.write(f"[{self.split.upper()}] Loading index {idx} → {os.path.basename(folder_path)}")

        try:
            bbox3d = np.load(os.path.join(folder_path, 'bbox3d.npy'))
            pc = np.load(os.path.join(folder_path, 'pc.npy'))

            pc = np.transpose(pc, (1, 2, 0)).reshape(-1, 3)
            valid = ~np.isnan(pc).any(axis=1) & (pc[:, 2] > 0)
            pc = pc[valid]

            if self.split == 'train' and self.do_rotation:
                angles = tuple(
                        random.uniform(-self.rotation_degrees, self.rotation_degrees)
                        for _ in range(3)
                )
                pc = torch.from_numpy(pc).float()
                corners = torch.from_numpy(bbox3d).float()

                r = R.from_euler('xyz', angles, degrees=True)
                rot_mat = torch.from_numpy(r.as_matrix()).float()

                pc = torch.matmul(pc, rot_mat.T)
                corners = torch.matmul(corners.view(-1, 3), rot_mat.T).view(corners.shape)

                pc = pc.numpy()
                bbox3d = corners.numpy()

            if len(pc) > self.config['max_points']:
                indices = np.random.choice(len(pc), self.config['max_points'], replace=False)
                pc = pc[indices]
            elif len(pc) < self.config['max_points']:
                pad = np.zeros((self.config['max_points'] - len(pc), 3))
                pc = np.vstack([pc, pad])

            centers = np.mean(bbox3d, axis=1)

            max_objects = self.config['max_objects']
            num_objects = centers.shape[0]

            if num_objects < max_objects:
                pad = max_objects - num_objects
                centers = np.concatenate([centers, np.zeros((pad, 3))], axis=0)
            elif num_objects > max_objects:
                centers = centers[:max_objects]
                num_objects = max_objects

            pc = torch.from_numpy(pc).float()
            centers = torch.from_numpy(centers).float()
            num_objects = torch.tensor(num_objects, dtype=torch.long)

            bbox_corners = torch.zeros((max_objects, 8, 3), dtype=torch.float32)
            n = min(num_objects, max_objects)
            bbox_corners[:n] = torch.from_numpy(bbox3d[:n]).float()

            return {
                'pointcloud': pc,
                'bbox_centers': centers,
                'num_objects': num_objects,
                'bbox_corners': bbox_corners
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {folder_path}: {e}")
            with open("bad_samples.txt", "a") as f:
                f.write(folder_path + "\n")
            return {
                'pointcloud': torch.zeros(self.config['max_points'], 3),
                'bbox_centers': torch.zeros(self.config['max_objects'], 3),
                'num_objects': torch.tensor(0, dtype=torch.long),
                'bbox_corners': torch.zeros(self.config['max_objects'], 8, 3, dtype=torch.float32)
            }
