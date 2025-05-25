import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms



def corners_to_center_size_yaw(corners):
    """Convert 8x3 corner points to center, size, rotation parameters"""
    center = np.mean(corners, axis=0)
    
    edges = np.array([
        corners[1] - corners[0],  # width
        corners[3] - corners[0],  # length  
        corners[4] - corners[0]   # height
    ])
    
    sizes = np.linalg.norm(edges, axis=1)
    
    front_vec = corners[1] - corners[0]
    front_vec = front_vec / np.linalg.norm(front_vec)
    yaw = np.arctan2(front_vec[1], front_vec[0])
    
    return np.concatenate([center, sizes, [yaw, 0, 0]])

class EnhancedBBox3DDataset(Dataset):
    def __init__(self, folder_paths, config, split='train'):
        self.folder_paths = folder_paths
        self.config = config
        self.split = split
        self.point_augment = split == 'train'

        if split == 'train':
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(config['image_size']),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.RandomCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def normalize_bbox_params(self, bbox_params):
        """Normalize bbox parameters for better training stability"""
        if len(bbox_params) == 0:
            return bbox_params
        
        bbox_params[:, :3] = bbox_params[:, :3] / 5.0  # Normalize positions
        bbox_params[:, 3:6] = bbox_params[:, 3:6] / 2.0  # Normalize sizes
        bbox_params[:, 6:] = bbox_params[:, 6:] / np.pi  # Normalize rotations
        return bbox_params

    def augment_pointcloud(self, pc):
        """Enhanced point cloud augmentation"""
        if not self.point_augment:
            return pc

        noise = np.random.normal(0, self.config['augmentation']['point_noise_std'], pc.shape)
        pc = pc + noise

        dropout_ratio = self.config['augmentation']['point_dropout_ratio']
        keep_mask = np.random.random(len(pc)) > dropout_ratio
        if keep_mask.sum() > self.config['max_points'] // 2:
            pc = pc[keep_mask]

        return pc

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]

        try:
            rgb = cv2.imread(os.path.join(folder_path, 'rgb.jpg'))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            bbox3d = np.load(os.path.join(folder_path, 'bbox3d.npy'))
            pc = np.load(os.path.join(folder_path, 'pc.npy'))
            
            try:
                mask = np.load(os.path.join(folder_path, 'mask.npy'))
            except:
                mask = np.array([])

            pc = np.transpose(pc, (1, 2, 0)).reshape(-1, 3)
            valid = ~np.isnan(pc).any(axis=1) & (pc[:, 2] > 0) & (pc[:, 2] < 10)
            pc = pc[valid]

            pc_mean = np.mean(pc, axis=0)
            pc_std = np.std(pc, axis=0) + 1e-8
            pc = (pc - pc_mean) / pc_std

            pc = self.augment_pointcloud(pc)

            if len(pc) > self.config['max_points']:
                indices = np.random.choice(len(pc), self.config['max_points'], replace=False)
                pc = pc[indices]
            elif len(pc) < self.config['max_points']:
                if len(pc) > 0:
                    repeat_factor = self.config['max_points'] // len(pc) + 1
                    pc = np.tile(pc, (repeat_factor, 1))[:self.config['max_points']]
                else:
                    pc = np.zeros((self.config['max_points'], 3))

            valid_bbox_mask = ~(bbox3d == 0).all(axis=(1, 2))
            valid_bboxes = bbox3d[valid_bbox_mask]

            if len(valid_bboxes) > 0:
                bbox_params = np.array([corners_to_center_size_yaw(b) for b in valid_bboxes])
                bbox_params = self.normalize_bbox_params(bbox_params)
            else:
                bbox_params = np.zeros((0, 9))

            max_objects = self.config['max_objects']
            num_objects = len(bbox_params)

            if num_objects < max_objects:
                pad = max_objects - num_objects
                bbox_params = np.pad(bbox_params, ((0, pad), (0, 0)), mode='constant')
                confidence_target = np.zeros(max_objects)
                confidence_target[:num_objects] = 1.0
            else:
                bbox_params = bbox_params[:max_objects]
                num_objects = max_objects
                confidence_target = np.ones(max_objects)

            out_H, out_W = self.config['image_size']
            if len(mask) > 0 and len(mask) > max_objects:
                mask = mask[:max_objects]

            resized_masks = []
            if len(mask) > 0:
                for k in range(min(len(mask), max_objects)):
                    m = mask[k].astype(np.float32)
                    m = cv2.resize(m, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(m)

            while len(resized_masks) < max_objects:
                resized_masks.append(np.zeros((out_H, out_W), dtype=np.float32))

            mask = np.stack(resized_masks[:max_objects], axis=0)

            rgb = self.img_transform(rgb)
            pc = torch.from_numpy(pc).float()
            bbox_params = torch.from_numpy(bbox_params).float()
            mask = torch.from_numpy(mask).float()
            confidence_target = torch.from_numpy(confidence_target).float()
            num_objects = torch.tensor(min(num_objects, max_objects), dtype=torch.long)

            return {
                'rgb': rgb,
                'pointcloud': pc,
                'bbox_params': bbox_params,
                'mask': mask,
                'confidence_target': confidence_target,
                'num_objects': num_objects
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {folder_path}: {e}")
            return self.get_dummy_sample()