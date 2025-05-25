import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

class PointNetEncoder(nn.Module):
    def __init__(self, point_dim=3, feat_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(point_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feat_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        return x
