import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetPPFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, k=20, output_dim=256):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Conv2d(2 * input_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(256, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def knn(self, x, k):
        B, C, N = x.size()
        x = x.transpose(2, 1)  # (B, N, C)
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k):
        B, C, N = x.size()
        idx = self.knn(x, k)
        device = x.device
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()
        feature = x.view(B * N, -1)[idx, :]
        feature = feature.view(B, N, k, C)
        x = x.view(B, N, 1, C).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # (B, 3, N)
        feat = self.get_graph_feature(x, k=self.k)
        x = self.mlp1(feat)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = torch.max(x, dim=-1)[0]  # (B, 256, N)
        x = self.proj(x)
        x = torch.max(x, dim=-1)[0]  # (B, output_dim)
        return x
