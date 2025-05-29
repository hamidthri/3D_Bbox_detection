import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    """
    EdgeConv layer as described in the DGCNN paper.
    This layer computes local features by considering the k-nearest neighbors of each point.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors to consider for each point.
    
    Attributes:
        k (int): Number of nearest neighbors.
        conv (nn.Sequential): Sequential container for convolutional layers.
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def knn(self, x, k):
        B, C, N = x.size()
        x_t = x.transpose(2, 1)
        inner = -2 * torch.matmul(x_t, x_t.transpose(2, 1))
        xx = torch.sum(x_t ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k, idx=None):
        B, C, N = x.size()
        if idx is None:
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
        x = self.get_graph_feature(x, self.k)  
        x = self.conv(x) 
        x = x.max(dim=-1)[0] 
        return x

class DGCNN(nn.Module):
    def __init__(self, input_dim=3, k=20, output_dim=256):
        super().__init__()
        self.k = k

        self.edge_conv1 = EdgeConv(input_dim, 64, k)
        self.edge_conv2 = EdgeConv(64, 64, k)
        self.edge_conv3 = EdgeConv(64, 64, k)

        self.conv1 = nn.Conv1d(192, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)

        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 3:
            x = x.transpose(2, 1)  # (B, 3, N)

        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)

        x = torch.cat((x1, x2, x3), dim=1)  # (B, 192, N)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # (B, 1024, N)
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        return self.linear(x)  # (B, output_dim)
