import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import get_resnet

# TODO
class CRW(nn.Module):
    def __init__(self):
        super(CRW, self).__init__()
        self.temperature = 0.05
        self.dropout = 0.1
        self.edge_dropout = 0.1

        self.encoder = get_resnet(18)
        self.find_vector_dim()
        self.head = self.add_head(3)

        self.dropout = nn.Dropout(p=self.dropout, inplace=False)
        self.featdrop = nn.Dropout(p=self.edge_dropout, inplace=False)

    def find_vector_dim(self):
        out = self.encoder(torch.zeros(1, 3, 256, 256).to(
            next(self.encoder.parameters()).device))
        self.encoder_out_dim = out.shape[1]
        self.map_scale = 256 // out.shape[-1]

    def add_head(self, depth_head=0):
        layers = []
        if depth_head:
            for _ in range(depth_head - 1):
                layers.append(nn.Linear(self.encoder_out_dim, self.encoder_out_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.encoder_out_dim, 128))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x is (B, T, C*N, H, W)
        T - number of frames
        N - number of patches
        """
        B, T, CN, H, W = x.shape
        N = CN // 3
        x = x.transpose(1, 2).view(B, N, 3, T, H, W) # (B, N, C, T, H, W)

        x = x.flatten(0, 1) # (BN, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous(). \
            view(-1, C, H, W) # (BNT, C, H, W)
        
        # to nodes
        maps = self.encoder(x) # (BNT, c, h, w)
        c, h, w = maps.shape[-3:]
        maps = maps.view(B*N, T, c, h, w).transpose(1, 2) # (BN, c, T, h, w)

        if self.edge_dropout > 0:
            maps = self.featdrop(maps)
        
        q = maps.sum(-1).sum(-1) / (h * w) # (BN, c, T)
        q = self.head(q.transpose(-1, -2)).transpose(-1,-2) # (BN, T, c)
        q = F.normalize(q, p=2, dim=1) # l2 norm vector (BN, c, T)
        
        # to embed patches (B x c x T x N) and maps (B, N, c, T, h, w)
        q = q.view(B, N, c, T).permute(0, 2, 3, 1)
        maps = maps.view(B, N, c, T, h, w)