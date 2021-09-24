import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import get_resnet

EPS = 1e-20

class CRW(nn.Module):
    def __init__(self, opt):
        super(CRW, self).__init__()
        self.temperature = opt.temperature
        self.featdrop_rate = opt.featdrop
        self.edgedrop_rate = opt.edgedrop

        self.encoder = get_resnet(opt.depth)
        self.find_vector_dim()
        self.head = self.add_head(opt.head_depth)

        self.featdrop = nn.Dropout(p=self.featdrop_rate, inplace=False)
        
        self.criterion = nn.CrossEntropyLoss()

    def find_vector_dim(self):
        out = self.encoder(torch.zeros(1, 3, 256, 256).to(
            next(self.encoder.parameters()).device))
        self.encoder_out_dim = out.shape[1]
        self.map_scale = 256 // out.shape[-1]

    def add_head(self, depth_head=0):
        layers = []
        if depth_head >= 0:
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
        C, N = 3, CN // 3
        x = x.transpose(1, 2).view(B, N, 3, T, H, W) # (B, N, C, T, H, W)

        x = x.flatten(0, 1) # (BN, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous(). \
            view(-1, C, H, W) # (BNT, C, H, W)
        
        # to nodes
        maps = self.encoder(x) # (BNT, c, h, w)
        c, h, w = maps.shape[-3:]
        maps = maps.view(B*N, T, c, h, w).transpose(1, 2) # (BN, c, T, h, w)

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)
        
        q = maps.sum(-1).sum(-1) / (h * w) # (BN, c, T)
        q = self.head(q.transpose(-1, -2)).transpose(-1,-2) # (BN, T, c)
        q = F.normalize(q, p=2, dim=1) # l2 norm vector (BN, c, T)
        
        # to embed patches (B x c x T x N) and maps (B, N, c, T, h, w)
        q = q.view(B, N, q.shape[1], T).permute(0, 2, 3, 1)
        maps = maps.view(B, N, *maps.shape[1:])

        # transitions from t to t+1 (B x T-1 x N x N)
        A = torch.einsum('bctn,bctm->btnm', q[:, :, :-1],
                          q[:, :, 1:]) / self.temperature

        ## Transition energies for palindrome graph
        AA = torch.cat((A, torch.flip(A, dims=[1]).transpose(-1,-2)), dim=1)
        AA[torch.rand_like(AA) < self.edgedrop_rate] = -1e10
        At = torch.diag_embed(torch.ones((B, N))).to(A.device)

        ## Compute walks
        for t in range(2 * T - 2):
            At = torch.bmm(F.softmax(AA[:, t], dim=-1), At)
        
        ## Walk Loss
        target = torch.arange(At.shape[-1])[None]. \
            repeat(At.shape[0], 1).view(-1).to(At.device)
        logits = torch.log(At + EPS).flatten(0, -2)
        loss = self.criterion(logits, target)
        acc = (torch.argmax(logits, dim=-1) == target).float().mean()

        return q, loss, acc
