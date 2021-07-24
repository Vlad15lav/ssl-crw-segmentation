import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, planes, stride=1,
                downsample=None, groups=1, base_width=64,
                dilation=1, norm_layer=None):
        
        super(Bottleneck, self).__init__()
        self.stride = stride
        
        up_channel = int(planes * (base_width / 64.0)) * groups
        
        self.Residual = nn.Sequential(
            nn.Conv2d(in_channels, up_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(),
            nn.Conv2d(up_channel, up_channel, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(),
            nn.Conv2d(up_channel, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4))
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.Residual(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
