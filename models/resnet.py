import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Using block repeater for ResNet18 and ResNet34
    """
    def __init__(self, in_channels, planes, stride=1,
                downsample=None, groups=1, base_width=64,
                dilation=1, norm_layer=None):
        
        super(BasicBlock, self).__init__()
        self.stride = stride
        # F(x) sequential
        self.Residual = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(planes))
        # add skip connection
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # save for skip connection
        identity = x
        
        # F(x) layers
        out = self.Residual(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # skip connecion
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    Using block repeater for ResNet50, ResNet101, ResNet152
    """
    def __init__(self, in_channels, planes, stride=1,
                downsample=None, groups=1, base_width=64,
                dilation=1, norm_layer=None):
        
        super(Bottleneck, self).__init__()
        self.stride = stride
        
        up_channel = int(planes * (base_width / 64.0)) * groups
        # F(x) sequential
        self.Residual = nn.Sequential(
            nn.Conv2d(in_channels, up_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(),
            nn.Conv2d(up_channel, up_channel, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False, dilation=dilation),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(),
            nn.Conv2d(up_channel, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4))
        # add skip connection
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # save for skip connection
        identity = x
        
        # F(x) layers
        out = self.Residual(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # skip connecion
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet Architecture
    """
    def __init__(self, type_block, count_blocks, 
                 groups=1, widen=1, start_channel=64):
        """
        type_block: type of repeated blocks (Basic or Bottleneck)
        count_blocks: count of repeated blocks
        widen: scale width model (ResNet50 w2 w4 w8...)
        start_channel: first count of filters block
        """
        super(ResNet, self).__init__()
        self.type_block = type_block
        self.expansion = 4 if type_block.__name__ == 'Bottleneck' else 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                    kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                    dilation=1, ceil_mode=False)
        
        self.groups = groups
        self.in_channels = widen * start_channel
        out_filters = widen * start_channel
        self.dilation = 1

        self.layer1 = self.__add_blocks(out_filters, count_blocks[0])
        
        out_filters *= 2
        self.layer2 = self.__add_blocks(out_filters, count_blocks[1], stride=2)
        
        out_filters *= 2
        self.layer3 = self.__add_blocks(out_filters, count_blocks[2], stride=2)
        
        out_filters *= 2
        self.layer4 = self.__add_blocks(out_filters, count_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def __add_blocks(self, out_filters, count, stride=1):
        """
        Add Basic or Bottleneck k times
        """
        layers_list = []
        # add first with downsample block
        downsample_layer = nn.Sequential(
                nn.Conv2d(self.in_channels, out_filters * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters * self.expansion))
        
        layers_list.append(
            self.type_block(in_channels=self.in_channels, planes=out_filters,
                       stride=stride, downsample=downsample_layer))
        # add others blocks
        self.in_channels = out_filters * self.expansion
        for _ in range(1, count):
            layers_list.append(
                self.type_block(in_channels=self.in_channels, planes=out_filters))
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
