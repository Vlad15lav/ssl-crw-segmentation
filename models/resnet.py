import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

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
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # add skip connection
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # save for skip connection
        residual = x
        
        # F(x) layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # skip connecion
        out += residual
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
        self.conv1 = nn.Conv2d(in_channels, up_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(up_channel)
        self.conv2 = nn.Conv2d(up_channel, up_channel, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(up_channel)
        self.conv3 = nn.Conv2d(up_channel, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # add skip connection
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # save for skip connection
        residual = x
        
        # F(x) layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # skip connecion
        out += residual
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
        #self.in_planes = [int(x * widen) for x in [64, 128, 256, 512]] 

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
        self.fc = nn.Linear(in_features=out_filters * self.expansion, out_features=1000, bias=True)
        
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
        downsample_layer = None
        if stride != 1 or self.in_channels != out_filters * self.expansion:
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
    
    def modify(self, remove_layers=[], padding=''):
        # change stride from 2 to 1 for layer3 and layer4
        filter_layers = lambda x: [l for l in x if getattr(self, l) is not None]
        for layer in filter_layers(['layer3', 'layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = tuple(1 for _ in m.stride)

        if padding != '':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding_mode = padding

        # remove last layers
        remove_layers += ['fc', 'avgpool']
        for layer in filter_layers(remove_layers):
            setattr(self, layer, None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = x if self.layer3 is None else self.layer3(x) 
        x = x if self.layer4 is None else self.layer4(x) 
    
        return x

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def get_resnet(depth, pretrained=False):
    if depth == 18:
        model = resnet18()
        model.modify(padding='reflect')
    elif depth == 34:
        model = resnet34()
    elif depth == 50:
        model = resnet50()
    elif depth == 101:
        model = resnet101()
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[f'resnet{depth}'],
                                      progress=True)
        model.load_state_dict(state_dict)
    return model
