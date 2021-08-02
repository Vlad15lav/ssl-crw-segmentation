import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, base_model, num_class):
        super(RotNet, self).__init__()
        self.model = base_model
        net_list = list(self.model.children())
        in_features = self.model.fc.in_features
        self.model = nn.Sequential(*net_list[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

        self.fc_cls = nn.Linear(in_features, num_class)
        self.fc_rot = nn.Linear(in_features, 4)
    
    def forward(self, x):
        cam = self.model(x)

        out = self.avgpool(cam)
        out = self.flatten(out)
        
        out_cls = self.fc_cls(out)
        if self.training:
            out_rot = self.fc_rot(out)
            return out_cls, out_rot

        return out_cls
