import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, base_model, num_class):
        super(Network, self).__init__()
        self.model = base_model

        net_list = list(self.model.children())
        in_features = self.model.fc.in_features
        self.model = nn.Sequential(*net_list[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

        self.fc_cls = nn.Linear(in_features, num_class)
    
    def forward(self, x):
        cam = self.model(x)

        out = self.avgpool(cam)
        out = self.flatten(out)
        
        cls_out = self.fc_cls(out)
        if self.training:
            return cls_out

        cls_label = cls_out.argmax(axis=1)
        cam_weight = self.fc_cls.weight[cls_label]
        cam_weight = cam_weight.reshape(cam_weight.shape[0],\
                                        cam_weight.shape[1], 1, 1)
        classmaps = torch.sum(cam * cam_weight, axis=1)
        return classmaps, cls_label
