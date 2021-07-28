import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, base_model, num_class):
        super(RotNet, self).__init__()
        self.model = base_model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
    
    def forward(self, x):
        out = self.model.forward(x)
        return out