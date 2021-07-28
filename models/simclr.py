import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, base_model, num_class):
        super(SimCLR, self).__init__()
        self.feature_network = base_model
        self.out_filter_gap = self.feature_network.fc.in_features
        del self.feature_network.fc


        self.project_head = nn.Sequential(
            nn.Linear(self.out_filter_gap, self.out_filter_gap),
            nn.ReLU(),
            nn.Linear(self.out_filter_gap, num_class)
        )
    
    def forward(self, x):
        rep = self.feature_network.forward(x)
        agr = self.project_head.forward(rep)
        return rep, agr