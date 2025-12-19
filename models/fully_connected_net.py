import torch 
from torch import nn

class FullyConnectedNet(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_layers,
                 batch_normalization, 
                 output_dim,
                 ):

        super().__init__()

        layers = []
        in_features = input_dim

        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        logits = self.network(x)
        return logits
    
