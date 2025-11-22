import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 phi_layers: list,
                 rho_layers: list,  
                 output_dim: int,
                 pooling: str = "sum" # mean, max, or sum
                 ):
        super().__init__()


        # phi network (point encoder)
        phi = []
        last_dim = input_dim

        for hidden in phi_layers:
            phi.append(nn.Linear(last_dim, hidden))
            phi.append(nn.GELU())
            last_dim = hidden

        phi.append(nn.Linear(last_dim, last_dim))
        self.phi_output_dim = last_dim
        self.phi = nn.Sequential(*phi)

        # rho network (set encoder)
        rho = []
        last_dim = self.phi_output_dim

        for hidden in rho_layers:
            rho.append(nn.Linear(last_dim, hidden))
            rho.append(nn.GELU())
            last_dim = hidden

        # classification layer
        rho.append(nn.Linear(last_dim, output_dim))
        self.rho = nn.Sequential(*rho)

        if pooling not in ["mean", "sum", "max"]:
            raise ValueError("pooling must be 'mean', 'sum', or 'max'")
        self.pooling = pooling


    def forward(self, x):
        """
        x: [batch, num_points, input_dim]
        """
        # Apply phi to each point
        phi_x = self.phi(x)  # [B, N, H]

        # Pool over points (DeepSets uses sum pooling)

        if self.pooling == "mean":
            pooled = phi_x.mean(dim=1)
        elif self.pooling == "sum":
            pooled = phi_x.sum(dim=1)
        elif self.pooling == "max":
            pooled = phi_x.max(dim=1)[0]

        # Apply rho
        logits = self.rho(pooled)  # [B, output_dim]

        return logits