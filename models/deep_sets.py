import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 phi_layers: list,
                 rho_layers: list,  
                 output_dim: int,
                 activation: str,
                 layer_norm: bool =  True,
                 residual_block: bool = False,
                 sparse_batching: bool = True,
                 pooling: str = "sum" # mean, max, or sum
                 ):
        super().__init__()

        # phi network (point encoder)
        phi = []
        last_dim = input_dim

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        

        # for hidden in phi_layers:
        #     phi.append(nn.Linear(last_dim, hidden))
        #     if layer_norm:
        #         phi.append(nn.LayerNorm(hidden))
        #     if residual_block and last_dim == hidden:
        #         phi.append(ResidualBlock(hidden, self.activation))  
        #     else:
        #         phi.append(self.activation) 
        #     last_dim = hidden

        for hidden in phi_layers:
            if residual_block and last_dim == hidden:
                phi.append(ResidualBlock(hidden, self.activation, layer_norm=layer_norm))
            else:
                phi.append(nn.Linear(last_dim, hidden))
                if layer_norm:
                    phi.append(nn.LayerNorm(hidden))
                phi.append(self.activation)
            last_dim = hidden

        phi.append(nn.Linear(last_dim, last_dim))
        self.phi_output_dim = last_dim
        self.phi = nn.Sequential(*phi)

        # rho network (set encoder)
        rho = []
        last_dim = self.phi_output_dim

        for hidden in rho_layers:
            rho.append(nn.Linear(last_dim, hidden))
            if layer_norm:
                rho.append(nn.LayerNorm(hidden))
            rho.append(self.activation)
            last_dim = hidden

        # classification layer
        rho.append(nn.Linear(last_dim, output_dim))
        self.rho = nn.Sequential(*rho)

        if pooling not in ["mean", "sum", "max"]:
            raise ValueError("pooling must be 'mean', 'sum', or 'max'")
        self.pooling = pooling

        self.sparse_batching = sparse_batching


    def _forward_sparse(self, x: torch.Tensor, idx: torch.Tensor):
        """
        Sparse forward pass.
        
        x: [sum(N_hits_i), input_dim] all points concatenated
        idx: [sum(N_hits_i)] event index for each point
        """
        # encode each point
        phi_x = self.phi(x)  # [sum(N_hits_i), H]

        counts = torch.bincount(idx)
        chunks = torch.split(phi_x, counts.tolist(), dim=0)

        pooled_list = []

        for chunk in chunks:
            if self.pooling == "sum":
                #pooled_list.append(chunk.sum(dim=0) / chunk.size(0).sqrt())
                pooled_list.append(chunk.sum(dim=0) / torch.sqrt(torch.tensor(chunk.size(0), dtype=chunk.dtype)))
                #pooled_list.append(chunk.sum(dim=0))
            elif self.pooling == "mean":
                pooled_list.append(chunk.mean(dim=0))
            elif self.pooling == "max":
                pooled_list.append(chunk.max(dim=0)[0])

        pooled = torch.stack(pooled_list)  # [B, H]

        # print("pooled mean:", pooled.mean(dim=0))
        # print("pooled std:", pooled.std(dim=0))

        # encode set
        logits = self.rho(pooled)  # [B, output_dim]

        return logits

    def _forward_padded(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Padded forward pass.
        x:    [B, N_max, input_dim]
        mask: [B, N_max]  1 for valid points, 0 for padding
        """
        phi_x = self.phi(x)  # [B, N_max, H]
        phi_x = phi_x * mask.unsqueeze(-1)  # zero out padded points

        if self.pooling == "sum":
            pooled = phi_x.sum(dim=1)
        elif self.pooling == "mean":
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid division by 0
            pooled = phi_x.sum(dim=1) / lengths
        elif self.pooling == "max":
            # mask padded points with a very negative value so they don't affect max
            phi_x = phi_x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled = phi_x.max(dim=1)[0]

        logits = self.rho(pooled)  # [B, output_dim]
        return logits

  
    def forward(self, *args):
        if self.sparse_batching:
            return self._forward_sparse(*args)            
        else:
            return self._forward_padded(*args)



class ResidualBlock(nn.Module):
    def __init__(self, dim, activation, layer_norm=False):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.activation = activation

    def forward(self, x):
        out = self.linear(x)
        out = self.layer_norm(out)
        out = self.activation(out)
        return x + out