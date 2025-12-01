import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, GATConv, SAGPooling, global_mean_pool, global_add_pool, global_max_pool


class GraphNet(nn.Module):

    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 activation,
                 use_gat=False,
                 gat_heads=4,
                 sag_pool=False,
                 pool_ratio=0.5,
                 local_pooling="add", # add, mean, max
                 global_pooling="mean", # add, mean, max
                 deepchem_style=False
                 ):

        super().__init__()

        if global_pooling == "mean":
            self.global_pooling = global_mean_pool
        elif global_pooling == "add":
            self.global_pooling = global_add_pool
        elif global_pooling == "max":
            self.global_pooling = global_max_pool

        self.deepchem_style = deepchem_style
        self.local_pooling = local_pooling
        self.sag_pool = sag_pool
        self.use_gat = use_gat 

        if activation == "tanh":
            self.activation = nn.Tanh()  
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()

        # GraphConv / GATConv layers
        if use_gat:
            self.conv1 = GATConv(input_dim, hidden_dim // gat_heads, heads=gat_heads, concat=True)
            self.conv2 = GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads, concat=True)
        else:
            self.conv1 = GraphConv(input_dim, hidden_dim, aggr=self.local_pooling)
            self.conv2 = GraphConv(hidden_dim, hidden_dim, aggr=self.local_pooling)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
   
        # self-attention pooling
        if self.sag_pool:
            self.pool1 = SAGPooling(hidden_dim, ratio=pool_ratio)

        # dense layers
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x, membership, edges, weights=None):

        # shapes:
        # x:            [n_nodes, F]
        # membership:   [n_nodes,]     
        # edges:        [2, num_edges] 
        # weights:      [num_edges,] optional

        x = self.conv1(x, edges, edge_weight=weights)
        
        x = self.activation(x)
        x = self.bn1(x)

        if self.sag_pool:
            # use weights as edge features instead
            x, edges, weights, membership, _, _ = self.pool1(x, edges, weights, membership)
    
        x = self.conv2(x, edges, edge_weight=weights)  
        x = self.activation(x)
        x = self.bn2(x)

        if self.deepchem_style:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.bn3(x)

            # global pooling
            x = global_mean_pool(x, membership)  # shape: [num_graphs, 256]

        else:
            # global pooling
            x = global_mean_pool(x, membership)  # shape: [num_graphs, 128]

            x = self.fc1(x)
            x = self.activation(x)
            x = self.bn3(x)

        logits = self.fc2(x)

        return logits 