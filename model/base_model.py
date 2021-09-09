from torch import nn
from torch_geometric.nn import GATConv, GCNConv

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
            
        out = self.layers[-1](x)
            
        return out
    
class GNN(nn.Module):
    def __init__(self, layers, GNN_type='GAT'):
        super(GNN, self).__init__()
        self.GNN_type = GNN_type
        GNNconv = GCNConv if GNN_type=='GCN' else GATConv if GNN_type=='GAT' else None
        self.layers = nn.ModuleList([
            GNNconv(in_dim, out_dim) for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ])
        
    def forward(self, *x, edge_weights=None):
        embeds, edges = x
        if self.GNN_type=='GAT' or self.GNN_type=='GCN':
            edge_weights=None
        for layer in self.layers[:-1]:
            embeds = layer(embeds, edges, edge_weights).relu()
            
        out = self.layers[-1](embeds, edges, edge_weights)
        
        return out