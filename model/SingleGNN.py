import copy
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from model.base_model import GNN
    
class SingleGNN(nn.Module):
    def __init__(self, model_parameters):
        super(SingleGNN, self).__init__()
        self.gnn_params = copy.deepcopy(model_parameters.gnn_params)
        self.entity_dim = model_parameters.entity_dim
        self.static_dense_dims = model_parameters.static_dense_dims
        self.static_sparse_dims = model_parameters.static_sparse_dims
        
        self.dynamic_dense_dims = model_parameters.dynamic_dense_dims
        self.dynamic_sparse_dims = model_parameters.dynamic_sparse_dims
        self.window_size = model_parameters.window_size
        
        self.static_embeds = nn.ModuleList([
            nn.Embedding(dims, self.entity_dim) for dims in self.static_sparse_dims])                
        
        self.dynamic_embeds = nn.ModuleList([
            nn.Embedding(dims, self.entity_dim) for dims in self.dynamic_sparse_dims])
        
        
        self.gnn_params['layers'][0] += model_parameters.static_dim

        self.graph_encoders = nn.ModuleList([
            GNN(self.gnn_params['layers'], self.gnn_params['GNN_type']) for i in range(self.window_size)])

    
    def forward(self, *x):
        static_dense_x, static_sparse_x, dynaminc_dense_x, dynamic_sparse_x, edges, weights = x
        
        static_x = torch.cat([layer(static_sparse_x[:, i]) for i, layer in enumerate(self.static_embeds)], -1)
        static_x = torch.cat([static_x, static_dense_x], -1)  
        
        dynamic_xs = list()
        for dense_x, spare_x in zip(dynaminc_dense_x, dynamic_sparse_x):
            dynamic_x = torch.cat([layer(spare_x[:, i]) for i, layer in enumerate(self.dynamic_embeds)], -1)

            
            dynamic_xs.append(torch.cat([dynamic_x, dense_x, static_x], -1))
            
        graph_embeds = [encoder(dynamic_x, edge, edge_weights=weight) for dynamic_x, edge, weight, encoder in zip(dynamic_xs, edges, weights, self.graph_encoders)]
        
    

        return graph_embeds[0]
    
    def decode(self, embeds, n_users):
        
        user_embeds = embeds[:n_users]
        shop_embeds = embeds[n_users:]
        
        return torch.mm(user_embeds, shop_embeds.T)
