import copy
import torch
from torch import nn
from model.base_model import *

    
class TemGNN(nn.Module):
    def __init__(self, model_parameters):
        super(TemGNN, self).__init__()
        self.gnn_params = copy.deepcopy(model_parameters.gnn_params)
        self.static_layers = model_parameters.static_layers
        
        self.entity_dim = model_parameters.entity_dim
        self.static_dense_dims = model_parameters.static_dense_dims
        self.static_sparse_dims = model_parameters.static_sparse_dims
        
        self.dynamic_dense_dims = model_parameters.dynamic_dense_dims
        self.dynamic_sparse_dims = model_parameters.dynamic_sparse_dims
        
        self.window_size = model_parameters.window_size
        self.rnn_in_dim = model_parameters.rnn_in_dim
        self.rnn_hidden_dim = model_parameters.rnn_hidden_dim
        self.rnn_layer_num = model_parameters.rnn_layer_num
        self.rnn_type = model_parameters.rnn_type
        
        self.static_embeds = nn.ModuleList([
            nn.Embedding(dims, self.entity_dim) for dims in self.static_sparse_dims])                
        
        self.dynamic_embeds = nn.ModuleList([
            nn.Embedding(dims, self.entity_dim) for dims in self.dynamic_sparse_dims])
        
        self.static_model = model_parameters.static_model
        if self.static_model:
            self.static_encoder = MLP(self.static_layers)
        else:
            self.gnn_params['layers'][0] += self.static_layers[0]

        self.graph_encoders = nn.ModuleList([
            GNN(self.gnn_params['layers'], self.gnn_params['GNN_type']) for i in range(self.window_size)])
        
        rnn_model = nn.LSTM if self.rnn_type=='LSTM' else nn.GRU if self.rnn_type=='GRU' else nn.RNN
        self.rnn = rnn_model(self.rnn_in_dim, self.rnn_hidden_dim, self.rnn_layer_num, batch_first=True, bidirectional=False)
        
        self.att_w = nn.Linear(self.rnn_hidden_dim, 1)
            
    def calc_attention(self, x):
        att_score = nn.Softmax(dim=1)(self.att_w(x.tanh()))

        return att_score
    
    def Embedding_layer(self, *x):
        
        static_dense_x, static_sparse_x, dynaminc_dense_x, dynamic_sparse_x = x
        
        static_x = torch.cat([layer(static_sparse_x[:, i]) for i, layer in enumerate(self.static_embeds)], -1)
        static_x = torch.cat([static_x, static_dense_x], -1)  
        
        dynamic_xs = list()
        for dense_x, spare_x in zip(dynaminc_dense_x, dynamic_sparse_x):
            dynamic_x = torch.cat([layer(spare_x[:, i]) for i, layer in enumerate(self.dynamic_embeds)], -1)
            if self.static_model:
                dynamic_xs.append(torch.cat([dynamic_x, dense_x], -1))
            else:
                dynamic_xs.append(torch.cat([dynamic_x, dense_x, static_x], -1))
                
        if self.static_model:
            static_embed = self.static_encoder(static_x)
            return dynamic_xs, static_embed
        else:
            return dynamic_xs
        
    def forward(self, *x):
        
        static_dense_x, static_sparse_x, dynaminc_dense_x, dynamic_sparse_x, edges, weights = x
        
        if self.static_model:
            dynamic_xs, static_embed = self.Embedding_layer(static_dense_x, static_sparse_x, dynaminc_dense_x, dynamic_sparse_x)
        else:
            dynamic_xs = self.Embedding_layer(static_dense_x, static_sparse_x, dynaminc_dense_x, dynamic_sparse_x)
       
        graph_embeds = [encoder(dynamic_x, edge, edge_weights=weight) for dynamic_x, edge, weight, encoder in zip(dynamic_xs, edges, weights, self.graph_encoders)]
        
        graph_embed = torch.cat([graph_embed.unsqueeze(1) for graph_embed in graph_embeds], 1)
        
        dynamic_embed, hidden = self.rnn(graph_embed)
        
        if self.static_model:
            embed = torch.cat([static_embed.unsqueeze(1), dynamic_embed], dim=1)
        else:
            embed = dynamic_embed
            
        att_score = self.calc_attention(embed)
        
        embed = (embed*att_score).sum(dim=1)

        return embed
    
class DualGNN(nn.Module):
    def __init__(self, model_parameters):
        super(DualGNN, self).__init__()
        
        self.state_gnn = TemGNN(model_parameters)
        self.influ_gnn = TemGNN(model_parameters)
        self.state_decoder = MLP(model_parameters.state_layers)
        
    def forward(self, *x):
        
        state_embeds = self.state_gnn(*x)
        influ_embeds = self.influ_gnn(*x)
        
        state = self.state_decoder(state_embeds).sigmoid()
        
        output = influ_embeds * state
            
        return output, state
    
    
    def decode(self, embeds, n_users):
        
        user_embeds = embeds[:n_users]
        shop_embeds = embeds[n_users:]
        
        return torch.mm(user_embeds, shop_embeds.T)
        