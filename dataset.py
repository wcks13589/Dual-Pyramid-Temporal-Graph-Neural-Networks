import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import is_undirected
from utils.utils import read_sample_files

class Graph():
    def __init__(self, shop_col):
    
        data_root = './data'
        data_path = {'chid':os.path.join(data_root, 'sample/sample_50k_idx_map.npy'),
                     'cdtx':os.path.join(data_root, 'sample/sample_50k_cdtx.csv'),
                     'cust':os.path.join(data_root, 'preprocessed/df_cust_5000.csv')}

        self.shop_col = shop_col
        self.df_cdtx, self.df_cust, self.n_users, self.n_shops = read_sample_files(data_path['cdtx'], data_path['cust'], data_path['chid'], shop_col, 5000)

        self.list_months = sorted(self.df_cust.data_dt.unique())[:24]
        self.df_cdtx = self.df_cdtx[self.df_cdtx.csmdt.isin(self.list_months)]
        self.df_cust = self.df_cust[self.df_cust.data_dt.isin(self.list_months)]

        self.static_features = {'dense': ['cycam', 'monin', 'wrky'], 
                           'sparse':['masts', 'educd', 'naty', 'poscd', 'cuorg']}

        self.dynamic_features = {'dense':list(set(self.df_cust.columns) - set(self.static_features['dense'] + self.static_features['sparse'])\
                                        -{'chid', 'data_dt','trdtp'}), 'sparse':['trdtp']}


        self.static_dense, self.static_sparse = self.sparse_feat()
        self.dynamic_dense, self.dynamic_sparse = self.dense_feat()

        self.edge_indices, self.edge_weights = self.get_graphs()
        self.adj = self.get_adj()
        self.y = self.get_activate_state()

    
    def sparse_feat(self):
        static_dense = self.df_cust[self.df_cust.data_dt==self.list_months[-1]][self.static_features['dense']].values
        static_sparse = self.df_cust[self.df_cust.data_dt==self.list_months[-1]][self.static_features['sparse']].values
        static_dense = torch.Tensor(static_dense)
        static_sparse = torch.LongTensor(static_sparse)
        
        shop_dense_features = torch.zeros(self.n_shops, static_dense.shape[1])
        shop_sparse_features = torch.zeros(self.n_shops, static_sparse.shape[1]).long()#-1

        static_dense = torch.cat([static_dense, shop_dense_features], 0)
        static_sparse = torch.cat([static_sparse, shop_sparse_features], 0)    
    
        return static_dense, static_sparse
    
    def dense_feat(self):
        dynamic_dense = {}
        dynamic_sparse = {}
        for month in self.list_months:
            cust_dense_features = self.df_cust[self.df_cust.data_dt==month][self.dynamic_features['dense']].to_numpy()
            cust_dense_features = torch.Tensor(cust_dense_features)
            cust_sparse_features = self.df_cust[self.df_cust.data_dt==month][self.dynamic_features['sparse']].to_numpy()
            cust_sparse_features = torch.LongTensor(cust_sparse_features)

            shop_dense_features = torch.zeros(self.n_shops, cust_dense_features.shape[1])
            shop_sparse_features = torch.zeros(self.n_shops, cust_sparse_features.shape[1]).long()-1

            dynamic_dense[month] = torch.cat([cust_dense_features, shop_dense_features], 0)
            dynamic_sparse[month] = torch.cat([cust_sparse_features, shop_sparse_features], 0)
    
        return dynamic_dense, dynamic_sparse

    def get_graphs(self):
        edge_indices_dict = {}
        edge_weights_dict = {}
        max_objam = np.log1p(self.df_cdtx.groupby(['chid', self.shop_col]).objam.sum().max())
        for month in self.list_months:
            edges = self.df_cdtx[self.df_cdtx.csmdt==month].groupby(['chid', self.shop_col]).objam.sum()
            edge_pairs = np.stack([np.array(i) for i in edges.index]).T
            edge_pairs = torch.LongTensor(edge_pairs)

            edge_weights = np.log1p(edges.values)/max_objam
            edge_weights = torch.Tensor(edge_weights)

            if not is_undirected(edge_pairs):
                edge_pairs = torch.cat([edge_pairs, edge_pairs[[1,0],:]], -1)
                edge_weights = edge_weights.repeat(2)

            edge_indices_dict[month] = edge_pairs
            edge_weights_dict[month] = edge_weights

        return edge_indices_dict, edge_weights_dict

    def get_adj(self):
        adj = {}
        for month in self.list_months:
            edges, weights = self.edge_indices[month], self.edge_weights[month]
            mask = edges[0] < edges[1]
            edges = edges[:,mask]
            edges[1] = edges[1] - self.n_users
            adj[month] = torch.sparse_coo_tensor(edges, weights[mask], [self.n_users, self.n_shops])
            
        return adj
    
    
    def get_activate_state(self):
        activate_state = []
        for month in self.list_months:
            cust = (self.adj[month].coalesce().to_dense().sum(1) > 0).float().view(-1,1)
            shop = (self.adj[month].coalesce().to_dense().sum(0) > 0).float().view(-1,1)
            
            activate_state.append(torch.cat([cust, shop],0))
            
        return activate_statex