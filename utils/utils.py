import numpy as np
import pandas as pd

import torch
from torch_sparse import coalesce
from torch_geometric.utils import is_undirected, to_undirected
from tqdm import tqdm

def read_sample_files(sample_cdtx_file, sample_cust_file, sample_chid_dict, shop_col, n_users=None):
    
    print('Start reading cdtx file...')
    df_cdtx = pd.read_csv(sample_cdtx_file) 
    df_cdtx.sort_values('csmdt') # sort by date
    print('Finish reading cdtx file !')
    
    print('Start reading cust file...')
    df_cust = pd.read_csv(sample_cust_file)
    df_cust.drop_duplicates(ignore_index=True, inplace=True) # drop duplicate row
    print('Finish reading cust file !')
    
    idx_map = np.load(sample_chid_dict, allow_pickle=True).tolist()
    df_cdtx.chid = df_cdtx.chid.map(idx_map)
    if n_users != None:
        df_cdtx = df_cdtx[df_cdtx.chid.isin(list(range(n_users)))]
        df_cdtx.reset_index(inplace=True)
       
    else:
        n_users = df_cdtx.chid.nunique()
        
    n_shops = df_cdtx[shop_col].nunique()
    
    for i , j in enumerate(sorted(df_cdtx[shop_col].unique())):
        idx_map[j] = i+n_users
    
    print('Start mapping encoding...')
    df_cdtx[shop_col] = df_cdtx[shop_col].map(idx_map)

    df_cdtx.csmdt = df_cdtx.csmdt.apply(lambda x: x[:8]+'01')
    df_cdtx.objam = df_cdtx.objam.apply(lambda x: int(x))
    
    print('Finish !!')
    return df_cdtx, df_cust, n_users, n_shops

def get_domain_values(df, domains):
    domain_values = {}
    for domain, condit in domains.items():
        temp_dic = {}
        for value in df[domain].unique():
            if value != condit:
                temp_dic[value] = 1
            else:
                temp_dic[value] = 0
        domain_values[domain] = temp_dic

    return domain_values

def get_graph_monthly(df_cdtx, list_months, shop_col):
    edge_indices_dict = {}
    edge_weights_dict = {}
    max_objam = np.log1p(df_cdtx.groupby(['chid', shop_col]).objam.sum().max())
    for month in list_months:
        edges = df_cdtx[df_cdtx.csmdt==month].groupby(['chid', shop_col]).objam.sum()
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

def get_adj_monthly(edge_indices, edge_weight, n_users, n_shops):
    adj = []
    for edges, weights in zip(edge_indices.values(), edge_weights.values()):
        mask = edges[0] < edges[1]
        edges = edges[:,mask]
        edges[1] = edges[1] - n_users
        adj.append(torch.sparse_coo_tensor(edges, v[mask], [n_users, n_shops]).to(device))
    

def negative_sampling(pos_edges, n_users, n_shops, sampling_ratio=5):
    device = pos_edges.device
    all_items = set(range(n_users, n_users+n_shops))
    neg_edges = []
    for user in range(n_users):
        pos_items = set(pos_edges[1, pos_edges[0] == user].cpu().numpy())
        neg_items = torch.LongTensor(list(all_items - pos_items))
        n_neg = len(neg_items)
        n_pos = n_shops - n_neg
        n_neg_sample = n_pos * sampling_ratio if n_pos!=0 else sampling_ratio
        
        if n_neg > n_neg_sample:
            neg_items[torch.randperm(neg_items.shape[0])]
            neg_items = neg_items[:n_neg_sample]

        for item in neg_items:
            neg_edges.append([user, item])

    neg_edges = torch.LongTensor(neg_edges)
    
    return neg_edges.T.to(device)


