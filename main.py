import argparse
import argparse
import numpy as np
from tqdm import trange

import torch
from torch_geometric.data import Data

import model
from dataset import Graph

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='TemGNN')
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('-e', '--epochs', type=int, default=400)
parser.add_argument('-d', '--embed_dim', type=int, default=32)
parser.add_argument('-l', '--lr', type=float, default=1e4)
parser.add_argument('-w', '--window_size', type=int, default=6)
parser.add_argument('-g', '--gpu', type=str, default=0)

parser.add_argument('--entity_dim', type=int, default=4)
parser.add_argument('--shop_col', type=str, default='stonc_6_label')
parser.add_argument('--static_model', dest='static_model', action='store_true', default=False)

args = parser.parse_args()



def train():
    
    weight_loss_ = []
    if args.dual:
        state_loss_ = []

    static_dense_x = data.static_dense.to(device)
    static_sparse_x = data.static_sparse.long().to(device)
    
    for i in range(training_stride):
        dynamic_dense_x = list(data.dynamic_dense.values())[i:i+args.window_size]
        dynamic_sparse_x = list(data.dynamic_sparse.values())[i:i+args.window_size]
        dynamic_dense_x = [x.to(device) for x in dynamic_dense_x]
        dynamic_sparse_x = [x.long().to(device) for x in dynamic_sparse_x]

        edges = list(data.edge_indices.values())[i:i+args.window_size]
        edges = [e.to(device) for e in edges]
        weights = list(data.edge_weights.values())[i:i+args.window_size]
        weights = [w.to(device) for w in weights]
        
        model.train()
        optimizer.zero_grad()
        if args.dual:
            z, y_pred = model(static_dense_x, static_sparse_x, dynamic_dense_x, dynamic_sparse_x, edges, weights)
        else:
            z = model(static_dense_x, static_sparse_x, dynamic_dense_x, dynamic_sparse_x, edges, weights)
        adj_pred = model.decode(z, data.n_users)
        adj_true = data.adj[data.list_months[i+args.window_size]].to(device)
        weight_loss = torch.square(adj_pred - adj_true).mean()
        weight_loss_.append(weight_loss.item())
        
        loss = weight_loss
        
        if args.dual:
            y_true = data.y[i+args.window_size].to(device)
            state_loss = criterion(y_pred, y_true)
            state_loss_.append(state_loss.item())
            loss += state_loss
        
        loss.backward()
        optimizer.step()
        
    if args.dual:
        return np.mean(weight_loss_), np.mean(state_loss_)
    else:
        return np.mean(weight_loss_)


def test(stride, months):
    
    weight_loss_ = []
    if args.dual:
        state_auc_ = []
        
    user_loss_ = []
    
    static_dense_x = data.static_dense.to(device)
    static_sparse_x = data.static_sparse.long().to(device)
    
    for i in range(stride, stride+months):
        dynamic_dense_x = list(data.dynamic_dense.values())[i:i+args.window_size]
        dynamic_sparse_x = list(data.dynamic_sparse.values())[i:i+args.window_size]
        dynamic_dense_x = [x.to(device) for x in dynamic_dense_x]
        dynamic_sparse_x = [x.long().to(device) for x in dynamic_sparse_x]

        edges = list(data.edge_indices.values())[i:i+args.window_size]
        edges = [e.to(device) for e in edges]
        weights = list(data.edge_weights.values())[i:i+args.window_size]
        weights = [w.to(device) for w in weights]
        
        model.eval()
        
        if args.dual:
            z, y_pred = model(static_dense_x, static_sparse_x, dynamic_dense_x, dynamic_sparse_x, edges, weights)
        else:
            z = model(static_dense_x, static_sparse_x, dynamic_dense_x, dynamic_sparse_x, edges, weights)
        
        adj_pred = model.decode(z, data.n_users).detach().cpu()
        adj_true = data.adj[data.list_months[i+args.window_size]]

        weight_loss = torch.sqrt(torch.square(adj_pred - adj_true).mean())
        user_loss = np.square(adj_pred.sum(1).numpy() - adj_true.to_dense().sum(1).numpy()).mean()
        
        weight_loss_.append(weight_loss.item())
        user_loss_.append(user_loss)
        
        if args.dual:
            y_true = data.y[i+args.window_size].numpy()
            state_auc = roc_auc_score(y_true, y_pred.detach().cpu().numpy())
            state_auc_.append(state_auc)

    if args.dual:
        return np.mean(weight_loss_), np.mean(user_loss_), np.mean(state_auc_),
    else:
        return np.mean(weight_loss_), np.mean(user_loss_)



def main(args):
    
    data = Graph(shop_col = args.shop_col)
    
    # shop -1 to 0
    for sparse in data.dynamic_sparse.values():
        sparse += 1

    node_num = data.n_users+data.n_shops
    dynamic_dense_dim = data.dynamic_dense['2018-01-01'].shape[1]
    dynamic_sparse_dims = (max([sparse.max(axis=0)[0] for sparse in data.dynamic_sparse.values()])+1).tolist()

    in_channels = dynamic_dense_dim + len(dynamic_sparse_dims)*args.entity_dim

    gnn_params = {
        'layers': [in_channels, args.embed_dim*2, args.embed_dim], 
        'GNN_type': 'GCN'
    }

    static_dense_dims = data.static_dense.shape[1]
    static_sparse_dims = (data.static_sparse.max(axis=0)[0]+1).int().tolist()
    mlp_in_dim = static_dense_dims + len(static_sparse_dims)*args.entity_dim
    
    training_months = 18 # number of training months
    val_months = 3 # number of validation months
    testing_months = 3 # number of testing months
    training_stride = training_months - args.window_size
    val_stride = training_stride + val_months
    testing_stride = val_stride + testing_months
    
    args.gnn_params = gnn_params
    args.static_dense_dim = static_dense_dims
    args.static_sparse_dims = static_sparse_dims
    args.dynamic_dense_dims = dynamic_dense_dims
    args.dynamic_sparse_dims = dynamic_sparse_dims
    args.rnn_in_dim = args.embed_dim
    args.rnn_hidden_dim = args.embed_dim
    args.rnn_layer_num = 1
    args.rnn_type = 'LSTM' # {'LSTM', 'GRU'}
    args.static_layers = [mlp_in_dim, args.embed_dim*2, args.embed_dim]
    args.state_layers = [args.embed_dim, args.embed_dim//2, 1]
    args.dual = True if args.model in ['TemGNN_pyramid', 'DualGNN_pyramid'] else False
    
    if arg.model == 'SingleGNN':
        model = model.SingleGNN(args)
    elif arg.model == 'GCN_RNN':
        model = model.GCN_RNN(args)
    elif arg.model == 'TemGNN':
        model = model.TemGNN(args)
    elif arg.model == 'TemGNN_pyramid':
        model = model.TemGNN_pyramid(args)
    elif arg.model == 'DualGNN':
        model = model.DualGNN(args)
    elif arg.model == 'DualGNN_pyramid':
        model = model.DualGNN_pyramid(args)
    else:
        raise ValueError("Model name must be in : ['SingleGNN', 'GCN_RNN', 'TemGNN', 'TemGNN_pyramid', 'DualGNN', 'DualGNN_pyramid']")
    
    device = torch.device(f'cuda:{arg.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.dual:
        criterion = torch.nn.BCELoss() # if use dual model, task2 binary classification

    train_loss = []
    val_loss = []
    test_loss = []

    val_user = []
    test_user = []

    for epoch in range(epochs):
        print(f'epoch:{epoch+1}')
        loss = train()
        
        val_metric = list(test(training_stride, val_months))
        
        metrics = ['RMSE','User_Loss']
        if args.dual:
            metrics.append('AUC')
            
        output = [' ']
        for metric, value in zip(metrics, val_metric):
            output.append(f'val_{metric}:{value}')
            
        print(*output)
        

if __name__ == '__main__':
    main(args)
    

        
        
    
        
    



