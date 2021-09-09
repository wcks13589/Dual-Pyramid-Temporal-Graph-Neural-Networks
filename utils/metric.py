import numpy as np
from sklearn.metrics import roc_auc_score

def Recall_K(y_pred, y_true, k):
    recall_k = []
    for i, j  in zip(y_pred, y_true):
        if j.sum() == 0:
            continue        
        
        top_k = np.argsort(i)[-k:]
        ground_truth = np.argwhere(j == 1)
        a = np.intersect1d(top_k, ground_truth).shape[0]
        b = ground_truth.shape[0]

        recall_k.append(a/b)
        
    return np.mean(recall_k)

def Precision_K(y_pred, y_true, k):
    precision_k = []
    for i, j  in zip(y_pred, y_true):        
        if j.sum() == 0:
            continue
            
        top_k = np.argsort(i)[-k:]
        top_k_ground_truth = j[top_k]            
        
        precision_k.append(top_k_ground_truth.sum()/k)
        
    return np.mean(precision_k)

def Ndcg_k(y_pred, y_true, k):
    ndcg_k = []
    for i, j  in zip(y_pred, y_true):
        ground_truth = np.argwhere(j == 1)
        
        if j.sum() == 0:
            continue
        
        ndcg_k.append(ndcg_score(j.reshape(1, -1), i.reshape(1, -1), k=k))
        
    return np.mean(ndcg_k)    

def AUC(y_pred, y_true):
    auc = []
    for i, j in zip(y_true, y_pred):
        if i.sum() == 0:
            continue
        auc.append(roc_auc_score(i, j))
    
    return np.mean(auc)