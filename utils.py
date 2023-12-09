# 作者：York
# 时间：2022/5/30 18:32
import os
import time

import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
import torch.nn.functional as F
import torch.nn as nn

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(config):
    f = np.loadtxt(config.feature_path.format(config.name), dtype=float)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(config):
    struct_edges = np.genfromtxt(config.structgraph_path.format(config.name), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return nsadj


def get_metric_matrix(matrix1, matrix2, method):

    assert method in ['Euclid', 'cosine'], "Unkown operation!"  # Assert the reasonability of measurement method
    if method == 'cosine':
        metric_matrix = cosine_similarity(matrix1, matrix2)
    else:
        metric_matrix = euclidean_distances(matrix1, matrix2)
    if method == 'cosine':  # In the case of cosine, logarithmic operations are required to ensure that the lower
        # the value for the better performance
        metric_matrix = np.exp(-metric_matrix)
    return metric_matrix


from collections import Counter
def augement_nodes(s_a, t_a):
    s = torch.zeros(s_a.shape[0])
    t = torch.zeros(t_a.shape[0])
    d_S = np.nonzero(s_a).t()[0].cpu().numpy().tolist()
    d_T = np.nonzero(t_a).t()[0].cpu().numpy().tolist()
    dix_s = Counter(d_S)
    dix_t = Counter(d_T)
    list_dix_s = sorted(dict(dix_s).items(), key= lambda x:x[1], reverse=True)
    list_dix_t = sorted(dict(dix_t).items(), key= lambda x:x[1], reverse=True)
    tensor_dix_s = torch.from_numpy(np.array(list_dix_s))
    tensor_dix_t = torch.from_numpy(np.array(list_dix_t))

    s_10 = tensor_dix_s[:10, :]
    t_10 = tensor_dix_t[:10, :]
    s_degree = s_10.t()[0].tolist()
    t_degree = t_10.t()[0].tolist()

    return torch.LongTensor(s_degree), torch.LongTensor(t_degree)

def get_topological_all(address):
    s, t = np.load(os.path.join('./data', address, address + '_s_centrality.npy'), allow_pickle=True), np.load(os.path.join('./data', address, address + '_t_centrality.npy'), allow_pickle=True)
    
    return s, t

if __name__ == '__main__':
    pass