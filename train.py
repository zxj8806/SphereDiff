import numpy as np
import torch
import scipy.sparse as sp
import math

from model import SphereDiff
from preprocessing import load_ogbn_arxiv, sparse_to_tuple, preprocess_graph

def map_vector_to_clusters(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    y_true_mapped = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        y_true_mapped[i] = col_ind[y_true[i]]
    return y_true_mapped.astype(int)

dataset = "ogbn-arxiv"
adj, features, labels = load_ogbn_arxiv()
nClusters = int(labels.max() + 1)

num_neurons = 128
embedding_size = 128
alpha = 1.
gamma_1 = gamma_2 = gamma_3 = 1.
save_path = "./results/"

adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
adj_norm = torch.sparse_coo_tensor(
    torch.LongTensor(adj_norm[0].T),
    torch.FloatTensor(adj_norm[1]),
    torch.Size(adj_norm[2]),
    device=device
)
features = torch.sparse_coo_tensor(
    torch.LongTensor(features[0].T),
    torch.FloatTensor(features[1]),
    torch.Size(features[2]),
    device=device
)

adj_label = (adj + sp.eye(adj.shape[0])).tocoo()
adj_label = torch.sparse_coo_tensor(
    torch.LongTensor(np.vstack([adj_label.row, adj_label.col])),
    torch.ones(adj_label.nnz, dtype=torch.float32),
    torch.Size(adj_label.shape),
    device=device
)

network = SphereDiff(
    num_neurons=num_neurons,
    num_features=num_features,
    embedding_size=embedding_size,
    nClusters=nClusters,
    activation="ReLU",
    alpha=alpha,
    gamma_1=gamma_1,
    gamma_2=gamma_2,
    gamma_3=gamma_3,
    T=10,
).to(device)

y_pred, y = network.train(
    features,
    adj_norm,
    adj_label,
    labels,
    norm=1.0,
    optimizer="Adam",
    epochs=120,
    lr=0.0001,
    save_path=save_path,
    dataset=dataset,
    run_id="run1",
    pos_per_step=20_000,
    neg_ratio=1.0,
    steps_per_epoch=3,
    pair_micro_bs=2000,
)

y_mapped = map_vector_to_clusters(y, y_pred)
