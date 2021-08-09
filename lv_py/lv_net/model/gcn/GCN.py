import torch.nn as nn
import scipy.sparse as sp
import torch
import numpy as np
import torch.nn.functional as F
import math

from typing import *
from enum import Enum
from torch.nn.parameter import Parameter


class InitMode(Enum):
    xavier = 1
    kaiming = 2
    uniform = 3

class ModeErrorException(Exception):

    def __init__(self, msg):
        self.msg = msg
    
    def __repr__(self) -> str:
        return f"expected xavier, kaiming or uniform but got {self.msg}"

    def __str__(self) -> str:
        return self.__repr__()

class GraphConvolution(nn.Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: InitMode = InitMode.xavier):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = self.init_weight()
        self.bias = self.init_bias(bias)
        self.init_parameters(init)

    def __repr__(self):
        return f"in_features = {self.in_features} and out_features = {self.out_features}\
                \nweights = {self.weights}\
                \nbias = {self.bias}" 
    
    def forward(self, x, adjcent_matrix):
        """
            The inputs of GCN layer are feature matrix and graph(adjacent matrix)
            self.weight is a row * column matrix which represent in_features
            and out_features, for Saliency Object Detection, it may be channels
            (every channel is a feature vector(feature map) or a graph node)
        """
        # for Y = AXW
        # Y is output or H(i+1), 
        # A is adjacent matrix or degrees matrix
        # X is H(i), 
        # W is weight which is used to change channels
        support = torch.mm(x, self.weights)
        output = torch.spmm(adjcent_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    
    """Here are some initialize operator for GCN's parameters"""
    def init_weight(self):
        float_tensor = torch.Tensor(self.in_features, self.out_features)
        float_tensor = Parameter(float_tensor)
        return float_tensor

    def init_bias(self, bias):
        return Parameter(torch.Tensor(self.out_features)) if bias else None

    def init_parameters(self, mode: InitMode):
        if mode == InitMode.xavier:
            self.xavier()
        elif mode == InitMode.kaiming:
            self.kaiming()
        elif mode == InitMode.uniform:
            self.uniform()
        else:
            raise ModeErrorException(mode)
    
    def xavier(self):
        # Implement Xavier Uniform
        nn.init.xavier_normal_(self.weights.data, gain=0.02) 
        if self.bias is not None:
            nn.init.constant_(self.bias.data, val=0.)

    def uniform(self):
        # standard variance
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def kaming(self):
        nn.init.kaiming_normal_(self.weights.data, a=0, mode="fan_in")
        if self.bias is not None:
            nn.init.constant_(self.bias.data, val=0.0)

class GraphAttention(nn.Module):
    """
        Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, concat: bool = True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.device = torch.cuda.is_avaliable()
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adjacent_matrix):
        h = torch.mm(x, self.W)
        N = h.size(0)
        f1 = torch.matmul(h, self.a1)
        f2 = torch.matmul(h, self.a2)
        
class GCN(nn.Module):
    """
        A simple GCN with two layers
    """
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# https://blog.csdn.net/d179212934/article/details/108093614

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == "__main__":
    pass
