import math

import torch
import torch.nn as nn
import torch.functional as F

from torch.nn.parameter import Parameter


class GCN_ZSIH(nn.Module):
    """
    GCN in "Zero-shot Sketch-Image Hashing"
    """
    def __init__(self, in_size, hidden_size, out_size, dropout, adj_scaler):
        super(GCN_ZSIH, self).__init__()
        self.gc1 = GraphConvolution(in_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, out_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.adj_scaler = adj_scaler

    def forward(self, x, semantics):
        """
        x [batch_size, concat_size]
        semantics [batch_size, semantics_size]
        """
        adj = self.build_adj(semantics)
        adj = self.graph_laplacian(adj)
        x = self.relu(self.dropout(self.gc1(x, adj)))
        x = self.sigmoid(self.dropout(self.gc2(x, adj)))
        return x
    
    def build_adj(self, x):
        squared_sum = torch.reshape(torch.sum(x*x, 1), [-1, 1])
        distance = squared_sum - 2 * torch.matmul(x, x.transpose(1,0)) + squared_sum.transpose(1,0)
        adj = torch.exp(-1 * distance / self.adj_scaler)
        return adj
    
    def graph_laplacian(self, adj):
        graph_size = adj.shape[0]
        a = adj
        d = a @ torch.ones([graph_size, 1]).to(adj.device)
        d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
        d_inv_sqrt = torch.eye(graph_size).to(adj.device) * d_inv_sqrt
        laplacian = d_inv_sqrt @ a @ d_inv_sqrt
        return laplacian

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = x @ self.weight
        output = adj @ support
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


