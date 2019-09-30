import math

import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.functional as F


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

    def forward(self, input, adj, mask):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return torch.mul(output + self.bias, mask)
        else:
            return torch.mul(output, mask)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self,opt):
        super(GCN, self).__init__()
        nhid = int((opt.sg_feat_size+opt.feat_size)/2)
        self.gc1 = GraphConvolution(opt.sg_feat_size, nhid)
        self.gc2 = GraphConvolution(nhid, opt.feat_size)
        self.dropout = opt.dropout
        self.MLP = nn.Linear(opt.obj_size, 1)

    def forward(self, x, adj, mask):
        batch_size = x.shape[0]
        story_size = x.shape[1]
        obj_size = x.shape[2]
        x = x.reshape(batch_size*story_size, obj_size, -1)
        adj = adj.reshape(batch_size*story_size, obj_size, obj_size)
        mask = mask.reshape(batch_size*story_size, obj_size, 1)
        x = F.relu(self.gc1(x, adj, mask))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, mask)
        x = x.reshape(batch_size, story_size, -1, obj_size)
        x = self.MLP(x).reshape(batch_size, story_size, -1)
        return x
