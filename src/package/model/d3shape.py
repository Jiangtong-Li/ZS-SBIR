import torch
import torch.nn as nn
from collections import OrderedDict
# import tensorboardX as tbx
from package.model.vgg import vgg16, vgg16_bn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        div = torch.sqrt(torch.sum(x * x,1))
        x = (x.T / (div + 1e-10)).T
        return x


class D3Shape(nn.Module):
    """
    We use vgg16 for the network.
    """
    def __init__(self, from_pretrain=True, batch_normalization=True, debug=False):
        super(D3Shape, self).__init__()
        self.debug = debug
        _vgg16 = vgg16_bn if batch_normalization else vgg16

        # two structure-identical networks that share no parameters
        self.features_sk = _vgg16(pretrained=from_pretrain, return_type=1)
        self.features_imsk = _vgg16(pretrained=from_pretrain, return_type=1)

    def forward(self, sk=None, imsk=None):
        rets = []
        if sk is not None:
            rets.append(self.features_sk(sk) if not self.debug else torch.zeros(sk.size(0), 10))
        if imsk is not None:
            rets.append(self.features_imsk(imsk) if not self.debug else torch.zeros(sk.size(0), 10))
        return rets


if __name__=='__main__':
    pass