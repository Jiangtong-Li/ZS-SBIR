import torch
import torch.nn as nn
from collections import OrderedDict
# import tensorboardX as tbx


class Flatten(nn.Module):
    def __init__(self, do_norm=True):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class L2Normalization(nn.Module):
    def __init__(self, do_norm=True):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        div = torch.sqrt(torch.sum(x * x,1))
        x = (x.T / (div + 1e-10)).T
        return x


class SaN(nn.Module):
    def __init__(self):
        super(SaN, self).__init__()
        self.features = self._make_layers()

    def forward(self, x):
        # print("input.shape=", input.shape) # input.shape= torch.Size([4, 1, 256, 256])
        return self.features(x)

    def _make_layers(self):
        '''
        Refer to https://github.com/yuchuochuo1023/Deep_SBIR_tf/blob/master/triplet_sbir_train.py
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        '''
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 15, 3, 0)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(3,2,0)),

            ('conv2', nn.Conv2d(64, 128, 5, 1, 0)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(3,2,0)),

            ('conv3', nn.Conv2d(128, 256, 3, 1, 1)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu5', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(3,2,0)),

            ('conv6', nn.Conv2d(256, 512, 7, 1, 0)),
            ('relu6', nn.ReLU()),
            # ('dp6', nn.Dropout(0.50)),

            ('conv7', nn.Conv2d(512, 256, 1, 1, 0)),
            # ('relu7', nn.ReLU()),
            # ('dp7', nn.Dropout(0.50)),
            ('flatten', Flatten()),
            ('l2norm', L2Normalization()),
            ]))


if __name__=='__main__':
    pass