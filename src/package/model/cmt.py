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


class CMT(nn.Module):
    """
    We use vgg16 for the network.
    """
    def __init__(self, from_pretrain=True, batch_normalization=True, d=300, h=500, back_bone=None, sz=32):
        """
        We minimize || w - t2 * f(t1 * xi) ||, where w is the semantics, t2 ~ d * h
        :param from_pretrain: Whether to load pretrained parameters to VGG16.
        :param batch_normalization: Whether to apply batch normalization.
        :param d: size of output from f. Also size of the codebook.
        :param h: size of input to f
        :param back_bone:
            1. Set back_bone to None or default: use the default setting of CMT:
                a. x is resized to 32 and then squeezed to an array.
                b. t1 ~ h * I
                c. f = tanh
            2. Set back_bone to vgg:
                a. x is a 3-channel image
                b. Ignore t1.
                c. f = CNN
            3. Otherwise:
                ERROR.
        :param sz: size of image
        """
        super(CMT, self).__init__()
        self.d = d
        self.h = h
        self.sz = sz
        self._vgg16 = vgg16_bn if batch_normalization else vgg16
        self.from_pretrain = from_pretrain
        self.back_bone = back_bone
        if back_bone is None or back_bone == 'default':
            feat_fun = self._make_layers_default
        elif back_bone == 'vgg':
            feat_fun = self._make_layers_vgg
        else:
            raise Exception("back_bone should be None or vgg, but got {}".format(back_bone))
        # feat_fun = self._make_layers_debug
        self.features_sk = feat_fun()
        self.features_im = feat_fun()

    def _make_layers_default(self):
        return nn.Sequential(OrderedDict([
            ('flatten', Flatten()),
            ('t1', nn.Linear(self.sz * self.sz * 3, self.h)),
            ('bn1', nn.BatchNorm1d(self.h)),
            ('tanh', nn.Tanh()),
            ('t2', nn.Linear(self.h, self.d))
            ]))

    def _make_layers_debug(self):
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)),
            ('bn1', nn.BatchNorm2d(6)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)),
            ('bn2', nn.BatchNorm2d(16)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),

            ('flatten', Flatten()),

            ('fc1', nn.Linear(1600, self.h)),
            ('bn3', nn.BatchNorm1d(self.h)),
            ('relu3', nn.ReLU(inplace=True)),

            ('fc2', nn.Linear(self.h, self.d)),
            ]))

    '''
    
    ('bn3', nn.BatchNorm2d(120)),
    ('relu3', nn.ReLU(inplace=True)),

    ('fc2', nn.Linear(400, self.d)),
    '''

    def _make_layers_vgg(self):
        return nn.Sequential(OrderedDict([
            ('vgg', self._vgg16(pretrained=self.from_pretrain, return_type=1)),
            ('t2_after_vgg', nn.Linear(4096, self.d))
            ]))

    def fix_vgg(self):
        if self.back_bone != 'vgg':
            return None
        for k, v in self.named_parameters():
            if k.count('t2_after_vgg') == 0:
                v.requires_grad = False

    def forward(self, sk=None, im=None):
        if sk is not None:
            # print(self.features_sk(sk).shape)
            # exit()
            return self.features_sk(sk)
        elif im is not None:
            # print(im.shape)
            # print(self.features_im(im).shape)
            # exit()
            return self.features_im(im)
        else:
            raise Exception("either sk or im should be provided")


import numpy as np
def _test():
    a = np.zeros([10,3,48,48])
    cmt = CMT(d=300, h=500, back_bone='vgg', sz=48)
    cmt.fix_vgg()
    for k, v in cmt.named_parameters():
        if v.requires_grad:
            print(k)
    # print(cmt.parameters())
    # print(cmt(torch.tensor(a).float()).shape)


if __name__=='__main__':
    _test()