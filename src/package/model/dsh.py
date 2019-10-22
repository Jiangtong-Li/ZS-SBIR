import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# import tensorboardX as tbx

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


class DSH(nn.Module):
    def __init__(self, m, config=1):
        """
        :param m: number f bits of the hash codes
        :param config: 1 or 2.
            1: Perform cross weight fusion as MDL-CW does.
                Fill 0 for C2 network for sketch hash code generation.
            2. Perform cross weight fusion like https://github.com/ymcidence/DeepSketchHashing/blob/master/Sketchy/Image/deploy_it_256.prototxt does.
        """
        super(DSH, self).__init__()
        self.m = m
        self.config = config
        self.featuresC1conv = self._make_layersC1conv()
        self.featuresC2conv = self._make_layersC2conv()

        self.featuresC1fc1 = nn.Sequential(OrderedDict([
            ('conv_fc_a', nn.Conv2d(in_channels=256 * 2, out_channels=4096, kernel_size=7, stride=1, padding=0)),
            ('conv_fc_a_bn', nn.BatchNorm2d(4096)),
            ('conv_fc_a_relu', nn.ReLU(inplace=True)),
        ]))
        self.featuresC1fc2 = nn.Sequential(OrderedDict([
            ('conv_fc_b', nn.Conv2d(in_channels=4096 * ((config == 1) + 1), out_channels=1024, kernel_size=1, stride=1, padding=0)),
            ('conv_fc_b_bn', nn.BatchNorm2d(1024)),
            ('conv_fc_b_relu', nn.ReLU(inplace=True)),
        ]))
        self.featuresC1hash = nn.Sequential(OrderedDict([
            ('conv_hash_c1', nn.Conv2d(in_channels=1024 * ((config == 1) + 1), out_channels=self.m, kernel_size=1, stride=1, padding=0)),
            ('conv_hash_c1_flatten', Flatten()),
        ]))

        self.featuresC2fc1 = nn.Sequential(OrderedDict([
            ('conv_fc_a', nn.Conv2d(in_channels=256 * ((config == 1) + 1), out_channels=4096, kernel_size=7, stride=1, padding=0)),
            ('conv_fc_a_bn', nn.BatchNorm2d(4096)),
            ('conv_fc_a_relu', nn.ReLU(inplace=True)),
        ]))
        self.featuresC2fc2 = nn.Sequential(OrderedDict([
            ('conv_fc_b', nn.Conv2d(in_channels=4096 * ((config == 1) + 1), out_channels=1024, kernel_size=1, stride=1, padding=0)),
            ('conv_fc_b_bn', nn.BatchNorm2d(1024)),
            ('conv_fc_b_relu', nn.ReLU(inplace=True)),
        ]))
        self.featuresC2hash = nn.Sequential(OrderedDict([
            ('conv_hash_c2', nn.Conv2d(in_channels=1024 * ((config == 1) + 1), out_channels=self.m, kernel_size=1, stride=1, padding=0)),
            ('conv_hash_c2_flatten', Flatten()),
        ]))

    def _cat(self, ts, other=None, dim=1):
        if self.config == 1 or other is not None:
            return torch.cat([ts, torch.zeros(ts.shape).cuda() if other is None else other], dim=dim)
        else:
            return ts

    def forward(self, sk=None, st=None, im=None):
        """
        :param sk: sketch as C2 input to produce sketch hash
        :param st: sketch token as C2 input to produce image hash
        :param im: image as C1 input to produce image hash
        :return: Decided on whether the correponsind inputs are provided,
                    raw sketch and image outputs are generated
        """
        rets = []
        if sk is not None:
            c2conv = self.featuresC2conv(sk)
            c2fc1 = self.featuresC2fc1(self._cat(c2conv))
            c2fc2 = self.featuresC2fc2(self._cat(c2fc1))
            c2hash = self.featuresC2hash(self._cat(c2fc2))
            rets.append(c2hash)
        if st is not None and im is not None:
            # print(st.shape, im.shape)
            c2conv = self.featuresC2conv(st)
            c1conv = self.featuresC1conv(im)
            if self.config == 1:
                # print(c1conv.shape, c2conv.shape)
                c1fc1 = self.featuresC1fc1(self._cat(c1conv, c2conv))
                c2fc1 = self.featuresC2fc1(self._cat(c2conv, c1conv))
                c1fc2 = self.featuresC1fc2(self._cat(c1fc1, c2fc1))
                c2fc2 = self.featuresC2fc2(self._cat(c2fc1, c1fc1))
                c1hash = self.featuresC1hash(self._cat(c1fc2, c2fc2))
                rets.append(c1hash)
            else:
                c1fc1 = self.featuresC1fc1(self._cat(c1conv, c2conv))
                c1fc2 = self.featuresC1fc2(c1fc1)
                c1hash = self.featuresC1hash(c1fc2)
                rets.append(c1hash)
        return rets

    def _make_layersC1conv(self):
        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        [0 0 0 0 (0 0 0 0 0 0 0] 0 0 0 0) =>> 11-1, 15-2 : output = (n - 7) / 4
        input image is sized 3*227*227
        '''
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)),
            ('bn1', nn.BatchNorm2d(96)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),


            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)),
            ('bn2', nn.BatchNorm2d(256)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),


            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),


            ('conv4', nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)),
            ('bn4', nn.BatchNorm2d(384)),
            ('relu4', nn.ReLU(inplace=True)),


            # the number of output channels should be 256, not 384 given in the paper.
            # Check the repo https://github.com/ymcidence/DeepSketchHashing/blob/master/Sketchy/Image/deploy_it_128.prototxt
            # for details.
            ('conv5', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ]))


    def _make_layersC2conv(self):
        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        input sketches/sketch tokens is sized 1*200*200
        '''
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=14, stride=3, padding=0)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),

            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),

            ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('bn2_1', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),

            ('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('bn2_2', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),

            ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('bn3_1', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),

            ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('bn3_2', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ]))


def update_D(vec_bi, vec_bs, bi, bs):
    """
    :param vec_bi: word vector of images, ni * d
    :param vec_bs: word vector of sketches, ns * d
    :param bi: hash_codes_of(image), ni * m
    :param bs: hash_codes_of(sketch),  ns * m
    :return: updated D, d * m
    """
    vec_bi = vec_bi.T
    vec_bs = vec_bs.T
    bi = bi.T
    bs = bs.T

    #   (d * ni)(ni * m) + (d * ns)(ns * m) => (d * m)
    #   (m * ni)(ni * m) + (m * ns)(ns * m) => (m * m)
    #   (d * m)(m * m) => (d * m)
    #   Operand @ is equivalent to np.matmul
    return (vec_bi @ bi.T + vec_bs @ bs.T) @\
           (bi @ bi.T + bs @ bs.T)


def _update_Bi(bi, bs, vec_bi , W, D, Fi, lamb, gamma):
    """
    :param bi: hash_codes_of(image), ni * m
    :param bs: hash_codes_of(sketch),  ns * m
    :param vec_bi: word vector of images, ni * d
    :param W: similarity matrix(1/-1) between images and sketches, ni * ns
    :param D: d * m
    :param Fi: raw output of image hashing network, ni * m
    :param lamb: Weight of the second loss term
    :param gamma: Weight of the third loss term
    :return: updated bi, ni * m
    """
    '''
    assert bi.shape[1] == bs.shape[1], '_update_Bi: bi and bs should have the same code dimension. Get {} and {}'.\
        format(bi.shape[1], bs.shape[1])
    assert Fi.shape == bi.shape, '_update_Bi: Fi and bi should have the same shape. Get {} and {}'.\
        format(Fi.shape, bi.shape)
    '''
    # print(bi.shape, bs.shape, vec_bi.shape, W.shape, D.shape, Fi.shape)
    # (593, 128) (661, 128) (593, 300) (593, 661) (300, 128) (592, 128)

    ni = Fi.shape[0]
    ns = bs.shape[0]
    m = Fi.shape[1]

    # Some data could be dropped because of batch size
    W = W[:ni, :ns]
    bi = bi[:ni].T # m * ni
    bs = bs.T # m * ns
    Fi = Fi.T # m * ni
    vec_bi = vec_bi[:ni].T

    # R ~ m * ni
    R = bs @ (W.T * m) + lamb * D.T @ vec_bi + gamma * Fi
    bi_new = []
    for k in range(m):
        bi_hatk = np.delete(bi, k, axis=0) # (m-1) * ni
        bs_hatk = np.delete(bs, k, axis=0) # (m-1) * ns
        D_hatk = np.delete(D, k, axis=1) # d * (m - 1)
        term2 = (bs[k].reshape([1, -1]) @ bs_hatk.T @ bi_hatk).reshape(-1) # bs_hatk * bi_hatk ~ ns * ni, bs[k] * (*) ~ 1 * ni
        # print(D[:, k].reshape([1, -1]).shape, D_hatk.shape, bi_hatk.shape)
        term3 = (D[:, k].reshape([1, -1]) @ D_hatk @ bi_hatk).reshape(-1) # D_hatk * bi_hatk ~ d * ni, D[:, k] * (*) ~ 1 * ni

        bi_k_new = np.sign(R[k] # 1 * ni
                           - term2 - lamb * term3)
        bi_new.append(bi_k_new)
    return np.vstack(bi_new).T


def update_B(bi, bs, vec_bi, vec_bs, W, D, Fi, Fs, lamb, gamma):
    """
    :param bi: hash_codes_of(image), ni * m
    :param bs: hash_codes_of(sketch),  ns * m
    :param vec_bi: word vector of images, ni * d
    :param vec_bs: word vector of sketches, ns * d
    :param Ws: similarity matrix(1/-1) between images and sketches, ni * ns
    :param D: d * m
    :param Fi: raw output of image hashing network, ni * m
    :param Fs: raw output of sketch hashing network, ns * m
    :param lamb: Weight of the second loss term, scalar
    :param gamma: Weight of the third loss term, scalar
    :return: [bi_new(ni * m), bs_new(ns * m)]
    """
    bi = _update_Bi(bi, bs, vec_bi, W, D, Fi, lamb, gamma)
    bs = _update_Bi(bs, bi, vec_bs, W.T, D, Fs, lamb, gamma)
    return bi, bs


def _test_update_D():
    print("_test_update_D...")
    ni = 600
    ns = 700
    m = 500
    d = 300

    bi = np.random.rand(ni,m)
    bs = np.random.rand(ns,m)
    vec_bi = np.random.rand(ni, d)
    vec_bs = np.random.rand(ns, d)

    print(update_D(bi=bi, bs=bs, vec_bs=vec_bs, vec_bi=vec_bi).shape)


def _test_update_B():
    print("_test_update_B...")
    ni = 1000
    ns = 1200
    m = 70
    d = 80
    lamb = 1
    gamma = 1

    def z(shape):
        return np.zeros(shape, dtype=np.float32)
    # z = np.zeros
    bi = z([ni, m])
    bs = z([ns, m])
    vec_bi = z([ni, d])
    vec_bs = z([ns, d])
    W = z([ni, ns])
    D = z([d, m])
    Fi = z([ni, m])
    Fs = z([ns, m])
    print(bi.shape, bs.shape)
    bi, bs = update_B(bi=bi, bs=bs, vec_bi=vec_bi, vec_bs=vec_bs, W=W, D=D, Fi=Fi, Fs=Fs, lamb=lamb, gamma=gamma)
    print(bi.shape, bs.shape)


if __name__=='__main__':
    _test_update_D()
    _test_update_B()
    pass

'''
(1000, 200) (1200, 200)
shapes (1000, 1200) (200, 1200) (1200, 1000)
shapes (1200, 1000) (1000, 1200) (1000, 1200)
(1200, 1000) (1000, 1200)

'''