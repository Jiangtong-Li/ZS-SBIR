import torch
import torch.nn as nn

# _SaN_loss = nn.CrossEntropyLoss()


class _D3Shape_loss(nn.Module):
    def __init__(self, cp=0.2, cn=10):
        super(_D3Shape_loss, self).__init__()
        self.alpha = 1 / cp
        self.beta = cn
        self.gamma = -2.77 / cn

    def _d(self, feat1, feat2):
        return torch.sum(torch.abs(feat1 - feat2), 1).cuda()

    def _l(self, d, is_same):
        return is_same * self.alpha * d * d + (1 - is_same) * self.beta * torch.exp(self.gamma * d)

    def forward(self, sketch1_feat, imsk1_feat, sketch2_feat, imsk2_feat, is_same):
        d_sk2sk = self._d(sketch1_feat, sketch2_feat)
        d_is2is = self._d(imsk1_feat, imsk2_feat)
        d_sk2is = self._d(sketch1_feat, imsk1_feat)
        is_same = is_same.view(is_same.size(0) * is_same.size(1))
        loss = self._l(d_sk2sk, is_same) + \
               self._l(d_is2is, is_same) + \
               self._l(d_sk2is, is_same)
        return torch.mean(loss)
