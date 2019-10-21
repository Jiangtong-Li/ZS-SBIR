import torch
import torch.nn as nn

# _SaN_loss = nn.CrossEntropyLoss()


class _DSH_loss(nn.Module):
    def __init__(self, gamma=1):
        super(_DSH_loss, self).__init__()
        self.gamma = gamma
        self.d = nn.PairwiseDistance()

    def forward(self, sk_feat, im_feat, bs, bi):
        """
        :param sk_feat: features of sketches. bs * m.
        :param im_feat: features of images. bs * m.
        :param bs: hash codes of sketches. bs * m.
        :param bi: hash codes of images. bs * m.
        :return: loss
        """
        # print(sk_feat.type(), im_feat.type(), bs.type(), bi.type())
        return torch.mean(
            self.d(sk_feat, bs) ** 2 +
            self.d(im_feat, bi) ** 2
        ) * self.gamma