import torch
import torch.nn as nn

# _SaN_loss = nn.CrossEntropyLoss()


class _CMT_loss(nn.Module):
    def __init__(self):
        super(_CMT_loss, self).__init__()
        self.d = nn.PairwiseDistance()

    def forward(self, feat, sematics):
        """
        :param feat: features of images or images. bs * d. d is the length of word vector.
        :param sematics: sematics of sketches. bs * d. d is the length of word vector.
        :return: loss
        """
        # print(sk_feat.type(), im_feat.type(), bs.type(), bi.type())
        return torch.mean(self.d(feat.float(), sematics.float()) ** 2)