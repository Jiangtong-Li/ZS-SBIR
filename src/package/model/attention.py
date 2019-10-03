import torch
import torch.nn as nn

class CNN_attention(nn.Module):
    def __init__(self, channel_size):
        super(CNN_attention, self).__init__()
        self.attention = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

    def forward(self, conv_feature):
        # conv_feature: [batch_size, channel_size, height, width]
        conv_shape = conv_feature.shape
        assert len(conv_shape) == 4
        att_weight = self.attention(conv_feature)
        att_weight = att_weight.reshape((conv_shape[0], conv_shape[1], conv_shape[2]*conv_shape[3]))
        att_weight = self.softmax(att_weight)
        att_weight = att_weight.reshape((conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[3]))
        assert att_weight.shape == conv_feature.shape
        weighted_conv_feature = att_weight * conv_feature
        weighted_conv_feature = weighted_conv_feature.mean([2, 3])
        return weighted_conv_feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class attention(nn.Module):
    def __init__(self, channel_size):
        super(attention, self).__init__()
        self.attention = nn.Linear(4096, 512)

    def forward(self, conv_feature):
        # conv_feature: [batch_size, channel_size, height, width]
        conv_shape = conv_feature.shape
        weighted_conv_feature = self.attention(conv_feature.reshape(conv_shape[0], -1))
        return weighted_conv_feature

        