import torch
import torch.nn as nn
from package.model.vgg import vgg16, vgg16_bn

class Siamese(nn.Module):
    def __init__(self, from_pretrain=True, batch_normalization=True):
        super(Siamese, self).__init__()
        if batch_normalization:
            self.model = vgg16_bn(pretrained=from_pretrain, return_type=0)
        else:
            self.model = vgg16(pretrained=from_pretrain, return_type=0)

    def forward(self, sketch, image):
        sketch_feature = self.model(sketch)
        image_feature = self.model(image)
        return sketch_feature, image_feature