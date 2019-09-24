import torch
import torch.nn as nn
from package.model.vgg import vgg16, vgg16_bn
from package.loss.siamese_loss import _Siamese_loss
from package.loss.regularization import _Regularization

class Siamese(nn.Module):
    def __init__(self, margin, loss_type, distance_type, logger, from_pretrain=True, batch_normalization=True):
        super(Siamese, self).__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.distance_type = distance_type
        self.logger = logger
        if batch_normalization:
            self.model = vgg16_bn(pretrained=from_pretrain, return_type=1)
        else:
            self.model = vgg16(pretrained=from_pretrain, return_type=1)

    def forward(self, sketch, image):
        sketch_feature = self.model(sketch)
        image_feature = self.model(image)
        return sketch_feature, image_feature
    
    def get_feature(self, input):
        return self.model(input)