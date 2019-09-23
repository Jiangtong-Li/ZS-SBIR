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
        self.siamese_loss = _Siamese_loss()
        self.l1_regularization = _Regularization(self.model, 0.1, p=1, logger=logger)
        self.l2_regularization = _Regularization(self.model, 0.1, p=2, logger=logger)

    def forward(self, sketch, image, label):
        sketch_feature = self.model(sketch)
        image_feature = self.model(image)
        loss_siamese, sim, dis_sim = self.siamese_loss(sketch_feature, image_feature, label, \
                            self.margin, self.loss_type, self.distance_type)
        loss_l1 = self.l1_regularization(self.model)
        loss_l2 = self.l2_regularization(self.model)
        return loss_siamese, sim, dis_sim, loss_l1, loss_l2
    
    def get_feature(self, input):
        return self.model(input)