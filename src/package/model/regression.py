import math

import torch
import torch.nn as nn
import torch.functional as F

from package.loss.triplet_loss import _Triplet_loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x_true, x_pred):
        return torch.sqrt(torch.mean(torch.pow(x_pred-x_true, 2), dim=-1))

class Regressor(nn.Module):
    """
    The overall model of our zero-shot sketch based image retrieval using cross-modal domain translation
    """
    def __init__(self, raw_size, hidden_size, dropout_prob=0.3, logger=None):
        super(Regressor, self).__init__()
        # Dist Matrics
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.hidden_size = hidden_size

        # Modules
        middle_size = 2048
        self.sketch_encoder = nn.Sequential(nn.Linear(raw_size, middle_size), 
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(middle_size, eps=0.001, momentum=0.99), 
                                            nn.Dropout(dropout_prob), 
                                            nn.Linear(middle_size, hidden_size))
        self.image_encoder = nn.Sequential(nn.Linear(raw_size, middle_size), 
                                           nn.ReLU(inplace=True),
                                           nn.BatchNorm1d(middle_size, eps=0.001, momentum=0.99), 
                                           nn.Dropout(dropout_prob), 
                                           nn.Linear(middle_size, hidden_size))
        # Triplet loss
        self.triplet_loss = _Triplet_loss(dist=self.l2_dist, margin=10)

    def forward(self, sketch, image_p, image_n):
        """
        image [batch_size, pca_size]
        sketch [batch_size, pca_size]
        """
        # recode size info
        _batch_size = image_p.shape[0]
        _raw_size = image_n.shape[1]
        # model
        sketch_encoded = self.sketch_encoder(sketch)
        image_p_encoded = self.sketch_encoder(image_p)
        image_n_encoded = self.sketch_encoder(image_n)
        # loss
        loss = self.triplet_loss(image_p_encoded, image_n_encoded, sketch_encoded)
        return loss

    def inference_sketch(self, sketch):
        return self.sketch_encoder(sketch)

    def inference_image(self, image):
        return self.sketch_encoder(image)
