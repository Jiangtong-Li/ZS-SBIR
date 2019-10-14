import math

import torch
import torch.nn as nn
import torch.functional as F

from src.package.model.variational_dropout import VariationalDropout
from src.package.model.vgg import vgg16

class Encoder_Decoder(nn.Module):
    """
    This is a default encoder/decoder to map features from one domain to another one.
    This general encoder contains several layers of MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob, num_layers):
        super(Encoder_Decoder, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.mid_size = int((input_size+output_size)/2)
        encoder = list()
        encoder += [nn.Linear(input_size, self.mid_size), nn.LeakyReLU(inplace=True), nn.Dropout(dropout_prob)]
        for _ in range(num_layers-2):
            encoder += [nn.Linear(self.mid_size, self.mid_size), nn.LeakyReLU(inplace=True), nn.Dropout(dropout_prob)]
        encoder += [nn.Linear(self.mid_size, output_size), nn.LeakyReLU(inplace=True), nn.Dropout(dropout_prob)]
        self.encoder = nn.Sequential(*encoder)
    
    def forward(self, features):
        out_feature = self.encoder(features)
        return out_feature

class Variational_Sampler(nn.Module):
    """
    Variational sampler for image apperance features
    """
    def __init__(self, hidden_size):
        super(Variational_Sampler, self).__init__()
        self.hidden_size = hidden_size
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.logvar_encoder = nn.Linear(hidden_size, hidden_size)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def lossfn(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    def forward(self, x):
        x_mean = self.mean_encoder(x)
        x_logvar = self.logvar_encoder(x)
        z = self.reparameterize(x_mean, x_logvar)
        loss = self.lossfn(x_mean, x_logvar)
        return z, loss

class Semantic_Preservation(nn.Module):
    """
    Semantic Preservation module for reconstructured image feature
    """
    def __init__(self, feature_size, semantic_size, dropout_prob, loss_fn):
        super(Semantic_Preservation, self).__init__()
        self.fea2sem = Encoder_Decoder(feature_size, semantic_size, dropout_prob, 2)
        self.loss_fn = loss_fn

    def forward(self, x, target):
        predict = self.fea2sem(x)
        loss = self.loss_fn(x, target)
        return predict, loss

class CMDT(nn.Module):
    """
    The overall model of our zero-shot sketch based image retrieval using cross-modal domain translation
    """
    def __init__(self, pca_size, hidden_size, semantic_size, pretrain_embedding, 
                 from_pretrain=True, dropout_prob=0.3, fix_backbone=True, fix_embedding=True, 
                 seman_dist='l2'):
        super(CMDT, self).__init__()
        # Dist Matrics
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.cosine = nn.CosineSimilarity(dim=-1)
        if seman_dist == 'l2':
            self.seman_dist = self.l2_dist
        elif seman_dist == 'cosine':
            self.seman_dist = self.cosine
        else:
            raise ValueError('The seman_dist should be l2 or cosine')
        # Modules
        self.semantics = nn.Embedding.from_pretrained(pretrain_embedding)
        if fix_embedding:
            self.semantics.weight.requires_grad=False
        #self.backbone = vgg16(pretrained=from_pretrain, return_type=3, dropout=dropout_prob)
        #if fix_backbone:
        #    for param in self.backbone.parameters():
        #        param.requires_grad = False
        self.sketch_encoder = Encoder_Decoder(pca_size, hidden_size, dropout_prob, 3)
        self.image_encoder_S = Encoder_Decoder(pca_size, hidden_size, dropout_prob, 3)
        self.image_encoder_A = Encoder_Decoder(pca_size, hidden_size, dropout_prob, 3)
        self.variational_sample = Variational_Sampler(hidden_size)
        self.sketch_decoder = Encoder_Decoder(hidden_size, pca_size, dropout_prob, 3)
        self.image_decoder = Encoder_Decoder(hidden_size, pca_size, dropout_prob, 3)
        self.semantic_preserve = Semantic_Preservation(pca_size, semantic_size, dropout_prob, self.seman_dist)
    
    def forward(self, image_p, image_n, sketch, semantics):
        """
        image_p [batch_size, pca_size]
        image_n [batch_size, pca_size]
        sketch [batch_size, pca_size]
        semantics [batch_size, 1]
        """
        # recode size info
        batch_size = image_p.shape[0]
        pca_size = image_p.shape[1]
        # load semantics info
        semantics_embedding = self.semantics(semantics)
        semantics_embedding = semantics_embedding.reshape([batch_size, -1])


        