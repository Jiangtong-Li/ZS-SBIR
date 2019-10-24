import math

import torch
import torch.nn as nn
import torch.functional as F

from package.model.variational_dropout import VariationalDropout
from package.model.vgg import vgg16
from package.loss.triplet_loss import _Triplet_loss, _Ranking_loss

class Encoder(nn.Module):
    """
    This is a default encoder/decoder to map features from one domain to another one.
    This general encoder contains several layers of MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob):
        super(Encoder, self).__init__()
        self.mid_size = 2048
        encoder = [nn.Linear(input_size, self.mid_size), 
                   nn.ReLU(inplace=True), 
                   nn.BatchNorm1d(self.mid_size), 
                   nn.Dropout(dropout_prob), 
                   nn.Linear(self.mid_size, output_size), 
                   nn.ReLU(inplace=True), 
                   nn.BatchNorm1d(output_size)]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, features):
        out_feature = self.encoder(features)
        return out_feature

class Decoder(nn.Module):
    """
    This is a default encoder/decoder to map features from one domain to another one.
    This general encoder contains several layers of MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.mid_size = 2048
        decoder = [nn.Linear(input_size, self.mid_size), 
                   nn.ReLU(inplace=True), 
                   nn.Linear(self.mid_size, output_size), 
                   nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, features):
        out_feature = self.decoder(features)
        return out_feature

class Variational_Sampler(nn.Module):
    """
    Variational sampler for image apperance features
    """
    def __init__(self, hidden_size):
        super(Variational_Sampler, self).__init__()
        self.hidden_size = hidden_size
        self.mean_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.logvar_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def lossfn(self, mean, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

    def forward(self, x):
        x_mean = self.mean_encoder(x)
        x_logvar = self.logvar_encoder(x)
        z = self.reparameterize(x_mean, x_logvar)
        loss = self.lossfn(x_mean, x_logvar)
        return z, loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x_true, x_pred):
        return torch.mean(torch.pow(x_pred-x_true, 2), dim=-1)

class CosineDistance(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
    
    def forward(self, input1, input2):
        """
        batch_size * hidden_dim
        """
        assert input1.shape == input2.shape
        num = torch.sum(input1*input2, dim=self.dim)
        denorm = torch.sqrt(torch.sum(torch.pow(input1, 2), dim=self.dim)) * torch.sqrt(torch.sum(torch.pow(input2, 2), dim=self.dim))
        cosine_distance = 1 - num/denorm
        return cosine_distance

class CMDTrans_model(nn.Module):
    """
    The overall model of our zero-shot sketch based image retrieval using cross-modal domain translation
    """
    def __init__(self, pca_size, raw_size, hidden_size, semantic_size, pretrain_embedding, dropout_prob=0.3, 
                 fix_embedding=True, seman_dist='cosine', triplet_dist='l2', margin1=0, margin2=10, logger=None):
        super(CMDTrans_model, self).__init__()
        # Dist Matrics
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.mse = MSE()
        self.cosine = nn.CosineSimilarity(dim=-1)
        if seman_dist == 'l2':
            self.seman_dist = self.l2_dist
        elif seman_dist == 'cosine':
            self.seman_dist = self.cosine
        else:
            raise ValueError('The seman_dist should be l2 or cosine')
        if triplet_dist == 'l2':
            self.triplet_dist = self.l2_dist
        elif triplet_dist == 'cosine':
            self.triplet_dist = self.cosine
        else:
            raise ValueError('The triplet_dist should be l2 or cosine')

        # Modules
        self.sketch_encoder = Encoder(raw_size, hidden_size, dropout_prob)
        self.image_encoder_S = Encoder(raw_size, hidden_size, dropout_prob)
        self.image_encoder_A = Encoder(raw_size, hidden_size, dropout_prob)
        self.variational_sample = Variational_Sampler(hidden_size)
        self.sketch_decoder = Decoder(hidden_size, raw_size)
        #self.image_decoder = Decoder(hidden_size*2, raw_size)
        #self.image_decoder = Decoder(hidden_size+raw_size, raw_size)
        self.image_decoder = Decoder(hidden_size*2+raw_size, raw_size)

        # Loss
        self.triplet_loss = _Ranking_loss(self.triplet_dist, margin1, margin2)
    
    def forward(self, sketch, image_pair, image_unpair, image_n):
        """
        image_p [batch_size, pca_size]
        image_n [batch_size, pca_size]
        sketch [batch_size, pca_size]
        semantics [batch_size, 1]
        """
        # recode size info
        batch_size = image_pair.shape[0]
        _raw_size = image_pair.shape[1]
        # model
        sketch_encode_feature = self.sketch_encoder(sketch)
        image_n_encode_feature_s = self.image_encoder_S(image_n)
        image_unpaired_encode_feature_s = self.image_encoder_S(image_unpair)
        image_paired_encode_feature_s = self.image_encoder_S(image_pair)
        image_paired_encode_feature_a = self.image_encoder_A(image_pair)
        image_paired_encode_feature_a_resampled, kl_loss = self.variational_sample(image_paired_encode_feature_a) # kl loss(1)
        #image_paired_sketch_feature_combine = torch.cat([image_paired_encode_feature_a_resampled, sketch_encode_feature], dim=1)
        #image_paired_sketch_feature_combine = torch.cat([image_paired_encode_feature_a_resampled, sketch], dim=1)
        image_paired_sketch_feature_combine = torch.cat([image_paired_encode_feature_a_resampled, sketch_encode_feature, sketch], dim=1)
        image_translate = self.image_decoder(image_paired_sketch_feature_combine)
        sketch_translate = self.sketch_decoder(image_paired_encode_feature_s)
        # loss
        triplet_loss = self.triplet_loss(sketch_encode_feature, image_paired_encode_feature_s, image_unpaired_encode_feature_s, image_n_encode_feature_s) # triplet loss(3)
        image_translate_loss = torch.mean(self.mse(image_translate, image_pair)) # image loss(5)
        sketch_translate_loss = torch.mean(self.mse(sketch_translate, sketch)) # sketch loss(6)
        orthogonality_loss = torch.mean(self.cosine(image_paired_encode_feature_s, image_paired_encode_feature_a)) # orthogonality loss(4)
        loss = dict()
        loss['kl'] = kl_loss
        loss['triplet'] = triplet_loss
        loss['orthogonality'] = orthogonality_loss
        loss['image'] = image_translate_loss
        loss['sketch'] = sketch_translate_loss
        return loss
    
    def inference_structure(self, x, mode):
        """
        map sketch and image to structure space
        """
        if mode=='image':
            return self.image_encoder_S(x)
        elif mode=='sketch':
            return self.sketch_encoder(x)
        else:
            raise ValueError('The mode must be image or sketch')
    
    def inference_generation(self, x, sample_times=200):
        x_hidden = self.sketch_encoder(x)
        generated = list()
        for _ in range(sample_times):
            eps = torch.randn_like(x_hidden)
            #z = torch.cat([eps, x_hidden], dim=1)
            #z = torch.cat([eps, x], dim=1)
            z = torch.cat([eps, x_hidden, x], dim=1)
            image_translate = self.image_decoder(z)
            generated.append(image_translate)
        generated = torch.mean(torch.stack(generated, dim=-1),dim=-1)
        return generated
