import math

import torch
import torch.nn as nn
import torch.functional as F

from package.model.variational_dropout import VariationalDropout
from package.model.vgg import vgg16
from package.loss.triplet_loss import _Triplet_loss

class Encoder(nn.Module):
    """
    This is a default encoder/decoder to map features from one domain to another one.
    This general encoder contains several layers of MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob, num_layers):
        super(Encoder, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.mid_size = int((input_size+output_size)/2)
        encoder = list()
        encoder += [nn.Linear(input_size, self.mid_size), nn.LeakyReLU(inplace=True), nn.BatchNorm1d(self.mid_size), nn.Dropout(dropout_prob)]
        for _ in range(num_layers-2):
            encoder += [nn.Linear(self.mid_size, self.mid_size), nn.LeakyReLU(inplace=True), nn.BatchNorm1d(self.mid_size)]
        encoder += [nn.Linear(self.mid_size, output_size), nn.LeakyReLU(inplace=True), nn.BatchNorm1d(output_size)]
        self.encoder = nn.Sequential(*encoder)
    
    def forward(self, features):
        out_feature = self.encoder(features)
        return out_feature

class Decoder(nn.Module):
    """
    This is a default encoder/decoder to map features from one domain to another one.
    This general encoder contains several layers of MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob, num_layers):
        super(Decoder, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.mid_size = int((input_size+output_size)/2)
        encoder = list()
        encoder += [nn.Linear(input_size, self.mid_size), nn.LeakyReLU(inplace=True), nn.Dropout(dropout_prob)]
        for _ in range(num_layers-2):
            encoder += [nn.Linear(self.mid_size, self.mid_size), nn.LeakyReLU(inplace=True)]
        encoder += [nn.Linear(self.mid_size, output_size), nn.LeakyReLU(inplace=True)]
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
        self.mean_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.logvar_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size))
    
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
        self.fea2sem = Decoder(feature_size, semantic_size, dropout_prob, 2)
        self.activation = nn.LeakyReLU()
        self.loss_fn = loss_fn

    def forward(self, x, target):
        predict = self.activation(self.fea2sem(x))
        loss = torch.mean(self.loss_fn(predict, target))
        return predict, loss

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
                 fix_embedding=True, seman_dist='cosine', triplet_dist='l2', margin=1, logger=None):
        super(CMDTrans_model, self).__init__()
        # Dist Matrics
        self.l2_dist = nn.PairwiseDistance(p=2)
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
        self.semantics = nn.Embedding.from_pretrained(pretrain_embedding)
        if fix_embedding:
            self.semantics.weight.requires_grad=False
        #self.backbone = vgg16(pretrained=from_pretrain, return_type=3, dropout=dropout_prob)
        #if fix_backbone:
        #    for param in self.backbone.parameters():
        #        param.requires_grad = False
        self.sketch_encoder = Encoder(pca_size, hidden_size, dropout_prob, 3)
        self.image_encoder_S = Encoder(raw_size, hidden_size, dropout_prob, 3)
        self.image_encoder_A = Encoder(raw_size, hidden_size, dropout_prob, 3)
        #self.image_encoder_A = Encoder(pca_size*2, hidden_size, dropout_prob, 3)
        self.variational_sample = Variational_Sampler(hidden_size)
        self.sketch_decoder = Decoder(hidden_size, pca_size, dropout_prob, 3)
        self.image_decoder = Decoder(hidden_size*2, raw_size, dropout_prob, 3)
        #self.image_decoder = Encoder_Decoder(hidden_size+pca_size, pca_size, dropout_prob, 3)
        #self.image_decoder = Encoder_Decoder(hidden_size*2, pca_size, dropout_prob, 3)
        self.semantic_preserve = Semantic_Preservation(raw_size, semantic_size, dropout_prob, self.seman_dist)

        # Loss
        self.triplet_loss = _Triplet_loss(self.triplet_dist, margin)
    
    def forward(self, sketch, image_p, image_n, semantics):
        """
        image_p [batch_size, pca_size]
        image_n [batch_size, pca_size]
        sketch [batch_size, pca_size]
        semantics [batch_size, 1]
        """
        # recode size info
        batch_size = image_p.shape[0]
        _pca_size = image_p.shape[1]
        # load semantics info
        semantics_embedding = self.semantics(semantics)
        semantics_embedding = semantics_embedding.reshape([batch_size, -1])
        # model
        sketch_encode_feature = self.sketch_encoder(sketch)
        image_n_encode_feature_s = self.image_encoder_S(image_n)
        image_p_encode_feature_s = self.image_encoder_S(image_p)
        image_p_encode_feature_a = self.image_encoder_A(image_p)
        #image_p_encode_feature_a = self.image_encoder_A(torch.cat([image_p,sketch], dim=1))
        image_p_encode_feature_a_resampled, kl_loss = self.variational_sample(image_p_encode_feature_a) # kl loss(1)
        image_p_sketch_feature_combine = torch.cat([image_p_encode_feature_a_resampled, sketch_encode_feature], dim=1)
        #image_p_sketch_feature_combine = torch.cat([image_p_encode_feature_a_resampled, sketch], dim=1)
        #image_p_sketch_feature_combine = torch.cat([torch.zeros_like(image_p_encode_feature_a_resampled).to(sketch_encode_feature.device), sketch_encode_feature], dim=1)
        image_translate = self.image_decoder(image_p_sketch_feature_combine)
        sketch_translate = self.sketch_decoder(image_p_encode_feature_s)
        _predicted_semantic, semantics_loss = self.semantic_preserve(image_translate, semantics_embedding) # seman loss(2)
        # loss
        triplet_loss = self.triplet_loss(image_p_encode_feature_s, image_n_encode_feature_s, sketch_encode_feature) # triplet loss(3)
        image_translate_loss = torch.mean(self.l2_dist(image_translate, image_p)) # image loss(5)
        sketch_translate_loss = torch.mean(self.l2_dist(sketch_translate, sketch)) # sketch loss(6)
        orthogonality_loss = torch.mean(self.cosine(image_p_encode_feature_s, image_p_encode_feature_a)) # orthogonality loss(4)
        loss = dict()
        loss['kl'] = kl_loss
        loss['seman'] = semantics_loss
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
            z = torch.cat([eps, x_hidden], dim=1)
            image_translate = self.image_decoder(z)
            generated.append(image_translate)
        generated = torch.mean(torch.stack(generated, dim=-1),dim=-1)
        return generated
