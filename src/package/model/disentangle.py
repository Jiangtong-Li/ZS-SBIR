import math

import torch
import torch.nn as nn
import torch.functional as F

class general_encoder(nn.Module):
    """
    This is a default encoder to encode image/sketch feature from the encode result of backbone(usually VGG-16).
    This general encoder contains a three layer MLP with Leaky ReLU as activation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob):
        self.mid_size = int((input_size+output_size)/2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.mid_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob), 
            nn.Linear(self.mid_size, self.mid_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob), 
            nn.Linear(self.mid_size, output_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, features):
        out_feature = self.encoder(features)
        return out_feature

class general_decoder(nn.Module):
    """
    This is a default decoder to generator reconstructure image/sketch feature from middle representation.
    Samiliar to the general encoder, it also have a three layer MLP with Leaky ReLU as avtivation function. Dropout is also added during training
    """
    def __init__(self, input_size, output_size, dropout_prob):
        self.mid_size = int((input_size+output_size)/2)
        self.decoder = nn.Sequential(
            nn.Linear(input_size, self.mid_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob), 
            nn.Linear(self.mid_size, self.mid_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob), 
            nn.Linear(self.mid_size, output_size), 
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, features):
        out_feature = self.decoder(features)
        return out_feature

class disentangle_vae(nn.Module):
    """
    This is the overall model of disentangle VAE, which contains three encoder and two decoder.
    """
    def __init__(self, hidden_size, semantics_size, pretrained_embedding, \
                 dropout=0.5, from_pretrain=True, VGG_path=None, fix_cnn=True, \
                 fix_embedding=True, logger=None):
        self.hidden_size = hidden_size
        self.semantics_size = semantics_size