import math

import torch
import torch.nn as nn
import torch.functional as F

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

class CVAE(nn.Module):
    """
    The overall model of our zero-shot sketch based image retrieval using cross-modal domain translation
    """
    def __init__(self, raw_size, hidden_size, dropout_prob=0.3, logger=None):
        super(CVAE, self).__init__()
        # Dist Matrics
        self.l2_dist = MSE()
        self.hidden_size = hidden_size

        # Modules
        middle_size = 2048
        self.encoder = nn.Sequential(nn.Linear(raw_size*2, middle_size), 
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(middle_size, eps=0.001, momentum=0.99), 
                                     nn.Dropout(dropout_prob), 
                                     nn.Linear(middle_size, hidden_size), 
                                     nn.ReLU(inplace=True), 
                                     nn.BatchNorm1d(hidden_size, eps=0.001, momentum=0.99))
        self.variational_sample = Variational_Sampler(hidden_size)
        self.image_decoder = nn.Sequential(nn.Linear(hidden_size+raw_size, middle_size), 
                                           nn.ReLU(inplace=True), 
                                           nn.Linear(middle_size, raw_size), 
                                           nn.ReLU(inplace=True))
        self.sketch_decoder = nn.Sequential(nn.Linear(raw_size, middle_size), 
                                            nn.ReLU(inplace=True), 
                                            nn.Linear(middle_size, raw_size), 
                                            nn.ReLU(inplace=True))
    
    def forward(self, sketch, image):
        """
        image [batch_size, pca_size]
        sketch [batch_size, pca_size]
        """
        # recode size info
        _batch_size = image.shape[0]
        _raw_size = image.shape[1]
        # model
        combine_feature = torch.cat([sketch, image], dim=1)
        encoded = self.encoder(combine_feature)
        encoded_resampled, kl_loss = self.variational_sample(encoded) # kl loss(1)
        encoded_resampled_combine = torch.cat([encoded_resampled, sketch], dim=1)
        image_recon = self.image_decoder(encoded_resampled_combine)
        sketch_recon = self.sketch_decoder(image_recon)
        # loss
        image_translate_loss = torch.mean(self.l2_dist(image_recon, image)) # image loss(5)
        sketch_translate_loss = torch.mean(self.l2_dist(sketch_recon, sketch)) # sketch loss(6)
        loss = dict()
        loss['kl'] = kl_loss
        loss['image'] = image_translate_loss
        loss['sketch'] = sketch_translate_loss
        return loss
    
    def inference_generation(self, x, sample_times=200):
        generated = list()
        for _ in range(sample_times):
            eps = torch.randn([x.shape[0], self.hidden_size]).to(x.device)
            z = torch.cat([eps, x], dim=1)
            image_translate = self.image_decoder(z)
            generated.append(image_translate)
        generated = torch.mean(torch.stack(generated, dim=-1),dim=-1)
        return generated
