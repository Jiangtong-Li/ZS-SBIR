#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from package.model.vgg import vgg16, vgg16_bn
from package.loss.pcyc_loss import _GAN_Loss


class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = vgg16(pretrained=pretrained, return_type=0)
        # model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(4096, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Generator(nn.Module):
    def __init__(self, in_dim=512, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator, self).__init__()
        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))

        self.gen = nn.Sequential(*modules)

    def forward(self, x):
        return self.gen(x)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.cuda()
            x = x + noise
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False,
                 use_sigmoid=False):
        super(Discriminator, self).__init__()
        hid_dim = int(in_dim / 2)
        modules = list()
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.disc = nn.Sequential(*modules)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        return self.disc(x)


class AutoEncoder(nn.Module):
    def __init__(self, dim=300, hid_dim=300, nlayer=1):
        super(AutoEncoder, self).__init__()
        steps_down = np.linspace(dim, hid_dim, num=nlayer + 1, dtype=np.int).tolist()
        modules = []
        for i in range(nlayer):
            modules.append(nn.Linear(steps_down[i], steps_down[i + 1]),)
            modules.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*modules)

        steps_up = np.linspace(hid_dim, dim, num=nlayer + 1, dtype=np.int).tolist()
        modules = []
        for i in range(nlayer):
            modules.append(nn.Linear(steps_up[i], steps_up[i + 1]))
            modules.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*modules)

    def forward(self, x):
        xenc = self.enc(x)
        xrec = self.dec(xenc)
        return xenc, xrec


class PCYC(nn.Module):
    def __init__(self, num_clss, args):
        super(PCYC, self).__init__()

        # Dimension of embedding
        self.dim_enc = args.dim_enc

        # Number of classes
        self.num_clss = num_clss
        # Sketch model: pre-trained on ImageNet
        self.sketch_model = VGGNetFeats(pretrained=True, finetune=False)
        # self.load_weight(self.sketch_model, args.path_sketch_model, 'sketch')
        # Image model: pre-trained on ImageNet
        self.image_model = VGGNetFeats(pretrained=True, finetune=False)

        # Generators
        # Sketch to semantic generator
        self.gen_sk2se = Generator(in_dim=512, out_dim=self.dim_enc, noise=False, use_dropout=True)
        # Image to semantic generator
        self.gen_im2se = Generator(in_dim=512, out_dim=self.dim_enc, noise=False, use_dropout=True)
        # Semantic to sketch generator
        self.gen_se2sk = Generator(in_dim=self.dim_enc, out_dim=512, noise=False, use_dropout=True)
        # Semantic to image generator
        self.gen_se2im = Generator(in_dim=self.dim_enc, out_dim=512, noise=False, use_dropout=True)
        # Discriminators
        # Common semantic discriminator
        self.disc_se = Discriminator(in_dim=self.dim_enc, noise=True, use_batchnorm=True)
        # Sketch discriminator
        self.disc_sk = Discriminator(in_dim=512, noise=True, use_batchnorm=True)
        # Image discriminator
        self.disc_im = Discriminator(in_dim=512, noise=True, use_batchnorm=True)
        # Classifiers
        self.classifier_sk = nn.Linear(512, self.num_clss, bias=False)
        self.classifier_im = nn.Linear(512, self.num_clss, bias=False)
        self.classifier_se = nn.Linear(self.dim_enc, self.num_clss, bias=False)
        for param in self.classifier_sk.parameters():
            param.requires_grad = False
        for param in self.classifier_im.parameters():
            param.requires_grad = False
        for param in self.classifier_se.parameters():
            param.requires_grad = False
        # print('Done')

        # Optimizers
        # print('Defining optimizers...', end='')
        self.lr = args.lr
        self.gamma = 0.1
        self.momentum = 0.9
        self.optimizer_gen = optim.Adam(list(self.gen_sk2se.parameters()) + list(self.gen_im2se.parameters()) +
                                        list(self.gen_se2sk.parameters()) + list(self.gen_se2im.parameters()),
                                        lr=self.lr)
        self.optimizer_disc = optim.SGD(list(self.disc_se.parameters()) + list(self.disc_sk.parameters()) +
                                        list(self.disc_im.parameters()), lr=self.lr, momentum=self.momentum)
        self.scheduler_gen = optim.lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=[],
                                                            gamma=self.gamma)
        self.scheduler_disc = optim.lr_scheduler.MultiStepLR(self.optimizer_disc, milestones=[],
                                                             gamma=self.gamma)
        # print('Done')

        # Loss function
        # print('Defining losses...', end='')
        self.lambda_se = args.lambda_se
        self.lambda_im = args.lambda_im
        self.lambda_sk = args.lambda_sk
        self.lambda_gen_cyc = args.lambda_gen_cyc
        self.lambda_gen_adv = args.lambda_gen_adv
        self.lambda_gen_cls = args.lambda_gen_cls
        self.lambda_gen_reg = args.lambda_gen_reg
        self.lambda_disc_se = args.lambda_disc_se
        self.lambda_disc_sk = args.lambda_disc_sk
        self.lambda_disc_im = args.lambda_disc_im
        self.lambda_regular = args.lambda_regular
        self.criterion_gan = _GAN_Loss(use_lsgan=True)
        self.criterion_cyc = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

    def forward(self, sk, im):
        # vgg feats
        if sk is not None:
            self.sk_fe = self.sketch_model(sk)
            self.sk2se_em = self.gen_sk2se(self.sk_fe)

        if im is not None:
            self.im_fe = self.image_model(im)
            self.im2se_em = self.gen_im2se(self.im_fe)  # Y -> S

        if sk is None:
            return self.im2se_em

        if im is None:
            return self.sk2se_em

        # Reconstruct original examples for cycle consistency
        self.im_em_hat = self.gen_se2im(self.im2se_em) # Y -> S -> Y'
        self.sk_em_hat = self.gen_se2sk(self.sk2se_em)

    def backward(self, cl):
        # ranking loss
        loss_gen_adv = self.criterion_gan(self.disc_se(self.im2se_em), True) + \
            self.criterion_gan(self.disc_se(self.sk2se_em), True)
        loss_gen_adv = self.lambda_gen_adv * loss_gen_adv

        # Cycle consistency loss
        loss_gen_cyc = self.lambda_im * self.criterion_cyc(self.im_em_hat, self.im_fe) + \
            self.lambda_sk * self.criterion_cyc(self.sk_em_hat, self.sk_fe)
        loss_gen_cyc = self.lambda_gen_cyc * loss_gen_cyc

        # Classification loss
        loss_gen_cls = self.lambda_se * (self.criterion_cls(self.classifier_se(self.im2se_em), cl) +
                                         self.criterion_cls(self.classifier_se(self.sk2se_em), cl))
        loss_gen_cls = self.lambda_gen_cls * loss_gen_cls

        # Regression loss
        # self.criterion_reg = nn.MSELoss()
        loss_gen_reg = self.lambda_se * (self.criterion_reg(self.im2se_em, self.sk2se_em) +
                                         self.criterion_reg(self.sk2se_em, self.im2se_em))
        loss_gen_reg = self.lambda_gen_reg * loss_gen_reg

        # Sum the above generator losses for back propagation and displaying
        loss_gen = loss_gen_cyc + loss_gen_cls + loss_gen_reg

        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        self.optimizer_gen.step()

        # initialize optimizer
        self.optimizer_disc.zero_grad()

        # Semantic discriminator loss


        # Sketch discriminator loss
        loss_disc_sk = self.criterion_gan(self.disc_sk(self.sk_fe), True)
        loss_disc_sk = self.lambda_disc_sk * loss_disc_sk
        loss_disc_sk.backward(retain_graph=True)

        # Image discriminator loss
        loss_disc_im = self.criterion_gan(self.disc_im(self.im_fe), True)
        loss_disc_im = self.lambda_disc_im * loss_disc_im
        loss_disc_im.backward()

        # Optimizer step
        self.optimizer_disc.step()

        # Sum the above discriminator losses for displaying
        loss_disc = loss_disc_sk + loss_disc_im

        loss = {'gen_adv': loss_gen_adv, 'gen_cyc': loss_gen_cyc, 'gen_cls': loss_gen_cls,
                'gen_reg': loss_gen_reg, 'gen': loss_gen, 'disc_sk': loss_disc_sk, 'disc_im':
                    loss_disc_im, 'disc': loss_disc}

        return loss_gen_adv, loss_gen, loss_disc

    def optimize_params(self, sk, im, cl):

        # Forward pass
        self.forward(sk, im)

        # Backward pass
        loss = self.backward(cl)

        return loss

    def get_sketch_embeddings(self, sk):

        # sketch embedding
        sk_em = self.gen_sk2se(self.sketch_model(sk))

        return sk_em

    def get_image_embeddings(self, im):

        # image embedding
        im_em = self.gen_im2se(self.image_model(im))

        return im_em
