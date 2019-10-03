import time

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from package.model.vgg import vgg16, vgg16_bn
from package.model.attention import CNN_attention, attention
from package.model.gcn import GCN_ZSIH
from package.loss.regularization import _Regularization

class ZSIM(nn.Module):
    def __init__(self, hidden_size, hashing_bit, semantics_size, pretrain_embedding, adj_scaler=1, dropout=0.5, from_pretrain=True, fix_cnn=True, fix_embedding=True, logger=None):
        super(ZSIM, self).__init__()
        # hyper-param
        self.hidden_size = hidden_size
        self.hashing_bit = hashing_bit
        self.adj_scaler = adj_scaler
        self.dropout = dropout
        self.from_pretrain = from_pretrain
        self.fix_cnn = fix_cnn
        self.logger = logger

        # model
        self.semantics = nn.Embedding.from_pretrained(pretrain_embedding)
        if fix_embedding:
            self.semantics.weight.requires_grad=False
        self.backbone = vgg16(pretrained=from_pretrain, return_type=1)
        if fix_cnn:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.sketch_attention = CNN_attention(512)
        self.sketch_linear1 = nn.Linear(512, hidden_size)
        self.sketch_linear2 = nn.Linear(hidden_size, hashing_bit)
        self.image_attention = CNN_attention(512)
        self.image_linear1 = nn.Linear(512, hidden_size)
        self.image_linear2 = nn.Linear(hidden_size, hashing_bit)
        self.gcn = GCN_ZSIH(512*512, hidden_size, hashing_bit, dropout, adj_scaler)
        self.doubly_sn = Doubly_SN_Function.apply
        self.mean_linear = nn.Linear(hashing_bit, semantics_size)
        self.var_linear = nn.Linear(hashing_bit, semantics_size)

        # activation function
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.l2 = L2()
        self.bce = CE()
    
    def forward(self, sketch, image, semantics):
        """
        sketch [batch_size, channel_size, H, W]
        image [batch_size, channel_size, H, W]
        semantices [batch_size, 1]
        """
        # get embedding
        semantics = self.semantics(semantics)
        semantics = semantics.squeeze()

        # recode the batch information
        batch_size = sketch.shape[0]
        _semantics_size = semantics.shape[1]

        # encode the sketch
        feat_sketch = self.backbone(sketch) # [bs, 512, 7, 7]
        att_sketch = self.sketch_attention(feat_sketch) # [bs, 512]
        fc_sketch = self.relu(self.sketch_linear1(att_sketch)) # [bs, hs]
        enc_sketch = self.sigmoid(self.sketch_linear2(fc_sketch)) # [bs, hb]

        # encode the image
        feat_image = self.backbone(image) # [bs, 512, 7, 7]
        att_image = self.image_attention(feat_image) # [bs, 512]
        fc_image = self.relu(self.image_linear1(att_image)) # [bs, hs]
        enc_image = self.sigmoid(self.image_linear2(fc_image)) # [bs, hb]

        # kronecker product
        fusion = self.kronecker(att_sketch, att_image) # [bs, 512*512]

        # gcn semantics representation
        gcn_out_1 = self.gcn(fusion, semantics) # [bs, hb]
        gcn_out = self.sigmoid(gcn_out_1)

        # VAE sampling
        eps_one = torch.ones([batch_size, self.hashing_bit]).to(sketch.device) * 0.5
        eps_rand = torch.rand([batch_size, self.hashing_bit]).to(sketch.device)
        codes_one = self.doubly_sn(gcn_out, eps_one) # [bs, hb]
        codes_rand = self.doubly_sn(gcn_out, eps_rand)
        codes = codes_rand
        dec_mean = self.mean_linear(codes) # [bs, semantics_size]
        _dec_var = self.var_linear(codes) # [bs, semantics_size]
        
        # calculate loss
        loss = self.loss(enc_sketch, enc_image, gcn_out, codes, dec_mean, semantics)
        return loss

    def kronecker(self, feat1, feat2):
        batch_size = feat1.shape[0]
        feat1 = torch.unsqueeze(feat1, 2)
        feat2 = torch.unsqueeze(feat2, 1)
        fusion = torch.bmm(feat1, feat2).reshape(batch_size, -1)
        return fusion
    
    def loss(self, enc_sketch, enc_image, codes_logits, codes, dec_mean, semantics):
        p_xz = self.l2(dec_mean, semantics)
        no_grad_code = codes.detach()
        q_zx = self.bce(codes_logits, no_grad_code)

        loss_image = self.l2(enc_image, no_grad_code)
        loss_sketch = self.l2(enc_sketch, no_grad_code)
        loss = dict()
        loss['p_xz'] = (p_xz, 0.1)
        loss['q_zx'] = (q_zx, 1.0)
        loss['image_l2'] = (loss_image, 1.0)
        loss['sketch_l2'] = (loss_sketch, 1.0)
        return loss
    
    def hash(self, figure, label):
        _batch_size = figure.shape[0]
        feat_figure = self.backbone(figure) # [bs, 512, 7, 7]
        if label == 0:
            att_figure = self.sketch_attention(feat_figure) # [bs, 512]
            fc_figure = self.relu(self.sketch_linear1(att_figure)) # [bs, hs]
            enc_figure = self.sigmoid(self.sketch_linear2(fc_figure)) # [bs, hb]
        else:
            att_figure = self.image_attention(feat_figure) # [bs, 512]
            fc_figure = self.relu(self.image_linear1(att_figure)) # [bs, hs]
            enc_figure = self.sigmoid(self.image_linear2(fc_figure)) # [bs, hb]

        yout = (torch.sign(enc_figure - 0.5) + 1.0) / 2.0
        return yout

class Doubly_SN_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, epsilon):
        ctx.save_for_backward(logits)
        yout = (torch.sign(logits - epsilon) + 1.0) / 2.0
        return yout
    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        grad_input = logits * (1 - logits) * grad_output
        return grad_input, grad_input
    
class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, mat1, mat2):
        return torch.mean(torch.pow(mat1-mat2, 2))

class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
    
    def forward(self, mat1, mat2):
        return -torch.mean(mat2*torch.log(mat1+1e-10)+(1-mat2)*torch.log((1-mat1+1e-10)))