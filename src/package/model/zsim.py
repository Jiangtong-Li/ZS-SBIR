import torch
import torch.nn as nn

from package.model.vgg import vgg16, vgg16_bn
from package.model.attention import CNN_attention
from package.model.gcn import GCN_ZSIH
from package.loss.regularization import _Regularization

class ZSIM(nn.Module):
    def __init__(self, hidden_size, hashing_bit, semantices_size, adj_scaler=0.1, dropout=0.5, from_pretrain=True, fix_cnn=True):
        super(ZSIM, self).__init__()
        # hyper-param
        self.hidden_size = hidden_size
        self.hashing_bit = hashing_bit
        self.adj_scaler = adj_scaler
        self.dropout = dropout
        self.from_pretrain = from_pretrain
        self.fix_cnn = fix_cnn

        # model
        self.backbone = vgg16(pretrained=from_pretrain, return_type=2)
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
        self.doubly_sn = Doubly_SN()
        self.mean_linear = nn.Linear(hashing_bit, semantices_size)
        self.var_linear = nn.Linear(hashing_bit, semantices_size)

        # activation function
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.l2 = nn.MSELoss(reduction='mean')
        self.cross_entropy = nn.KLDivLoss(reduction='mean')

    
    def forward(self, sketch, image, semantics):
        """
        sketch [batch_size, channel_size, H, W]
        image [batch_size, channel_size, H, W]
        semantices [batch_size, hidden_size]
        """
        # recode the batch information
        batch_size = sketch.shape[0]
        semantics_size = semantics.shape[1]

        # encode the sketch
        feat_sketch = self.backbone(sketch) # [bs, 512, 7, 7]
        att_sketch = self.sketch_attention(feat_sketch) # [bs, 512]
        fc_sketch = self.relu(self.sketch_linear1(att_sketch)) # [bs, hs]
        enc_sketch = self.sigmoid(self.sketch_linear2(fc_sketch)) # [bs, hb]

        # encode the image
        feat_image = self.backbone(image) # [bs, 512, 7, 7]
        att_image = self.sketch_attention(feat_image) # [bs, 512]
        fc_image = self.relu(self.image_linear1(att_image)) # [bs, hs]
        enc_image = self.sigmoid(self.image_linear2(fc_image)) # [bs, hb]

        # kronecker product
        fusion = self.kronecker(att_sketch, att_sketch) # [bs, 512*512]

        # gcn semantics representation
        gcn_out = self.gcn(fusion, semantics) #[bs, hb]

        # VAE sampling
        eps = torch.rand([batch_size, self.hashing_bit]).to(sketch.device)
        codes = self.doubly_sn(gcn_out, eps)
        dec_mean = self.mean_linear(codes)
        dec_var = self.mean_linear(codes)
        
        # calculate loss
        loss = self.loss(enc_sketch, enc_image, gcn_out, codes, dec_mean, semantics)
    
    def kronecker(self, feat1, feat2):
        batch_size = feat1.shape[0]
        feat1 = torch.unsqueeze(feat1, 2)
        feat2 = torch.unsqueeze(feat2, 1)
        fusion = torch.bmm(feat1, feat2).reshape(batch_size, -1)
        return fusion
    
    def loss(self, enc_sketch, enc_image, codes_logits, codes, dec_mean, semantics):
        p_xz = self.l2(dec_mean, semantics)
        q_zx = self.cross_entropy(codes_logits, codes)

        loss_image = self.l2(enc_image, codes)
        loss_sketch = self.l2(enc_sketch, codes)
        loss = dict()
        loss['p_xz'] = p_xz
        loss['q_zx'] = q_zx
        loss['image_l2'] = loss_image
        loss['sketch_l2'] = loss_sketch
        return loss
    
    def hash(self, figure):
        batch_size = figure.shape[0]
        feat_figure = self.backbone(figure) # [bs, 512, 7, 7]
        att_figure = self.sketch_attention(feat_figure) # [bs, 512]
        fc_figure = self.relu(self.sketch_linear1(att_figure)) # [bs, hs]
        enc_figure = self.sigmoid(self.sketch_linear2(fc_figure)) # [bs, hb]

        yout = (torch.sign(enc_figure - 0.5) + 1.0) / 2.0

class Doubly_SN(nn.Module):
    def __init__(self):
        super(Doubly_SN, self).__init__()
    
    def forward(self, logits, epsilon):
        yout = (torch.sign(logits - epsilon) + 1.0) / 2.0
        return yout
    
    def backward(self, logits, epsilon, dprev):
        dlogits = logits * (1 - logits) * dprev
        depsilon = dprev
        return dlogits, depsilon