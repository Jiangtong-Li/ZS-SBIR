import torch
import torch.nn as nn
import torch.functional as F

class _Triplet_loss(nn.Module):
    def __init__(self, dist, margin):
        super(_Triplet_loss, self).__init__()
        self.dist = dist
        self.margin = margin

    def forward(self, image_p, image_n, sketch):
        dist_p = self.dist(image_p, sketch)
        dist_n = self.dist(image_n, sketch)
        loss = torch.mean(torch.clamp(dist_p-dist_n+self.margin, min=0.0))
        return loss

