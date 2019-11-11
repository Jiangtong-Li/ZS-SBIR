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
        loss = torch.mean(0.5*torch.pow(dist_p, 2) + 0.5*torch.pow(torch.clamp(-dist_n+self.margin, min=0.0), 2))
        return loss

    
class _Ranking_loss(nn.Module):
    def __init__(self, dist, margin1, margin2):
        super(_Ranking_loss, self).__init__()
        self.dist = dist
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, sketch, image_paired, image_unpaired, image_n):
        dist_paired = self.dist(image_paired, sketch)
        dist_unpaired = self.dist(image_unpaired, sketch)
        dist_n = self.dist(image_n, sketch)
        loss = torch.mean(torch.pow(dist_paired, 2) + 
                          torch.pow(torch.clamp(dist_paired - dist_unpaired + self.margin1, min=0.0), 2) +
                          torch.pow(torch.clamp(dist_unpaired - dist_n + self.margin2, min=0.0), 2))/3
        return loss