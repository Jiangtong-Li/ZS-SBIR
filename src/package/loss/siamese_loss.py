import torch
import torch.nn as nn
import torch.functional as F

class _Siamese_loss(nn.Module):
    def __init__(self):
        super(_Siamese_loss, self).__init__()
        self.eduli = nn.PairwiseDistance()
        self.cosine = nn.CosineSimilarity()

    def forward(self, model1, model2, label, margin, loss_type=0, distance_type=0):
        """
        loss_type - [0, 1] - corresponding to "A Zero-Shot Framework for Sketch Based Image Retrieval."'s [1,2]
        distance_type - [0, 1] - corresponding to l2 and cosine
        """
        batch_size = model1.shape[0]
        assert model1.shape == model2.shape
        assert label.shape[0] == batch_size

        if distance_type == 0:
            distance = self.eduli(model1, model2)
        elif distance_type == 1:
            distance = self.cosine(model1, model2)
        else:
            raise ValueError('The distance type should be 0 or 1')
        distance = distance.reshape(distance.shape[0], 1)

        if loss_type == 0:
            similarity = torch.mul(label, torch.pow(distance, 2))
            dissimilarity = torch.mul((1-label), torch.pow(torch.clamp(margin-distance,min=0.0), 2))
        elif loss_type == 1:
            Q = margin
            alpha = 2 / Q
            beta = 2 * Q
            gamma = -2.77 / Q
            similarity = torch.mul(label, (alpha*(torch.pow(distance, 2))))
            dissimilarity = torch.mul((1-label), beta*torch.exp(gamma*distance))
        else:
            raise ValueError('The loss type should be 0 or 1')

        assert similarity.shape == (batch_size, 1)
        assert dissimilarity.shape == (batch_size, 1)

        loss = torch.mean(similarity+dissimilarity)
        return loss, torch.mean(similarity), torch.mean(dissimilarity)
