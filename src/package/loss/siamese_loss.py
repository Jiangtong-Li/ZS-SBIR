import torch
import torch.nn as nn

class _Siamese_loss(nn.Module):
    def __init__(self):
        super(_Siamese_loss, self).__init__()

    def forward(self, model1, model2, label, margin, loss_type=0, distance_type=0):
        """
        loss_type - [0, 1] - corresponding to "A Zero-Shot Framework for Sketch Based Image Retrieval."'s [1,2]
        distance_type - [0, 1] - corresponding to l2 and cosine
        """
        batch_size = model1.shape[0]
        hidden_size = model1.shape[1]
        assert model1.shape == model2.shape
        assert label.shape[0] == batch_size

        if distance_type == 0:
            distance = torch.sqrt(torch.sum(torch.pow(model1-model2, 2), dim=1)/hidden_size)
        elif distance_type == 1:
            distance = torch.sum(torch.mul(model1, model2), dim=1)/hidden_size
        else:
            raise ValueError('The distance type should be 0 or 1')
        
        if loss_type == 0:
            zero = torch.Tensor([0]).to(margin.device)
            similarity = torch.mul(label, torch.pow(distance, 2))
            dissimilarity = torch.mul((1-label), torch.pow(torch.max(margin-distance, zero), 2))
        elif loss_type == 1:
            Q = margin
            alpha = 2 / Q
            beta = 2 * Q
            gamma = -2.77 / Q
            similarity = torch.mul(label, alpha*(torch.pow(distance, 2)))
            dissimilarity = torch.mul((1-label), beta*torch.exp(gamma*distance))
        else:
            raise ValueError('The loss type should be 0 or 1')

        loss = torch.sum(similarity+dissimilarity) / (batch_size*2)
        return loss, torch.sum(similarity) / batch_size, torch.sum(dissimilarity) / batch_size
