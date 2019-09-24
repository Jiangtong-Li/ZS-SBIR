import torch
from torchvision import transforms
import torchvision.datasets as dsets
import numpy as np
import random
import cv2

class Dataset(object):

    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size


def create_pairs(data, digit_indices):
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        # make n pairs with each number
        for i in range(n):
            # make pairs of the same class
            # label is 1
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0 = cv2.resize(data[z1], (224,224))
            x1 = cv2.resize(data[z2], (224,224))
            x0_data.append(np.stack((x0,x0,x0),0) / 255.0)
            x1_data.append(np.stack((x1,x1,x1),0) / 255.0)
            label.append(1)

            # make pairs of different classes
            # since the minimum value is 1, it is not the same class
            # label is 0
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0 = cv2.resize(data[z1], (224,224))
            x1 = cv2.resize(data[z2], (224,224))
            x0_data.append(np.stack((x0,x0,x0),0) / 255.0)
            x1_data.append(np.stack((x1,x1,x1),0) / 255.0)
            label.append(0)

    x0_data = np.array(x0_data, dtype=np.float32)
    x0_data = x0_data.reshape([-1, 3, 224, 224])
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 3, 224, 224])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_iterator(data, label, batchsize, shuffle=False):
    digit_indices = [np.where(label == i)[0] for i in range(10)]
    x0, x1, label = create_pairs(data, digit_indices)
    ret = Dataset(x0, x1, label)
    return ret

def load_train():
    train = dsets.MNIST(
                root='../data/',
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                download=True
            )
    train_iter = create_iterator(
        train.train_data.numpy(),
        train.train_labels.numpy(),
        32)
    return train_iter