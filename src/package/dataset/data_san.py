import os
import csv
import time
import random
import pickle
import torchvision
from torchvision import transforms

import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset
from torchvision.transforms import Normalize
from PIL import Image
from package.dataset.utils import *
import package.dataset.tv_functional as F



join = os.path.join
SK = 0
IM = 1

# SaN requires the input image/sketch to be 25x25
IMAGE_SIZE = 225

SKETCH_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010'
IMAGE_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/EXTEND_image_sketchy'


SKETCH_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/ImageResized'
IMAGE_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/png'


try:
    TEST_CLASS_SKETCHY = list(TEST_CLASS_SKETCHY)
    TRAIN_CLASS_SKETCHY = list(TRAIN_CLASS_SKETCHY)
except:
    # default
    TEST_CLASS_SKETCHY = list(TEST_CLASS)
    TRAIN_CLASS_SKETCHY = list(TRAIN_CLASS)

try:
    TEST_CLASS_TUBERLIN = list(TEST_CLASS_TUBERLIN)
    TRAIN_CLASS_TUBERLIN = list(TRAIN_CLASS_SKETCHY)
except:
    TEST_CLASS_TUBERLIN = list()
    TRAIN_CLASS_TUBERLIN = list()


class RandomShift(object):
    # randomly move the sketch
    def __init__(self, maxpix=32):
        self.maxpix = maxpix
        self.fill = (255,255,255)
        self.padding_mode = 'constant'

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        l = int(random.random() * self.maxpix)
        t = int(random.random() * self.maxpix)
        size = img.size
        img = F.pad(img, (l, t, self.maxpix - l, self.maxpix - t), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, size)
        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(maxpix={0})'.format(self.maxpix)


class SaN_dataloader(torchDataset):
    """
    ATTENTION: access to the dataset via the same index can result in different elements
    """
    def __init__(self, folder_sk, folder_im, clss, normalize01=False, doaug=True, exp3ch=True):
        super(SaN_dataloader, self).__init__()
        self.idx2skim_pair = []
        self.normalize01 = normalize01
        self.doaug = doaug
        self.exp3ch = exp3ch
        self._build_trans()
        self.cls2idx = {}
        self.idx2cls = []
        self.lens = [0, 0]
        for name in clss:
            sks_folder = join(folder_sk, name) if folder_sk is not None else None
            ims_folder = join(folder_im, name) if folder_im is not None else None
            self.idx2skim_pair.append(
                [[self._prep_img(join(sks_folder, path)) for path in os.listdir(sks_folder)
                            if path.endswith('.jpg') or path.endswith('.png')],
                 [self._prep_img(join(ims_folder, path)) for path in os.listdir(ims_folder)
                           if path.endswith('.jpg') or path.endswith('.png')]])
            self.cls2idx[name] = len(self.idx2cls)
            self.idx2cls.append(name)
            self.lens[SK] += len(self.idx2skim_pair[-1][SK])
            self.lens[SK] += len(self.idx2skim_pair[-1][SK])
        self.clss = clss
        self.sks_folder = sks_folder
        self.ims_folder = ims_folder
        # print("len(self.idx2skim_pair)=", len(self.idx2skim_pair))
            
    def _build_trans(self):
        self.trans = transforms.Compose([
            RandomShift(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor()
            ]) if self.doaug else \
                transforms.Compose([transforms.ToTensor()])

    def _prep_img(self, path):
        # print(path)
        img = cv2.imread(path)
        if not self.exp3ch and path.endswith('.png'):
            img = img[:, :, 0]
        else:
            if img.shape[2] == 1:
                img = np.concatenate([img, img, img], 2)
            else:
                img = img.copy()[:,:,::-1]
        if img.shape != (IMAGE_SIZE,IMAGE_SIZE,3):
            img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        if self.normalize01:
            img = img / 255.0
        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        """
        :param index: index of the data
        :return: a tensor list [sketch, positive_image, negative_image, positive_class_id]
        """
        rets = []
        index %= len(self.idx2skim_pair)
        sk_idx = np.random.randint(0, len(self.idx2skim_pair[index][SK]))
        rets.append(self.trans(self.idx2skim_pair[index][SK][sk_idx]))
        im_idx = np.random.randint(0, len(self.idx2skim_pair[index][IM]))
        rets.append(self.trans(self.idx2skim_pair[index][IM][im_idx]))

        neg_idx = index
        while neg_idx == index: neg_idx = np.random.randint(0, len(self.idx2skim_pair))
        im_neg_idx = np.random.randint(0, len(self.idx2skim_pair[neg_idx][IM]))
        rets.append(self.trans(self.idx2skim_pair[neg_idx][IM][im_neg_idx]))

        rets.append(index)
        return rets

    def traverse(self, what, batch_size=16, skip=1):
        """
        :param what: SK or IM
        :param skip: skip >= 2 allows to skip some images/sketches to reduce computation
        :return: yield a two-element list [image/sketches, id]
        """
        it = 0
        rets_ims = []
        rets_ids = []
        for id, xs in enumerate(self.idx2skim_pair):
            for x in xs[what]:
                it += 1
                if it % skip == 0:
                    rets_ims.append(self.trans(x))
                    rets_ids.append(id)
                    if len(rets_ims) == batch_size:
                        yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)

    def __len__(self):
        return max(self.lens)



def _test():
    pass


if __name__=="__main__":
    _test()