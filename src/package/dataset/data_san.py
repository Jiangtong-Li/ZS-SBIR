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
PKL_FOLDER_SKETCHY = '../datasets/npy_sketchy'
PKL_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/pkl_sketchy'


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


def npfn(fn):
    if not fn.endswith('.npy'):
        fn += '.npy'
    return fn


class SaN_dataloader(torchDataset):
    """
    ATTENTION: access to the dataset via the same index can result in different elements
    """
    def __init__(self, folder_sk, folder_im, clss, normalize01=False, doaug=True, exp3ch=True, folder_nps=None):
        """
        :param folder_sk: sketch folder
        :param folder_im: image folder
        :param clss: classes to load
        :param normalize01: whether normalize data to 0-1
        :param doaug: whether do data augmentation
        :param exp3ch: whether force the sketches to expand to 3 channels
        :param folder_nps: the folder that saves npy files. This allow fewer inodes to save the datasets(the server
                    does not allow too many inodes allocated). The folder should contain
                            classname1_sk.npy, classname1_im.npy,
                            classname2_sk.npy, classname2_im.npy,
                            ...
                    1. If folder_nps is None, folder_sk and folder_im must be provided.
                    2. If folder_nps is not None but no files exist in the folder, folder_sk and folder_im must be
                        provided, and such files would be created in folder_nps.
                    3. If folder_nps is not None and files exist in the folder, load the files instead of those
                        in folder_sk and folder_im for training.
        """
        super(SaN_dataloader, self).__init__()
        self.idx2skim_pair = []
        self.normalize01 = normalize01
        self.doaug = doaug
        self.exp3ch = exp3ch
        self._build_trans()
        self.cls2idx = {}
        self.idx2cls = []
        self.lens = [0, 0]
        if folder_nps and not os.path.exists(folder_nps):
            os.mkdir(folder_nps)
        for name in clss:
            if os.path.exists(folder_sk) and os.path.exists(folder_im):
                sks_folder = join(folder_sk, name)
                ims_folder = join(folder_im, name)
            if folder_nps and os.path.exists(join(folder_nps, npfn(name + '_sk'))):
                to_app = [np.load(join(folder_nps, npfn(name + '_sk'))), np.load(join(folder_nps, npfn(name + '_im')))]
                # print(to_app[SK].shape, to_app[IM].shape)
            else:
                to_app = [[self._prep_img(join(sks_folder, path)) for path in os.listdir(sks_folder)
                                if path.endswith('.jpg') or path.endswith('.png')],
                     [self._prep_img(join(ims_folder, path)) for path in os.listdir(ims_folder)
                               if path.endswith('.jpg') or path.endswith('.png')]]
                to_app[SK] = np.asarray(to_app[SK], dtype=np.uint8)
                to_app[IM] = np.asarray(to_app[IM], dtype=np.uint8)
            if folder_nps and not os.path.exists(join(folder_nps, npfn(name + '_sk'))):
                np.save(join(folder_nps, npfn(name + '_sk')), to_app[SK])
                np.save(join(folder_nps, npfn(name + '_im')), to_app[IM])
            to_app[SK] = [Image.fromarray(img) for img in to_app[SK]]
            to_app[IM] = [Image.fromarray(img) for img in to_app[IM]]
            self.idx2skim_pair.append(to_app)
            self.cls2idx[name] = len(self.idx2cls)
            self.idx2cls.append(name)
            self.lens[SK] += len(to_app[SK])
            self.lens[IM] += len(to_app[IM])
        print('Dataset loaded from folder_sk:{}, folder_im:{}, folder_nps:{}, sk_len:{}, im_len:{}'.format(
            folder_sk, folder_im, folder_nps, self.lens[SK], self.lens[IM]
        ))
        self.clss = clss
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
        # img = Image.fromarray(img)
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
                        rets_ims = []
                        rets_ids = []
        if len(rets_ims) != 0:
            yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)

    def __len__(self):
        return max(self.lens)


def _test():
    print('train')
    sands_train = SaN_dataloader(SKETCH_FOLDER_SKETCHY, IMAGE_FOLDER_SKETCHY, TRAIN_CLASS_SKETCHY, normalize01=False,
                   doaug=False, exp3ch=True, folder_nps=PKL_FOLDER_SKETCHY)
    print("test")
    sands_test = SaN_dataloader(SKETCH_FOLDER_SKETCHY, IMAGE_FOLDER_SKETCHY, TEST_CLASS_SKETCHY, normalize01=False,
                   doaug=False, exp3ch=True, folder_nps=PKL_FOLDER_SKETCHY)
    # for ims, ids in sands.traverse(what=SK, skip=50):
    #     print(ims, ids)



if __name__=="__main__":
    _test()


"""
Dataset loaded from folder_sk:G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010, folder_im:G:/f/SJTUstudy/labNL/ZS_SBIR/EXTEND_image_sketchy, folder_nps:G:/f/SJTUstudy/labNL/ZS_SBIR/pkl_sketchy, sk_len:62787, im_len:62549
test

Dataset loaded from folder_sk:G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010, folder_im:G:/f/SJTUstudy/labNL/ZS_SBIR/EXTEND_image_sketchy, folder_nps:G:/f/SJTUstudy/labNL/ZS_SBIR/pkl_sketchy, sk_len:12694, im_len:10453
"""