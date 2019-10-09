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
from package.dataset.data_san import RandomShift, npfn



join = os.path.join
SK = 0
IMSK = 1
IMSKAGE_SIZE = IMAGE_SIZE


SKETCH_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010'
IMSKAGE_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/SBIR_datasets/sketchy/sketch_tokens'
NPY_FOLDER_SKETCHY = '../datasets/npy_sketchy'
# NPY_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/pkl_sketchy'


SKETCH_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/ImageResized'
IMSKAGE_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/png'


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


class D3Shape_dataloader(torchDataset):
    """
    ATTENTION: access to the dataset via the same index can result in different elements
    """
    def __init__(self, folder_sk, folder_imsk, clss, normalize01=False, doaug=True, exp3ch=True, folder_nps=None, dis2sim=10):
        """
        :param folder_sk: sketch folder
        :param folder_imsk: image's sketch token folder.
            ATTENTION: this folder contains sketch tokens of the corresponding images, not images!
        :param clss: classes to load
        :param normalize01: whether normalize data to 0-1
        :param doaug: whether do data augmentation
        :param exp3ch: whether force the sketches to expand to 3 channels
        :param folder_nps: the folder saves npy files. This allow fewer inodes to save the datasets(the server
                    does not allow too many inodes allocated). The folder should contain
                            classname1_sk.npy, classname1_imsk.npy,
                            classname2_sk.npy, classname2_imsk.npy,
                            ...
                    1. If folder_nps is None, folder_sk and folder_imsk must be provided.
                    2. If folder_nps is not None but no files exist in the folder, folder_sk and folder_im must be
                        provided, and such files would be created in folder_nps.
                    3. If folder_nps is not None and files exist in the folder, load the files instead of those
                        in folder_sk and folder_imsk for training.
        :param dis2sim: The ratio of dissimilar pairs to similar pairs.
        """
        super(D3Shape_dataloader, self).__init__()
        self.idx2skim_pair = []
        self.normalize01 = normalize01
        self.dis2sim = dis2sim
        self.doaug = doaug
        self.exp3ch = exp3ch
        self._build_trans()
        self.cls2idx = {}
        self.idx2cls = []
        self.lens = [0, 0]
        if folder_nps and not os.path.exists(folder_nps):
            os.mkdir(folder_nps)
        for name in clss:
            if os.path.exists(folder_sk) and os.path.exists(folder_imsk):
                sks_folder = join(folder_sk, name)
                imsks_folder = join(folder_imsk, name)
            if folder_nps and os.path.exists(join(folder_nps, npfn(name + '_imsk'))):
                to_app = [np.load(join(folder_nps, npfn(name + '_sk'))), np.load(join(folder_nps, npfn(name + '_imsk')))]
                # print(to_app[SK].shape, to_app[IMSK].shape)
            else:
                to_app = [[self._prep_img(join(sks_folder, path)) for path in os.listdir(sks_folder)
                                if path.endswith('.jpg') or path.endswith('.png')],
                     [self._prep_img(join(imsks_folder, path)) for path in os.listdir(imsks_folder)
                               if path.endswith('.jpg') or path.endswith('.png')]]
                to_app[SK] = np.asarray(to_app[SK], dtype=np.uint8)
                to_app[IMSK] = np.asarray(to_app[IMSK], dtype=np.uint8)
            if folder_nps and not os.path.exists(join(folder_nps, npfn(name + '_imsk'))):
                # np.save(join(folder_nps, npfn(name + '_sk')), to_app[SK])
                np.save(join(folder_nps, npfn(name + '_imsk')), to_app[IMSK])
            to_app[SK] = [Image.fromarray(img) for img in to_app[SK]]
            to_app[IMSK] = [Image.fromarray(img) for img in to_app[IMSK]]
            self.idx2skim_pair.append(to_app)
            self.cls2idx[name] = len(self.idx2cls)
            self.idx2cls.append(name)
            self.lens[SK] += len(to_app[SK])
            self.lens[IMSK] += len(to_app[IMSK])
        print('Dataset loaded from folder_sk:{}, folder_imsk:{}, folder_nps:{}, sk_len:{}, imsk_len:{}'.format(
            folder_sk, folder_imsk, folder_nps, self.lens[SK], self.lens[IMSK]
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
        if img.shape != (IMSKAGE_SIZE,IMSKAGE_SIZE,3):
            img = cv2.resize(img, (IMSKAGE_SIZE,IMSKAGE_SIZE))
        if self.normalize01:
            img = img / 255.0
        # img = Image.fromarray(img)
        return img

    def _get_sk_imsk_pair(self, cls_idx):
        sk_idx = np.random.randint(0, len(self.idx2skim_pair[cls_idx][SK]))
        imsk_idx = np.random.randint(0, len(self.idx2skim_pair[cls_idx][IMSK]))
        return [self.trans(self.idx2skim_pair[cls_idx][SK][sk_idx]),
                           self.trans(self.idx2skim_pair[cls_idx][IMSK][imsk_idx])]

    def __getitem__(self, _):
        """
        :param index: index of the data
        :return: a tensor list [sketch1, imsk1, sketch2, imsk2, is_same]
        """
        is_same = not (np.random.rand() * self.dis2sim > 1)
        i1 = np.random.randint(0, len(self.idx2skim_pair))
        i2 = i1
        if not is_same:
            while i2 == i1:
                i2 = np.random.randint(0, len(self.idx2skim_pair))
        pairs1 = self._get_sk_imsk_pair(i1)
        pairs2 = self._get_sk_imsk_pair(i2)
        return pairs1 + pairs2 + [torch.Tensor([is_same])]

    def traverse(self, what, batch_size=16, skip=1):
        """
        :param what: SK or IMSK
        :param batch_size: batch size of the traversing
        :param skip: skip >= 2 allows to skip some images/sketches to reduce computation. (Used for debugging)
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
    sands_train = D3Shape_dataloader(SKETCH_FOLDER_SKETCHY, IMSKAGE_FOLDER_SKETCHY, TRAIN_CLASS_SKETCHY + TEST_CLASS_SKETCHY, normalize01=False,
                   doaug=False, exp3ch=True, folder_nps=NPY_FOLDER_SKETCHY)


if __name__=="__main__":
    _test()


