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
from package.dataset.utils import *
from package.dataset.data_san import RandomShift, npfn

try:
    import gensim
except:
    pass


join = os.path.join
SK = 0
IM = 1
IM_SIZE = IMAGE_SIZE


PATH_SEMANTIC_SKETCHY = '../datasets/vecs/semantic_sketchy.pkl'
SKETCH_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010'
IMAGE_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/SBIR_datasets/sketchy/EXTEND_image_sketchy'
NPY_FOLDER_SKETCHY = '../datasets/npy_sketchy'
PATH_NAMES = '../datasets/sketchy_clsname_what_idx_2_imfilename.pkl'

if False:
    PATH_SEMANTIC_SKETCHY = r'G:\f\SJTUstudy\labNL\SBIR_datasets\vecs\semantic_sketchy.pkl'
    NPY_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/SBIR_datasets/sketchy/npy_sketchy'
    PATH_NAMES = r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\clsname_what_idx_2_imfilename.pkl'



PATH_SEMANTIC_MODEL = '../datasets/vecs/GoogleNews-vectors-negative300.bin'
PATH_SEMANTIC_TUBERLIN = '../datasets/vecs/semantic_tuberlin.pkl'
SKETCH_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/ImageResized'
IMAGE_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/png'
NPY_FOLDER_TUBERLIN = ''



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


class CMT_dataloader(torchDataset):
    """
    ATTENTION: access to the dataset via the same index can result in different elements
    """
    def __init__(self, path_semantic, folder_sk=None, folder_im=None, clss=None, normalize01=False, doaug=True,
                 folder_nps=None, mode=IM, logger=None, sz=None, names=None, paired=False):
        """
        Attirbute:
            mode: IM/SK. Indicating it is image or sketch to be retrieved. Default: IM.
        :param folder_sk: sketch folder
        :param folder_im: image folder
        :param clss: classes to load
        :param normalize01: whether normalize data to 0-1
        :param doaug: whether do data augmentation
        :param folder_nps: the folder saves npy files. This allow fewer inodes to save the datasets(the server
                    does not allow too many inodes allocated). The folder should contain
                            classname1_sk.npy, classname1_im.npy,
                            classname2_sk.npy, classname1_im.npy,
                            ...
                    1. If folder_nps is None, folder_sk and folder_imsk must be provided.
                    2. If folder_nps is not None but no files exist in the folder, folder_sk and folder_im must be
                        provided, and such files would be created in folder_nps.
                    3. If folder_nps is not None and files exist in the folder, load the files instead of those
                        in folder_sk and folder_imsk for training.
        :param path_semantic: path of the semantic vector(xxx.pkl). It should be a dict: {class_name1: [b1, b2, ...],
                    class_name2: [b1, b2, ...]}
        :param logger: logger to debug.
        :param sz: resize or not.
        :param names: the clsname_what_idx_2_imfilename.pkl file. Or a dict.
                clsname_what_idx_2_imfilename[class_name][im/st/sk][index] = image filename without postfix.
                Neccessary if data are paired.
        :param paired: paired data or not
        """
        super(CMT_dataloader, self).__init__()
        self.idx2skim_pair = []

        # self.idx2cls_item_sk[SK][i] = [class_index, item_index], where 0 <= i <= self.lens[SK]
        # self.idx2cls_item_sk[IM][i] = [class_index, item_index], where 0 <= i <= self.lens[IM]
        self.idx2cls_items = [[], []]
        self.normalize01 = normalize01
        self.doaug = doaug
        self._build_trans()
        self.cls2idx = {}
        self.idx2cls = []
        self.semantics = []
        self.lens = [0, 0]
        self.logger = logger
        self.clss = clss
        self.paired = paired
        if isinstance(names, str):
            names = pickle.load(open(names, 'rb'))
        if paired and names is None:
            self._print("{} is None. But paired data is required".format(names))
            raise Exception("{} is None. But paired data is required".format(names))
        folders = [folder_sk, folder_im]
        semantics = pickle.load(open(path_semantic, 'rb'))
        if folder_nps and not os.path.exists(folder_nps):
            os.mkdir(folder_nps)
        for name in clss:
            self.semantics.append(semantics[name])
            if all([os.path.exists(str(fd)) for fd in folders]):
                sks_folder = join(folders[SK], name)
                ims_folder = join(folders[IM], name)

            if folder_nps and os.path.exists(join(folder_nps, npfn(name + '_sk'))) \
                    and os.path.exists(join(folder_nps, npfn(name + '_im'))):
                data_of_name = [np.load(join(folder_nps, npfn(name + '_sk'))),
                                  np.load(join(folder_nps, npfn(name + '_im')))]

            else:
                data_of_name = self._get_data_from_ims(sks_folder=sks_folder,
                                                       ims_folder=ims_folder)
            data_of_name = self._process(data_of_name, sz)
            self._try_save_ims(folder_nps=folder_nps, name=name, data_of_name=data_of_name)
            if paired:
                data_of_name[SK], data_of_name[IM] = self._get_paired_image(
                    im_fns=names[name]['im'], sk_fns=names[name]['sk'], sk_arr=data_of_name[SK], im_arr=data_of_name[IM])
            for i in range(2):
                self.lens[i] += len(data_of_name[i])
            self.idx2skim_pair.append(data_of_name)
            self.idx2cls_items[SK] += [(len(self.idx2cls), i) for i in range(len(data_of_name[SK]))]
            self.idx2cls_items[IM] += [(len(self.idx2cls), i) for i in range(len(data_of_name[IM]))]
            self.cls2idx[name] = len(self.idx2cls)
            self.idx2cls.append(name)

        self.semantics = np.asarray(self.semantics)
        self.idx2cls_items[SK] = np.asarray(self.idx2cls_items[SK], dtype=np.uint32)
        self.idx2cls_items[IM] = np.asarray(self.idx2cls_items[IM], dtype=np.uint32)
        self._print('Dataset loaded from folder_sk:{},folder_im:{}, folder_nps:{}, sk_len:{}, im_len:{}'.format(
            folder_sk, folder_im, folder_nps, len(self.idx2cls_items[SK]), len(self.idx2cls_items[IM])))
        self.mode = mode

    def _print(self, debug):
        logger = self.logger
        print(debug) if logger is None else logger.info(debug)

    def d(self):
        return len(self.semantics[0])

    def _get_paired_image(self, im_fns, sk_fns, sk_arr, im_arr):
        im_fn2idx = {}
        for i, fn in enumerate(im_fns):
            im_fn2idx[fn] = i
        ims = []
        bad_sk_i = []
        for i in range(len(sk_arr)):
            fn = sk_fns[i].split('-')[0]
            try:
                ims.append(im_arr[im_fn2idx[fn]])
            except:
                bad_sk_i.append(i)
        for i in bad_sk_i:
            sk_arr = np.delete(sk_arr, i, 0)
        im_arr = np.asarray(ims)
        assert len(im_arr) == len(sk_arr), 'Length of image array and sketch array should be identical. ' \
                                           'But get {} and {}'.format(len(im_arr), len(sk_arr))
        return sk_arr, im_arr

    def _get_data_from_ims(self, sks_folder, ims_folder):
        paths = [path.split('.')[0] for path in sorted(os.listdir(sks_folder))]
        data_of_name = [[self._prep_img(join(sks_folder, path)) for path in os.listdir(sks_folder)
                   if path.endswith('.jpg') or path.endswith('.png')],
                  [self._prep_img(join(ims_folder, path)) for path in sorted(os.listdir(ims_folder))
                   if ((path.endswith('.jpg') or path.endswith('.png')) and path.split('.')[0] in paths)]]
        return data_of_name

    def _process(self, data_of_name, sz):
        sz = sz if sz is not None else IM_SIZE
        for i in range(2):
            data_of_name[i] = [cv2.resize(img, (sz, sz)) \
                             if img.shape != (sz, sz, 3) else img for img in data_of_name[i]]
            data_of_name[i] = np.asarray(data_of_name[i], dtype=np.uint8)
        return data_of_name

    def _try_save_ims(self, folder_nps, name, data_of_name):
        if folder_nps:
            if not os.path.exists(join(folder_nps, npfn(name + '_sk'))):
                np.save(join(folder_nps, npfn(name + '_sk')), data_of_name[SK])
            if not os.path.exists(join(folder_nps, npfn(name + '_im'))):
                np.save(join(folder_nps, npfn(name + '_im')), data_of_name[IM])

    def _build_trans(self):
        self.trans_im = transforms.Compose([
            RandomShift(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) if self.doaug else \
                transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
        self.trans_sk = transforms.Compose([
            RandomShift(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor(),
            ]) if self.doaug else \
                transforms.Compose([transforms.ToTensor(),
                                    ])
        self.trans = [0,0]
        self.trans[SK] = self.trans_sk
        self.trans[IM] = self.trans_im

    def _prep_img(self, path):
        # print(path)
        img = cv2.imread(path)
        img = img.copy()[:,:,::-1]
        if img.shape != (IM_SIZE,IM_SIZE,3):
            img = cv2.resize(img, (IM_SIZE, IM_SIZE))
        if self.normalize01:
            img = img / 255.0
        return img

    def __getitem__(self, idx):
        """
        :param index: index of the data
        :return: a tensor list [sketch/image, semantics]. Whether the returned object is sketch or image is decided by
            self.mode
        """
        cls, idx = self.idx2cls_items[self.mode][idx]
        return self.trans[self.mode](self.idx2skim_pair[cls][self.mode][idx]), self.semantics[cls]

    def traverse(self, what, batch_size=16, skip=1):
        """
        :param what: SK or IM
        :param batch_size: batch size of the traversing
        :param skip: skip >= 2 allows to skip some images/sketches to reduce computation. (Used for debugging)
        :return: yield a 2-element list [sketch/image, id]. Whether the first returned object is sketch or image is decided by
            self.mode
        """
        it = 0
        assert what == IM or what == SK, "DSH_dataloader.traverse: what must be IM({})/SK({}), but get {}"\
            .format(IM, SK, what)
        rets_ims = []; rets_ids = []
        for id, xs in enumerate(self.idx2skim_pair):
            for i, x in enumerate(xs[what]):
                it += 1
                if it % skip == 0:
                    rets_ims.append(self.trans[what](x))
                    rets_ids.append(id)
                    if len(rets_ims) == batch_size:
                        yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)
                        rets_ims = []; rets_ids = []

        # Avoid single element returned fot batch normalization's convenience
        if len(rets_ids) >= 1:
            yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)

    def __len__(self):
        return len(self.idx2cls_items[self.mode])


def _create_im2(folder_im, folder_imsk, folder_im2):
    if not os.path.exists(folder_im2):
        os.mkdir(folder_im2)
    for cls in os.listdir(folder_im):
        ims = []
        fils_imsk = set([f.split('.')[0] for f in os.listdir(join(folder_imsk, cls))])
        for name in os.listdir(join(folder_im, cls)):
            if name.split('.')[0] not in fils_imsk:
                continue
            img = cv2.imread(join(folder_im, cls, name))
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            ims.append(img)
        print(cls, len(fils_imsk), len(ims))
        np.save(join(folder_im2, cls + npfn('_im2')), np.asarray(ims, dtype=np.uint8))


def gen_vec_sketchy():
    path_vec = '../datasets/vecs/GoogleNews-vectors-negative300.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_vec, binary=True, unicode_errors='....')
    semantics = {}
    for idx, item in enumerate(TEST_CLASS_SKETCHY + TRAIN_CLASS_SKETCHY):
        semantics[item] = word2vec[item] if item in word2vec else word2vec[SEMANTICS_REPLACE[item]]
    print(len(semantics), len(semantics[item]))
    pickle.dump(semantics, '../datasets/vecs/sketchy.pkl')


def _test_traverse():
    ds = CMT_dataloader(clss=TRAIN_CLASS_SKETCHY[:2], doaug=False, folder_nps=NPY_FOLDER_SKETCHY,
                        path_semantic=PATH_SEMANTIC_SKETCHY)
    for data, id in ds.traverse(IM):
        data = data[0].numpy()
        img = np.transpose(data, (1, 2, 0))
        print(id[0])
        cv2.imshow("show", img)
        cv2.waitKeyEx()
        # print("data.shape=", data.shape)

def _test():
    ds = CMT_dataloader(clss=TRAIN_CLASS_SKETCHY[:2], doaug=False, folder_nps=NPY_FOLDER_SKETCHY,
                        path_semantic=PATH_SEMANTIC_SKETCHY)
    ds.mode = IM
    for i, (item, semantic) in enumerate(ds):
        print(i, item.shape, semantic.shape)
    ds.mode = SK
    for i, (item, semantic) in enumerate(ds):
        print(i, item.shape, semantic.shape)


if __name__=="__main__":
    _test_traverse()
    pass
    # create_im2(r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\EXTEND_image_sketchy', r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\sketch_tokens', r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\im2')


