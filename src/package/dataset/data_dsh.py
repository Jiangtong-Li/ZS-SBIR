import os
import csv
import time
import random
import pickle
import torchvision
from torchvision import transforms
from sklearn.decomposition import PCA
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
IMSK = 1
IM = 2
IM_SIZE = 227
SK_SIZE = 200


PATH_SEMANTIC_MODEL = '../datasets/vecs/GoogleNews-vectors-negative300.bin'
PATH_SEMANTIC_SKETCHY = '../datasets/vecs/semantic_sketchy.pkl'
PATH_SEMANTIC_TUBERLIN = '../datasets/vecs/semantic_tuberlin.pkl'


SKETCH_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/ZS_SBIR/256x256/sketch/tx_000000000010'
IMSKAGE_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/SBIR_datasets/sketchy/sketch_tokens'
IMAGE_FOLDER_SKETCHY = 'G:/f/SJTUstudy/labNL/SBIR_datasets/sketchy/EXTEND_image_sketchy'
NPY_FOLDER_SKETCHY = '../datasets/npy_sketchy'


SKETCH_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/ImageResized'
IMSKAGE_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/png'
IMAGE_FOLDER_TUBERLIN = 'G:/f/SJTUstudy/labNL/SBIR_datasets/tuberlin/png'
NPY_FOLDER_TUBERLIN = ''


if True:
    PATH_SEMANTIC_SKETCHY = r'G:\f\SJTUstudy\labNL\SBIR_datasets\vecs\semantic_sketchy.pkl'
    NPY_FOLDER_SKETCHY = r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\npy_sketchy'


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


class DSH_dataloader(torchDataset):
    """
    ATTENTION: access to the dataset via the same index can result in different elements
    """
    def __init__(self, folder_saving, path_semantic, folder_sk=None, folder_im=None, folder_imsk=None, clss=None, normalize01=False, doaug=True,
                 folder_nps=None, m=300, logger=None):
        """
        Attirbute:
            BS, ns * m
            BI, ni * m
            D, d * m
            vec_bs, ns * d
            vec_bi, ni * d
            Ws ~ ni * ns, implemented with memory mapping.
        :param folder_sk: sketch folder
        :param folder_im: image folder
        :param folder_imsk: image's sketch token folder.
            ATTENTION: this folder contains sketch tokens of the corresponding images, not images!
        :param clss: classes to load
        :param normalize01: whether normalize data to 0-1
        :param doaug: whether do data augmentation
        :param folder_nps: the folder saves npy files. This allow fewer inodes to save the datasets(the server
                    does not allow too many inodes allocated). The folder should contain
                            classname1_sk.npy, classname1_imsk.npy, classname1_im.npy,
                            classname2_sk.npy, classname2_imsk.npy, classname1_im.npy,
                            ...
                    1. If folder_nps is None, folder_sk and folder_imsk must be provided.
                    2. If folder_nps is not None but no files exist in the folder, folder_sk and folder_im must be
                        provided, and such files would be created in folder_nps.
                    3. If folder_nps is not None and files exist in the folder, load the files instead of those
                        in folder_sk and folder_imsk for training.
        :param path_semantic: path of the semantic vector(xxx.pkl). It should be a dict: {class_name1: [b1, b2, ...],
                    class_name2: [b1, b2, ...]}
        :param m: number of binary bits
        :param folder_saving: folder to save/load binary codes
        :param logger: logger to debug.
        """
        super(DSH_dataloader, self).__init__()
        self.idx2skim_pair = []
        self.logger = logger
        self.normalize01 = normalize01
        self.doaug = doaug
        self._build_trans()
        self.cls2idx = {}
        self.idx2cls = []
        self.semantics = []
        self.lens = [0, 0, 0]
        self.folder_saving = folder_saving
        self.clss = clss
        self.m = m
        self.vec_bi = []
        self.vec_bs = []
        self.label_all_i = []
        self.label_all_s = []
        folders = [folder_sk, folder_imsk, folder_im]
        if not os.path.exists(folder_saving):
            os.mkdir(folder_saving)
        semantics = pickle.load(open(path_semantic, 'rb'))
        if folder_nps and not os.path.exists(folder_nps):
            os.mkdir(folder_nps)
        for name in clss:
            self.semantics.append(semantics[name])
            if all([os.path.exists(str(fd)) for fd in folders]):
                sks_folder = join(folders[SK], name)
                imsks_folder = join(folders[IMSK], name)
                ims_folder = join(folders[IM], name)
            # print(folder_nps, name, join(folder_nps, npfn(name + '_imsk')), os.path.exists(join(folder_nps, npfn(name + '_imsk'))))
            if folder_nps and os.path.exists(join(folder_nps, npfn(name + '_imsk'))):
                data_of_name = [np.load(join(folder_nps, npfn(name + '_sk'))),
                          np.load(join(folder_nps, npfn(name + '_imsk'))),
                                  np.load(join(folder_nps, npfn(name + '_im2')))]
                # print(data_of_name[SK].shape, data_of_name[IMSK].shape, data_of_name[IM].shape)
            else:
                data_of_name = self._get_data_from_ims(sks_folder=sks_folder, imsks_folder=imsks_folder,
                                                       ims_folder=ims_folder)
            data_of_name = self._process(data_of_name)
            self._try_save_ims(folder_nps=folder_nps, name=name, data_of_name=data_of_name)

            for i in range(3):
                self.lens[i] += len(data_of_name[i])
            self.vec_bi += [semantics[name] for _ in range(len(data_of_name[IM]))]
            self.vec_bs += [semantics[name] for _ in range(len(data_of_name[SK]))]
            self.idx2skim_pair.append(data_of_name)
            self.cls2idx[name] = len(self.idx2cls)
            self.idx2cls.append(name)
            self.label_all_i.append(np.zeros(len(data_of_name[IM])) + self.cls2idx[name])
            self.label_all_s.append(np.zeros(len(data_of_name[SK])) + self.cls2idx[name])

        self.semantics = np.asarray(self.semantics)

        self._print('Dataset loaded from folder_sk:{}, folder_imsk:{}, folder_im:{}, folder_nps:{}, sk_len:{},\
            imsk_len:{}, im_len:{}'.format(
            folder_sk, folder_imsk, folder_im, folder_nps, self.lens[SK], self.lens[IMSK], self.lens[IM]))
        self.vec_bs = np.asarray(self.vec_bs)
        self.vec_bi = np.asarray(self.vec_bi)
        self.label_all_i = np.hstack(self.label_all_i)
        self.label_all_s = np.hstack(self.label_all_s)
        self._init_W(label_all_i=self.label_all_i, label_all_s=self.label_all_s)
        self._init_B()
        self._init_D()
        self._print('Dataset init done.')
        # print("len(self.idx2skim_pair)=", len(self.idx2skim_pair))

    def _init_W(self, label_all_i, label_all_s):
        file = join(self.folder_saving, 'W_tmp.npmm')
        if os.path.exists(file):
            self.W = np.memmap(file, dtype=np.float16, shape=(len(label_all_i), len(label_all_s)), mode='r')
            return None
        W = np.memmap(file, dtype=np.float16, shape=(len(label_all_i), len(label_all_s)), mode='w+')
        for i in range(len(label_all_i)):
            W[i] = [(s == label_all_i[i]) * 2 - 1 for s in label_all_s]
        self.W = W

    def _print(self, s):
        print(s) if self.logger is None else self.logger.info(s)

    def _init_D(self):
        d_file = join(self.folder_saving, npfn('d'))
        if os.path.exists(d_file):
            self._print("Init D matrix from {}".format(d_file))
            self.D = np.load(d_file)
        else:
            self._print("Reinit D matrix. It is OK since D can be inferred from BI and BS.")
            self.D = np.random.rand(self.d(), self.m)

    def d(self):
        return len(self.semantics[0])

    def _get_data_from_ims(self, sks_folder, ims_folder, imsks_folder):
        paths = [path.split('.')[0] for path in sorted(os.listdir(sks_folder))]
        paths = [fn.split('.')[0] for fn in sorted(os.listdir(imsks_folder)) if fn.split('.')[0] in paths]
        data_of_name = [[self._prep_img(join(sks_folder, path)) for path in os.listdir(sks_folder)
                   if path.endswith('.jpg') or path.endswith('.png')],
                  [self._prep_img(join(imsks_folder, path)) for path in sorted(os.listdir(imsks_folder))
                   if ((path.endswith('.jpg') or path.endswith('.png')) and path.split('.')[0] in paths)],
                  [self._prep_img(join(ims_folder, path)) for path in sorted(os.listdir(ims_folder))
                   if ((path.endswith('.jpg') or path.endswith('.png')) and path.split('.')[0] in paths)]]
        return data_of_name

    def _init_B(self):
        folder_saving = self.folder_saving
        m = self.m
        folder_bi = join(folder_saving, npfn('bi'))
        folder_bs = join(folder_saving, npfn('bs'))
        if os.path.exists(folder_bi) and os.path.exists(folder_bs):
            self._print("Init BI matrix from {} and BS matrix from {}.".format(folder_bi, folder_bs))
            self.BI = np.load(join(folder_saving, npfn('bi')))
            self.BS = np.load(join(folder_saving, npfn('bs')))
        else:
            self._print("Reinit BI and BS matrix!")
            self.BI = np.random.randint(0, 2, [self.lens[IM], m]) * 2 - 1
            self.BS = np.random.randint(0, 2, [self.lens[SK], m]) * 2 - 1

    def _process(self, data_of_name):
        for i in range(3):
            # In the DSH configuration, input of C2-Net is a single-channel image
            # while input of C1-Net is a three-channel image
            if i == IM:
                data_of_name[i] = [cv2.resize(img, (IM_SIZE, IM_SIZE)) \
                                 if img.shape != (IM_SIZE, IM_SIZE, 3) else img for img in data_of_name[i]]
            else:
                data_of_name[i] = [cv2.resize(img, (SK_SIZE, SK_SIZE))[:, :, 0] \
                                 if img.shape != (SK_SIZE, SK_SIZE, 3) else img[:, :, 0] for img in data_of_name[i]]
            data_of_name[i] = np.asarray(data_of_name[i], dtype=np.uint8)
        return data_of_name

    def _try_save_ims(self, folder_nps, name, data_of_name):
        if folder_nps:
            if not os.path.exists(join(folder_nps, npfn(name + '_imsk'))):
                np.save(join(folder_nps, npfn(name + '_imsk')), data_of_name[IMSK])
            if not os.path.exists(join(folder_nps, npfn(name + '_sk'))):
                np.save(join(folder_nps, npfn(name + '_sk')), data_of_name[SK])
            if not os.path.exists(join(folder_nps, npfn(name + '_im2'))):
                np.save(join(folder_nps, npfn(name + '_im2')), data_of_name[IM])
        assert len(data_of_name[IM]) == len(data_of_name[IMSK]), 'Sketch token and images must satisfy one-to-one \
                correspondence. (Error while disposing class {})'.format(name)

    def _build_trans(self):
        self.trans_im = transforms.Compose([
            RandomShift(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) if self.doaug else \
                transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.trans_sk = transforms.Compose([
            RandomShift(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor()
            ]) if self.doaug else \
                transforms.Compose([transforms.ToTensor()])
        self.trans = [0,0,0]
        self.trans[IM] = self.trans_im
        self.trans[SK] = self.trans_sk
        self.trans[IMSK] = self.trans_sk

    def save_params(self):
        np.save(join(self.folder_saving, npfn('bi')), self.BI)
        np.save(join(self.folder_saving, npfn('bs')), self.BS)
        np.save(join(self.folder_saving, npfn('d')), self.D)

    def __del__(self):
        # super(DSH_dataloader, self).__del__()
        self.save_params()

    def _prep_img(self, path):
        # print(path)
        img = cv2.imread(path)
        img = img.copy()[:,:,::-1]
        if img.shape != (IM_SIZE,IM_SIZE,3):
            img = cv2.resize(img, (IM_SIZE, IM_SIZE))
        if self.normalize01:
            img = img / 255.0
        return img

    def __getitem__(self, cls_idx):
        """
        :param index: index of the data
        :return: a tensor list [sketch, code_of_sketch, image, sketch_token, code_of_image]
        """
        cls_idx %= len(self.idx2skim_pair)
        # print(cls_idx, SK, IM, len(self.idx2skim_pair[cls_idx][SK]), len(self.idx2skim_pair[cls_idx][IM]))
        sk_idx = np.random.randint(0, len(self.idx2skim_pair[cls_idx][SK]))
        im_idx = np.random.randint(0, len(self.idx2skim_pair[cls_idx][IM]))
        return [self.trans_sk(self.idx2skim_pair[cls_idx][SK][sk_idx]), self.BS[sk_idx].astype(np.float32),
                self.trans_im(self.idx2skim_pair[cls_idx][IM][im_idx]),
                self.trans_sk(self.idx2skim_pair[cls_idx][IMSK][im_idx]), self.BI[im_idx].astype(np.float32)]

    def traverse(self, what, batch_size=16, skip=1):
        """
        :param what: SK or IM
        :param batch_size: batch size of the traversing
        :param skip: skip >= 2 allows to skip some images/sketches to reduce computation. (Used for debugging)
        :return: yield a four-element list [sketches, sketch_tokens, images, id]
        """
        it = 0
        assert what == IM or what == SK, "DSH_dataloader.traverse: what must be IM({})/SK({}), but get {}"\
            .format(IM, SK, what)
        rets_ims = []; rets_ids = []; rets_imsks = []
        for id, xs in enumerate(self.idx2skim_pair):
            for i, x in enumerate(xs[what]):
                it += 1
                if it % skip == 0:
                    rets_ims.append(self.trans[what](x))
                    if what == IM:
                        rets_imsks.append(self.trans_sk(xs[IMSK][i]))
                    rets_ids.append(id)
                    if len(rets_ims) == batch_size:
                        if what == IM:
                            yield None, torch.stack(rets_imsks, dim=0), torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)
                        else:
                            yield torch.stack(rets_ims, dim=0), None, None, torch.tensor(rets_ids)
                        rets_ims = []; rets_ids = []; rets_imsks = []

        # Avoid single element returned fot batch normalization's convenience
        if len(rets_ids) >= 1:
            if what == IM:
                yield None, torch.stack(rets_imsks, dim=0), torch.stack(rets_ims, dim=0), torch.tensor(rets_ids)
            else:
                yield torch.stack(rets_ims, dim=0), None, None, torch.tensor(rets_ids)

    def __len__(self):
        return max(self.lens)


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


def _test():
    ds = DSH_dataloader(clss=TRAIN_CLASS_SKETCHY[:2], doaug=False, folder_nps=NPY_FOLDER_SKETCHY,
                        path_semantic=PATH_SEMANTIC_SKETCHY, folder_saving='test_dsh_ds')
    sketch, code_of_sketch, image, sketch_token, code_of_image = ds[4]
    print(sketch.shape, code_of_sketch.shape, image.shape, sketch_token.shape, code_of_image.shape)


def _test_W():
    label_all_s = np.array([0,0,1,2,2])
    label_all_i = np.array([0,1,1,2])
    W = np.asarray([[((i == s) * 2 - 1) for s in label_all_s] for i in label_all_i], dtype=np.int16)
    print('A1=\n', label_all_i, '\n', 'A2=\n',label_all_s, '\n', 'B=\n',W)
    eqs = (label_all_s == label_all_i)


if __name__=="__main__":
    _test_W()
    pass
    # create_im2(r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\EXTEND_image_sketchy', r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\sketch_tokens', r'G:\f\SJTUstudy\labNL\SBIR_datasets\sketchy\im2')


