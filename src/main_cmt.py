import os
import random

import numpy as np
from scipy.spatial.distance import cdist
import cv2
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
# from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import cdist
from package.model.cmt import CMT
from package.loss.cmt_loss import _CMT_loss
from package.dataset.data_cmt import *
from package.args.cmt_args import parse_config
from package.dataset.utils import make_logger
from package.model.utils import *
from package.loss.regularization import _Regularization

import numpy as np
from sklearn.neighbors import NearestNeighbors as NN


DEBUG = False


def dr_dec(optimizer, args):
    args.lr *= 0.5
    args.lr = max(args.lr, 5e-5)
    optimizer.param_groups[0]['lr'] = args.lr


def _get_pre_from_matches(matches):
    """
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
    :return: precision
    """
    return np.mean(matches)


def _map_change(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if idx != 0:
            # dup cannot be bool type
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)


def _get_map_from_matches(matches):
    """
    mAP's calculation refers to https://github.com/ShivaKrishnaM/ZS-SBIR/blob/master/trainCVAE_pre.py.
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
            matches[i][j] == 1 indicates the j-th retrieved test image j belongs to the same class as test sketch i,
            otherwise, matches[i][j] = 0.
    :return: mAP
    """
    temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    mAP_term = 1.0 / (np.stack(temp, axis=0) + 1.0)
    precisions = np.multiply(_map_change(matches), mAP_term)
    mAP = np.mean(precisions, axis=1)
    return np.mean(mAP)


def _eval(feats_labels_sk, feats_labels_im, n=200):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n, mAP@all
    """
    nn = NN(n_neighbors=feats_labels_im[0].shape[0], metric='cosine', algorithm='brute').fit(feats_labels_im[0])
    _, indices = nn.kneighbors(feats_labels_sk[0])
    retrieved_classes = np.array(feats_labels_im[1])[indices]
    matches = np.vstack([(retrieved_classes[i] == feats_labels_sk[1][i])
                       for i in range(retrieved_classes.shape[0])]).astype(np.uint16)
    return _get_pre_from_matches(matches[:, :n]), _get_map_from_matches(matches[:, :n])


def _test_and_save(epochs, optimizer, data_test, model, logger, args, loss_sum):
    if not hasattr(_test_and_save, 'best_acc'):
        _test_and_save.best_acc = 0
    n = 200
    start_cpu_t = time.time()
    feats_labels_sk, feats_labels_im = _extract_feats_sk_im(data=data_test, model=model,
                                                              batch_size=args.batch_size)
    pre, mAPn = _eval(feats_labels_sk, feats_labels_im, n)
    logger.info("Precision@{}: {}, mAP@{}: {}, bestPrecsion: {}".format(n, pre, n, mAPn, max(pre, _test_and_save.best_acc)) +
                "  " + 'epochs: {},  loss_sk: {},  loss_im: {},  (eval cpu time: {}s)'.
                format(epochs, np.mean(loss_sum[SK]), np.mean(loss_sum[IM]), time.time() - start_cpu_t))
    if pre > _test_and_save.best_acc:
        _test_and_save.best_acc = pre
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epochs': epochs,
                    'args': args},
                   save_fn(args.save_dir, epochs, pre, mAPn))
    torch.cuda.empty_cache()


def save_fn(save_dir, it, pre=0, mAP=0):
    return join(mkdir(join(save_dir, 'models')), 'Iter__{}__{}_{}.pkl'.format(it, int(pre * 1000), int(mAP * 1000)))


def _try_load(args, logger, model, optimizer):
    if args.start_from is None:
        # try to find the latest checkpoint
        files = os.listdir(mkdir(join(mkdir(args.save_dir), 'models')))
        if len(files) == 0:
            logger.info("Cannot find any checkpoint. Start new training.")
            return 0
        latest = max(files, key=lambda name: int(name.split('\\')[-1].split('/')[-1].split('.')[0].split('__')[1]))
        checkpoint = join(args.save_dir, 'models', latest)
    else:
        try: checkpoint = save_fn(args.save_dir, str(int(args.start_from)))
        except: checkpoint = args.start_from
    logger.info("Load model from {}".format(checkpoint))
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epochs']


def _extract_feats_sk_im(data, model, batch_size=64):
    skip = 1
    model.eval()
    feats_labels_sk = _extract_feats(data, lambda x: model(sk=x), SK, skip=skip,
                                     batch_size=batch_size)
    feats_labels_im = _extract_feats(data, lambda x: model(im=x), IM, skip=skip,
                                     batch_size=batch_size)
    model.train()
    return feats_labels_sk, feats_labels_im


def _extract_feats(data_test, model, what, skip=1, batch_size=16):
    """
    :param data_test: test Dataset
    :param model: network model
    :param what: SK or IM
    :param skip: skip a certain number of image/sketches to reduce computation
    :return: a two-element list [extracted_features, extracted_labels]
    """
    labels = []
    feats = []
    for batch_idx, (xs, id) in \
            enumerate(data_test.traverse(what, skip=skip, batch_size=batch_size)):
        feats.append(model(xs.cuda()).data.cpu().numpy())
        # print(type(labels[0]), labels[0].shape)#     <class 'numpy.ndarray'> (16, 256)
        # print(type(id), id) # <class 'torch.Tensor'> tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels.append(id.numpy())
        # print(feats[-1][-1][:4])
    return np.concatenate(feats), np.concatenate(labels)


def _parse_args_paths(args):
    if args.dataset == 'sketchy':
        sketch_folder = SKETCH_FOLDER_SKETCHY
        im_folder = IMAGE_FOLDER_SKETCHY
        path_semantic = PATH_SEMANTIC_SKETCHY
        train_class = TRAIN_CLASS_SKETCHY
        test_class = TEST_CLASS_SKETCHY
        npy_folder = NPY_FOLDER_SKETCHY
    elif args.dataset == 'tuberlin':
        sketch_folder = SKETCH_FOLDER_TUBERLIN
        im_folder = IMAGE_FOLDER_TUBERLIN
        path_semantic = PATH_SEMANTIC_TUBERLIN
        train_class = TRAIN_CLASS_TUBERLIN
        test_class = TEST_CLASS_TUBERLIN
        npy_folder = NPY_FOLDER_TUBERLIN
    else: raise Exception("dataset args error!")
    if args.sketch_dir != '': sketch_folder = args.sketch_dir
    if args.image_dir != '': im_folder = args.image_dir
    if args.path_semantic != '': im_folder = args.path_semantic
    if args.npy_dir == '0': args.npy_dir = npy_folder
    elif args.npy_dir == '': args.npy_dir = None
    if args.ni_path == '0': args.ni_path = PATH_NAMES
    return sketch_folder, im_folder, path_semantic, train_class, test_class


def train(args):
    # srun -p gpu --gres=gpu:1 --exclusive --output=san10.out python main_san.py --epochs 50000 --print_every 500 --save_every 2000 --batch_size 96 --dataset sketchy --margin 10 --npy_dir 0 --save_dir san_sketchy10
    # srun -p gpu --gres=gpu:1 --exclusive --output=san1.out python main_san.py --epochs 50000 --print_every 500 --save_every 2000 --batch_size 96 --dataset sketchy --margin 1 --npy_dir 0 --save_dir san_sketchy1

    # srun -p gpu --gres=gpu:1 --output=san_sketchy03.out python main_san.py --epochs 30000 --print_every 200 --save_every 3000 --batch_size 96 --dataset sketchy --margin 0.3 --npy_dir 0 --save_dir san_sketchy03 --lr 0.0001
    sketch_folder, image_folder, path_semantic, train_class, test_class = _parse_args_paths(args)

    if DEBUG:
        args.back_bone = 'default'
        args.npy_dir = NPY_FOLDER_SKETCHY
        args.ni_path = PATH_NAMES
        args.print_every = 1
        args.save_every = 5
        args.paired = True
        args.epochs = 20000
        # args.lr = 0.001
        args.sz = 32
        # args.l2_reg = 0.0001
        args.back_bone = 'default'
        args.batch_size = 32
        args.h = 500

        test_class = train_class[5:7]
        train_class = train_class[:5]
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    data_train = CMT_dataloader(folder_sk=sketch_folder, clss=train_class, folder_nps=args.npy_dir,
                                path_semantic=path_semantic, paired=args.paired, names=args.ni_path,
                                folder_im=image_folder, normalize01=False, doaug=False, logger=logger,
                                sz=None if args.back_bone=='vgg' else args.sz)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True)

    data_test = CMT_dataloader(folder_sk=sketch_folder, clss=test_class, folder_nps=args.npy_dir,
                               path_semantic=path_semantic, folder_im=image_folder, normalize01=False, doaug=False,
                               logger=logger, sz=None if args.back_bone=='vgg' else args.sz)

    model = CMT(d=data_train.d(), h=args.h, back_bone=args.back_bone, batch_normalization=args.bn, sz=args.sz)
    model.cuda()

    if not args.ft:
        model.fix_vgg()
    optimizer = SGD(params=model.parameters(), lr=args.lr, momentum=0.6)

    epochs = _try_load(args, logger, model, optimizer)
    logger.info(str(args))
    args.epochs += epochs
    cmt_loss = _CMT_loss()
    model.train()

    l2_regularization = _Regularization(model, args.l2_reg, p=2, logger=None)
    loss_sum = [[0], [0]]
    logger.info("Start training:\n train_classes: {}\n test_classes: {}".format(train_class, test_class))
    _test_and_save(epochs=epochs, optimizer=optimizer, data_test=data_test,
                   model=model, logger=logger, args=args, loss_sum=loss_sum)
    while True:
        for mode, get_feat in [[IM, lambda data: model(im=data)],
                               [SK, lambda data: model(sk=data)]]:
            data_train.mode = mode
            for _, (data, semantics) in enumerate(dataloader_train):

                # Skip one-element batch in consideration of batch normalization
                if data.shape[0] == 1:
                    continue
                # print(data.shape)
                optimizer.zero_grad()
                loss = cmt_loss(get_feat(data.cuda()),
                                semantics.cuda()) \
                        + l2_regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum[mode].append(float(loss.item()))
        epochs += 1
        dr_dec(optimizer=optimizer, args=args)
        if (epochs + 1) % args.save_every == 0:
            _test_and_save(epochs=epochs, optimizer=optimizer, data_test=data_test,
                           model=model, logger=logger, args=args, loss_sum=loss_sum)

        if (epochs + 1) % args.print_every == 0:
            logger.info('epochs: {},  loss_sk: {},  loss_im: {},'.
                        format(epochs, np.mean(loss_sum[SK]), np.mean(loss_sum[IM])))
            loss_sum = [[], []]

        if epochs >= args.epochs: break



def gen_args(h=500, dataset='sketchy', back_bone='vgg', sz=32, ft=True, paired=False):
    ft = int(ft)
    paired = int(paired)
    return \
"""
###

#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=cmt_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4

source activate lzxtc2
python main_cmt.py --npy_dir 0 --dataset {} --save_dir cmts/cmt{}{}_{}_{}_{}_{} --h {} --back_bone {} --sz {} --ft {} --paired {}  --ni_path 0

""".format(dataset, int(ft), int(paired) , dataset, h, back_bone, sz if back_bone=='default' else "", h, back_bone, sz, ft, paired)


if __name__ == '__main__':
    if False:
        print(gen_args(back_bone='vgg', ft=False, paired=True))
        print(gen_args(back_bone='vgg', ft=True, paired=False))
        print(gen_args(back_bone='vgg', ft=True, paired=True))
        print(gen_args(back_bone='vgg', ft=False, paired=False))
        print(gen_args(back_bone='default'))
        exit()
    args = parse_config()
    print(str(args))
    # train(args)


# srun --gres=gpu:1 --output=cmt_%j.out python main_cmt.py
'''
#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=cmt_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4

source activate lzxtc2

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt11_sketchy_500_default_32 --h 500 --back_bone default --sz 32 --ft 1 --paired 1 --ni_path 0

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt01_sketchy_500_vgg_ --h 500 --back_bone vgg --sz 32 --ft 0 --paired 1 --ni_path 0

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt10_sketchy_500_vgg_ --h 500 --back_bone vgg --sz 32 --ft 1 --paired 0 --ni_path 0

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt11_sketchy_500_vgg_ --h 500 --back_bone vgg --sz 32 --ft 1 --paired 1 --ni_path 0

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt00_sketchy_500_vgg_ --h 500 --back_bone vgg --sz 32 --ft 0 --paired 0 --ni_path 0

python main_cmt.py --npy_dir 0 --dataset sketchy --save_dir cmts/cmt10_sketchy_500_default_32 --h 500 --back_bone default --sz 32 --ft 1 --paired 0 --ni_path 0

'''