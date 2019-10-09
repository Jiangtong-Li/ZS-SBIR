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
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import cdist
from package.model.d3shape import D3Shape
from package.loss.d3shape_loss import _D3Shape_loss
from package.dataset.data_d3shape import *
from package.args.d3shape_args import parse_config
from package.dataset.utils import make_logger
from package.model.utils import *


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    return ckpt['steps']


def _extract_feats(data_test, model, what, skip=1, batch_size=16):
    """
    :param data_test: test Dataset
    :param model: network model
    :param what: SK or IMSK
    :param skip: skip a certain number of image/sketches to reduce computation
    :return: a two-element list [extracted_labels, extracted_features]
    """
    labels = []
    feats = []
    for batch_idx, (xs, id) in \
            enumerate(data_test.traverse(what, skip=skip, batch_size=batch_size)):
        labels.append(model(xs.cuda()).data.cpu().numpy())
        # print(type(labels[0]), labels[0].shape)#     <class 'numpy.ndarray'> (16, 256)
        # print(type(id), id) # <class 'torch.Tensor'> tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        feats.append(id.numpy())
    return np.concatenate(labels), np.concatenate(feats)


def _get_pre_from_matches(matches):
    return np.mean(matches)


def _get_map_from_matches(matches):
    s = 0
    for match in matches:
        count = 0
        for rank, ismatch in enumerate(match):
            s += ismatch * (count + 1) / (rank + 1)
            count += ismatch
    return s / matches.size


def _eval(feats_labels_sk, feats_labels_im, n):
    """
    Refer to https://blog.csdn.net/JNingWei/article/details/78955536 for mAP calculation
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n
    """
    # print("eval\n", len(feats_labels_sk[1]), len(feats_labels_im[1])) # 1099 581

    dists = cdist(feats_labels_sk[0], feats_labels_im[0], 'euclidean')
    # print("dists.shape=", dists.shape)  # (1099, 581)
    ranks = np.argsort(dists, 1) # (1099, 581)

    ranksn = ranks[:, :n] # (1099, 50)
    # print("dists ranks ranksn shape=", dists.shape, ranks.shape, ranksn.shape)

    classesn = np.array([[feats_labels_im[1][i] == feats_labels_sk[1][r] for i in ranksn[r]] for r in range(len(ranksn))])

    return _get_pre_from_matches(classesn), _get_map_from_matches(classesn)


def _parse_args_paths(args):
    if args.dataset == 'sketchy':
        sketch_folder = SKETCH_FOLDER_SKETCHY
        imsk_folder = IMSKAGE_FOLDER_SKETCHY
        train_class = TRAIN_CLASS_SKETCHY
        test_class = TEST_CLASS_SKETCHY
    elif args.dataset == 'tuberlin':
        sketch_folder = SKETCH_FOLDER_TUBERLIN
        imsk_folder = IMSKAGE_FOLDER_TUBERLIN
        train_class = TRAIN_CLASS_TUBERLIN
        test_class = TEST_CLASS_TUBERLIN
    else: raise Exception("dataset args error!")
    if args.sketch_dir != '': sketch_folder = args.sketch_dir
    if args.imsk_dir != '': imsk_folder = args.imsk_dir
    if args.npy_dir == '0': args.npy_dir = NPY_FOLDER_SKETCHY
    elif args.npy_dir == '': args.npy_dir = None
    return sketch_folder, imsk_folder, train_class, test_class


def train(args):
    # srun -p gpu --gres=gpu:1 --output=d3shape1.out python main_d3shape.py --steps 50000 --print_every 500 --npy_dir 0 --save_every 2000 --batch_size 96 --dataset sketchy --save_dir d3shape_sketchy

    sketch_folder, imsk_folder, train_class, test_class = _parse_args_paths(args)

    data_train = D3Shape_dataloader(folder_sk=sketch_folder, clss=train_class, folder_nps=args.npy_dir,
                                folder_imsk=imsk_folder, normalize01=False, doaug=False)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=False)

    data_test = D3Shape_dataloader(folder_sk=sketch_folder, clss=test_class, folder_nps=args.npy_dir,
                               folder_imsk=imsk_folder, normalize01=False, doaug=False)

    model = D3Shape()
    model.cuda()
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    steps = _try_load(args, logger, model, optimizer)
    logger.info(str(args))
    args.steps += steps
    d3shape_loss = _D3Shape_loss(cp=args.cp, cn=args.cn)
    model.train()
    while True:
        loss_sum = []
        for _, (sketch1, imsk1, sketch2, imsk2, is_same) in enumerate(dataloader_train):
            optimizer.zero_grad()
            sketch1_feats, imsk1_feats = model(sketch1.cuda(), imsk1.cuda())
            sketch2_feats, imsk2_feats = model(sketch2.cuda(), imsk2.cuda())
            loss = d3shape_loss(sketch1_feats, imsk1_feats, sketch2_feats, imsk2_feats, is_same.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum.append(float(loss.item()))
            if (steps + 1) % args.save_every == 0:
                model.eval()
                n = 50; skip = 1
                start_cpu_t = time.time()
                feats_labels_sk = _extract_feats(data_test, lambda sk: model(sk, None)[0], SK, skip=skip,
                                                 batch_size=args.batch_size)
                feats_labels_imsk = _extract_feats(data_test, lambda imsk: model(None, imsk)[0], IMSK, skip=skip,
                                                   batch_size=args.batch_size)
                pre, mAP = _eval(feats_labels_sk, feats_labels_imsk, n)
                logger.info("Precision@{}: {}, mAP@{}: {}".format(n, pre, n, mAP) +
                            "  " + 'step: {},  loss: {},  (eval cpu time: {}s)'.format(steps, np.mean(loss_sum),
                                                                                       time.time() - start_cpu_t))
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'steps': steps,
                            'args': args},
                           save_fn(args.save_dir, steps, pre, mAP))
                model.train()

            if (steps + 1) % args.print_every == 0:
                logger.info('step: {},  loss: {}'.format(steps, np.mean(loss_sum)))
                loss_sum = []

            steps += 1
            if steps >= args.steps: break
        if steps >= args.steps: break


if __name__ == '__main__':
    args = parse_config()
    train(args)
