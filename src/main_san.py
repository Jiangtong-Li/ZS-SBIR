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
from package.model.san import SaN
from package.loss.san_loss import _SaN_loss
from package.dataset.data_san import *
from package.args.san_args import parse_config
from package.dataset.utils import make_logger
from package.model.utils import *
from package.loss.regularization import _Regularization
from sklearn.neighbors import NearestNeighbors as NN


DEBUG = True


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
    :param what: SK or IM
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
    :return: mAP
    """
    temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    mAP_term = 1.0 / (np.stack(temp, axis=0) + 1.0)
    mAP = np.mean(np.multiply(_map_change(matches), mAP_term), axis=1)
    return np.mean(mAP)


def _eval(feats_labels_sk, feats_labels_im, n=200):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n
    """
    nn = NN(n_neighbors=n, metric='cosine', algorithm='brute').fit(feats_labels_im[0])
    _, indices = nn.kneighbors(feats_labels_sk[0])
    retrieved_classes = np.array(feats_labels_im[1])[indices]

    # astype(np.uint16) is necessary
    ranks = np.vstack([(retrieved_classes[i] == feats_labels_sk[1][i])
                       for i in range(retrieved_classes.shape[0])]).astype(np.uint16)
    return _get_pre_from_matches(ranks), _get_map_from_matches(ranks)


def _parse_args_paths(args):
    if args.dataset == 'sketchy':
        sketch_folder = SKETCH_FOLDER_SKETCHY
        image_folder = IMAGE_FOLDER_SKETCHY
        train_class = TRAIN_CLASS_SKETCHY
        test_class = TEST_CLASS_SKETCHY
    elif args.dataset == 'tuberlin':
        sketch_folder = SKETCH_FOLDER_TUBERLIN
        image_folder = IMAGE_FOLDER_TUBERLIN
        train_class = TRAIN_CLASS_TUBERLIN
        test_class = TEST_CLASS_TUBERLIN
    else: raise Exception("dataset args error!")
    if args.sketch_dir != '': sketch_folder = args.sketch_dir
    if args.image_dir != '': image_folder = args.image_dir
    if args.npy_dir == '0': args.npy_dir = NPY_FOLDER_SKETCHY
    elif args.npy_dir == '': args.npy_dir = None
    return sketch_folder, image_folder, train_class, test_class


def train(args):
    # srun -p gpu --gres=gpu:1 --exclusive --output=san10.out python main_san.py --steps 50000 --print_every 500 --save_every 2000 --batch_size 96 --dataset sketchy --margin 10 --npy_dir 0 --save_dir san_sketchy10
    # srun -p gpu --gres=gpu:1 --exclusive --output=san1.out python main_san.py --steps 50000 --print_every 500 --save_every 2000 --batch_size 96 --dataset sketchy --margin 1 --npy_dir 0 --save_dir san_sketchy1

    # srun -p gpu --gres=gpu:1 --output=san_sketchy03.out python main_san.py --steps 30000 --print_every 200 --save_every 3000 --batch_size 96 --dataset sketchy --margin 0.3 --npy_dir 0 --save_dir san_sketchy03 --lr 0.0001
    sketch_folder, image_folder, train_class, test_class = _parse_args_paths(args)

    if DEBUG:
        args.print_every = 5
        args.save_every = 20
        args.steps = 100
        args.batch_size = 32
        train_class = train_class[:2]
        test_class = test_class[:2]

    data_train = SaN_dataloader(folder_sk=sketch_folder, clss=train_class, folder_nps=args.npy_dir,
                                folder_im=image_folder, normalize01=False, doaug=False)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=False)

    data_test = SaN_dataloader(folder_sk=sketch_folder, exp3ch=True, clss=test_class, folder_nps=args.npy_dir,
                               folder_im=image_folder, normalize01=False, doaug=False)

    model = SaN()
    model.cuda()
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    steps = _try_load(args, logger, model, optimizer)
    logger.info(str(args))
    args.steps += steps
    san_loss = _SaN_loss(args.margin)
    model.train()
    l2_regularization = _Regularization(model, args.l2_reg, p=2, logger=None)
    while True:
        loss_sum = []
        for _, (sketch, positive_image, negative_image, positive_class_id) in enumerate(dataloader_train):
            optimizer.zero_grad()
            loss = san_loss(model(sketch.cuda()),
                            model(positive_image.cuda()),
                            model(negative_image.cuda())) \
                    + l2_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum.append(float(loss.item()))
            if (steps + 1) % args.save_every == 0:
                model.eval()
                n = 200; skip = 1
                start_cpu_t = time.time()
                feats_labels_sk = _extract_feats(data_test, model, SK, skip=skip, batch_size=args.batch_size)
                feats_labels_im = _extract_feats(data_test, model, IM, skip=skip, batch_size=args.batch_size)
                pre, mAP = _eval(feats_labels_sk, feats_labels_im, n)
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


def gen_args(margin=0.3, dataset='sketchy'):
    margins = str(margin).replace('.', '')
    return \
"""
###

#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=san_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4

source activate lzxtc
python main_san.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --dataset {} --save_dir san_{}_{} --lr 0.0001 --margin {}

sbatch san.slurm
""".format(dataset, dataset, margins, margin)


if __name__ == '__main__':
    # print(gen_args(1))
    # exit()
    args = parse_config()
    train(args)


'''
#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=san_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4

source activate lzxtc
python main_san.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --dataset sketchy --save_dir san_sketchy_03 --lr 0.0001 --margin 0.3
python main_san.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --dataset sketchy --save_dir san_sketchy_01 --lr 0.0001 --margin 0.1
python main_san.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --dataset sketchy --save_dir san_sketchy_05 --lr 0.0001 --margin 0.5
python main_san.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --dataset sketchy --save_dir san_sketchy_1 --lr 0.0001 --margin 1
sbatch san.slurm
'''