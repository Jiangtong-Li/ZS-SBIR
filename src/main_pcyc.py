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
from package.model.pcyc import PCYC
from package.dataset.data_pcyc import *
from package.args.pcyc_args import parse_config
from package.dataset.utils import make_logger
from package.model.utils import *
from package.loss.regularization import _Regularization

import numpy as np
from sklearn.neighbors import NearestNeighbors as NN


DEBUG = False


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
                "  " + 'epochs: {},  loss: {},  (eval cpu time: {}s)'.
                format(epochs, [np.mean(loss) for loss in loss_sum], time.time() - start_cpu_t))
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
    feats_labels_sk = _extract_feats(data, lambda x: model(sk=x, im=None), SK, skip=skip,
                                     batch_size=batch_size)
    feats_labels_im = _extract_feats(data, lambda x: model(sk=None, im=x), IM, skip=skip,
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
        labels.append(id.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _parse_args_paths(args):
    if args.dataset == 'sketchy':
        sketch_folder = SKETCH_FOLDER_SKETCHY
        im_folder = IMAGE_FOLDER_SKETCHY
        train_class = TRAIN_CLASS_SKETCHY
        test_class = TEST_CLASS_SKETCHY
        npy_folder = NPY_FOLDER_SKETCHY
        path_names = PATH_NAMES_SKETCHY
    elif args.dataset == 'tuberlin':
        sketch_folder = SKETCH_FOLDER_TUBERLIN
        im_folder = IMAGE_FOLDER_TUBERLIN
        train_class = TRAIN_CLASS_TUBERLIN
        test_class = TEST_CLASS_TUBERLIN
        npy_folder = NPY_FOLDER_TUBERLIN
        path_names = ''
    else: raise Exception("dataset args error!")
    if args.sketch_dir != '': sketch_folder = args.sketch_dir
    if args.image_dir != '': im_folder = args.image_dir
    if args.npy_dir == '0': args.npy_dir = npy_folder
    elif args.npy_dir == '': args.npy_dir = None
    if args.ni_path == '0': args.ni_path = path_names
    return sketch_folder, im_folder, train_class, test_class


def _init_dataset(args):
    sketch_folder, image_folder, train_class, test_class = _parse_args_paths(args)

    if DEBUG:
        test_class = train_class[0:2]
        train_class = train_class[:2]

    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    data_train = PCYC_dataloader(folder_sk=sketch_folder, clss=train_class, folder_nps=args.npy_dir,
                                 paired=args.paired, names=args.ni_path,
                                folder_im=image_folder, normalize01=False, doaug=False, logger=logger,
                                sz=None)

    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True)

    data_test = PCYC_dataloader(folder_sk=sketch_folder, clss=test_class, folder_nps=args.npy_dir,
                                folder_im=image_folder, normalize01=False, doaug=False,
                               logger=logger, sz=None)
    logger.info("Datasets initialized:\n train_classes: {}\n test_classes: {}".format(train_class, test_class))
    return data_train, dataloader_train, data_test, logger


def train(args):
    data_train, dataloader_train, data_test, logger = _init_dataset(args=args)

    model = PCYC(args=args, num_clss=len(data_train.clss))
    print(33333)
    model.cuda()
    print(11111)
    # this optimizer is not used, ignore it (just to keep the code consistent)
    optimizer = SGD(params=model.parameters(), lr=args.lr, momentum=0.6)
    print(22222)
    epochs = _try_load(args, logger, model, optimizer)
    logger.info(str(args))
    args.epochs += epochs

    model.train()

    loss_sum = [[0], [0], [0]]

    _test_and_save(epochs=epochs, optimizer=optimizer, data_test=data_test,
                   model=model, logger=logger, args=args, loss_sum=loss_sum)
    while True:
        model.scheduler_gen.step()
        model.scheduler_disc.step()
        for _, (sk, im, cl) in enumerate(dataloader_train):
            cl = cl.long()
            if DEBUG:
                sk, im, cl = sk[:2].cuda(), im[:2].cuda(), cl[:2].cuda()
            else:
                sk, im, cl = sk.cuda(), im.cuda(), cl.cuda()
            loss = model.optimize_params(sk, im, cl)
            for i in range(len(loss_sum)):
                loss_sum[i].append(loss[i].item())
            if DEBUG:
                break

        epochs += 1

        if (epochs + 1) % args.save_every == 0:
            _test_and_save(epochs=epochs, optimizer=optimizer, data_test=data_test,
                           model=model, logger=logger, args=args, loss_sum=loss_sum)

        if (epochs + 1) % args.print_every == 0:
            logger.info('epochs: {},  loss: {}'.
                        format(epochs, [np.mean(loss) for loss in loss_sum]))
            loss_sum = [[loss[-1]] for loss in loss_sum]

        if epochs >= args.epochs: break


def gen_args(dim_enc=128, dataset='sketchy', paired=False):
    """
#!/bin/bash
#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=pcyc_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4; source activate lzxtc2
mkdir pcycs

#!/bin/bash
#SBATCH --job-name=ZXLing
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --output=pcyc_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
module load cuda/10.0.130-gcc-4.8.5   gcc/8.3.0-gcc-4.8.5 miniconda3/4.6.14-gcc-4.8.5; source activate lzxtc
mkdir pcycs
    """
    paired = int(paired)
    return \
"""
python main_pcyc.py --npy_dir 0 --dataset {} --save_dir pcycs/pcyc_{}_{}_{} --dim_enc {} --paired {}  --ni_path 0
""".format(dataset, int(paired) , dataset, dim_enc, dim_enc, int(paired))


if __name__ == '__main__':
    if False:
        print(gen_args(paired=True, dataset='sketchy'))
        print(gen_args(paired=False, dataset='sketchy'))
        print(gen_args(paired=True, dataset='tuberlin'))
        print(gen_args(paired=False, dataset='tuberlin'))
        exit()
    args = parse_config()
    train(args)


# srun --gres=gpu:1 --output=cmt_%j.out python main_cmt.py
'''
python main_pcyc.py --npy_dir 0 --dataset sketchy --save_dir pcycs/pcyc_1_sketchy_128 --dim_enc 128 --paired 1  --ni_path 0
python main_pcyc.py --npy_dir 0 --dataset sketchy --save_dir pcycs/pcyc_0_sketchy_128 --dim_enc 128 --paired 0  --ni_path 0

python main_pcyc.py --npy_dir 0 --dataset tuberlin --save_dir pcycs/pcyc_0_tuberlin_128 --dim_enc 128 --paired 0
'''