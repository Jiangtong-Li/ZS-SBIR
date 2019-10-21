import os
import random

import numpy as np
from scipy.spatial.distance import cdist
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
from package.model.dsh import DSH, update_B, update_D
from package.loss.dsh_loss import _DSH_loss
from package.dataset.data_dsh import *
from package.args.dsh_args import parse_config
from package.dataset.utils import make_logger
from package.model.utils import *
from package.loss.regularization import _Regularization
from sklearn.neighbors import NearestNeighbors as NN


DEBUG = False


def dr_dec(optimizer, args):
    args.lr *= 0.3
    optimizer.param_groups[0]['lr'] = args.lr


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
    for batch_idx, (sketches, sketch_tokens, images, id) in \
            enumerate(data_test.traverse(what, skip=skip, batch_size=batch_size)):
        '''
        print(sketches.shape if sketches is not None else None,
              sketch_tokens.shape if sketch_tokens is not None else None,
              images.shape if images is not None else None)
        # None torch.Size([4, 1, 200, 200]) torch.Size([4, 3, 227, 227])
        '''
        labels.append(model(sk=sketches.cuda() if sketches is not None else None,
                            st=sketch_tokens.cuda() if sketch_tokens is not None else None,
                            im=images.cuda() if images is not None else None)[0].data.cpu().numpy())
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
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
            matches[i][j] == 1 indicates the j-th retrieved test image j belongs to the same class as test sketch i,
            otherwise, matches[i][j] = 0.
    :return: mAP
    """
    temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    mAP_term = 1.0 / (np.stack(temp, axis=0) + 1.0)
    precisions = np.multiply(_map_change(matches), mAP_term)
    ap = np.sum(precisions, 1) * (1.0 / np.sum(np.abs(precisions) > 1e-10, 1))
    return np.mean(ap)


def _eval(feats_labels_sk, feats_labels_im, n=200):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@all
    """
    nn = NN(n_neighbors=feats_labels_im[0].shape[0], metric='hamming', algorithm='brute').fit(feats_labels_im[0])
    _, indices = nn.kneighbors(feats_labels_sk[0])
    retrieved_classes = np.array(feats_labels_im[1])[indices]
    matches = np.vstack([(retrieved_classes[i] == feats_labels_sk[1][i])
                       for i in range(retrieved_classes.shape[0])]).astype(np.uint16)
    return _get_pre_from_matches(matches[:, :n]), _get_map_from_matches(matches)



def _parse_args_paths(args):
    if args.dataset == 'sketchy':
        sketch_folder = SKETCH_FOLDER_SKETCHY
        imsk_folder = IMSKAGE_FOLDER_SKETCHY
        im_folder = IMAGE_FOLDER_SKETCHY
        path_semantic = PATH_SEMANTIC_SKETCHY
        train_class = TRAIN_CLASS_SKETCHY
        test_class = TEST_CLASS_SKETCHY
        npy_folder = NPY_FOLDER_SKETCHY
    elif args.dataset == 'tuberlin':
        sketch_folder = SKETCH_FOLDER_TUBERLIN
        imsk_folder = IMSKAGE_FOLDER_TUBERLIN
        im_folder = IMAGE_FOLDER_TUBERLIN
        path_semantic = PATH_SEMANTIC_TUBERLIN
        train_class = TRAIN_CLASS_TUBERLIN
        test_class = TEST_CLASS_TUBERLIN
        npy_folder = NPY_FOLDER_TUBERLIN
    else: raise Exception("dataset args error!")
    if args.sketch_dir != '': sketch_folder = args.sketch_dir
    if args.imsk_dir != '': imsk_folder = args.imsk_dir
    if args.im_dir != '': im_folder = args.im_dir
    if args.path_semantic != '': im_folder = args.path_semantic
    if args.npy_dir == '0': args.npy_dir = npy_folder
    elif args.npy_dir == '': args.npy_dir = None
    return sketch_folder, imsk_folder, im_folder, path_semantic, train_class, test_class


def _extract_feats_sk_im(data, model, batch_size=64):
    skip = 1
    model.eval()
    feats_labels_sk = _extract_feats(data, model, SK, skip=skip,
                                     batch_size=batch_size)
    feats_labels_im = _extract_feats(data, model, IM, skip=skip,
                                    batch_size=batch_size)
    model.train()
    return feats_labels_sk, feats_labels_im


def _test_and_save(steps, optimizer, data_test, model, logger, args, loss_sum):
    if not hasattr(_test_and_save, 'best_acc'):
        _test_and_save.best_acc = 0
    n = 100
    start_cpu_t = time.time()
    feats_labels_sk, feats_labels_im = _extract_feats_sk_im(data=data_test, model=model,
                                                              batch_size=args.batch_size)
    pre, mAP = _eval(feats_labels_sk, feats_labels_im, n)
    logger.info("Precision@{}: {}, mAP@all: {}, bestPrecision: {}".format(n, pre, mAP, max(pre, _test_and_save.best_acc)) +
                "  " + 'step: {},  loss: {},  (eval cpu time: {}s)'.format(steps, np.mean(loss_sum),
                                                                           time.time() - start_cpu_t))
    if pre > _test_and_save.best_acc:
        _test_and_save.best_acc = pre
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'args': args},
                   save_fn(args.save_dir, steps, pre, mAP))
    torch.cuda.empty_cache()


def train(args):
    # srun -p gpu --gres=gpu:1 python main_dsh.py
    sketch_folder, imsk_folder, im_folder, path_semantic, train_class, test_class = _parse_args_paths(args)
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    if DEBUG:
        train_class = train_class[:2]
        test_class = test_class[:2]
        args.print_every = 2
        args.save_every = 8
        args.steps = 20
        args.batch_size = 2
        args.npy_dir = NPY_FOLDER_SKETCHY


    # logger.info("try loading data_train")
    data_train = DSH_dataloader(folder_sk=sketch_folder, folder_im=im_folder, clss=train_class, folder_nps=args.npy_dir,
                                folder_imsk=imsk_folder, normalize01=False, doaug=False, m=args.m, path_semantic=path_semantic,
                                folder_saving=join(mkdir(args.save_dir), 'train_saving'), logger=logger)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=False)
    # logger.info("try loading data_test")
    data_test = DSH_dataloader(folder_sk=sketch_folder, clss=test_class, folder_nps=args.npy_dir, path_semantic=path_semantic,
                               folder_imsk=imsk_folder, normalize01=False, doaug=False, m=args.m,
                               folder_saving=join(mkdir(args.save_dir), 'test_saving'), logger=logger)

    model = DSH(m=args.m, config=args.config)
    model.cuda()

    optimizer = SGD(params=model.parameters(), lr=args.lr, momentum=0.9)

    # logger.info("optimizer inited")
    steps = _try_load(args, logger, model, optimizer)
    logger.info(str(args))
    args.steps += steps
    dsh_loss = _DSH_loss(gamma=args.gamma)
    model.train()
    l2_regularization = _Regularization(model, args.l2_reg, p=2, logger=None)
    loss_sum = []
    # logger.info("iterations")
    # iterations
    while True:
        # logger.info("update D")
        # 1. update D
        data_train.D = update_D(bi=data_train.BI, bs=data_train.BS,
                                vec_bi=data_train.vec_bi, vec_bs=data_train.vec_bs)
        # logger.info("update BI/BS")
        # 2. update BI/BS
        feats_labels_sk, feats_labels_im = _extract_feats_sk_im(data=data_train, model=model,
                                                                  batch_size=args.batch_size)

        data_train.BI, data_train.BS = update_B(bi=data_train.BI, bs=data_train.BS,
                                                vec_bi=data_train.vec_bi, vec_bs=data_train.vec_bs,
                                                W=data_train.W, D=data_train.D, Fi=feats_labels_im[0],
                                                Fs=feats_labels_sk[0], lamb=args.lamb, gamma=args.gamma)
        # logger.info("update network parameters")
        # 3. update network parameters
        for _, (sketch, code_of_sketch, image, sketch_token, code_of_image) in enumerate(dataloader_train):

            sketch_feats, im_feats = model(sketch.cuda(), sketch_token.cuda(), image.cuda())
            loss = dsh_loss(sketch_feats, im_feats, code_of_sketch.cuda(), code_of_image.cuda()) \
                    + l2_regularization()
            loss = loss / args.update_every
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (steps + 1) % args.update_every == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum.append(float(loss.item() * args.update_every))
            if (steps + 1) % args.save_every == 0:
                _test_and_save(steps=steps, optimizer=optimizer, data_test=data_test,
                                model=model, logger=logger, args=args,
                                loss_sum=loss_sum)
                data_train.save_params()

            if (steps + 1) % args.print_every == 0:
                loss_sum = [np.mean(loss_sum)]
                logger.info('step: {},  loss: {}'.format(steps, loss_sum[0]))

            steps += 1
            if steps >= args.steps: break
        dr_dec(optimizer=optimizer, args=args)
        if steps >= args.steps: break


def gen_args(m=128, lamb=0.01, gamma=0.00001, dataset='sketchy', config=1):
    lambs = str(lamb).replace('.', '')
    gammas = str(gamma).replace('.', '')
    return \
"""
###

#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=dsh_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4
source activate lzxtc2
python main_dsh.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 64 --m {} --dataset {} --save_dir dsh{}_{}{}_{} --lr 0.0001 --lamb {} --gamma {} --config {}

sbatch dsh.slurm
""".format(m, dataset, config, dataset, lambs, gammas, lamb, gamma, config)


if __name__ == '__main__':
    # print(gen_args())
    # exit()
    args = parse_config()
    train(args)
    pass

'''
#!/bin/bash

#SBATCH --job-name=ZXLing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=dsh_%j.out
#SBATCH --time=7-00:00:00
module load gcc/7.3.0 anaconda/3 cuda/9.2 cudnn/7.1.4
source activate lzxtc
python main_dsh.py --steps 30000 --print_every 500 --npy_dir 0 --save_every 3000 --batch_size 32 --m 128 --dataset sketchy --save_dir dsh_sketchy001_1e-05 --lr 0.0001 --lamb 0.01 --gamma 1e-05

sbatch dsh.slurm
'''


