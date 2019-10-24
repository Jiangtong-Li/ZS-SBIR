import os
import random
import time

import numpy as np
from scipy.spatial.distance import cdist
import cv2

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from package.model.regression import Regressor
from package.loss.regularization import _Regularization
from package.dataset.data_cmd_translate import CMDTrans_data
from package.args.cvae_args import parse_config
from package.dataset.utils import make_logger
from package import cal_matrics_single

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    writer = SummaryWriter()
    logger = make_logger(args.log_file)

    if args.zs:
        packed = args.packed_pkl_zs
    else:
        packed = args.packed_pkl_nozs

    logger.info('Loading the data ...')
    data = CMDTrans_data(args.sketch_dir, args.image_dir, args.stats_file, args.embedding_file, 
                         packed, args.preprocess_data, args.raw_data, zs=args.zs, sample_time=1, 
                         cvae=True, paired=False, cut_part=False)
    dataloader_train = DataLoader(dataset=data, num_workers=args.num_worker, \
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle)
    logger.info('Training sketch size: {}'.format(len(data.path2class_sketch.keys())))
    logger.info('Training image size: {}'.format(len(data.path2class_image.keys())))
    logger.info('Testing sketch size: {}'.format(len(data.path2class_sketch_test.keys())))
    logger.info('Testing image size: {}'.format(len(data.path2class_image_test.keys())))

    logger.info('Building the model ...')
    model = Regressor(args.raw_size, args.hidden_size, dropout_prob=args.dropout, logger=logger)
    logger.info('Building the optimizer ...')
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1_regularization = _Regularization(model, args.l1_weight, p=1, logger=logger)
    l2_regularization = _Regularization(model, args.l2_weight, p=2, logger=logger)

    if args.start_from is not None:
        logger.info('Loading pretrained model from {} ...'.format(args.start_from))
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    if args.gpu_id != -1:
        model.cuda(args.gpu_id)
    optimizer.zero_grad()

    loss_tri_acm = 0.
    loss_l1_acm = 0.
    loss_l2_acm = 0.
    batch_acm = 0
    global_step = 0
    best_precision = 0.
    best_iter = 0
    patience = args.patience
    logger.info('Hyper-Parameter:')
    logger.info(args)
    logger.info('Model Structure:')
    logger.info(model)
    logger.info('Begin Training !')
    while True:
        if patience <= 0:
            break
        for sketch_batch, image_p_batch, image_n_batch, _semantics_batch in dataloader_train:
            sketch_batch = sketch_batch.float()
            image_p_batch = image_p_batch.float()
            image_n_batch = image_n_batch.float()
            if global_step % args.print_every == 0 % args.print_every and global_step and batch_acm % args.cum_num == 0:
                logger.info('*** Iter {} ***'.format(global_step))
                logger.info('        Loss/Triplet {:.3}'.format(loss_tri_acm/args.print_every/args.cum_num))
                logger.info('        Loss/L1 {:.3}'.format(loss_l1_acm/args.print_every/args.cum_num))
                logger.info('        Loss/L2 {:.3}'.format(loss_l2_acm/args.print_every/args.cum_num))
                loss_tri_acm = 0.
                loss_l1_acm = 0.
                loss_l2_acm = 0.

            if global_step % args.save_every == 0 % args.save_every and batch_acm % args.cum_num == 0 and global_step :
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()},
                           '{}/Iter_{}.pkl'.format(args.save_dir,global_step))

                ### Evaluation
                model.eval()

                image_label = list()
                image_feature = list()
                for image, label in data.load_test_images(batch_size=args.batch_size):
                    image = image.float()
                    if args.gpu_id != -1:
                        image = image.cuda(args.gpu_id)
                    image_label += label
                    tmp_feature = model.inference_image(image).cpu().detach().numpy()
                    image_feature.append(tmp_feature)
                image_feature = np.vstack(image_feature)

                sketch_label = list()
                sketch_feature = list()
                for sketch, label in data.load_test_sketch(batch_size=args.batch_size):
                    sketch = sketch.float()
                    if args.gpu_id != -1:
                        sketch = sketch.cuda(args.gpu_id)
                    sketch_label += label
                    tmp_feature = model.inference_sketch(sketch).cpu().detach().numpy()
                    sketch_feature.append(tmp_feature)
                sketch_feature = np.vstack(sketch_feature)

                Precision, mAP,  = cal_matrics_single(image_feature, image_label, sketch_feature, sketch_label)

                writer.add_scalar('Precision_200/cosine', Precision, global_step)
                writer.add_scalar('mAP_200/cosine', mAP, global_step)
                logger.info('*** Evaluation Iter {} ***'.format(global_step))
                logger.info('        Precision {:.3}'.format(Precision))
                logger.info('        mAP {:.3}'.format(mAP))

                if best_precision < Precision:
                    patience = args.patience
                    best_precision = Precision
                    best_iter = global_step
                    writer.add_scalar('Best/Precision_200', best_precision, best_iter)
                    logger.info('Iter {}, Best Precision_200 {:.3}'.format(global_step, best_precision))
                    torch.save({'args':args, 'model':model.state_dict(), \
                        'optimizer':optimizer.state_dict()}, '{}/Best.pkl'.format(args.save_dir))
                else:
                    patience -= 1
            if patience <= 0:
                break

            model.train()
            batch_acm += 1
            if global_step <= args.warmup_steps:
                update_lr(optimizer, args.lr*global_step/args.warmup_steps)

            if args.gpu_id != -1:
                sketch_batch = sketch_batch.cuda(args.gpu_id)
                image_p_batch = image_p_batch.cuda(args.gpu_id)
                image_n_batch = image_n_batch.cuda(args.gpu_id)

            loss = model(sketch_batch, image_p_batch, image_n_batch)

            loss_l1 = l1_regularization()
            loss_l2 = l2_regularization()
            loss_tri = loss.item()

            loss_l1_acm += (loss_l1.item() / args.l1_weight)
            loss_l2_acm += (loss_l2.item() / args.l2_weight)
            loss_tri_acm += loss_tri

            writer.add_scalar('Loss/Triplet', loss_tri, global_step)
            writer.add_scalar('Loss/Reg_l1', (loss_l1.item() / args.l1_weight), global_step)
            writer.add_scalar('Loss/Reg_l2', (loss_l2.item() / args.l2_weight), global_step)

            loss_ = 0
            loss_ += loss
            loss_.backward()

            if batch_acm % args.cum_num == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                optimizer.zero_grad()

if __name__ == '__main__':
    args = parse_config()
    train(args)
