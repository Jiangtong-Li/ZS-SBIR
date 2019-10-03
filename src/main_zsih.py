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

from package.model.zsim import ZSIM
from package.loss.regularization import _Regularization
from package.dataset.data_zsih import ZSIH_dataloader
from package.args.zsih_args import parse_config
from package.dataset.utils import make_logger


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

    data = ZSIH_dataloader(args.sketch_dir, args.image_dir, args.stats_file, args.embedding_file, packed, zs=args.zs)
    print(len(data))
    dataloader_train = DataLoader(dataset=data, num_workers=args.num_worker, \
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle)

    logger.info('Building the model ...')
    model = ZSIM(args.hidden_size, args.hashing_bit, args.semantics_size, data.pretrain_embedding.float(), 
                 adj_scaler=args.adj_scaler, dropout=args.dropout, fix_cnn=args.fix_cnn, 
                 fix_embedding=args.fix_embedding, logger=logger)
    logger.info('Building the optimizer ...')
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    #optimizer = SGD(params=model.parameters(), lr=args.lr, momentum=0.9)
    l1_regularization = _Regularization(model, 1, p=1, logger=logger)
    l2_regularization = _Regularization(model, 0.005, p=2, logger=logger)

    if args.start_from is not None:
        logger.info('Loading pretrained model from {} ...'.format(args.start_from))
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    if args.gpu_id != -1:
        model.cuda(args.gpu_id)

    batch_acm = 0
    global_step = 0
    loss_p_xz_acm, loss_q_zx_acm, loss_image_l2_acm, loss_sketch_l2_acm, loss_reg_l2_acm, loss_reg_l1_acm = 0., 0., 0., 0., 0., 0.,
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
        for sketch_batch, image_batch, semantics_batch in dataloader_train:
            if global_step % args.print_every == 0 % args.print_every and global_step and batch_acm % args.cum_num == 0:
                logger.info('Iter {}, Loss/p_xz {:.3f}, Loss/q_zx {:.3f}, Loss/image_l2 {:.3f}, Loss/sketch_l2 {:.3f}, Loss/reg_l2 {:.3f}, Loss/reg_l1 {:.3f}'.format(global_step, \
                             loss_p_xz_acm/args.print_every/args.cum_num, \
                             loss_q_zx_acm/args.print_every/args.cum_num, \
                             loss_image_l2_acm/args.print_every/args.cum_num, \
                             loss_sketch_l2_acm/args.print_every/args.cum_num, \
                             loss_reg_l2_acm/args.print_every/args.cum_num, \
                             loss_reg_l1_acm/args.print_every/args.cum_num))
                loss_p_xz_acm, loss_q_zx_acm, loss_image_l2_acm, loss_sketch_l2_acm, loss_reg_l2_acm, loss_reg_l1_acm = 0., 0., 0., 0., 0., 0.,

            if global_step % args.save_every == 0 % args.save_every and batch_acm % args.cum_num == 0 and global_step :
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args':args, 'model':model.state_dict(), \
                        'optimizer':optimizer.state_dict()},
                        '{}/Iter_{}.pkl'.format(args.save_dir,global_step))

                ### Evaluation
                model.eval()

                image_label = list()
                image_feature = list()
                for image, label in data.load_test_images(batch_size=args.batch_size):
                    image = image.cuda(args.gpu_id)
                    image_label += label
                    tmp_feature = model.hash(image, 1).cpu().detach().numpy()
                    image_feature.append(tmp_feature)
                image_feature = np.vstack(image_feature)

                sketch_label = list()
                sketch_feature = list()
                for sketch, label in data.load_test_sketch(batch_size=args.batch_size):
                    sketch = sketch.cuda(args.gpu_id)
                    sketch_label += label
                    tmp_feature = model.hash(sketch, 0).cpu().detach().numpy()
                    sketch_feature.append(tmp_feature)
                sketch_feature = np.vstack(sketch_feature)

                dists_cosine = cdist(image_feature, sketch_feature, 'hamming')

                rank_cosine = np.argsort(dists_cosine, 0)

                for n in [5, 100, 200]:
                    ranksn_cosine = rank_cosine[:n, :].T

                    classesn_cosine = np.array([[image_label[i] == sketch_label[r] \
                                                for i in ranksn_cosine[r]] for r in range(len(ranksn_cosine))])

                    precision_cosine = np.mean(classesn_cosine)

                    writer.add_scalar('Precision_{}/cosine'.format(n),
                            precision_cosine, global_step)

                    logger.info('Iter {}, Precision_{}/cosine {}'.format(global_step, n, precision_cosine))

                if best_precision < precision_cosine:
                    patience = args.patience
                    best_precision = precision_cosine
                    best_iter = global_step
                    writer.add_scalar('Best/Precision_200', best_precision, best_iter)
                    logger.info('Iter {}, Best Precision_200 {}'.format(global_step, best_precision))
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
            """
            #code for testing if the images and the sketches are corresponding to each other correctly

            for i in range(args.batch_size):
                sk = sketch_batch[i].numpy().reshape(224, 224, 3)
                im = image_batch[i].numpy().reshape(224, 224, 3)
                print(label[i])
                ims = np.vstack((np.uint8(sk), np.uint8(im)))
                cv2.imshow('test', ims)
                cv2.waitKey(3000)
            """

            sketch = sketch_batch.cuda(args.gpu_id)
            image = image_batch.cuda(args.gpu_id)
            semantics = semantics_batch.long().cuda(args.gpu_id)

            optimizer.zero_grad()
            loss = model(sketch, image, semantics)
            loss_l1 = l1_regularization()
            loss_l2 = l2_regularization()
            loss_p_xz_acm += loss['p_xz'][0].item()
            loss_q_zx_acm += loss['q_zx'][0].item()
            loss_image_l2_acm += loss['image_l2'][0].item()
            loss_sketch_l2_acm += loss['sketch_l2'][0].item()
            loss_reg_l1_acm += loss_l1.item()
            loss_reg_l2_acm += (loss_l2.item() / 0.005)
            writer.add_scalar('Loss/p_xz', loss['p_xz'][0].item(), global_step)
            writer.add_scalar('Loss/q_zx', loss['q_zx'][0].item(), global_step)
            writer.add_scalar('Loss/image_l2', loss['image_l2'][0].item(), global_step)
            writer.add_scalar('Loss/sketch_l2', loss['sketch_l2'][0].item(), global_step)
            writer.add_scalar('Loss/reg_l2', (loss_l2.item() / 0.005), global_step)
            writer.add_scalar('Loss/reg_l1', loss_l1.item(), global_step)
            
            loss_ = loss_l2
            for item in loss.values():
                loss_ += item[0]*item[1]
            loss_.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if batch_acm % args.cum_num == 0:
                optimizer.step()
                global_step += 1

if __name__ == '__main__':
    args = parse_config()
    train(args)
