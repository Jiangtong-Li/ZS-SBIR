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

from package.model.cmd_translate import CMDTrans_model
from package.loss.regularization import _Regularization
from package.dataset.data_cmd_translate import CMDTrans_data
from package.args.cmd_trans_args import parse_config
from package.dataset.utils import make_logger
from package import cal_matrics, cosine_distance

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
                         cvae=True, paired=True, cut_part=False, ranking=True)
    dataloader_train = DataLoader(dataset=data, num_workers=args.num_worker, 
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle)
    logger.info('Training sketch size: {}'.format(len(data.path2class_sketch.keys())))
    logger.info('Training image size: {}'.format(len(data.path2class_image.keys())))
    logger.info('Testing sketch size: {}'.format(len(data.path2class_sketch_test.keys())))
    logger.info('Testing image size: {}'.format(len(data.path2class_image_test.keys())))

    logger.info('Building the model ...')
    model = CMDTrans_model(args.pca_size, args.raw_size, args.hidden_size, args.semantics_size, data.pretrain_embedding.float(), 
                 dropout_prob=args.dropout, fix_embedding=args.fix_embedding, seman_dist=args.seman_dist, 
                 triplet_dist=args.triplet_dist, margin1=args.margin1, margin2=args.margin2, logger=logger)
    logger.info('Building the optimizer ...')
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    #optimizer = SGD(params=model.parameters(), lr=args.lr, momentum=0.9)
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

    # six design loss and two reg loss
    loss_triplet_acm = 0.
    loss_orth_acm = 0.
    loss_kl_acm = 0.
    loss_img_acm = 0.
    loss_ske_acm = 0.
    loss_l1_acm = 0.
    loss_l2_acm = 0.
    # loading batch and optimization step
    batch_acm = 0
    global_step = 0
    # best recoder
    best_precision = 0.
    best_iter = 0
    patience = args.patience
    logger.info('Hyper-Parameter:')
    logger.info(args)
    logger.info('Model Structure:')
    logger.info(model)
    logger.info('Begin Training !')
    loss_weight = dict([('kl',1.0), ('triplet', 1.0), ('orthogonality', 0.01), ('image', 1.0), ('sketch', 10.0)])
    while True:
        if patience <= 0:
            break
        for sketch_batch, image_pair_batch, image_unpair_batch, image_n_batch in dataloader_train:
            if global_step % args.print_every == 0 % args.print_every and global_step and batch_acm % args.cum_num == 0:
                logger.info('*** Iter {} ***'.format(global_step))
                logger.info('        Loss/Triplet {:.3}'.format(loss_triplet_acm/args.print_every/args.cum_num))
                logger.info('        Loss/Orthogonality {:.3}'.format(loss_orth_acm/args.print_every/args.cum_num))
                logger.info('        Loss/KL {:.3}'.format(loss_kl_acm/args.print_every/args.cum_num))
                logger.info('        Loss/Image {:.3}'.format(loss_img_acm/args.print_every/args.cum_num))
                logger.info('        Loss/Sketch {:.3}'.format(loss_ske_acm/args.print_every/args.cum_num))
                logger.info('        Loss/L1 {:.3}'.format(loss_l1_acm/args.print_every/args.cum_num))
                logger.info('        Loss/L2 {:.3}'.format(loss_l2_acm/args.print_every/args.cum_num))
                loss_triplet_acm = 0.
                loss_orth_acm = 0.
                loss_kl_acm = 0.
                loss_img_acm = 0.
                loss_ske_acm = 0.
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
                image_feature1 = list() # S
                image_feature2 = list() # G
                for image, label in data.load_test_images(batch_size=args.batch_size):
                    if args.gpu_id != -1:
                        image = image.float().cuda(args.gpu_id)
                    image_label += label
                    tmp_feature1 = model.inference_structure(image, 'image').detach() # S
                    tmp_feature2 = image.detach() # G
                    image_feature1.append(tmp_feature1)
                    image_feature2.append(tmp_feature2)
                image_feature1 = torch.cat(image_feature1)
                image_feature2 = torch.cat(image_feature2)

                sketch_label = list()
                sketch_feature1 = list() # S
                sketch_feature2 = list() # G
                for sketch, label in data.load_test_sketch(batch_size=args.batch_size):
                    if args.gpu_id != -1:
                        sketch = sketch.float().cuda(args.gpu_id)
                    sketch_label += label
                    tmp_feature1 = model.inference_structure(sketch, 'sketch').detach() # S
                    tmp_feature2 = model.inference_generation(sketch).detach() # G
                    sketch_feature1.append(tmp_feature1)
                    sketch_feature2.append(tmp_feature2)
                sketch_feature1 = torch.cat(sketch_feature1)
                sketch_feature2 = torch.cat(sketch_feature2)

                dists_cosine1 = cosine_distance(image_feature1, sketch_feature1).cpu().detach().numpy()
                dists_cosine2 = cosine_distance(image_feature2, sketch_feature2).cpu().detach().numpy()

                Precision_list, mAP_list, lambda_list, Precision_c, mAP_c = \
                    cal_matrics(dists_cosine1, dists_cosine2, image_label, sketch_label)
                
                logger.info('*** Evaluation Iter {} ***'.format(global_step))
                for idx, item in enumerate(lambda_list):
                    writer.add_scalar('Precision_200/{}'.format(item), Precision_list[idx], global_step)
                    writer.add_scalar('mAP_200/{}'.format(item), mAP_list[idx], global_step)
                    logger.info('        Precision/{} {:.3}'.format(item, Precision_list[idx]))
                    logger.info('        mAP/{} {:.3}'.format(item, mAP_list[idx]))
                writer.add_scalar('Precision_200/Compare', Precision_c, global_step)
                writer.add_scalar('mAP_200/Compare', mAP_c, global_step)
                logger.info('        Precision/Compare {:.3}'.format(Precision_c))
                logger.info('        mAP/Compare {:.3}'.format(mAP_c))
                
                Precision_list.append(Precision_c)
                Precision = max(Precision_list)
                if best_precision < Precision:
                    patience = args.patience
                    best_precision = Precision
                    best_iter = global_step
                    writer.add_scalar('Best/Precision_200', best_precision, best_iter)
                    logger.info('=== Iter {}, Best Precision_200 {:.3} ==='.format(global_step, best_precision))
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
                sketch_batch = sketch_batch.float().cuda(args.gpu_id)
                image_pair_batch = image_pair_batch.float().cuda(args.gpu_id)
                image_unpair_batch = image_unpair_batch.float().cuda(args.gpu_id)
                image_n_batch = image_n_batch.float().cuda(args.gpu_id)

            loss = model(sketch_batch, image_pair_batch, image_unpair_batch, image_n_batch)

            loss_l1 = l1_regularization()
            loss_l2 = l2_regularization()
            loss_kl = loss['kl'].item()
            loss_orth = loss['orthogonality'].item()
            loss_triplet = loss['triplet'].item()
            loss_img = loss['image'].item()
            loss_ske = loss['sketch'].item()

            loss_l1_acm += (loss_l1.item() / args.l1_weight)
            loss_l2_acm += (loss_l2.item() / args.l2_weight)
            loss_kl_acm += loss_kl
            loss_orth_acm += loss_orth
            loss_triplet_acm += loss_triplet
            loss_img_acm += loss_img
            loss_ske_acm += loss_ske

            writer.add_scalar('Loss/KL', loss_kl, global_step)
            writer.add_scalar('Loss/Orthogonality', loss_orth, global_step)
            writer.add_scalar('Loss/Triplet', loss_triplet, global_step)
            writer.add_scalar('Loss/Image', loss_img, global_step)
            writer.add_scalar('Loss/Sketch', loss_ske, global_step)
            writer.add_scalar('Loss/Reg_l1', (loss_l1.item() / args.l1_weight), global_step)
            writer.add_scalar('Loss/Reg_l2', (loss_l2.item() / args.l2_weight), global_step)

            loss_ = 0
            loss_ += loss['image']*loss_weight['image']
            loss_ += loss['sketch']*loss_weight['sketch']
            loss_ += loss['triplet']*loss_weight['triplet']
            loss_ += loss['kl']*loss_weight['kl']
            #loss_ += loss['orthogonality']*loss_weight['orthogonality']
            loss_.backward()

            if batch_acm % args.cum_num == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                optimizer.zero_grad()

if __name__ == '__main__':
    args = parse_config()
    train(args)
