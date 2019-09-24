import os
import random

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

from package.model.siamese import Siamese
from package.loss.regularization import _Regularization
from package.loss.siamese_loss import _Siamese_loss
from package.dataset.data import Siamese_dataloader
from package.args.siamese_args import parse_config
from package.dataset.utils import make_logger

from package.dataset.data_mnist import load_train

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    writer = SummaryWriter()
    logger = make_logger(args.log_file)

    data = Siamese_dataloader(args.sketch_dir_train, args.image_dir_train, args.stats_file_train, \
                              args.sketch_dir_test, args.image_dir_test, args.stats_file_test, \
                              args.packed_pkl)
    dataloader_train = DataLoader(dataset=data, num_workers=args.num_worker, \
                                  batch_size=args.batch_size, shuffle=False)
    
    #data_mnist = load_train()
    #dataloader_mnist = DataLoader(data_mnist, batch_size=32)

    logger.info('Building the model ...')
    model = Siamese(args.margin, args.loss_type, args.distance_type, batch_normalization=False, from_pretrain=True, logger=logger)
    logger.info('Building the optimizer ...')
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    #optimizer = SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    siamese_loss = _Siamese_loss()
    l1_regularization = _Regularization(model, 0.1, p=1, logger=logger)
    l2_regularization = _Regularization(model, 1e-4, p=2, logger=logger)

    if args.start_from is not None:
        logger.info('Loading pretrained model from {} ...'.format(args.start_from))
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    if args.gpu_id != -1:
        model.cuda(args.gpu_id)

    batch_acm = 0
    loss_siamese_acm, sim_acm, dis_sim_acm, loss_l1_acm, loss_l2_acm = 0., 0., 0., 0., 0.,
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
        for sketch, image, label in dataloader_train:#data.load_same_class_image(batch_size=32):
            model.train()
            batch_acm += 1
            if batch_acm <= args.warmup_steps:
                update_lr(optimizer, args.lr*batch_acm/args.warmup_steps)

            """
            #code for testing if the images and the sketches are corresponding to each other correctly

            for i in range(args.batch_size):
                sk = sketch[i].numpy().reshape(224, 224, 3)
                im = image[i].numpy().reshape(224, 224, 3)
                print(label[i])
                ims = np.vstack((np.uint8(sk), np.uint8(im)))
                cv2.imshow('test', ims)
                cv2.waitKey(3000)
            """

            sketch = sketch.cuda(args.gpu_id)
            image = image.cuda(args.gpu_id)
            label = label.float().cuda(args.gpu_id)

            optimizer.zero_grad()
            sketch_feature, image_feature = model(sketch, image)
            loss_siamese, sim, dis_sim = siamese_loss(sketch_feature, image_feature, label, args.margin, loss_type=args.loss_type, distance_type=args.distance_type)
            loss_l1 = l1_regularization()
            loss_l2 = l2_regularization()
            loss_siamese_acm += loss_siamese.item()
            sim_acm += sim.item()
            dis_sim_acm += dis_sim.item()
            loss_l1_acm += loss_l1.item()
            loss_l2_acm += loss_l2.item()
            writer.add_scalar('Loss/Siamese', loss_siamese.item(), batch_acm)
            writer.add_scalar('Loss/L1', loss_l1.item(), batch_acm)
            writer.add_scalar('Loss/L2', loss_l2.item(), batch_acm)
            writer.add_scalar('Siamese/Similar', sim.item(), batch_acm)
            writer.add_scalar('Siamese/Dis-Similar', dis_sim.item(), batch_acm)
            loss = loss_siamese + loss_l2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(loss_siamese.item())
            break
            if batch_acm % args.print_every == -1 % args.print_every:
                logger.info('Iter {}, Loss/siamese {:.3f}, Loss/l1 {:.3f}, Loss/l2 {:.3f}, Siamese/sim {:.3f}, Siamese/dis_sim {:.3f}'.format(batch_acm, \
                             loss_siamese_acm/args.print_every, loss_l1_acm/args.print_every, \
                             loss_l2_acm/args.print_every, sim_acm/args.print_every, \
                             dis_sim_acm/args.print_every))
                loss_siamese_acm, sim_acm, dis_sim_acm, loss_l1_acm, loss_l2_acm = 0., 0., 0., 0., 0.,

            if batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args':args, 'model':model.state_dict(), \
                        'optimizer':optimizer.state_dict()}, '{}/Iter_{}.pkl'.format(args.save_dir,batch_acm))

                ### Evaluation
                model.eval()

                image_label = list()
                image_feature = list()
                for image, label in data.load_test_images(batch_size=args.batch_size):
                    shape = image.shape
                    image = image.cuda(args.gpu_id)
                    image_label += label
                    tmp_feature = model.get_feature(image).cpu().detach().numpy()
                    image_feature.append(tmp_feature)
                image_feature = np.vstack(image_feature)

                sketch_label = list()
                sketch_feature = list()
                for sketch, label in data.load_test_sketch(batch_size=args.batch_size):
                    shape = sketch.shape
                    sketch = sketch.cuda(args.gpu_id)
                    sketch_label += label
                    tmp_feature = model.get_feature(sketch).cpu().detach().numpy()
                    sketch_feature.append(tmp_feature)
                sketch_feature = np.vstack(sketch_feature)

                dists_cosine = cdist(image_feature, sketch_feature, 'cosine')
                dists_euclid = cdist(image_feature, sketch_feature, 'euclidean')

                rank_cosine = np.argsort(dists_cosine, 0)
                rank_euclid = np.argsort(dists_euclid, 0)

                for n in [5, 20, 100, 200]:
                    ranksn_cosine = rank_cosine[:n, :].T
                    ranksn_euclid = rank_euclid[:n, :].T

                    classesn_cosine = np.array([[image_label[i] == sketch_label[r] \
                                                for i in ranksn_cosine[r]] for r in range(len(ranksn_cosine))])
                    classesn_euclid = np.array([[image_label[i] == sketch_label[r] \
                                                for i in ranksn_euclid[r]] for r in range(len(ranksn_euclid))])

                    precision_cosine = np.mean(classesn_cosine)
                    precision_euclid = np.mean(classesn_euclid)
                    writer.add_scalar('Precision_{}/cosine'.format(n), precision_cosine, batch_acm)
                    writer.add_scalar('Precision_{}/euclid'.format(n), precision_euclid, batch_acm)
                    logger.info('Iter {}, Precision_{}/cosine {}'.format(batch_acm, n, precision_cosine))
                    logger.info('Iter {}, Precision_{}/euclid {}'.format(batch_acm, n, precision_euclid))
                if best_precision < precision_cosine:
                    patience = args.patience
                    best_precision = precision_cosine
                    best_iter = batch_acm
                    writer.add_scalar('Best/Precision_200', best_precision, best_iter)
                    logger.info('Iter {}, Best Precision_200 {}'.format(batch_acm, best_precision))
                    torch.save({'args':args, 'model':model.state_dict(), \
                        'optimizer':optimizer.state_dict()}, '{}/Best.pkl'.format(args.save_dir))
                else:
                    patience -= 1
            if patience <= 0:
                break

if __name__ == '__main__':
    args = parse_config()
    train(args)