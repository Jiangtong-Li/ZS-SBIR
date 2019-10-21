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
from package.dataset.data_cmd_translate import CMDTrans_data
from package.args.zsih_args import parse_config
from package.dataset.utils import make_logger
relu = nn.ReLU()

#ckpt = torch.load('/home/jiangtongli/Lab_Work/ZS-SBIR/model/zsih_test1/Iter_4000.pkl', map_location='cpu')

#args = ckpt['args']

data = CMDTrans_data('./data/256x256/sketch/tx_000100000000', 
                     './data/256x256/EXTEND_image_sketchy', 
                     './data/info/stats.csv', 
                     './data/GoogleNews-vectors-negative300.bin', 
                     './data/preprocessed/cm_trans_sketch_all_unpair/zs_packed.pkl', 
                     './data/preprocessed/cm_trans_sketch_all_unpair/CNN_feature_1024.h5py', 
                     './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_5568.h5py',
                     cvae=True, sample_time=1, paired=False)

dataLoader = DataLoader(dataset=data, batch_size=10000, num_workers=8, shuffle=True)

# model = ZSIM(args.hidden_size, args.hashing_bit, args.semantics_size, data.pretrain_embedding.float(), 
#              adj_scaler=args.adj_scaler, dropout=args.dropout, fix_cnn=args.fix_cnn, 
#              fix_embedding=args.fix_embedding)
# 
# optimizer = Adam(params=model.parameters(), lr=args.lr)
# model.load_state_dict(ckpt['model'])
# optimizer.load_state_dict(ckpt['optimizer'])

iter = 0
b_time = time.time()
for sketch, image_p, image_n, semantics in dataLoader:
    nonzeroi1 = np.mean(image_p.cpu().detach().numpy() == 0)
    nonzeros1 = np.mean(sketch.cpu().detach().numpy() == 0)
    image_p = relu(image_p)
    sketch = relu(sketch)
    nonzeroi2 = np.mean(image_p.cpu().detach().numpy() == 0)
    nonzeros2 = np.mean(sketch.cpu().detach().numpy() == 0)
    print(nonzeroi1)
    print(nonzeroi2)
    print()
    print(nonzeros1)
    print(nonzeros2)
    print()
    iter += 1
    if iter and iter % 100 == 0:
        print(iter)
        #print(sketch.shape)
        #print(image_p.shape)
        #print(image_n.shape)
        #print(semantics.shape)
    #semantics = semantics.long()
    #loss = model(sketch, image, semantics)
    #for i in range(sketch.shape[0]):
    #    sk = sketch[i].numpy().reshape(224, 224, 3)
    #    im = image[i].numpy().reshape(224, 224, 3)
    #    print(label[i])
    #    ims = np.vstack((np.uint8(sk), np.uint8(im)))
    #    cv2.imshow('test', ims)
    #    cv2.waitKey(3000)
    #semantics = semantics.long()
    #_loss = model(sketch, image, semantics)
    #loss = 0
    #for key, value in _loss.items():
    #    loss += value[0] * value[1]
    #loss.backward()
    #optimizer.step()
    #break
print(time.time()-b_time)
