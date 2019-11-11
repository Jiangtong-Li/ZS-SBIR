import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn

def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def cal_matrics(dists_cosine1, dists_cosine2, image_l, sketch_l, n=200):
    precision_list = list()
    mAP_list = list()
    lambda_list = list()
    for lambda_i in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        dists = lambda_i*dists_cosine1 + (1-lambda_i)*dists_cosine2
        rank = np.argsort(dists, 0)
        ranksn = rank[:n, :].T
        classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
        precision = np.mean(classesn)
        mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
        precision_list.append(precision)
        mAP_list.append(mAP)
        lambda_list.append(lambda_i)
    min_dists = np.minimum(dists_cosine1, dists_cosine2)
    rank = np.argsort(min_dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision_c = np.mean(classesn)
    mAP_c = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision_list, mAP_list, lambda_list, precision_c, mAP_c

def cal_matrics_single(image_f, image_l, sketch_f, sketch_l, n=200):
    dists = cdist(image_f, sketch_f, 'cosine')
    rank = np.argsort(dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision = np.mean(classesn)
    mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision, mAP