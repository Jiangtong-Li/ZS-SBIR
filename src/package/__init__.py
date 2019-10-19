import numpy as np
from scipy.spatial.distance import cdist

def cal_matrics(image_f1, image_f2, image_l, sketch_f1, sketch_f2, sketch_l, lambda_i = 0.5, n=200):

    dists_cosine1 = cdist(image_f1, sketch_f1, 'cosine')
    dists_cosine2 = cdist(image_f2, sketch_f2, 'cosine')
    precision_b = 0
    mAP_b = 0.
    lambda_b = 0.
    for lambda_i in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
        dists = lambda_i*dists_cosine1 + (1-lambda_i)*dists_cosine2
        rank = np.argsort(dists, 0)
        ranksn = rank[:n, :].T
        classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
        precision = np.mean(classesn)
        # Cal MAP
        """
        Test case: np.array([[1,0,1,0,0,1,0,0,1,1],[0,1,0,0,1,0,1,0,0,0]])
        Answer: 0.53
        """
        mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
        if precision > precision_b:
            precision_b = precision
            lambda_b = lambda_i
        if mAP > mAP_b:
            mAP_b = mAP
    return precision_b, mAP_b, lambda_b

def cal_matrics_single(image_f, image_l, sketch_f, sketch_l, n=200):

    dists = cdist(image_f, sketch_f, 'cosine')
    precision_b = 0
    mAP_b = 0.
    rank = np.argsort(dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision = np.mean(classesn)
    # Cal MAP
    """
    Test case: np.array([[1,0,1,0,0,1,0,0,1,1],[0,1,0,0,1,0,1,0,0,0]])
    Answer: 0.53
    """
    mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision, mAP