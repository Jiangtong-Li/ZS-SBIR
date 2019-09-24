from torch.utils.data import DataLoader
import cv2
import numpy as np
from package.dataset.data import Siamese_dataloader

data = Siamese_dataloader('./data/preprocessed/sketch_train/', './data/preprocessed/photo_train/', \
                                './data/preprocessed/stats_train.csv', './data/preprocessed/sketch_test/', \
                                './data/preprocessed/photo_test/', './data/preprocessed/stats_test.csv', \
                                './data/preprocessed/packed.pkl')
dataloader = DataLoader(data, num_workers=0, batch_size=4096, shuffle=True)
