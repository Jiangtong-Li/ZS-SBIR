from torch.utils.data import DataLoader
import cv2
import numpy as np
from package.dataset.data import Siamese_dataloader

data = Siamese_dataloader('./data/preprocessed/sketch_train/', './data/preprocessed/photo_train/', \
                                './data/preprocessed/stats_train.csv', './data/preprocessed/sketch_test/', \
                                './data/preprocessed/photo_test/', './data/preprocessed/stats_test.csv', \
                                './data/preprocessed/packed.pkl')
dataloader = DataLoader(data, num_workers=0, batch_size=4096, shuffle=True)

#data.test()
for sk, im, la in data.load_train(batch_size=5):

print()

for item in data.load_test_images(batch_size=1024):
    print(item[0].shape)

print()

for item in data.load_test_sketch(batch_size=1024):
    print(item[0].shape)