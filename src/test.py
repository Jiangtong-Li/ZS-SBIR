from torch.utils.data import DataLoader
import cv2
import numpy as np
from package.dataset.data import Siamese_dataloader, Siamese_dataloader_nozs

data = Siamese_dataloader_nozs('./data/256x256/sketch/tx_000100000000', './data/256x256/photo/tx_000100000000', \
                             './data/info/stats.csv', './data/preprocessed/nozs_packed.pkl')
dataLoader = DataLoader(dataset=data, batch_size=16)

print(len(data))

for sketch, image, label in dataLoader:
    for i in range(sketch.shape[0]):
        sk = sketch[i].numpy().reshape(224, 224, 3)
        im = image[i].numpy().reshape(224, 224, 3)
        print(label[i])
        ims = np.vstack((np.uint8(sk), np.uint8(im)))
        cv2.imshow('test', ims)
        cv2.waitKey(3000)