import os
import csv
import time
import random
import pickle
import gensim

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset
from torchvision.transforms import Normalize, ToTensor

from package.dataset.utils import match_filename, TEST_CLASS, TRAIN_CLASS, IMAGE_SIZE, SEMANTICS_REPLACE

class ZSIH_dataloader(torchDataset):
    def __init__(self, sketch_dir, image_dir, stats_file, embedding_file, loaded_data, normalize=False, zs=True):
        super(ZSIH_dataloader, self).__init__()
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.stats_file = stats_file
        self.embedding_file = embedding_file
        self.normalize = normalize
        self.loaded_data = loaded_data
        self.ToTensor = ToTensor()
        self.Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_class = TRAIN_CLASS
        self.test_class = TEST_CLASS
        self.overall_class = self.train_class | self.test_class
        self.zs = zs
        self.class2path_sketch = dict() # class: set(path) | for sketch | for train
        self.class2path_image = dict() # class: set(path) | for image | for train
        self.path2class_sketch = dict() # path: class | for sketch | for train
        self.path2class_image = dict() # path: class | for image  | for train
        self.class2path_sketch_test = dict() # class: set(path) | for sketch | for test
        self.class2path_image_test = dict() # class: set(path) | for image | for test
        self.path2class_sketch_test = dict() # path: class | for sketch | for test
        self.path2class_image_test = dict() # path: class | for image | for test
        self.id2path = list() # path list corresponding to path2class_sketch 
        self.loaded_image = dict() # path: loaded image 
        self.class2id = dict() # class name to semantics index
        self.id2class = list() # semantics index to class name
        self.pretrain_embedding = np.zeros((len(list(self.train_class)) + len(list(self.test_class)), 300))
        self.load()
        random.shuffle(self.id2path)

    def __getitem__(self, index):
        sketch = self.load_each_image_use(self.id2path[index])
        cla = self.path2class_sketch[self.id2path[index]]
        image, _path = self.pair_similar(cla)
        semantics = np.zeros((1))
        semantics[0] = self.class2id[cla]
        return sketch, image, semantics

    def __len__(self):
        return len(self.id2path)

    def pair_similar(self, cls):
        path_list = list(self.class2path_image[cls])
        path = random.choice(path_list)
        return self.load_each_image_use(path), path

    def load_each_image_use(self, path):
        image = cv2.imread(path)
        try:
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], 2)
        except:
            print(path)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = self.ToTensor(image)
        image = self.Normalize(image)
        return image

    def load_test_images(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_image_test.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_image_test[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label

    def load_test_sketch(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_sketch_test.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_sketch_test[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label

    def load_train_images(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_image.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_image[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label

    def load_train_sketch(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_sketch.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_sketch[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label

    def load(self):
        if os.path.exists(self.loaded_data):
            with open(self.loaded_data, 'rb') as f:
                preloaded_data = pickle.load(f)
            # Semantics part
            self.class2id = preloaded_data['class2id']
            self.id2class = preloaded_data['id2class']
            self.pretrain_embedding = preloaded_data['pretrain_embedding']
            # Train part
            self.path2class_sketch = preloaded_data['path2class_sketch']
            self.class2path_sketch = preloaded_data['class2path_sketch']
            self.path2class_image = preloaded_data['path2class_image']
            self.class2path_image = preloaded_data['class2path_image']
            self.id2path = preloaded_data['id2path']
            # Test part
            self.class2path_sketch_test = preloaded_data['class2path_sketch_test']
            self.class2path_image_test = preloaded_data['class2path_image_test']
            self.path2class_sketch_test = preloaded_data['path2class_sketch_test']
            self.path2class_image_test = preloaded_data['path2class_image_test']
            return

        # train part
        for idx, cla in enumerate(self.overall_class):
            # semantics part
            self.class2id[cla] = idx
            self.id2class.append(cla)
            # image part
            image_cla_dir = [os.path.join(self.image_dir, cla, fname) for fname in os.listdir(os.path.join(self.image_dir, cla))]
            sketch_cla_dir = [os.path.join(self.sketch_dir, cla, fname) for fname in os.listdir(os.path.join(self.sketch_dir, cla))]
            if self.zs:
                if cla in self.train_class:
                    if cla not in self.class2path_image:
                        self.class2path_image[cla] = list()
                    if cla not in self.class2path_sketch:
                        self.class2path_sketch[cla] = list()
                    for path in image_cla_dir:
                        self.path2class_image[path] = cla
                        self.class2path_image[cla].append(path)
                    for path in sketch_cla_dir:
                        self.path2class_sketch[path] =cla
                        self.id2path.append(path)
                        self.class2path_sketch[cla].append(path)
                else:
                    if cla not in self.class2path_image_test:
                        self.class2path_image_test[cla] = list()
                    if cla not in self.class2path_sketch_test:
                        self.class2path_sketch_test[cla] = list()
                    for path in image_cla_dir:
                        self.path2class_image_test[path] = cla
                        self.class2path_image_test[cla].append(path)
                    for path in sketch_cla_dir:
                        self.path2class_sketch_test[path] =cla
                        self.class2path_sketch_test[cla].append(path)
            else:
                if cla in self.test_class:
                    random.shuffle(image_cla_dir)
                    random.shuffle(sketch_cla_dir)
                    train_im = image_cla_dir[:int(0.5*len(image_cla_dir))]
                    test_im = image_cla_dir[int(0.5*len(image_cla_dir)):]
                    train_sk = sketch_cla_dir[:int(0.5*len(sketch_cla_dir))]
                    test_sk = sketch_cla_dir[int(0.5*len(sketch_cla_dir)):]
                    self.class2path_image[cla] = train_im
                    self.class2path_sketch[cla] = train_sk
                    self.class2path_image_test[cla] = test_im
                    self.class2path_sketch_test[cla] = test_sk
                    for path in train_im:
                        self.path2class_image[path] = cla
                    for path in train_sk:
                        self.path2class_sketch[path] = cla
                        self.id2path.append(path)
                    for path in test_im:
                        self.path2class_image_test[path] = cla
                    for path in test_sk:
                        self.path2class_sketch_test[path] = cla
        
        # load embedding
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        for idx, item in enumerate(self.id2class):
            if item not in word2vec:
                item = SEMANTICS_REPLACE[item]
            self.pretrain_embedding[idx] = word2vec[item]
        self.pretrain_embedding = torch.from_numpy(self.pretrain_embedding)
        
        assert len(self.id2class) == 125
        assert len(self.path2class_sketch.keys()) == len(self.id2path)
        preloaded_data = dict()
        # Semantics part
        preloaded_data['class2id'] = self.class2id
        preloaded_data['id2class'] = self.id2class
        preloaded_data['pretrain_embedding'] = self.pretrain_embedding
        # Train part
        preloaded_data['path2class_sketch'] = self.path2class_sketch
        preloaded_data['class2path_sketch'] = self.class2path_sketch
        preloaded_data['path2class_image'] = self.path2class_image
        preloaded_data['class2path_image'] = self.class2path_image
        preloaded_data['id2path'] = self.id2path
        # Test part
        preloaded_data['class2path_sketch_test'] = self.class2path_sketch_test
        preloaded_data['class2path_image_test'] = self.class2path_image_test
        preloaded_data['path2class_sketch_test'] = self.path2class_sketch_test
        preloaded_data['path2class_image_test'] = self.path2class_image_test

        with open(self.loaded_data, 'wb') as f:
            pickle.dump(preloaded_data, f)
        return
