import os
import csv
import time
import random
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset
from torchvision.transforms import Normalize

from package.dataset.utils import match_filename, TEST_CLASS, TRAIN_CLASS, IMAGE_SIZE

class Siamese_dataloader(torchDataset):
    def __init__(self, sketch_dir, image_dir, stats_file, loaded_data, normalize=False, zs=True):
        super(Siamese_dataloader, self).__init__()
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.stats_file = stats_file
        self.normalize = normalize
        self.loaded_data = loaded_data
        self.Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_class = TRAIN_CLASS
        self.test_class = TEST_CLASS
        if zs:
            self.train_id, self.test_id = self.split_zs()
        else:
            self.train_id, self.test_id = self.split_nozs()
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
        self.load()
    
    def split_zs(self):
        cla = None
        id_list = list()
        train_ids = dict()
        test_ids = dict()
        with open(self.stats_file, 'r') as fin:
            reader = csv.reader(fin)
            _header = next(reader)
            for item in reader:
                if item[1].replace(' ', '_') != cla and cla is not None:
                    id_list = list(set(id_list))
                    random.shuffle(id_list)
                    if cla in self.train_class:
                        train_ids[cla] = id_list
                    else:
                        test_ids[cla] = id_list
                    id_list = list()
                    cla = item[1].replace(' ', '_')
                elif item[1] != cla and cla is None:
                    cla = item[1].replace(' ', '_')
                id_list.append(item[2])
            id_list = list(set(id_list))
            random.shuffle(id_list)
            if cla in self.train_class:
                train_ids[cla] = id_list
            else:
                test_ids[cla] = id_list
        return train_ids, test_ids

    def split_nozs(self):
        cla = None
        id_list = list()
        train_ids = dict()
        test_ids = dict()
        with open(self.stats_file, 'r') as fin:
            reader = csv.reader(fin)
            _header = next(reader)
            for item in reader:
                if item[1].replace(' ', '_') != cla and cla is not None:
                    id_list = list(set(id_list))
                    random.shuffle(id_list)
                    train_ids[cla] = id_list[:int(0.8*len(id_list))]
                    test_ids[cla] = id_list[int(0.8*len(id_list)):]
                    id_list = list()
                    cla = item[1].replace(' ', '_')
                elif item[1] != cla and cla is None:
                    cla = item[1].replace(' ', '_')
                id_list.append(item[2])
            id_list = list(set(id_list))
            random.shuffle(id_list)
            train_ids[cla] = id_list[:int(0.8*len(id_list))]
            test_ids[cla] = id_list[int(0.8*len(id_list)):]
        return train_ids, test_ids
    
    def __getitem__(self, index):
        sketch = self.load_each_image_use(self.id2path[int(index/2)])
        label = np.zeros(1)
        if index % 2 == 0:
            label[0] = 1
            image, _path = self.pair_similar(self.path2class_sketch[self.id2path[int(index/2)]])
        else:
            label[0] = 0
            image, _path = self.pair_dis_similar(self.path2class_sketch[self.id2path[int(index/2)]])
        return sketch, image, label

    def __len__(self):
        return 2*len(self.id2path)
    
    def pair_similar(self, cls):
        path_list = list(self.class2path_image[cls])
        path = random.choice(path_list)
        return self.load_each_image_use(path), path

    def pair_dis_similar(self, cls):
        class_list = list(self.class2path_image.keys())
        class_list.remove(cls)
        choice = random.choice(class_list)
        path_list = list(self.class2path_image[choice])
        path = random.choice(path_list)
        return self.load_each_image_use(path), path
    
    def load_each_image_use(self, path):
        if path in self.loaded_image:
            image = self.loaded_image[path]
        else:
            image = cv2.imread(path)
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], 2)
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            self.loaded_image[path] = image
        image = image.reshape(3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        image = torch.Tensor(image)
        image = image/255.0
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

    def load(self):
        if os.path.exists(self.loaded_data):
            with open(self.loaded_data, 'rb') as f:
                preloaded_data = pickle.load(f)
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
        for cla, id_list in self.train_id.items():
            image_cla_dir = [os.path.join(self.image_dir, cla, fname) for fname in os.listdir(os.path.join(self.image_dir, cla))]
            sketch_cla_dir = [os.path.join(self.sketch_dir, cla, fname) for fname in os.listdir(os.path.join(self.sketch_dir, cla))]
            if cla not in self.class2path_sketch:
                self.class2path_sketch[cla] = list()
            if cla not in self.class2path_image:
                self.class2path_image[cla] = list()
            for id in id_list:
                pattern = '.*' + id + '.*'
                # sketch part
                id_files = match_filename(pattern, sketch_cla_dir)
                for id_file in id_files:
                    self.class2path_sketch[cla].append(id_file)
                    self.path2class_sketch[id_file] = cla
                    self.id2path.append(id_file)
                # image part
                id_files = match_filename(pattern, image_cla_dir)
                for id_file in id_files:
                    self.class2path_image[cla].append(id_file)
                    self.path2class_image[id_file] = cla
        # test part
        for cla, id_list in self.test_id.items():
            image_cla_dir = [os.path.join(self.image_dir, cla, fname) for fname in (os.listdir(os.path.join(self.image_dir, cla)))]
            sketch_cla_dir = [os.path.join(self.sketch_dir, cla, fname) for fname in (os.listdir(os.path.join(self.sketch_dir, cla)))]
            if cla not in self.class2path_sketch_test:
                self.class2path_sketch_test[cla] = list()
            if cla not in self.class2path_image_test:
                self.class2path_image_test[cla] = list()
            for id in id_list:
                pattern = '.*' + id + '.*'
                # sketch part
                id_files = match_filename(pattern, sketch_cla_dir)
                for id_file in id_files:
                    self.class2path_sketch_test[cla].append(id_file)
                    self.path2class_sketch_test[id_file] = cla
                # image part
                id_files = match_filename(pattern, image_cla_dir)
                for id_file in id_files:
                    self.class2path_image_test[cla].append(id_file)
                    self.path2class_image_test[id_file] = cla
        
        preloaded_data = dict()
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