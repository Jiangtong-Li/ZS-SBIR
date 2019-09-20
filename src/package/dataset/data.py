import cv2
import csv
import torch
from torch.utils.data import Dataset as torchDataset

from package.dataset.utils import match_filename, TEST_CLASS, TRAIN_CLASS

class Siamese_dataloader_train(torchDataset):
    def __init__(self, sketch_dir, image_dir, stats_file):
        super(Siamese_dataloader_train, self).__init__()
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.stats_file = stats_file
        self.loaded_image = dict() # path: loaded image 
        self.class2id = dict() # class: set(id)
        self.path2class_sketch = dict() # path: class | for sketch 
        self.path2class_image = dict() # path: class | for image
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def pair_similar(self):
        pass

    def pair_dis_similar(self):
        pass

    def load(self):
        """
        this function will build the self.loaded_image, self.class2id, 
            self.path2class_sketch and self.path2class_image
        """
        with open(self.stats_file, 'r') as stats_in:
            stats_in_reader = csv.reader(stats_in)
            _header = next(stats_in_reader)
            for line in stats_in_reader:
                if line[1] not in self.class2id:
                    self.class2id[line[1]] = set()
                self.class2id[line[1]].add(line[2])
        
    
    def load_each_image(self, path):
        pass