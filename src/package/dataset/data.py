import cv2
import csv
import torch
from torch.utils.data import Dataset as torchDataset

class Siamese_dataloader(torchDataset):
    def __init__(self, sketch_dir, image_dir, stats_info):
        super(Siamese_dataloader, self).__init__()
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.stats_info = stats_info
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass