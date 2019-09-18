import torch
import numpy as np
import os 
import sys 
import csv

TEST_CLASS = ['bat', 'cabin', 'cow', 'dolphin', 'door', \
              'giraffe', 'helicopter', 'mouse', 'pear', \
              'raccoon', 'rhinoceros', 'saw', 'scissors', \
              'seagull', 'skyscraper', 'songbird', 'sword', \
              'tree', 'wheelchair', 'windmill', 'window']

def split_train_test(stats_file, sketch_dir, image_dir):
    stats_train = list()
    stats_test = list()
    with open(stats_file, 'r') as stats_in:
        stats_all_reader = csv.reader(stats_in)
        for line in stats_all_reader:
            if line[1] in TEST_CLASS:
                stats_test.append(line)
            else:
                stats_train.append(line)
    with open(stats_file.replace('csv', 'train.csv'), 'w') as stats_out:
        stats_train_writer = csv.writer(stats_out)
        for line in stats_train:
            stats_train_writer.writerow(line)
    
    with open(stats_file.replace('csv', 'test.csv'), 'w') as stats_out:
        stats_test_writer = csv.writer(stats_out)
        for line in stats_test:
            stats_test_writer.writerow(line)

if __name__ == '__main__':
    stats_file = './data/info/stats.csv'
    sketch_dir = None
    image_dir = None
    split_train_test(stats_file, None, None)