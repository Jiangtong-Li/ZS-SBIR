import torch
import numpy as np
import os 
import sys
import shutil
import csv
import re

TEST_CLASS = set(['bat', 'cabin', 'cow', 'dolphin', 'door', \
              'giraffe', 'helicopter', 'mouse', 'pear', \
              'raccoon', 'rhinoceros', 'saw', 'scissors', \
              'seagull', 'skyscraper', 'songbird', 'sword', \
              'tree', 'wheelchair', 'windmill', 'window'])
TRAIN_CLASS = set()


def split_train_test(stats_file, sketch_dir, image_dir, processed_dir):
    stats_train = list()
    stats_test = list()
    class2sketch_dirlist = dict() # dict -> list
    class2image_dirlist = dict() # dict -> list
    class_imageid_train = dict() # dixt -> set
    class_imageid_test = dict() # dict -> set
    stats_train_file = os.path.join(processed_dir, 'stats_train.csv')
    stats_test_file = os.path.join(processed_dir, 'stats_test.csv')
    sketch_train_dir = os.path.join(processed_dir, 'sketch_train')
    sketch_test_dir = os.path.join(processed_dir, 'sketch_test')
    image_train_dir = os.path.join(processed_dir, 'photo_train')
    image_test_dir = os.path.join(processed_dir, 'photo_test')
    if not os.path.exists(sketch_train_dir):
        os.makedirs(sketch_train_dir)
    if not os.path.exists(sketch_test_dir):
        os.makedirs(sketch_test_dir)
    if not os.path.exists(image_train_dir):
        os.makedirs(image_train_dir)
    if not os.path.exists(image_test_dir):
        os.makedirs(image_test_dir)
    with open(stats_file, 'r') as stats_in:
        stats_all_reader = csv.reader(stats_in)
        header = next(stats_in)
        for line in stats_all_reader:
            if line[1] in TEST_CLASS:
                if line[1] not in class_imageid_test:
                    class_imageid_test[line[1]] = set()
                class_imageid_test[line[1]].add(line[2])
                stats_test.append(line)
            else:
                TRAIN_CLASS.add(line[1])
                if line[1] not in class_imageid_train:
                    class_imageid_train[line[1]] = set()
                class_imageid_train[line[1]].add(line[2])
                stats_train.append(line)

    assert TRAIN_CLASS & TEST_CLASS == set()

    with open(stats_train_file, 'w') as stats_out:
        stats_train_writer = csv.writer(stats_out)
        stats_train_writer.writerow(header)
        for line in stats_train:
            stats_train_writer.writerow(line)

    with open(stats_test_file, 'w') as stats_out:
        stats_test_writer = csv.writer(stats_out)
        stats_test_writer.writerow(header)
        for line in stats_test:
            stats_test_writer.writerow(line)

    for item in list(TEST_CLASS | TRAIN_CLASS):
        class2image_dirlist[item] = os.listdir(os.path.join(image_dir, item.replace(' ', '_')))
        class2sketch_dirlist[item] = os.listdir(os.path.join(sketch_dir, item.replace(' ', '_')))
    
    for class_name in TEST_CLASS:
        for id_name in class_imageid_test[class_name]:
            pattern = id_name + '*'
            id_name_imagelist = [file for file in class2image_dirlist[class_name] if re.match(pattern, file)]
            id_name_sketchlist = [file for file in class2sketch_dirlist[class_name] if re.match(pattern, file)]
            for item in id_name_imagelist:
                shutil.copy(os.path.join(image_dir, class_name.replace(' ', '_'), item), \
                            image_test_dir)
            for item in id_name_sketchlist:
                shutil.copy(os.path.join(sketch_dir, class_name.replace(' ', '_'), item), \
                            sketch_test_dir)

    return 

if __name__ == '__main__':
    stats_file = './data/info/stats.csv'
    sketch_dir = './data/256x256/sketch/tx_000100000000/'
    image_dir = './data/256x256/photo/tx_000100000000/'
    processed_dir = './data/preprocessed/'
    split_train_test(stats_file, sketch_dir, image_dir, processed_dir)
