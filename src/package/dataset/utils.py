import torch 
import re 
import csv
import logging

TEST_CLASS = set(['bat', 'cabin', 'cow', 'dolphin', 'door', \
                  'giraffe', 'helicopter', 'mouse', 'pear', \
                  'raccoon', 'rhinoceros', 'saw', 'scissors', \
                  'seagull', 'skyscraper', 'songbird', 'sword', \
                  'tree', 'wheelchair', 'windmill', 'window'])
TRAIN_CLASS = set(['squirrel', 'turtle', 'tiger', 'bicycle', \
                   'crocodilian', 'frog', 'bread', 'hedgehog', \
                   'hot-air_balloon', 'ape', 'elephant', 'geyser', \
                   'chicken', 'ray', 'fan', 'hotdog', 'pizza', \
                   'duck', 'piano', 'armor', 'axe', 'hammer', \
                   'camel', 'horse', 'spider', 'kangaroo', \
                   'mushroom', 'owl', 'seal', 'table', 'hermit_crab', \
                   'zebra', 'car_(sedan)', 'shark', 'flower', 'guitar', \
                   'bench', 'wine_bottle', 'fish', 'snail', 'deer', \
                   'knife', 'airplane', 'sea_turtle', 'hat', 'eyeglasses', \
                   'parrot', 'bee', 'tank', 'lion', 'swan', 'penguin', \
                   'violin', 'rabbit', 'motorcycle', 'lobster', 'sheep', \
                   'snake', 'shoe', 'hamburger', 'teddy_bear', 'pretzel', \
                   'alarm_clock', 'church', 'ant', 'trumpet', 'candle', \
                   'chair', 'hourglass', 'cat', 'scorpion', 'bear', 'dog', \
                   'beetle', 'cannon', 'pig', 'cup', 'crab', 'pickup_truck', \
                   'pineapple', 'apple', 'lizard', 'sailboat', 'spoon', \
                   'umbrella', 'rocket', 'teapot', 'couch', 'butterfly', \
                   'blimp', 'jellyfish', 'rifle', 'starfish', 'banana', \
                   'wading_bird', 'bell', 'pistol', 'saxophone', 'strawberry', \
                   'jack-o-lantern', 'castle', 'racket', 'harp', 'volcano'])

IMAGE_SIZE = 224

def match_filename(pattern, listed_dir):
    return [f for f in listed_dir if re.match(pattern, f)]

def make_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = log_file
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('logfile = {}'.format(logfile))
    return logger
