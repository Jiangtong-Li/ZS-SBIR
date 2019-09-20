import torch 
import re 
import csv

TEST_CLASS = set(['bat', 'cabin', 'cow', 'dolphin', 'door', \
                  'giraffe', 'helicopter', 'mouse', 'pear', \
                  'raccoon', 'rhinoceros', 'saw', 'scissors', \
                  'seagull', 'skyscraper', 'songbird', 'sword', \
                  'tree', 'wheelchair', 'windmill', 'window'])
TRAIN_CLASS = set(['squirrel', 'turtle', 'tiger', 'bicycle', \
                   'crocodilian', 'frog', 'bread', 'hedgehog', \
                   'hot-air balloon', 'ape', 'elephant', 'geyser', \
                   'chicken', 'ray', 'fan', 'hotdog', 'pizza', \
                   'duck', 'piano', 'armor', 'axe', 'hammer', \
                   'camel', 'horse', 'spider', 'kangaroo', \
                   'mushroom', 'owl', 'seal', 'table', 'hermit crab', \
                   'zebra', 'car (sedan)', 'shark', 'flower', 'guitar', \
                   'bench', 'wine bottle', 'fish', 'snail', 'deer', \
                   'knife', 'airplane', 'sea turtle', 'hat', 'eyeglasses', \
                   'parrot', 'bee', 'tank', 'lion', 'swan', 'penguin', \
                   'violin', 'rabbit', 'motorcycle', 'lobster', 'sheep', \
                   'snake', 'shoe', 'hamburger', 'teddy bear', 'pretzel', \
                   'alarm clock', 'church', 'ant', 'trumpet', 'candle', \
                   'chair', 'hourglass', 'cat', 'scorpion', 'bear', 'dog', \
                   'beetle', 'cannon', 'pig', 'cup', 'crab', 'pickup truck', \
                   'pineapple', 'apple', 'lizard', 'sailboat', 'spoon', \
                   'umbrella', 'rocket', 'teapot', 'couch', 'butterfly', \
                   'blimp', 'jellyfish', 'rifle', 'starfish', 'banana', \
                   'wading bird', 'bell', 'pistol', 'saxophone', 'strawberry', \
                   'jack-o-lantern', 'castle', 'racket', 'harp', 'volcano'])

def match_filename(pattern, listed_dir):
    return [f for f in listed_dir if re.match(pattern, f)]