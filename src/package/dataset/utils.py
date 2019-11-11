import torch 
import re 
import csv
import logging


TEST_CLASS_SKETCHY = sorted(['bat', 'cabin', 'cow', 'dolphin', 'door',
                  'giraffe', 'helicopter', 'mouse', 'pear',
                  'raccoon', 'rhinoceros', 'saw', 'scissors',
                  'seagull', 'skyscraper', 'songbird', 'sword',
                  'tree', 'wheelchair', 'windmill', 'window'])

TRAIN_CLASS_SKETCHY = sorted(['squirrel', 'turtle', 'tiger', 'bicycle',
                   'crocodilian', 'frog', 'bread', 'hedgehog',
                   'hot-air_balloon', 'ape', 'elephant', 'geyser',
                   'chicken', 'ray', 'fan', 'hotdog', 'pizza',
                   'duck', 'piano', 'armor', 'axe', 'hammer',
                   'camel', 'horse', 'spider', 'kangaroo',
                   'mushroom', 'owl', 'seal', 'table', 'hermit_crab',
                   'zebra', 'car_(sedan)', 'shark', 'flower', 'guitar',
                   'bench', 'wine_bottle', 'fish', 'snail', 'deer',
                   'knife', 'airplane', 'sea_turtle', 'hat', 'eyeglasses',
                   'parrot', 'bee', 'tank', 'lion', 'swan', 'penguin',
                   'violin', 'rabbit', 'motorcycle', 'lobster', 'sheep',
                   'snake', 'shoe', 'hamburger', 'teddy_bear', 'pretzel',
                   'alarm_clock', 'church', 'ant', 'trumpet', 'candle',
                   'chair', 'hourglass', 'cat', 'scorpion', 'bear', 'dog',
                   'beetle', 'cannon', 'pig', 'cup', 'crab', 'pickup_truck',
                   'pineapple', 'apple', 'lizard', 'sailboat', 'spoon',
                   'umbrella', 'rocket', 'teapot', 'couch', 'butterfly',
                   'blimp', 'jellyfish', 'rifle', 'starfish', 'banana',
                   'wading_bird', 'bell', 'pistol', 'saxophone', 'strawberry',
                   'jack-o-lantern', 'castle', 'racket', 'harp', 'volcano'])
CLASS_SKETCHY = sorted(TEST_CLASS_SKETCHY + TRAIN_CLASS_SKETCHY)
assert len(CLASS_SKETCHY) == 125


# for backwards compatibility
TEST_CLASS = TEST_CLASS_SKETCHY
TRAIN_CLASS = TRAIN_CLASS_SKETCHY


TEST_CLASS_TUBERLIN = sorted(['arm', 'ashtray', 'axe', 'baseball bat', 'blimp', 'brain', 'bulldozer', 'bush',
                    'cake', 'chandelier', 'cloud', 'cow', 'crown', 'dolphin', 'donut', 'dragon', 'duck', 'eyeglasses',
                    'giraffe', 'grapes', 'grenade', 'head', 'head-phones', 'helicopter',
                    'horse', 'lightbulb', 'megaphone', 'microscope', 'mosquito', 'octopus', 'paper clip',
                    'pear', 'person walking', 'pigeon', 'pipe (for smoking)', 'pumpkin', 'rainbow',
                    'rooster', 'satellite', 'satellite dish', 'scissors', 'seagull', 'skateboard', 'skyscraper',
                    'snowboard', 'stapler', 'suitcase', 'sun', 'sword', 'tire', 'toilet', 'tomato',
                    'toothbrush', 'trousers', 'walkie talkie', 'windmill', 'wrist-watch', 'carrot', 'key', 'palm tree',
                    'parrot', 'rollerblades', 'suv', 'tree']) # 64


TRAIN_CLASS_TUBERLIN = sorted(['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'armchair', 'backpack', 'banana',
                               'barn', 'basket', 'bathtub', 'bear (animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench',
                               'bicycle', 'binoculars', 'book', 'bookshelf', 'boomerang', 'bottle opener', 'bowl',
                               'bread', 'bridge', 'bus', 'butterfly', 'cabinet', 'cactus', 'calculator', 'camel',
                               'camera', 'candle', 'cannon', 'canoe', 'car (sedan)', 'castle', 'cat', 'cell phone',
                               'chair', 'church', 'cigarette', 'comb', 'computer monitor', 'computer-mouse', 'couch',
                               'crab', 'crane (machine)', 'crocodile', 'cup', 'diamond', 'dog', 'door', 'door handle',
                               'ear', 'elephant', 'envelope', 'eye', 'face', 'fan', 'feather', 'fire hydrant', 'fish',
                               'flashlight', 'floor lamp', 'flower with stem', 'flying bird', 'flying saucer', 'foot',
                               'fork', 'frog', 'frying-pan', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
                               'hedgehog', 'helmet', 'hot air balloon', 'hot-dog', 'hourglass', 'house',
                               'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'keyboard', 'knife', 'ladder',
                               'laptop', 'leaf', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'mermaid',
                               'microphone', 'monkey', 'moon', 'motorbike', 'mouse (animal)', 'mouth', 'mug',
                               'mushroom', 'nose', 'owl', 'panda', 'parachute', 'parking meter', 'pen', 'penguin',
                               'person sitting', 'piano', 'pickup truck', 'pig', 'pineapple', 'pizza', 'potted plant',
                               'power outlet', 'present', 'pretzel', 'purse', 'rabbit', 'race car', 'radio', 'revolver',
                               'rifle', 'sailboat', 'santa claus', 'saxophone', 'scorpion', 'screwdriver', 'sea turtle',
                               'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skull', 'snail', 'snake', 'snowman',
                               'socks', 'space shuttle', 'speed-boat', 'spider', 'sponge bob', 'spoon', 'squirrel',
                               'standing bird', 'strawberry', 'streetlight', 'submarine', 'swan', 'syringe', 't-shirt',
                               'table', 'tablelamp', 'teacup', 'teapot', 'teddy-bear', 'telephone', 'tennis-racket',
                               'tent', 'tiger', 'tooth', 'tractor', 'traffic light', 'train', 'trombone', 'truck',
                               'trumpet', 'tv', 'umbrella', 'van', 'vase', 'violin', 'wheel',
                               'wheelbarrow', 'wine-bottle', 'wineglass', 'zebra'])

# use the file name of png folder
CLASS_TUBERLIN = sorted(TEST_CLASS_TUBERLIN + TRAIN_CLASS_TUBERLIN)
assert len(CLASS_TUBERLIN) == 250


IMAGE_SIZE = 224

SEMANTICS_REPLACE = dict([('car_(sedan)', 'car'), ('jack-o-lantern', 'pumpkin_lantern'),
                          ('wading_bird', 'bird'), ('hot-air_balloon','balloon'),
                          ('axe', 'hatchet'), ('wine_bottle', 'bottle')])


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


if __name__=='__main__':
    pass