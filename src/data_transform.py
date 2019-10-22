import numpy as np 
import h5py
import pickle

base_path = './data/ZSSBIR_data/'
h5file = './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_4096_raw.h5py'

image_path = np.load(base_path+'image_ext_paths.npy')
image_feature = np.load(base_path+'image_ext_vgg_features.npy')
sketch_path = np.load(base_path+'sketch_paths.npy')
sketch_feature = np.load(base_path+'sketch_vgg_features.npy')

assert len(image_feature) == len(image_path)
assert len(sketch_feature) == len(sketch_path)

with open('./data/preprocessed/cm_trans_sketch_all_unpair/image_pathes', 'rb') as fin:
    raw_image_path = pickle.load(fin)

with open('./data/preprocessed/cm_trans_sketch_all_unpair/sketch_pathes', 'rb') as fin:
    raw_sketch_path = pickle.load(fin)

with h5py.File(h5file, 'w') as f:
    for idx, item in enumerate(image_path):
        new_item = './data/256x256/EXTEND_image_sketchy/' + '/'.join(item.decode().split('/')[-2:])
        if new_item not in raw_image_path:
            print(new_item)
        else:
            raw_image_path.remove(new_item)
        _ = f.create_dataset(new_item, data=image_feature[idx])
    print(len(raw_image_path))
    
    for idx, item in enumerate(sketch_path):
        new_item = './data/256x256/sketch/tx_000100000000/' + '/'.join(item.decode().split('/')[-2:])
        if new_item not in raw_sketch_path:
            print(new_item)
        else:
            raw_sketch_path.remove(new_item)
        _ = f.create_dataset(new_item, data=sketch_feature[idx])
    print(raw_sketch_path)
