import numpy as np 
import h5py
import pickle

# base_path = './data/ZSSBIR_data/'
# h5file = './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_4096_raw_updated.h5py'

# image_ext_path = np.load(base_path+'image_ext_paths.npy')
# image_ext_feature = np.load(base_path+'image_ext_vgg_features.npy')
# image_path = np.load(base_path+'image_paths.npy')
# image_feature = np.load(base_path+'image_vgg_features.npy')
# sketch_path = np.load(base_path+'sketch_paths.npy')
# sketch_feature = np.load(base_path+'sketch_vgg_features.npy')

# assert len(image_ext_feature) == len(image_ext_path)
# assert len(sketch_feature) == len(sketch_path)

with open('./data/preprocessed/cm_trans_sketch_all_unpair/image_pathes', 'rb') as fin:
    raw_image_path = pickle.load(fin)

with open('./data/preprocessed/cm_trans_sketch_all_unpair/sketch_pathes', 'rb') as fin:
    raw_sketch_path = pickle.load(fin)

# origin_image_path = list()

# with h5py.File(h5file, 'w') as f:
#     for idx, item in enumerate(image_path):
#         new_item = './data/256x256/EXTEND_image_sketchy/' + '/'.join(item.decode().split('/')[-2:])
#         if new_item not in raw_image_path:
#             print(new_item)
#         else:
#             raw_image_path.remove(new_item)
#             origin_image_path.append(new_item)
#         _ = f.create_dataset(new_item, data=image_feature[idx])

#     for idx, item in enumerate(image_ext_path):
#         new_item = './data/256x256/EXTEND_image_sketchy/' + '/'.join(item.decode().split('/')[-2:])
#         if new_item not in raw_image_path:
#             if new_item not in origin_image_path:
#                 print(new_item)
#             else:
#                 continue
#         else:
#             raw_image_path.remove(new_item)
#         _ = f.create_dataset(new_item, data=image_ext_feature[idx])
#     print(len(raw_image_path))
    
#     for idx, item in enumerate(sketch_path):
#         new_item = './data/256x256/sketch/tx_000100000000/' + '/'.join(item.decode().split('/')[-2:])
#         if new_item not in raw_sketch_path:
#             print(new_item)
#         else:
#             raw_sketch_path.remove(new_item)
#         _ = f.create_dataset(new_item, data=sketch_feature[idx])
#     print(raw_sketch_path)

h5file_5568 = './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_5568.h5py'
h5file_4096_updated = './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_4096_raw_updated.h5py'
h5file_5568_updated = './data/preprocessed/cm_trans_sketch_all_unpair_relued/CNN_feature_5568_updated.h5py'

with h5py.File(h5file_5568, 'r') as f_5568:
    with h5py.File(h5file_4096_updated, 'r') as f_4096_updated:
        with h5py.File(h5file_5568_updated, 'w') as f_5568_updated:
            for item in raw_sketch_path:
                try:
                    data_5568 = f_5568[item][...]
                    data_4096_updated = f_4096_updated[item][...]
                except:
                    print(item)
                    continue
                data_5568_updated = np.concatenate((data_5568[:-4096], data_4096_updated))
                _ = f_5568_updated.create_dataset(item, data=data_5568_updated)
            for item in raw_image_path:
                try:
                    data_5568 = f_5568[item][...]
                    data_4096_updated = f_4096_updated[item][...]
                except:
                    print(item)
                    continue
                data_5568_updated = np.concatenate((data_5568[:-4096], data_4096_updated))
                _ = f_5568_updated.create_dataset(item, data=data_5568_updated)
