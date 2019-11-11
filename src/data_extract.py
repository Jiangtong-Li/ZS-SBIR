from package.dataset.data_cmd_translate import image2features, image2features_pcyc

#i2f = image2features(image_dir='./data/256x256/EXTEND_image_sketchy',
#                     sketch_dir='./data/256x256/sketch/tx_000100000000',
#                     save_dir='./data/preprocessed/cm_trans_sketch_all_unpair_relued')

#i2f.image2features(10)
#i2f.pca_from_h5(4096, 'CNN_feature_5568_updated.h5py')
#i2f.pca_from_h5(3072, 'CNN_feature_5568_updated.h5py')
#i2f.pca_from_h5(2048, 'CNN_feature_5568_updated.h5py')
#i2f.pca_from_h5(1024, 'CNN_feature_5568_updated.h5py')
#i2f.pca_from_h5(512, 'CNN_feature_5568_updated.h5py')

i2f = image2features_pcyc(image_dir='./data/256x256/EXTEND_image_sketchy',
                     sketch_dir='./data/256x256/sketch/tx_000100000000',
                     save_dir='./data/preprocessed/cm_trans_sketchy_pcyc', 
                     image_model_path='./data/pcyc/CheckPoints/Sketchy/image', 
                     sketch_model_path='./data/pcyc/CheckPoints/Sketchy/sketch')

i2f.image2features(300)
