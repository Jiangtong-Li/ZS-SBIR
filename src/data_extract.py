from package.dataset.data_cmd_translate import image2features

i2f = image2features(image_dir='./data/256x256/EXTEND_image_sketchy',
                     sketch_dir='./data/256x256/sketch/tx_000100000000',
                     save_dir='./data/preprocessed/cm_trans_sketch_all_unpair')

#i2f.image2features(300)
#i2f.pca(4096)
#i2f.pca(3072)
#i2f.pca(2048)
i2f.pca(1024)
i2f.pca(512)
