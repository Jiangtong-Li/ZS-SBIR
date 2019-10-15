from package.dataset.data_cm_translate import image2features

i2f = image2features(image_dir='/home/jiangtongli/Lab_Work/ZS-SBIR/data/256x256/EXTEND_image_sketchy',
                     sketch_dir='/home/jiangtongli/Lab_Work/ZS-SBIR/data/256x256/sketch/tx_000100000000',
                     save_dir='/home/jiangtongli/Lab_Work/ZS-SBIR/data/256x256/')

i2f.image2features(300)
i2f.pca(4096)
i2f.pca(3072)
i2f.pca(2048)