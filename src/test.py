from torch.utils.data import DataLoader
import cv2
import numpy as np
from package.dataset.data_zsih import ZSIH_dataloader
from package.model.zsim import ZSIM
from torch.optim import SGD

data = ZSIH_dataloader('/home/jiangtongli/Lab_Work/ZS-SBIR/data/256x256/sketch/tx_000100000000',
                          '/home/jiangtongli/Lab_Work/ZS-SBIR/data/256x256/EXTEND_image_sketchy',
                          '/home/jiangtongli/Lab_Work/ZS-SBIR/data/info/stats.csv',
                          '/home/jiangtongli/Lab_Work/ZS-SBIR/data/GoogleNews-vectors-negative300.bin',
                          '/home/jiangtongli/Lab_Work/ZS-SBIR/data/preprocessed/zsim_packed.pkl',
                          zs=True)
dataLoader = DataLoader(dataset=data, batch_size=3, num_workers=20,
        shuffle=True)

model = ZSIM(1024, 128, 300, data.pretrain_embedding.float())
optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.9)

print(len(data))
print(len(data.overall_class))
print(data.class2path_image.keys())
print(data.class2path_image_test.keys())
print(len(data.path2class_image.keys()))
print(len(data.path2class_image_test.keys()))
print(len(data.path2class_sketch.keys()))
print(len(data.path2class_sketch_test.keys()))

iter = 0
for sketch, image, semantics in dataLoader:
    iter += 1
    if iter and iter % 100 == 0:
        print(iter)
    #for i in range(sketch.shape[0]):
    #    sk = sketch[i].numpy().reshape(224, 224, 3)
    #    im = image[i].numpy().reshape(224, 224, 3)
    #    print(label[i])
    #    ims = np.vstack((np.uint8(sk), np.uint8(im)))
    #    cv2.imshow('test', ims)
    #    cv2.waitKey(3000)
    semantics = semantics.long()
    _loss = model(sketch, image, semantics)
    loss = 0
    for key, value in _loss.items():
        loss += value[0] * value[1]
    loss.backward()
    optimizer.step()
    break
