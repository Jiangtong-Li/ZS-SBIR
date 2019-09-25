class Siamese_dataloader(torchDataset):
    def __init__(self, sketch_dir_train, image_dir_train, stats_file_train, \
                 sketch_dir_test, image_dir_test, stats_file_test, loaded_data, normalize=False):
        super(Siamese_dataloader, self).__init__()
        self.sketch_dir_train = sketch_dir_train
        self.image_dir_train = image_dir_train
        self.stats_file_train = stats_file_train
        self.sketch_dir_test = sketch_dir_test
        self.image_dir_test = image_dir_test
        self.stats_file_test = stats_file_test
        self.normalize = normalize
        self.loaded_data = loaded_data
        self.Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.sketch_files_train = [os.path.join(self.sketch_dir_train, item) for item in os.listdir(self.sketch_dir_train)]
        self.image_files_train = [os.path.join(self.image_dir_train, item) for item in os.listdir(self.image_dir_train)]
        self.sketch_files_test = [os.path.join(self.sketch_dir_test, item) for item in os.listdir(self.sketch_dir_test)]
        self.image_files_test = [os.path.join(self.image_dir_test, item) for item in os.listdir(self.image_dir_test)]
        self.class2id = dict()
        self.id2path = list() # path list corresponding to path2class_sketch 
        self.loaded_image = dict() # path: loaded image 
        self.class2imgid = dict() # class: set(id)
        self.class2path_sketch = dict() # class: set(path) | for sketch | for train
        self.class2path_image = dict() # class: set(path) | for image | for train
        self.path2class_sketch = dict() # path: class | for sketch | for train
        self.path2class_image = dict() # path: class | for image  | for train
        self.class2path_sketch_test = dict() # class: set(path) | for sketch | for test
        self.class2path_image_test = dict() # class: set(path) | for image | for test
        self.path2class_sketch_test = dict() # path: class | for sketch | for test
        self.path2class_image_test = dict() # path: class | for image | for test
        self.load()

    def __getitem__(self, index):
        sketch = self.load_each_image_use(self.id2path[int(index/2)])
        label = np.zeros(1)
        if index % 2 == 0:
            label[0] = 1
            image, path = self.pair_similar(self.path2class_sketch[self.id2path[int(index/2)]])
        else:
            label[0] = 0
            image, path = self.pair_dis_similar(self.path2class_sketch[self.id2path[int(index/2)]])
        return sketch, image, label

    def __len__(self):
        return 2*len(self.id2path)

    def load_train(self, batch_size=512):
        image = []
        sketch = []
        label = []
        ranges = list(range(2*len(self.id2path)))
        random.shuffle(ranges)
        for index in ranges:
            path_sketch = self.id2path[int(index/2)]
            sketch.append(self.load_each_image_use(path_sketch))
            if index % 2 == 0:
                label.append(1)
                image_tmp, path = self.pair_similar(self.path2class_sketch[path_sketch])
            else:
                label.append(0)
                image_tmp, path = self.pair_dis_similar(self.path2class_sketch[path_sketch])
            image.append(image_tmp)
            if len(image) == batch_size:
                yield torch.from_numpy(np.asarray(sketch)), torch.from_numpy(np.asarray(image)), torch.from_numpy(np.asarray(label))
                image = []
                sketch = []
                label = []
        yield torch.from_numpy(np.asarray(sketch)), torch.from_numpy(np.asarray(image)), torch.from_numpy(np.asarray(label))

    def load_same_class_image(self, batch_size=512): 
        image = []
        sketch = []
        label = []
        ranges = list(range(2*len(self.id2path)))
        random.shuffle(ranges)
        for index in ranges:
            path_sketch = self.id2path[int(index/2)]
            #sketch.append(self.load_each_image_use(path_sketch))
            if index % 2 == 0:
                label.append(1)
                sketch_tmp, path = self.pair_similar(self.path2class_sketch[path_sketch])
                image_tmp, path = self.pair_similar(self.path2class_sketch[path_sketch])
            else:
                label.append(0)
                sketch_tmp, path = self.pair_similar(self.path2class_sketch[path_sketch])
                image_tmp, path = self.pair_dis_similar(self.path2class_sketch[path_sketch])
            sketch.append(sketch_tmp)
            image.append(image_tmp)
            if len(image) == batch_size:
                yield torch.stack(sketch), torch.stack(image), torch.from_numpy(np.asarray(label))
                image = []
                sketch = []
                label = []
        yield torch.stack(sketch), torch.stack(image), torch.from_numpy(np.asarray(label))

    def pair_similar(self, cls):
        path_list = list(self.class2path_image[cls])
        path = random.choice(path_list)
        return self.load_each_image_use(path), path

    def pair_dis_similar(self, cls):
        class_list = list(self.class2imgid.keys())
        class_list.remove(cls)
        path_list = list(self.class2path_image[random.choice(class_list)])
        path = random.choice(path_list)
        return self.load_each_image_use(path), path

    def load(self):
        """
        this function will build the self.loaded_image, self.class2imgid, 
            self.path2class_sketch and self.path2class_image
        """
        if os.path.exists(self.loaded_data):
            with open(self.loaded_data, 'rb') as f:
                preloaded_data = pickle.load(f)
            # Train part
            self.class2imgid = preloaded_data['class2imgid']
            self.path2class_sketch = preloaded_data['path2class_sketch']
            self.class2path_sketch = preloaded_data['class2path_sketch']
            self.path2class_image = preloaded_data['path2class_image']
            self.class2path_image = preloaded_data['class2path_image']
            self.id2path = preloaded_data['id2path']
            # Test part
            self.class2id = preloaded_data['class2id']
            self.id2class = TEST_CLASS
            self.class2imgid_test = preloaded_data['class2imgid_test']
            self.class2path_sketch_test = preloaded_data['class2path_sketch_test']
            self.class2path_image_test = preloaded_data['class2path_image_test']
            self.path2class_sketch_test = preloaded_data['path2class_sketch_test']
            self.path2class_image_test = preloaded_data['path2class_image_test']
            # Shared part
            self.loaded_image = preloaded_data['loaded_image']
            return
        self.id2class = TEST_CLASS
        self.class2id = dict()
        for idx, cls in enumerate(self.id2class):
            self.class2id[cls] = idx

        self.class2imgid, self.path2class_sketch, self.class2path_sketch, self.path2class_image, self.class2path_image = \
            self.load_stats(self.stats_file_train, TRAIN_CLASS, self.sketch_files_train, self.image_files_train)
        
        self.class2imgid_test, self.path2class_sketch_test, self.class2path_sketch_test, self.path2class_image_test, self.class2path_image_test = \
            self.load_stats(self.stats_file_test, TEST_CLASS, self.sketch_files_test, self.image_files_test)

        for path in self.path2class_sketch.keys():
            self.loaded_image[path] = self.load_each_image(path)
            self.id2path.append(path)

        for path in self.path2class_image.keys():
            self.loaded_image[path] = self.load_each_image(path)
        
        for path in self.path2class_sketch_test.keys():
            self.loaded_image[path] = self.load_each_image(path)

        for path in self.path2class_image_test.keys():
            self.loaded_image[path] = self.load_each_image(path)
        
        assert len(self.id2path) == len(self.path2class_sketch.keys())
        preloaded_data = dict()
        # Train part
        preloaded_data['class2imgid'] = self.class2imgid
        preloaded_data['path2class_sketch'] = self.path2class_sketch
        preloaded_data['class2path_sketch'] = self.class2path_sketch
        preloaded_data['path2class_image'] = self.path2class_image
        preloaded_data['class2path_image'] = self.class2path_image
        preloaded_data['id2path'] = self.id2path
        # Test part
        preloaded_data['class2id'] = self.class2id
        preloaded_data['class2imgid_test'] = self.class2imgid_test
        preloaded_data['class2path_sketch_test'] = self.class2path_sketch_test
        preloaded_data['class2path_image_test'] = self.class2path_image_test
        preloaded_data['path2class_sketch_test'] = self.path2class_sketch_test
        preloaded_data['path2class_image_test'] = self.path2class_image_test
        # Shared part
        preloaded_data['loaded_image'] = self.loaded_image
        
        with open(self.loaded_data, 'wb') as f:
            pickle.dump(preloaded_data, f)
        return

    def load_each_image(self, path):
        img = cv2.imread(path)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], 2)
        else:
            img = img.copy()[:, :, ::-1]
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img
    
    def load_each_image_use(self, path):
        image = self.loaded_image[path]
        image = image.copy()[:,:,::-1]
        image = image.reshape(3, IMAGE_SIZE, IMAGE_SIZE)
        image = torch.Tensor(image)
        image = image/255.0
        image = self.Normalize(image)
        #print(image)
        return image

    def load_test_images(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_image_test.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_image_test[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label
    
    def load_test_sketch(self, batch_size=512):
        ims = []
        label = []
        for path in self.path2class_sketch_test.keys():
            ims.append(self.load_each_image_use(path))
            label.append(self.path2class_sketch_test[path])
            if len(ims) == batch_size:
                yield torch.stack(ims), label
                ims = []
                label = []
        yield torch.stack(ims), label
    
    def load_stats(self, stats_file, part_set, sketch_files, image_files):
        class2imgid = dict()
        path2class_sketch = dict()
        class2path_sketch = dict()
        path2class_image = dict()
        class2path_image = dict()
        with open(stats_file, 'r') as stats_in:
            stats_in_reader = csv.reader(stats_in)
            _header = next(stats_in_reader)
            for line in stats_in_reader:
                if line[1] not in class2imgid:
                    class2imgid[line[1]] = set()
                class2imgid[line[1]].add(line[2])
        iter = 0
        b_time = time.time()
        for key, value in class2imgid.items():
            iter += 1
            for id in value:
                pattern = '.*' + id + '.*'
                tmp_sketchs = match_filename(pattern, sketch_files)
                tmp_images = match_filename(pattern, image_files)
                for tmp_sketch in tmp_sketchs:
                    sketch_files.remove(tmp_sketch)
                    path2class_sketch[tmp_sketch] = key
                    if key not in class2path_sketch:
                        class2path_sketch[key] = set()
                    class2path_sketch[key].add(tmp_sketch)
                for tmp_image in tmp_images:
                    image_files.remove(tmp_image)
                    path2class_image[tmp_image] = key
                    if key not in class2path_image:
                        class2path_image[key] = set()
                    class2path_image[key].add(tmp_image)
            print('Loaded {}, {}/{}, spend {:.2f} seconds'.format(key, iter, len(part_set), time.time()-b_time))
            b_time = time.time()
        return class2imgid, path2class_sketch, class2path_sketch, path2class_image, class2path_image
    
    def test(self):
        for cls, paths in self.class2path_image.items():
            for item in list(paths):
                print(cls)
                im1 = cv2.imread(item)
                cv2.imshow('test', im1)
                cv2.waitKey(6000)
                cv2.destroyAllWindows()
