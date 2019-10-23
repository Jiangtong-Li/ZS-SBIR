import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='cmt_test', help='The directory to save the model and logs')
    parser.add_argument('--h', type=int, default=500,
                        help='Size of hidden layer of default network. Ignored if set back_bone to vgg.')
    parser.add_argument('--back_bone', type=str, default='default',
                        help='Backbone of the network, default or vgg.')
    parser.add_argument('--sketch_dir', type=str,
                        help='The directory of sketches. The directory can be defined in dataset/data_san.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--bn', type=bool, default=True,
                        help='Use BN or not.')
    parser.add_argument('--sz', type=int, default=32,
                        help='Size to the image to be fit into.')
    parser.add_argument('--image_dir', type=str,
                        help='The directory of images. The directory can be defined in dataset/data_san.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--npy_dir', type=str,
                        help='The npy files directory. By set it 0 to load the default folder provided in '
                             'dataset/data_san.py. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default=None)
    parser.add_argument('--path_semantic', type=str,
                        help='The directory of semantics, a dict object. The directory can be defined in dataset/data_dsh.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--start_from', type=str, default=None,
                        help='The iteration the training starts from or the exact model file name. '
                             'Default: None -- Try to start from the latest checkpoint.')
    parser.add_argument('--paired', type=int, default=0,
                        help='Whether paired data must be given.')
    parser.add_argument('--ni_path', type=str, default=None,
                        help='A pkl file path. The object should be a dict:'
                             ' names[\'class_name\'][\'im\'/\'st\'/\'sk\'][i] = filename_without_postfix_of_the_class_image'
                             'Set it to \'to use default path defined in dataset/data_san.py.')
    parser.add_argument('--ft', type=int, default=0,
                        help='Fine-tune VGG or not. Work only when using VGG.')
    parser.add_argument('--print_every', type=int, default=1, help='Number of epochs to print information')
    parser.add_argument('--save_every', type=int, default=1, help='Number of epochs to save model')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--l2_reg', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='sketchy', help='[sketchy] or [tuberlin]')
    return parser.parse_args()
